package tensor

import (
	"math"
	"runtime"

	"github.com/fiber/ai/internal/blas"
	"github.com/fiber/ai/internal/kernel"
	"github.com/fiber/ai/internal/parallel"
)

// Attention building blocks. Masks are additive: 0 where attention is
// allowed and a large negative number where it is not, so that softmax
// gives those positions zero weight. Adding a mask broadcasts over batch
// and heads and needs no dedicated kernel.

const maskNeg = -1e9

// CausalMask returns an [n×n] additive mask that lets position i attend
// to positions ≤ i only.
func CausalMask(n int) *Tensor {
	m := newTensor(Shape{n, n})
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			m.data[i*n+j] = maskNeg
		}
	}
	return m
}

// PaddingMask returns a [len(lengths)×1×1×n] additive mask that hides the
// positions at or beyond each sequence's length, for batches padded to n.
func PaddingMask(lengths []int, n int) *Tensor {
	b := len(lengths)
	m := newTensor(Shape{b, 1, 1, n})
	for i, l := range lengths {
		for j := max(0, l); j < n; j++ {
			m.data[i*n+j] = maskNeg
		}
	}
	return m
}

// Attention is scaled dot-product attention: softmax(q·kᵀ/√d + mask)·v.
// q is [..., T, D], k and v are [..., S, D] with matching leading
// dimensions (typically batch and heads); mask is nil or broadcastable
// to [..., T, S]. The result is [..., T, D].
//
// Two implementations share the interface. While a gradient is being
// recorded the operation is composed from products and a softmax, whose
// backward passes autograd provides. Otherwise (NoGrad, or inputs that
// need no gradient) a fused path computes the scores of 64 query rows at
// a time, runs their softmax in cache and multiplies them with v, never
// materialising the [T×S] score matrix: for [8×8×512×64] that is 64 MiB
// per call that is not written, read three times and collected.
func Attention(q, k, v, mask *Tensor) *Tensor {
	d := q.shape[len(q.shape)-1]
	return AttentionScaled(q, k, v, mask, float32(1/math.Sqrt(float64(d))))
}

// AttentionScaled is Attention with an explicit score scale instead of the
// default 1/√d. Gemma-class models scale by query_pre_attn_scalar^-0.5,
// which differs from 1/√d when that hyper-parameter is not the head
// dimension. Grouped-query attention is expressed by giving k and v fewer
// heads than q via Expand (stride 0 on the head dimension), so several
// query heads read one key/value head with no copy.
func AttentionScaled(q, k, v, mask *Tensor, scale float32) *Tensor {
	nd := len(q.shape)
	if nd < 2 || len(k.shape) != nd || len(v.shape) != nd {
		fail("Attention", "q, k, v must have the same rank ≥ 2, got %v %v %v", q.shape, k.shape, v.shape)
	}
	d := q.shape[nd-1]
	if k.shape[nd-1] != d || v.shape[nd-2] != k.shape[nd-2] {
		fail("Attention", "shapes do not match: q %v, k %v, v %v", q.shape, k.shape, v.shape)
	}
	needGrad := GradEnabled() && (q.requiresGrad || k.requiresGrad || v.requiresGrad || mask != nil && mask.requiresGrad)
	if !needGrad {
		if out := attentionFused(q, k, v, mask, scale); out != nil {
			return out
		}
	}
	return attentionComposed(q, k, v, mask, scale)
}

func attentionComposed(q, k, v, mask *Tensor, scale float32) *Tensor {
	scores := q.MatMul(k.Transpose(-2, -1)).MulScalar(scale)
	if mask != nil {
		scores = scores.Add(mask)
	}
	return scores.Softmax(-1).MatMul(v)
}

// WindowMask returns an [n×n] additive mask for bidirectional sliding-window
// attention: 0 where |i−j| < w and a large negative number elsewhere, so a
// position attends only to the w−1 neighbours on each side. It composes
// with a padding mask by addition, like CausalMask.
func WindowMask(n, w int) *Tensor {
	m := newTensor(Shape{n, n})
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if j-i >= w || i-j >= w {
				m.data[i*n+j] = maskNeg
			}
		}
	}
	return m
}

// attentionRows is the number of query rows one fused task handles. The
// per-task working set no longer depends on it (see blas.AttentionBlock),
// so it only balances the load: 256 rows unless that leaves fewer than
// four tasks per worker, then halved down to 32.
func attentionRows(nb, T int) int {
	block := 256
	for block > 32 && nb*((T+block-1)/block) < 4*parallel.Workers() {
		block /= 2
	}
	return block
}

// attentionPackBudget bounds the packed keys and values held at once
// (floats); heads are processed in groups that fit.
const attentionPackBudget = 16 << 20

// attentionFused returns nil when the shapes need the composed path
// (leading dimensions that differ between q, k and v, or a mask whose
// key dimension is not contiguous). Otherwise it packs K and V once per
// head (heads that share storage, as grouped-query Expand views do, share
// the packing) and runs blas.AttentionBlock over (head, row block) tasks.
func attentionFused(q, k, v, mask *Tensor, scale float32) *Tensor {
	nd := len(q.shape)
	lead := q.shape[:nd-2]
	if !k.shape[:nd-2].Equal(lead) || !v.shape[:nd-2].Equal(lead) || v.shape[nd-1] != k.shape[nd-1] {
		return nil
	}
	T, D, S := q.shape[nd-2], q.shape[nd-1], k.shape[nd-2]
	if T == 0 || S == 0 || D == 0 {
		return nil
	}
	full := append(lead.clone(), T, S)
	var ms []int // mask strides over the full score shape, 0 where broadcast
	if mask != nil {
		if len(mask.shape) > len(full) {
			return nil
		}
		defer func() { recover() }() // broadcastStrides fails on incompatible shapes: composed path then
		ms = broadcastStrides(mask, full)
		if ms[len(ms)-1] != 1 && ms[len(ms)-1] != 0 {
			return nil
		}
	}
	nb := lead.Size()
	if nb == 0 {
		return newTensorUninit(q.shape)
	}
	// per leading index: element offsets into q, k, v, out and the mask
	qoff, koff, voff := make([]int, nb), make([]int, nb), make([]int, nb)
	moff := make([]int, nb)
	strides := [][]int{append(append([]int(nil), q.strides[:nd-2]...), 0), append(append([]int(nil), k.strides[:nd-2]...), 0), append(append([]int(nil), v.strides[:nd-2]...), 0)}
	if mask != nil {
		strides = append(strides, append(append([]int(nil), ms[:nd-2]...), 0))
	}
	walkRows(append(lead.clone(), 1), strides, 0, nb, func(b int, offs []int) {
		qoff[b], koff[b], voff[b] = offs[0], offs[1], offs[2]
		if mask != nil {
			moff[b] = offs[3]
		}
	})
	out := newTensorUninit(q.shape)
	qd, kd, vd := q.data, k.data, v.data
	var md []float32
	var mRowStride, mColStride int
	if mask != nil {
		md = mask.data
		mRowStride, mColStride = ms[nd-2], ms[nd-1]
	}

	workers := parallel.Workers()
	block := attentionRows(nb, T)
	blocks := (T + block - 1) / block
	per := blas.PackedKVSize(S, D)
	headsPerGroup := max(1, attentionPackBudget/per)
	buf := blas.GetBuf(headsPerGroup * per)
	defer blas.PutBuf(buf)
	type kvKey struct{ k, v int }
	for b0 := 0; b0 < nb; {
		// Take heads while their distinct (k, v) storages fit the budget.
		idx := map[kvKey]int{}
		keys := []kvKey{}
		b1 := b0
		for ; b1 < nb; b1++ {
			kk := kvKey{koff[b1], voff[b1]}
			if _, ok := idx[kk]; !ok {
				if len(keys) == headsPerGroup {
					break
				}
				idx[kk] = len(keys)
				keys = append(keys, kk)
			}
		}
		packs := make([]blas.PackedKV, len(keys))
		packOne := func(i, w int) {
			kk := keys[i]
			kt := blas.Mat{Data: kd[kk.k:], Rows: D, Cols: S, RS: k.strides[nd-1], CS: k.strides[nd-2]}
			vb := blas.Mat{Data: vd[kk.v:], Rows: S, Cols: D, RS: v.strides[nd-2], CS: v.strides[nd-1]}
			packs[i] = blas.PackKV(buf[i*per:(i+1)*per], kt, vb, w)
		}
		if len(keys) >= workers {
			parallel.For(len(keys), func(i int) { packOne(i, 1) })
		} else {
			for i := range keys {
				packOne(i, workers)
			}
		}
		heads := b1 - b0
		parallel.For(heads*blocks, func(task int) {
			b, blk := b0+task/blocks, task%blocks
			i0 := blk * block
			rows := min(block, T-i0)
			p := packs[idx[kvKey{koff[b], voff[b]}]]
			qb := blas.Mat{Data: qd[qoff[b]+i0*q.strides[nd-2]:], Rows: rows, Cols: D, RS: q.strides[nd-2], CS: q.strides[nd-1]}
			ob := blas.Mat{Data: out.data[(b*T+i0)*D:], Rows: rows, Cols: D, RS: D, CS: 1}
			var addMask func(r int, row []float32, invScale float32, k0 int)
			if md != nil {
				addMask = func(r int, row []float32, invScale float32, k0 int) {
					base := moff[b] + (i0+r)*mRowStride
					if mColStride == 1 {
						kernel.Axpy(invScale, md[base+k0:base+k0+len(row)], row)
					} else { // one mask value for the whole row
						kernel.AddScalar(row, md[base]*invScale, row)
					}
				}
			}
			blas.AttentionBlock(ob, qb, p, scale, addMask)
		})
		b0 = b1
	}
	runtime.KeepAlive(q)
	runtime.KeepAlive(k)
	runtime.KeepAlive(v)
	runtime.KeepAlive(mask)
	return out
}

// RMSNorm normalises the last dimension by its root mean square,
// x / √(mean(x²) + eps), and scales by g (size of the last dimension):
// the normalisation Gemma- and Llama-class models use.
func RMSNorm(x, g *Tensor, eps float32) *Tensor {
	nd := len(x.shape)
	if nd == 0 {
		fail("RMSNorm", "expected at least a 1-D input")
	}
	n := x.shape[nd-1]
	if g.size != n {
		fail("RMSNorm", "g must have %d elements, got %d", n, g.size)
	}
	if !GradEnabled() || !(x.requiresGrad || g.requiresGrad) {
		return rmsNormFused(x, g, eps)
	}
	rms := x.Square().Mean(-1).AddScalar(eps).Sqrt().Unsqueeze(-1)
	return x.Div(rms).Mul(g)
}

// rmsNormFused computes RMSNorm in one pass into a single output buffer,
// for inference. Composed RMSNorm allocates three full-size intermediates
// per call; a 24-layer encoder calls it 146 times, and those buffers are
// the bulk of its allocation traffic.
func rmsNormFused(x, g *Tensor, eps float32) *Tensor {
	xc := x
	if !x.IsContiguous() {
		xc = x.Contiguous()
	}
	n := x.shape[len(x.shape)-1]
	rows := xc.size / n
	out := newTensorUninit(xc.shape)
	xd, od, gd := xc.data, out.data, g.values()
	parallel.Range(rows, 1<<12/n+1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := xd[r*n : r*n+n]
			var ss float32
			for _, v := range row {
				ss += v * v
			}
			inv := float32(1 / math.Sqrt(float64(ss/float32(n)+eps)))
			dst := od[r*n : r*n+n]
			for i, v := range row {
				dst[i] = v * inv * gd[i]
			}
		}
	})
	runtime.KeepAlive(g)
	return out
}
