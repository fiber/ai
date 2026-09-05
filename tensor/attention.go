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
		if out := attentionFused(q, k, v, mask); out != nil {
			return out
		}
	}
	return attentionComposed(q, k, v, mask)
}

func attentionComposed(q, k, v, mask *Tensor) *Tensor {
	d := q.shape[len(q.shape)-1]
	scores := q.MatMul(k.Transpose(-2, -1)).MulScalar(float32(1 / math.Sqrt(float64(d))))
	if mask != nil {
		scores = scores.Add(mask)
	}
	return scores.Softmax(-1).MatMul(v)
}

// attentionRows is the number of query rows one fused task handles: as
// many as keep the [rows × S] scores within about 512 KiB (256 rows at
// S = 512, which measured best on the M2 Pro: fewer rows repack K and V
// more often, more spill L2), never fewer than 32.
func attentionRows(S int) int {
	return max(32, min(256, (512<<10)/(4*S)))
}

// attentionFused returns nil when the shapes need the composed path
// (leading dimensions that differ between q, k and v, or a mask whose
// key dimension is not contiguous).
func attentionFused(q, k, v, mask *Tensor) *Tensor {
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
	scale := float32(1 / math.Sqrt(float64(D)))
	block := attentionRows(S)
	blocks := (T + block - 1) / block
	parallel.For(nb*blocks, func(task int) {
		b, blk := task/blocks, task%blocks
		i0 := blk * block
		rows := min(block, T-i0)
		scratch := blas.GetBuf(rows * S)
		defer blas.PutBuf(scratch)
		scores := blas.Contiguous(scratch, rows, S)
		qb := blas.Mat{Data: qd[qoff[b]+i0*q.strides[nd-2]:], Rows: rows, Cols: D, RS: q.strides[nd-2], CS: q.strides[nd-1]}
		kt := blas.Mat{Data: kd[koff[b]:], Rows: D, Cols: S, RS: k.strides[nd-1], CS: k.strides[nd-2]}
		blas.GemmZeroWorkers(scores, qb, kt, 1)
		for r := 0; r < rows; r++ {
			row := scratch[r*S : (r+1)*S]
			kernel.Scale(row, scale, row)
			if md != nil {
				base := moff[b] + (i0+r)*mRowStride
				if mColStride == 1 {
					kernel.Add(row, md[base:base+S], row)
				} else { // one mask value for the whole row
					kernel.AddScalar(row, md[base], row)
				}
			}
			kernel.AddScalar(row, -kernel.Max(row), row)
			kernel.Exp(row, row)
			kernel.Scale(row, 1/kernel.Sum(row), row)
		}
		ob := blas.Mat{Data: out.data[(b*T+i0)*D:], Rows: rows, Cols: D, RS: D, CS: 1}
		vb := blas.Mat{Data: vd[voff[b]:], Rows: S, Cols: D, RS: v.strides[nd-2], CS: v.strides[nd-1]}
		blas.GemmZeroWorkers(ob, scores, vb, 1)
	})
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
	rms := x.Square().Mean(-1).AddScalar(eps).Sqrt().Unsqueeze(-1)
	return x.Div(rms).Mul(g)
}
