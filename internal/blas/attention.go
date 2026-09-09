package blas

import (
	"math"
	"os"

	"github.com/fiber/ai/internal/kernel"
)

// PackedKV holds one attention head's keys and values in micro-kernel
// panel layout: Kᵀ as NR-wide key panels of depth D (the B operand of
// Q·Kᵀ, keys zero-padded to SPad) and V as NR-wide column panels of depth
// S (the B operand of P·V, columns zero-padded to DPad). A head is packed
// once and read by every block of query rows that attends to it.
type PackedKV struct {
	K, V       []float32
	S, D       int
	SPad, DPad int
}

// PackedKVSize returns the floats PackKV needs for a head with S keys of
// dimension D.
func PackedKVSize(S, D int) int {
	nr := kernel.NR
	return roundUp(S, nr)*D + roundUp(D, nr)*S
}

// PackKV packs one head into buf (at least PackedKVSize(S, D) floats): kt
// is Kᵀ as a [D×S] Mat (a transposed view of the keys), v is [S×D].
// workers bounds the goroutines used for the panels.
func PackKV(buf []float32, kt, v Mat, workers int) PackedKV {
	nr := kernel.NR
	D, S := kt.Rows, kt.Cols
	if v.Rows != S || v.Cols != D {
		panic("blas: PackKV: v must be [S×D] for a [D×S] kt")
	}
	sPad, dPad := roundUp(S, nr), roundUp(D, nr)
	p := PackedKV{K: buf[:sPad*D], V: buf[sPad*D : sPad*D+dPad*S], S: S, D: D, SPad: sPad, DPad: dPad}
	packB(p.K, kt, 0, 0, D, S, nr, workers)
	packB(p.V, v, 0, 0, S, D, nr, workers)
	return p
}

// AttentionBlock computes out = softmax(scale·q·Kᵀ + mask)·V for a block
// of query rows against one packed head, straight from the micro-kernel:
// the query rows are packed once, then per MR-row group the scores
// [MR × SPad] are computed into L1-sized scratch, the softmax runs on
// them (scale, mask, max, one fused exp-and-sum pass), the probabilities
// are packed as the A operand and multiplied with the packed V, and the
// rows go to out scaled by their 1/Σ (a D-wide pass instead of an S-wide
// one). q is [rows×D] with any strides; out is [rows×D] with CS 1. mask,
// when not nil, adds invScale times the mask of block row r into the S
// raw scores of that row (the scale is applied afterwards), k0 being the
// first key of that slice. Nothing here depends on rows, so the caller
// chooses the block size for load balance alone.
//
// Where the back-end has a micro-kernel reading its left operand
// row-major (kernel.GemmRM: AVX2, AVX-512), the keys are processed in
// blocks with an online softmax and nothing is packed per row group (see
// attentionBlockRM); elsewhere (NEON, AMX, generic) the whole key range
// is scored at once and the probabilities are packed.
func AttentionBlock(out, q Mat, kv PackedKV, scale float32, mask func(r int, row []float32, invScale float32, k0 int)) {
	mr, nr := kernel.MR, kernel.NR
	rows, S, D := q.Rows, kv.S, kv.D
	if q.Cols != D || out.Rows != rows || out.Cols != D || out.CS != 1 {
		panic("blas: AttentionBlock: shape mismatch")
	}
	if rows == 0 {
		return
	}
	if kernel.GemmRM != nil && scale > 0 && !attentionPacked {
		attentionBlockRM(out, q, kv, scale, mask)
		return
	}
	sPad, dPad := kv.SPad, kv.DPad
	rPad := roundUp(rows, mr)
	qp := getBuf(rPad * D)[:rPad*D]   // A panels of q, depth D
	sc := getBuf(mr * sPad)[:mr*sPad] // scores, then probabilities, of one row group
	pp := getBuf(mr * S)[:mr*S]       // A panel of the probabilities, depth S
	ot := getBuf(mr * dPad)[:mr*dPad] // output tile of one row group
	defer putBuf(qp)
	defer putBuf(sc)
	defer putBuf(pp)
	defer putBuf(ot)
	var inv [256]float32 // 1/Σ per row of the group (MR ≤ 256 on every back-end)

	if kernel.GemmHooks {
		kernel.GemmBegin()
		defer kernel.GemmEnd()
	}
	packA(qp, q, 0, 0, rows, D, mr)
	gemmZero := kernel.GemmZero
	clearFirst := gemmZero == nil
	if clearFirst {
		gemmZero = kernel.Gemm
	}
	pm := Mat{Data: sc, Cols: S, RS: sPad, CS: 1}
	for ir := 0; ir < rPad; ir += mr {
		g := min(mr, rows-ir) // real rows in this group
		if clearFirst {
			clear(sc)
		}
		for jr := 0; jr < sPad; jr += nr {
			gemmZero(D, &qp[ir*D], &kv.K[jr*D], &sc[jr], sPad)
		}
		for r := 0; r < g; r++ {
			// softmax(scale·s + m) = exp(scale·(s + m/scale) − scale·max) / Σ:
			// the mask joins the raw scores, the max is taken once, and
			// ExpSum applies the scale and the shift in its own pass.
			row := sc[r*sPad : r*sPad+S]
			a, invScale := scale, float32(1)
			if scale > 0 {
				invScale = 1 / scale
			} else { // max(scale·s) ≠ scale·max(s): scale first
				kernel.Scale(row, scale, row)
				a = 1
			}
			if mask != nil {
				mask(ir+r, row, invScale, 0)
			}
			m := kernel.Max(row)
			inv[r] = 1 / kernel.ExpSum(row, row, a, -a*m)
		}
		pm.Rows = g
		packA(pp, pm, 0, 0, g, S, mr) // rows g..mr zero-filled
		if clearFirst {
			clear(ot)
		}
		for jr := 0; jr < dPad; jr += nr {
			gemmZero(S, &pp[0], &kv.V[jr*S], &ot[jr], dPad)
		}
		for r := 0; r < g; r++ {
			dst := out.Data[(ir+r)*out.RS : (ir+r)*out.RS+D]
			kernel.Scale(ot[r*dPad:r*dPad+D], inv[r], dst)
		}
	}
}

// attentionPacked forces the packed path on back-ends that have the
// row-major one (FIBERAI_ATTENTION=packed), for measuring the two against
// each other on the same machine.
var attentionPacked = os.Getenv("FIBERAI_ATTENTION") == "packed"

// attentionKeyBlock is the number of keys one online-softmax step covers
// (rounded up to NR): MR × 128 scores are 7 KB on AVX-512, L1-resident.
const attentionKeyBlock = 128

// attentionBlockRM is AttentionBlock for back-ends with a row-major
// micro-kernel: the query rows are copied once into a contiguous tile;
// per MR-row group the keys are walked in blocks, each block's scores
// computed straight into L1-sized scratch, the softmax kept running
// (max m, sum l, the output accumulator rescaled when the max moves),
// and the probabilities multiplied with the block's packed V columns as
// they are, row-major. Rows leave scaled by 1/l.
func attentionBlockRM(out, q Mat, kv PackedKV, scale float32, mask func(r int, row []float32, invScale float32, k0 int)) {
	mr, nr := kernel.MR, kernel.NR
	rows, S, D := q.Rows, kv.S, kv.D
	dPad := kv.DPad
	rPad := roundUp(rows, mr)
	kb := roundUp(attentionKeyBlock, nr)
	qs := getBuf(rPad * D)[:rPad*D]   // query rows, row-major, zero beyond rows
	sc := getBuf(mr * kb)[:mr*kb]     // one key block's scores, then probabilities
	ot := getBuf(mr * dPad)[:mr*dPad] // output accumulator of one row group
	defer putBuf(qs)
	defer putBuf(sc)
	defer putBuf(ot)
	for r := 0; r < rows; r++ {
		dst := qs[r*D : (r+1)*D]
		if q.CS == 1 {
			copy(dst, q.Data[r*q.RS:r*q.RS+D])
		} else {
			for d := range dst {
				dst[d] = q.Data[r*q.RS+d*q.CS]
			}
		}
	}
	clear(qs[rows*D:])
	var m, l [256]float32 // running max and sum per row of the group (MR ≤ 256)
	invScale := 1 / scale
	negInf := float32(math.Inf(-1))

	if kernel.GemmHooks {
		kernel.GemmBegin()
		defer kernel.GemmEnd()
	}
	for ir := 0; ir < rPad; ir += mr {
		g := min(mr, rows-ir)
		clear(ot)
		for r := 0; r < mr; r++ {
			m[r], l[r] = negInf, 0
		}
		for k0 := 0; k0 < S; k0 += kb {
			n := min(kb, S-k0)     // keys in this block
			nPad := roundUp(n, nr) // panels covering them (K is packed to SPad)
			for jr := 0; jr < nPad; jr += nr {
				kernel.GemmRMZero(D, &qs[ir*D], D, &kv.K[(k0+jr)*D], &sc[jr], kb)
			}
			for r := 0; r < g; r++ {
				row := sc[r*kb : r*kb+n]
				if mask != nil {
					mask(ir+r, row, invScale, k0)
				}
				mNew := max(m[r], kernel.Max(row))
				if mNew == negInf { // every key of the block masked: contributes nothing
					clear(row)
					continue
				}
				if m[r] != mNew {
					if m[r] != negInf {
						corr := float32(math.Exp(float64(scale * (m[r] - mNew))))
						l[r] *= corr
						orow := ot[r*dPad : r*dPad+D]
						kernel.Scale(orow, corr, orow)
					}
					m[r] = mNew
				}
				l[r] += kernel.ExpSum(row, row, scale, -scale*mNew)
			}
			// rows g..mr of sc hold the scores of zero query rows (0): finite, never copied out
			for jr := 0; jr < dPad; jr += nr {
				kernel.GemmRM(n, &sc[0], kb, &kv.V[jr*S+k0*nr], &ot[jr], dPad)
			}
		}
		for r := 0; r < g; r++ {
			dst := out.Data[(ir+r)*out.RS : (ir+r)*out.RS+D]
			kernel.Scale(ot[r*dPad:r*dPad+D], 1/l[r], dst)
		}
	}
}
