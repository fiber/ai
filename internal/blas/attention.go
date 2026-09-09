package blas

import (
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
// raw scores of that row (the scale is applied afterwards). Nothing here
// depends on rows, so the caller chooses the block size for load balance
// alone.
func AttentionBlock(out, q Mat, kv PackedKV, scale float32, mask func(r int, row []float32, invScale float32)) {
	mr, nr := kernel.MR, kernel.NR
	rows, S, D := q.Rows, kv.S, kv.D
	if q.Cols != D || out.Rows != rows || out.Cols != D || out.CS != 1 {
		panic("blas: AttentionBlock: shape mismatch")
	}
	if rows == 0 {
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
				mask(ir+r, row, invScale)
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
