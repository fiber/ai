package tensor

import (
	"math/rand/v2"
	"testing"
)

func TestMatMulFusedMatchesComposed(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 8))
	for _, sh := range [][3]int{{1, 32, 40}, {17, 300, 70}, {64, 512, 128}} {
		m, k, n := sh[0], sh[1], sh[2]
		x, w := RandnFrom(rng, m, k), RandnFrom(rng, k, n)
		bias := RandnFrom(rng, n)
		mul, res := RandnFrom(rng, m, n), RandnFrom(rng, m, n)
		scale := make([]float32, m)
		for i := range scale {
			scale[i] = 0.5 + rng.Float32()
		}
		cases := []Fused{
			{Bias: bias},
			{Bias: bias, Act: ReLUAct},
			{Act: GELUAct, Mul: mul},
			{RowScale: scale, Act: GELUAct, Mul: mul},
			{Bias: bias, RowScale: scale, Act: ReLUAct, Residual: res},
		}
		for ci, f := range cases {
			var got *Tensor
			NoGrad(func() { got = MatMulFused(x, w, f) })
			want := fusedComposed(x, w, f)
			if !got.AllClose(want, 1e-4, 1e-5) {
				t.Fatalf("shape %v case %d: fused differs from composed", sh, ci)
			}
			// and twice more, so the packed-operand cache path is exercised
			NoGrad(func() {
				for i := 0; i < 2; i++ {
					got = MatMulFused(x, w, f)
				}
			})
			if !got.AllClose(want, 1e-4, 1e-5) {
				t.Fatalf("shape %v case %d: fused (cached operand) differs", sh, ci)
			}
		}
	}
}

func TestMatMulFusedGradientPath(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 9))
	x := RandnFrom(rng, 6, 20).SetRequiresGrad(true)
	w := RandnFrom(rng, 20, 9).SetRequiresGrad(true)
	b := RandnFrom(rng, 9).SetRequiresGrad(true)
	y := MatMulFused(x, w, Fused{Bias: b, Act: ReLUAct})
	ref := x.MatMul(w).Add(b).ReLU()
	if !y.AllClose(ref, 1e-5, 1e-6) {
		t.Fatal("fused product under grad differs from composed")
	}
	y.Sum().Backward()
	if x.Grad() == nil || w.Grad() == nil || b.Grad() == nil {
		t.Fatal("fused product under grad recording must give gradients to every operand")
	}
}

func TestRowSumSquares(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	ss := x.RowSumSquares()
	if ss[0] != 14 || ss[1] != 77 {
		t.Fatalf("got %v", ss)
	}
	if ss := x.T().RowSumSquares(); ss[0] != 17 || ss[1] != 29 || ss[2] != 45 {
		t.Fatalf("transposed: %v", ss)
	}
}
