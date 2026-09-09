package kernel

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestExpSumMatchesExp checks the fused kernel on every implementation:
// the outputs equal Exp's, the sum equals a float64 sum of them, lengths
// hit every tail path, and inputs below the clamp contribute exact zeros.
func TestExpSumMatchesExp(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, im := range implementations() {
		t.Run(im.name, func(t *testing.T) {
			for _, n := range []int{0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 100, 512, 1001} {
				x := make([]float32, n)
				for i := range x {
					x[i] = rng.Float32()*30 - 20
				}
				if n > 8 {
					x[2], x[7] = -300, -600 // a·x + b below the clamp: flushed to 0
				}
				a, b := float32(0.35), float32(-1.5)
				want := make([]float32, n)
				for i := range x {
					want[i] = a*x[i] + b
				}
				im.exp(want, want)
				got := make([]float32, n)
				s := im.expSum(x, got, a, b)
				var ref float64
				for i := range want {
					ref += float64(want[i])
					// a·x + b is one fused multiply-add in the vector routines and
					// two roundings in Go; an ulp in the argument moves exp by
					// |arg| ulps: allow the exp kernels' own 2e-6 class.
					if math.Abs(float64(got[i]-want[i])) > 2e-6*math.Abs(float64(want[i])) {
						t.Fatalf("n=%d i=%d: expSum wrote %v, exp %v", n, i, got[i], want[i])
					}
				}
				if math.Abs(float64(s)-ref) > 1e-5*(1+math.Abs(ref)) {
					t.Fatalf("n=%d: sum %v, want %v", n, s, ref)
				}
			}
		})
	}
}
