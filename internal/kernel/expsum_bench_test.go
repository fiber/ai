package kernel

import "testing"

// BenchmarkExpSum measures the fused exp-and-sum on a softmax-sized row.
func BenchmarkExpSum(b *testing.B) {
	x, z := make([]float32, 512), make([]float32, 512)
	for i := range x {
		x[i] = float32(i%97)*0.1 - 5
	}
	b.SetBytes(4 * 512)
	for b.Loop() {
		ExpSum(x, z, 0.125, -0.5)
	}
}
