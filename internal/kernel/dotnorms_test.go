package kernel

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestDotNormsAccuracy checks the fused kernel against a float64 reference
// on every implementation, with lengths hitting all tail paths.
func TestDotNormsAccuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, im := range implementations() {
		t.Run(im.name, func(t *testing.T) {
			for _, n := range []int{0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 100, 768, 1001} {
				x, y := make([]float32, n), make([]float32, n)
				var dot, xx, yy float64
				for i := range x {
					x[i] = rng.Float32()*2 - 1
					y[i] = rng.Float32()*2 - 1
					dot += float64(x[i]) * float64(y[i])
					xx += float64(x[i]) * float64(x[i])
					yy += float64(y[i]) * float64(y[i])
				}
				d, a, b := im.dotNorms(x, y)
				for _, c := range []struct{ got, want float64 }{{float64(d), dot}, {float64(a), xx}, {float64(b), yy}} {
					if math.Abs(c.got-c.want) > 1e-4*(1+math.Abs(c.want)) {
						t.Fatalf("n=%d: got %v want %v", n, c.got, c.want)
					}
				}
			}
		})
	}
}

func BenchmarkDotNorms768(b *testing.B) {
	x, y := make([]float32, 768), make([]float32, 768)
	for i := range x {
		x[i], y[i] = float32(i%7), float32(i%5)
	}
	b.Run("fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			DotNorms(x, y)
		}
	})
	b.Run("three dots", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			Dot(x, y)
			Dot(x, x)
			Dot(y, y)
		}
	})
}
