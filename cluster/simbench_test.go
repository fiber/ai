package cluster

import (
	"math/rand/v2"
	"testing"

	"github.com/fiber/ai/tensor"
)

func BenchmarkSimilarities1000x10000(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	q := Normalize(tensor.RandnFrom(rng, 1000, 768))
	d := Normalize(tensor.RandnFrom(rng, 10000, 768))
	dt := d.T().Contiguous()
	b.Run("unit rows, transposed view", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			Similarities(q, d).Release()
		}
	})
	b.Run("unit rows, pre-transposed docs", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			q.MatMul(dt).Release()
		}
	})
	b.Run("Normalize docs only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			Normalize(d).Release()
		}
	})
}
