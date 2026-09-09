package blas

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/fiber/ai/internal/kernel"
)

// attentionReference is the scalar definition, in float64.
func attentionReference(q, k, v []float32, rows, S, D int, scale float32, mask func(r, j int) float32) []float64 {
	out := make([]float64, rows*D)
	p := make([]float64, S)
	for r := 0; r < rows; r++ {
		m := math.Inf(-1)
		for j := 0; j < S; j++ {
			var s float64
			for d := 0; d < D; d++ {
				s += float64(q[r*D+d]) * float64(k[j*D+d])
			}
			s *= float64(scale)
			if mask != nil {
				s += float64(mask(r, j))
			}
			p[j] = s
			m = math.Max(m, s)
		}
		var sum float64
		for j := range p {
			p[j] = math.Exp(p[j] - m)
			sum += p[j]
		}
		for j := 0; j < S; j++ {
			w := p[j] / sum
			for d := 0; d < D; d++ {
				out[r*D+d] += w * float64(v[j*D+d])
			}
		}
	}
	return out
}

// TestAttentionBlockMatchesReference covers row counts that are not a
// multiple of MR, key counts that are not a multiple of NR, small and odd
// head dimensions, and a mask.
func TestAttentionBlockMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	shapes := []struct{ rows, S, D int }{{1, 1, 1}, {5, 7, 3}, {33, 65, 40}, {64, 512, 64}, {17, 100, 256}, {kernel.MR, kernel.NR, 8}, {9, 129, 64}, {3, 300, 16}, {kernel.MR + 1, 2*kernel.NR + 1, 3}}
	for _, sh := range shapes {
		rows, S, D := sh.rows, sh.S, sh.D
		q, k, v := make([]float32, rows*D), make([]float32, S*D), make([]float32, S*D)
		for _, a := range [][]float32{q, k, v} {
			for i := range a {
				a[i] = rng.Float32()*2 - 1
			}
		}
		for _, masked := range []bool{false, true} {
			var mask func(r int, row []float32, invScale float32, k0 int)
			var ref func(r, j int) float32
			if masked {
				mask = func(r int, row []float32, invScale float32, k0 int) {
					for j := range row {
						if k0+j > r && (k0+j)%3 == 0 {
							row[j] -= 1e9 * invScale
						}
					}
				}
				ref = func(r, j int) float32 {
					if j > r && j%3 == 0 {
						return -1e9
					}
					return 0
				}
			}
			buf := make([]float32, PackedKVSize(S, D))
			kt := Mat{Data: k, Rows: D, Cols: S, RS: 1, CS: D}
			kv := PackKV(buf, kt, Contiguous(v, S, D), 2)
			out := make([]float32, rows*D)
			scale := float32(1 / math.Sqrt(float64(D)))
			AttentionBlock(Contiguous(out, rows, D), Contiguous(q, rows, D), kv, scale, mask)
			want := attentionReference(q, k, v, rows, S, D, scale, ref)
			for i := range out {
				if math.Abs(float64(out[i])-want[i]) > 1e-4*(1+math.Abs(want[i])) {
					t.Fatalf("rows=%d S=%d D=%d masked=%v: out[%d] = %v, want %v", rows, S, D, masked, i, out[i], want[i])
				}
			}
		}
	}
}
