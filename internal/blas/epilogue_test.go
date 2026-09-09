package blas

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/fiber/ai/internal/kernel"
)

// reference applies the epilogue to a plain product with scalar code.
func reference(c, a, b Mat, e Epilogue) []float32 {
	m, n, k := a.Rows, b.Cols, a.Cols
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for p := 0; p < k; p++ {
				s += float64(a.at(i, p)) * float64(b.at(p, j))
			}
			v := float32(s)
			if e.Bias != nil {
				v += e.Bias[j]
			}
			if e.RowScale != nil {
				v *= e.RowScale[i]
			}
			switch e.Act {
			case ActReLU:
				if v < 0 {
					v = 0
				}
			case ActGELU:
				z := []float32{0}
				kernel.GELU([]float32{v}, z)
				v = z[0]
			}
			if e.Mul.Data != nil {
				v *= e.Mul.at(i, j)
			}
			if e.Residual.Data != nil {
				v += e.Residual.at(i, j)
			}
			out[i*n+j] = v
		}
	}
	return out
}

func TestEpilogueMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 9))
	randMat := func(r, c int) Mat {
		d := make([]float32, r*c)
		for i := range d {
			d[i] = rng.Float32()*2 - 1
		}
		return Contiguous(d, r, c)
	}
	savedKC := KC
	defer func() { KC = savedKC }()
	shapes := [][3]int{{1, 64, 300}, {5, 700, 3}, {33, 130, 77}, {64, 1100, 96}, {200, 64, 1}, {7, 5, 9}}
	for _, kc := range []int{savedKC, 64} { // 64: several K blocks, the epilogue must run once
		KC = kc
		for _, sh := range shapes {
			m, k, n := sh[0], sh[1], sh[2]
			a, b := randMat(m, k), randMat(k, n)
			bias := make([]float32, n)
			scale := make([]float32, m)
			for i := range bias {
				bias[i] = rng.Float32()
			}
			for i := range scale {
				scale[i] = 0.5 + rng.Float32()
			}
			mul, res := randMat(m, n), randMat(m, n)
			cases := []Epilogue{
				{Bias: bias},
				{Act: ActReLU},
				{Act: ActGELU},
				{Bias: bias, Act: ActReLU},
				{RowScale: scale, Act: ActGELU, Mul: mul},
				{Bias: bias, RowScale: scale, Act: ActGELU, Mul: mul, Residual: res},
				{Residual: res},
			}
			for ci, e := range cases {
				for _, workers := range []int{1, 4} {
					c := Contiguous(make([]float32, m*n), m, n)
					GemmZeroEpilogue(c, a, b, nil, e, workers)
					want := reference(c, a, b, e)
					for i := range want {
						if d := math.Abs(float64(c.Data[i] - want[i])); d > 1e-4*(1+math.Abs(float64(want[i]))) {
							t.Fatalf("kc=%d shape %v case %d workers %d: C[%d] = %v, want %v", kc, sh, ci, workers, i, c.Data[i], want[i])
						}
					}
				}
			}
			// with a packed B
			p := PackB(b, 2)
			c := Contiguous(make([]float32, m*n), m, n)
			e := Epilogue{Bias: bias, Act: ActGELU, Residual: res}
			GemmZeroEpilogue(c, a, b, p, e, 3)
			want := reference(c, a, b, e)
			for i := range want {
				if d := math.Abs(float64(c.Data[i] - want[i])); d > 1e-4*(1+math.Abs(float64(want[i]))) {
					t.Fatalf("packed kc=%d shape %v: C[%d] = %v, want %v", kc, sh, i, c.Data[i], want[i])
				}
			}
		}
	}
}

func TestEpilogueRowsStrategy(t *testing.T) {
	saved := Strategy
	Strategy = StrategyRows
	defer func() { Strategy = saved }()
	rng := rand.New(rand.NewPCG(4, 4))
	m, k, n := 90, 300, 130
	ad, bd := make([]float32, m*k), make([]float32, k*n)
	for i := range ad {
		ad[i] = rng.Float32() - 0.5
	}
	for i := range bd {
		bd[i] = rng.Float32() - 0.5
	}
	a, b := Contiguous(ad, m, k), Contiguous(bd, k, n)
	bias := make([]float32, n)
	for i := range bias {
		bias[i] = float32(i) * 0.01
	}
	e := Epilogue{Bias: bias, Act: ActReLU}
	c := Contiguous(make([]float32, m*n), m, n)
	GemmZeroEpilogue(c, a, b, nil, e, 4)
	want := reference(c, a, b, e)
	for i := range want {
		if d := math.Abs(float64(c.Data[i] - want[i])); d > 1e-4*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("rows strategy: C[%d] = %v, want %v", i, c.Data[i], want[i])
		}
	}
}
