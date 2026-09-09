package blas

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/fiber/ai/internal/kernel"
	"github.com/fiber/ai/internal/parallel"
)

func randSlice(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = rng.Float32()*2 - 1
	}
	return s
}

// refGemm is the float64 reference: C += A·B on arbitrary strides.
func refGemm(c, a, b Mat) []float64 {
	out := make([]float64, len(c.Data))
	for i, v := range c.Data {
		out[i] = float64(v)
	}
	for i := 0; i < c.Rows; i++ {
		for j := 0; j < c.Cols; j++ {
			s := out[i*c.RS+j*c.CS]
			for p := 0; p < a.Cols; p++ {
				s += float64(a.at(i, p)) * float64(b.at(p, j))
			}
			out[i*c.RS+j*c.CS] = s
		}
	}
	return out
}

func checkResult(t *testing.T, ctx string, c Mat, want []float64) {
	t.Helper()
	for i := 0; i < c.Rows; i++ {
		for j := 0; j < c.Cols; j++ {
			idx := i*c.RS + j*c.CS
			g, w := float64(c.Data[idx]), want[idx]
			if math.Abs(g-w) > 1e-4*(1+math.Abs(w)) {
				t.Fatalf("%s: C[%d,%d] = %g, want %g", ctx, i, j, g, w)
			}
		}
	}
	// nothing outside the matrix may be touched
	for i := range c.Data {
		if float64(c.Data[i]) != want[i] && !(math.Abs(float64(c.Data[i])-want[i]) <= 1e-4*(1+math.Abs(want[i]))) {
			t.Fatalf("%s: padding element %d modified: %g vs %g", ctx, i, c.Data[i], want[i])
		}
	}
}

// layout describes how a test matrix is stored.
type layout int

const (
	rowMajor   layout = iota // RS = cols+pad, CS = 1
	colMajor                 // transposed storage: RS = 1, CS = rows+pad
	oddStrides               // both strides > 1
)

func (l layout) String() string { return [...]string{"rowmajor", "colmajor", "strided"}[l] }

func makeMat(rng *rand.Rand, rows, cols int, l layout) Mat {
	switch l {
	case rowMajor:
		rs := cols + 3
		return Mat{Data: randSlice(rng, rows*rs+1), Rows: rows, Cols: cols, RS: rs, CS: 1}
	case colMajor:
		cs := rows + 2
		return Mat{Data: randSlice(rng, cols*cs+1), Rows: rows, Cols: cols, RS: 1, CS: cs}
	default:
		cs := 3
		rs := cols*cs + 5
		return Mat{Data: randSlice(rng, rows*rs+1), Rows: rows, Cols: cols, RS: rs, CS: cs}
	}
}

var shapes = [][3]int{
	{1, 1, 1}, {1, 1, 100}, {1, 17, 5}, {1, 100, 300}, {5, 1, 7}, {300, 1, 100}, {3, 3, 3},
	{4, 16, 1}, {6, 16, 7}, {8, 12, 3}, {12, 32, 9}, {13, 33, 9}, {5, 17, 9}, {7, 37, 19},
	{64, 64, 64}, {129, 65, 33}, {133, 257, 300}, {100, 20, 513}, {257, 33, 1}, {31, 4097, 7},
	{300, 300, 300},
}

func TestGemm(t *testing.T) {
	savedStrategy := Strategy
	defer func() { Strategy = savedStrategy }()
	for _, strategy := range []int{StrategyShared, StrategyRows} {
		Strategy = strategy
		t.Run([]string{"shared", "rows"}[strategy], testGemm)
	}
}

func testGemm(t *testing.T) {
	saved := ParallelThreshold
	ParallelThreshold = 0 // exercise the parallel path even for tiny shapes
	defer func() { ParallelThreshold = saved }()

	rng := rand.New(rand.NewPCG(1, 2))
	for _, s := range shapes {
		m, n, k := s[0], s[1], s[2]
		for _, la := range []layout{rowMajor, colMajor, oddStrides} {
			for _, lb := range []layout{rowMajor, colMajor, oddStrides} {
				for _, workers := range []int{1, 4} {
					if (la == oddStrides || lb == oddStrides) && m*n*k > 100000 {
						continue // slow reference; strided paths are covered by small shapes
					}
					a := makeMat(rng, m, k, la)
					b := makeMat(rng, k, n, lb)
					c := makeMat(rng, m, n, rowMajor)
					want := refGemm(c, a, b)
					GemmWorkers(c, a, b, workers)
					checkResult(t, fmt.Sprintf("m=%d n=%d k=%d A=%v B=%v workers=%d", m, n, k, la, lb, workers), c, want)
				}
			}
		}
	}
}

func TestGemmTransposedViews(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	m, n, k := 70, 45, 33
	a := Contiguous(randSlice(rng, m*k), m, k)
	b := Contiguous(randSlice(rng, k*n), k, n)
	// (A·B)ᵀ = Bᵀ·Aᵀ
	c1 := Contiguous(make([]float32, m*n), m, n)
	Gemm(c1, a, b)
	c2 := Contiguous(make([]float32, n*m), n, m)
	Gemm(c2, b.T(), a.T())
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if x, y := c1.Data[i*n+j], c2.Data[j*m+i]; math.Abs(float64(x-y)) > 1e-5*(1+math.Abs(float64(x))) {
				t.Fatalf("transposed product mismatch at %d,%d: %g vs %g", i, j, x, y)
			}
		}
	}
}

func TestGemmBlockingBoundaries(t *testing.T) {
	// Shapes that straddle the blocking parameters exercise every block edge.
	savedKC, savedMC, savedNC, savedPT := KC, MC, NC, ParallelThreshold
	KC, MC, NC, ParallelThreshold = 8, 16, 48, 0
	defer func() { KC, MC, NC, ParallelThreshold = savedKC, savedMC, savedNC, savedPT }()

	rng := rand.New(rand.NewPCG(5, 6))
	savedStrategy := Strategy
	defer func() { Strategy = savedStrategy }()
	for _, strategy := range []int{StrategyShared, StrategyRows} {
		Strategy = strategy
		for _, s := range [][3]int{{15, 47, 7}, {16, 48, 8}, {17, 49, 9}, {33, 97, 25}, {40, 100, 24}} {
			m, n, k := s[0], s[1], s[2]
			for _, workers := range []int{1, 3} {
				a := makeMat(rng, m, k, rowMajor)
				b := makeMat(rng, k, n, colMajor)
				c := makeMat(rng, m, n, rowMajor)
				want := refGemm(c, a, b)
				GemmWorkers(c, a, b, workers)
				checkResult(t, fmt.Sprintf("blocking m=%d n=%d k=%d workers=%d strategy=%d", m, n, k, workers, strategy), c, want)
			}
		}
	}
}

func TestGemmZeroSizes(t *testing.T) {
	a := Contiguous(nil, 0, 5)
	b := Contiguous(nil, 5, 0)
	c := Contiguous(nil, 0, 0)
	Gemm(c, a, b) // must not panic
	a = Contiguous(nil, 3, 0)
	b = Contiguous(nil, 0, 4)
	c = Contiguous(make([]float32, 12), 3, 4)
	Gemm(c, a, b)
	for _, v := range c.Data {
		if v != 0 {
			t.Fatal("k=0 product must leave C unchanged")
		}
	}
}

func TestGemmPanics(t *testing.T) {
	expectPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: expected panic", name)
			}
		}()
		f()
	}
	expectPanic("shape mismatch", func() {
		Gemm(Contiguous(make([]float32, 4), 2, 2), Contiguous(make([]float32, 6), 2, 3), Contiguous(make([]float32, 4), 2, 2))
	})
	expectPanic("buffer too small", func() {
		Gemm(Contiguous(make([]float32, 4), 2, 2), Contiguous(make([]float32, 3), 2, 2), Contiguous(make([]float32, 4), 2, 2))
	})
	expectPanic("non-unit C column stride", func() {
		Gemm(Mat{Data: make([]float32, 8), Rows: 2, Cols: 2, RS: 4, CS: 2}, Contiguous(make([]float32, 4), 2, 2), Contiguous(make([]float32, 4), 2, 2))
	})
}

func BenchmarkGemm(b *testing.B) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, n := range []int{64, 128, 256, 512, 1024, 2048} {
		x := Contiguous(randSlice(rng, n*n), n, n)
		y := Contiguous(randSlice(rng, n*n), n, n)
		z := Contiguous(make([]float32, n*n), n, n)
		for _, workers := range []int{1, parallel.Workers()} {
			if workers == 1 && parallel.Workers() == 1 {
				continue
			}
			b.Run(fmt.Sprintf("%s/%s/n=%d/workers=%d", kernel.Impl, []string{"shared", "rows"}[Strategy], n, workers), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					GemmWorkers(z, x, y, workers)
				}
				flops := 2 * float64(n) * float64(n) * float64(n) * float64(b.N)
				b.ReportMetric(flops/b.Elapsed().Seconds()/1e9, "GFLOPS")
			})
		}
	}
}

func BenchmarkGemmSmallM(b *testing.B) {
	// batch-1 inference shape: [1×k]·[k×n]
	rng := rand.New(rand.NewPCG(9, 10))
	for _, s := range [][3]int{{1, 4096, 4096}, {8, 4096, 4096}, {32, 1024, 1024}} {
		m, n, k := s[0], s[1], s[2]
		x := Contiguous(randSlice(rng, m*k), m, k)
		y := Contiguous(randSlice(rng, k*n), k, n)
		z := Contiguous(make([]float32, m*n), m, n)
		b.Run(fmt.Sprintf("%s/m=%d/n=%d/k=%d", kernel.Impl, m, n, k), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Gemm(z, x, y)
			}
			flops := 2 * float64(m) * float64(n) * float64(k) * float64(b.N)
			b.ReportMetric(flops/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func TestPackARowContiguous(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, mr := range []int{6, 8, 14, 32} {
		for _, pb := range []int{1, 3, 4, 5, 7, 8, 13, 17, 64, 65, 513} {
			for _, ib := range []int{1, 3, 4, 5, mr, mr + 3, 2*mr + 1} {
				rs := pb + 7
				a := Mat{Data: make([]float32, (ib+2)*rs), Rows: ib, Cols: pb, RS: rs, CS: 1}
				for i := range a.Data {
					a.Data[i] = rng.Float32()
				}
				panels := (ib + mr - 1) / mr
				dst := make([]float32, panels*mr*pb)
				for i := range dst {
					dst[i] = -1
				}
				packA(dst, a, 0, 0, ib, pb, mr)
				for ir := 0; ir < ib; ir += mr {
					rows := min(mr, ib-ir)
					panel := dst[ir*pb : (ir+mr)*pb]
					for p := 0; p < pb; p++ {
						for i := 0; i < mr; i++ {
							want := float32(0)
							if i < rows {
								want = a.Data[(ir+i)*rs+p]
							}
							if panel[p*mr+i] != want {
								t.Fatalf("mr=%d pb=%d ib=%d: panel[p=%d][i=%d] = %v, want %v", mr, pb, ib, p, i, panel[p*mr+i], want)
							}
						}
					}
				}
			}
		}
	}
}

// TestPackBPanelTransposed checks the transposed-B path (columns of B
// contiguous), which shares the register transposes with packA.
func TestPackBPanelTransposed(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, nr := range []int{6, 8, 14, 16, 32} {
		for _, pb := range []int{1, 7, 8, 13, 64, 65} {
			for _, jb := range []int{1, 3, 5, nr, nr + 3, 2*nr + 1} {
				cs := pb + 5 // column stride of the source: element (p, j) at j*cs + p
				b := Mat{Data: make([]float32, (jb+2)*cs), Rows: pb, Cols: jb, RS: 1, CS: cs}
				for i := range b.Data {
					b.Data[i] = rng.Float32()
				}
				nPanels := (jb + nr - 1) / nr
				dst := make([]float32, nPanels*nr*pb)
				for i := range dst {
					dst[i] = -1
				}
				packB(dst, b, 0, 0, pb, jb, nr, 1)
				for pi := 0; pi < nPanels; pi++ {
					jr := pi * nr
					cols := min(nr, jb-jr)
					panel := dst[jr*pb : (jr+nr)*pb]
					for p := 0; p < pb; p++ {
						for j := 0; j < nr; j++ {
							want := float32(0)
							if j < cols {
								want = b.Data[(jr+j)*cs+p]
							}
							if panel[p*nr+j] != want {
								t.Fatalf("nr=%d pb=%d jb=%d: panel %d [p=%d][j=%d] = %v, want %v", nr, pb, jb, pi, p, j, panel[p*nr+j], want)
							}
						}
					}
				}
			}
		}
	}
}

func TestGemmZeroOverwrites(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	for _, strat := range []int{StrategyShared, StrategyRows} {
		old := Strategy
		Strategy = strat
		for _, dims := range [][3]int{{1, 1, 1}, {3, 5, 7}, {8, 12, 4}, {33, 45, 17}, {70, 130, 600}, {257, 129, 1030}, {1, 300, 40}, {300, 1, 40}, {64, 64, 0}} {
			m, n, k := dims[0], dims[1], dims[2]
			a := Mat{Data: make([]float32, m*k), Rows: m, Cols: k, RS: k, CS: 1}
			b := Mat{Data: make([]float32, k*n), Rows: k, Cols: n, RS: n, CS: 1}
			for i := range a.Data {
				a.Data[i] = rng.Float32() - 0.5
			}
			for i := range b.Data {
				b.Data[i] = rng.Float32() - 0.5
			}
			ldc := n + 3
			c := Mat{Data: make([]float32, m*ldc), Rows: m, Cols: n, RS: ldc, CS: 1}
			for i := range c.Data {
				c.Data[i] = 1e6 // garbage that must vanish
			}
			want := make([]float32, m*ldc)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					var s float64
					for p := 0; p < k; p++ {
						s += float64(a.Data[i*k+p]) * float64(b.Data[p*n+j])
					}
					want[i*ldc+j] = float32(s)
				}
			}
			GemmZero(c, a, b)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					got := c.Data[i*ldc+j]
					if d := got - want[i*ldc+j]; d > 1e-3 || d < -1e-3 {
						t.Fatalf("strategy %v %v: C[%d,%d] = %v, want %v", strat, dims, i, j, got, want[i*ldc+j])
					}
				}
				for j := n; j < ldc; j++ { // padding untouched
					if c.Data[i*ldc+j] != 1e6 {
						t.Fatalf("strategy %v %v: padding at [%d,%d] modified", strat, dims, i, j)
					}
				}
			}
		}
		Strategy = old
	}
}

func TestFewRowsMatchesPacked(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	k, n := 300, 2100
	b := Mat{Data: make([]float32, k*n), Rows: k, Cols: n, RS: n, CS: 1}
	for i := range b.Data {
		b.Data[i] = rng.Float32() - 0.5
	}
	for m := 1; m <= 8; m++ {
		a := Mat{Data: make([]float32, m*k), Rows: m, Cols: k, RS: k, CS: 1}
		for i := range a.Data {
			a.Data[i] = rng.Float32() - 0.5
		}
		at := Mat{Data: a.Data, Rows: m, Cols: k, RS: 1, CS: m} // transposed storage
		for i := 0; i < m; i++ {
			for p := 0; p < k; p++ {
				at.Data[p*m+i] = a.Data[i*k+p]
			}
		}
		for _, src := range []Mat{a, at} {
			ldc := n + 5
			c := Mat{Data: make([]float32, m*ldc), Rows: m, Cols: n, RS: ldc, CS: 1}
			want := Mat{Data: make([]float32, m*ldc), Rows: m, Cols: n, RS: ldc, CS: 1}
			for i := range c.Data {
				c.Data[i] = 0.25
				want.Data[i] = 0.25
			}
			saved := FewRows
			FewRows = 0
			Gemm(want, src, b) // packed path
			FewRows = saved
			Gemm(c, src, b)
			for i := range c.Data {
				if d := c.Data[i] - want.Data[i]; d > 1e-3 || d < -1e-3 {
					t.Fatalf("m=%d: mismatch at %d: %v vs %v", m, i, c.Data[i], want.Data[i])
				}
			}
			GemmZero(c, src, b)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					if d := c.Data[i*ldc+j] - (want.Data[i*ldc+j] - 0.25); d > 1e-3 || d < -1e-3 {
						t.Fatalf("m=%d zero: mismatch at %d,%d", m, i, j)
					}
				}
			}
		}
	}
}
