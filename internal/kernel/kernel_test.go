package kernel

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"testing"
	"unsafe"
)

// implementations returns every implementation that can run on this CPU,
// generic included, so tests exercise all of them regardless of which one
// init selected.
func implementations() []*impl {
	impls := candidates()
	if os.Getenv("FIBERAI_KERNEL_FORCE") == "1" {
		// Only the forced implementation is known to run on this machine.
		impls = nil
		for _, im := range allImpls() {
			if im.name == os.Getenv("FIBERAI_KERNEL") {
				impls = append(impls, im)
			}
		}
	}
	return append(impls, &generic)
}

func randSlice(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = rng.Float32()*2 - 1
	}
	return s
}

func closeEnough(got, want float64, tol float64) bool {
	return math.Abs(got-want) <= tol*(1+math.Abs(want))
}

var lengths = []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 1000, 1023}

func TestInitSelectedImplementation(t *testing.T) {
	t.Logf("active kernel implementation: %s (MR=%d, NR=%d), available: %v", Impl, MR, NR, Available())
	if len(Warnings) > 0 {
		t.Errorf("kernel warnings: %v", Warnings)
	}
	if cands := candidates(); len(cands) > 0 && Impl != cands[0].name && os.Getenv("FIBERAI_KERNEL") == "" {
		t.Errorf("init selected %q, but best candidate is %q", Impl, cands[0].name)
	}
}

func TestVerifyAll(t *testing.T) {
	for _, im := range implementations() {
		if err := verify(im); err != nil {
			t.Errorf("%s: %v", im.name, err)
		}
	}
}

func TestBinaryKernels(t *testing.T) {
	for _, im := range implementations() {
		t.Run(im.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, 2))
			for _, n := range lengths {
				x, y := randSlice(rng, n), randSlice(rng, n)
				for i := range y {
					if math.Abs(float64(y[i])) < 0.1 {
						y[i] = 0.5
					}
				}
				z := make([]float32, n)
				type tc struct {
					name string
					f    BinaryFunc
					ref  func(a, b float32) float32
				}
				for _, c := range []tc{
					{"add", im.add, func(a, b float32) float32 { return a + b }},
					{"sub", im.sub, func(a, b float32) float32 { return a - b }},
					{"mul", im.mul, func(a, b float32) float32 { return a * b }},
					{"div", im.div, func(a, b float32) float32 { return a / b }},
					{"maximum", im.maximum, func(a, b float32) float32 { return max(a, b) }},
				} {
					c.f(x, y, z)
					for i := range z {
						if want := c.ref(x[i], y[i]); !closeEnough(float64(z[i]), float64(want), 1e-6) {
							t.Fatalf("%s n=%d i=%d: got %v want %v", c.name, n, i, z[i], want)
						}
					}
				}
				// aliasing: z == x must work for element-wise kernels
				xc := append([]float32(nil), x...)
				im.add(xc, y, xc)
				for i := range xc {
					if xc[i] != x[i]+y[i] {
						t.Fatalf("add aliasing n=%d i=%d", n, i)
					}
				}
			}
		})
	}
}

func TestScalarKernels(t *testing.T) {
	for _, im := range implementations() {
		t.Run(im.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(3, 4))
			for _, n := range lengths {
				x := randSlice(rng, n)
				z := make([]float32, n)
				const s = float32(0.375)
				im.addScalar(x, s, z)
				for i := range z {
					if z[i] != x[i]+s {
						t.Fatalf("addScalar n=%d i=%d", n, i)
					}
				}
				im.scale(x, s, z)
				for i := range z {
					if z[i] != x[i]*s {
						t.Fatalf("scale n=%d i=%d", n, i)
					}
				}
				im.maxScalar(x, s, z)
				for i := range z {
					if z[i] != max(x[i], s) {
						t.Fatalf("maxScalar n=%d i=%d", n, i)
					}
				}
				y := randSlice(rng, n)
				yc := append([]float32(nil), y...)
				im.axpy(s, x, yc)
				for i := range yc {
					if !closeEnough(float64(yc[i]), float64(y[i])+float64(s)*float64(x[i]), 1e-6) {
						t.Fatalf("axpy n=%d i=%d", n, i)
					}
				}
			}
		})
	}
}

func TestReductionKernels(t *testing.T) {
	for _, im := range implementations() {
		t.Run(im.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(5, 6))
			for _, n := range lengths {
				x, y := randSlice(rng, n), randSlice(rng, n)
				var wd, ws float64
				wm := math.Inf(-1)
				for i := range x {
					wd += float64(x[i]) * float64(y[i])
					ws += float64(x[i])
					wm = math.Max(wm, float64(x[i]))
				}
				if d := im.dot(x, y); !closeEnough(float64(d), wd, 1e-5) {
					t.Fatalf("dot n=%d: got %v want %v", n, d, wd)
				}
				if s := im.sum(x); !closeEnough(float64(s), ws, 1e-5) {
					t.Fatalf("sum n=%d: got %v want %v", n, s, ws)
				}
				if n > 0 {
					if m := im.max(x); float64(m) != wm {
						t.Fatalf("max n=%d: got %v want %v", n, m, wm)
					}
					// the maximum must be found in every lane position
					for pos := 0; pos < n; pos += max(1, n/7) {
						xx := append([]float32(nil), x...)
						xx[pos] = 7
						if m := im.max(xx); m != 7 {
							t.Fatalf("max n=%d pos=%d: got %v want 7", n, pos, m)
						}
					}
				}
			}
		})
	}
}

func TestGemmMicroKernel(t *testing.T) {
	for _, im := range implementations() {
		t.Run(im.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(7, 8))
			mr, nr := im.mr, im.nr
			for _, k := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 64, 100} {
				a, b := randSlice(rng, k*mr), randSlice(rng, k*nr)
				ldc := nr + 5
				c := randSlice(rng, mr*ldc)
				want := make([]float64, len(c))
				for i, v := range c {
					want[i] = float64(v)
				}
				for p := 0; p < k; p++ {
					for i := 0; i < mr; i++ {
						for j := 0; j < nr; j++ {
							want[i*ldc+j] += float64(a[p*mr+i]) * float64(b[p*nr+j])
						}
					}
				}
				var ap, bp *float32
				if k > 0 {
					ap, bp = &a[0], &b[0]
				}
				im.beginGemm()
				im.gemm(k, ap, bp, &c[0], ldc)
				im.endGemm()
				for i := range c {
					if !closeEnough(float64(c[i]), want[i], 1e-5) {
						t.Fatalf("k=%d: c[%d]=%v want %v", k, i, c[i], want[i])
					}
				}
			}
		})
	}
}

func TestMathKernels(t *testing.T) {
	x := []float32{-2, -0.5, 0, 0.5, 1, 2, 3.5}
	z := make([]float32, len(x))
	Exp(x, z)
	for i := range x {
		if !closeEnough(float64(z[i]), math.Exp(float64(x[i])), 1e-6) {
			t.Errorf("Exp(%v) = %v", x[i], z[i])
		}
	}
	Tanh(x, z)
	for i := range x {
		if !closeEnough(float64(z[i]), math.Tanh(float64(x[i])), 1e-6) {
			t.Errorf("Tanh(%v) = %v", x[i], z[i])
		}
	}
	Sigmoid(x, z)
	for i := range x {
		if !closeEnough(float64(z[i]), 1/(1+math.Exp(-float64(x[i]))), 1e-6) {
			t.Errorf("Sigmoid(%v) = %v", x[i], z[i])
		}
	}
	GtZeroMask(x, z)
	for i := range x {
		want := float32(0)
		if x[i] > 0 {
			want = 1
		}
		if z[i] != want {
			t.Errorf("GtZeroMask(%v) = %v", x[i], z[i])
		}
	}
	Sign(x, z)
	if z[0] != -1 || z[2] != 0 || z[3] != 1 {
		t.Errorf("Sign = %v", z)
	}
	Clamp(x, -1, 1, z)
	if z[0] != -1 || z[6] != 1 || z[1] != -0.5 {
		t.Errorf("Clamp = %v", z)
	}
	Pow(x, 2, z)
	if z[0] != 4 || z[6] != 12.25 {
		t.Errorf("Pow2 = %v", z)
	}
}

func BenchmarkDot(b *testing.B) {
	rng := rand.New(rand.NewPCG(9, 10))
	for _, n := range []int{64, 4096, 1 << 20} {
		x, y := randSlice(rng, n), randSlice(rng, n)
		for _, im := range implementations() {
			b.Run(fmt.Sprintf("%s/n=%d", im.name, n), func(b *testing.B) {
				b.SetBytes(int64(8 * n))
				for i := 0; i < b.N; i++ {
					_ = im.dot(x, y)
				}
			})
		}
	}
}

func BenchmarkAdd(b *testing.B) {
	rng := rand.New(rand.NewPCG(11, 12))
	for _, n := range []int{64, 4096, 1 << 20} {
		x, y, z := randSlice(rng, n), randSlice(rng, n), make([]float32, n)
		for _, im := range implementations() {
			b.Run(fmt.Sprintf("%s/n=%d", im.name, n), func(b *testing.B) {
				b.SetBytes(int64(12 * n))
				for i := 0; i < b.N; i++ {
					im.add(x, y, z)
				}
			})
		}
	}
}

func BenchmarkGemmMicroKernel(b *testing.B) {
	rng := rand.New(rand.NewPCG(13, 14))
	const k = 256
	for _, im := range implementations() {
		a, bb := randSlice(rng, k*im.mr), randSlice(rng, k*im.nr)
		c := make([]float32, im.mr*im.nr)
		b.Run(im.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				im.beginGemm()
				im.gemm(k, &a[0], &bb[0], &c[0], im.nr)
				im.endGemm()
			}
			flops := 2 * float64(k) * float64(im.mr) * float64(im.nr) * float64(b.N)
			b.ReportMetric(flops/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func TestExpAccuracy(t *testing.T) {
	for _, im := range implementations() {
		t.Run(im.name, func(t *testing.T) {
			// dense sweep over the clamped range plus awkward lengths for the tails
			for _, n := range []int{1, 3, 7, 8, 9, 15, 16, 17, 1000, 4097} {
				x := make([]float32, n)
				for i := range x {
					x[i] = -90 + 180*float32(i)/float32(max(1, n-1))
				}
				z := make([]float32, n)
				im.exp(x, z)
				var maxRel float64
				for i := range x {
					if x[i] < expLo {
						// below the low clamp the kernels return exactly 0 (B-004)
						if z[i] != 0 {
							t.Fatalf("exp(%v) = %v, want 0", x[i], z[i])
						}
						continue
					}
					// the kernels clamp to the float32-rounded upper bound
					xc := min(x[i], float32(expHi))
					want := math.Exp(float64(xc))
					rel := math.Abs(float64(z[i])-want) / want
					maxRel = math.Max(maxRel, rel)
					if math.IsNaN(float64(z[i])) || math.IsInf(float64(z[i]), 0) {
						t.Fatalf("exp(%v) = %v", x[i], z[i])
					}
				}
				if maxRel > 3e-7 { // ~2.5 ulp
					t.Errorf("n=%d: max relative error %.3g", n, maxRel)
				}
			}
			// special values
			z := make([]float32, 3)
			im.exp([]float32{0, float32(math.Inf(1)), float32(math.Inf(-1))}, z)
			if z[0] != 1 || z[1] < 1e38 || z[2] > 1e-37 {
				t.Errorf("special values: %v", z)
			}
		})
	}
}

func BenchmarkExp(b *testing.B) {
	rng := rand.New(rand.NewPCG(15, 16))
	for _, n := range []int{4096, 1 << 20} {
		x := randSlice(rng, n)
		z := make([]float32, n)
		for _, im := range implementations() {
			b.Run(fmt.Sprintf("%s/n=%d", im.name, n), func(b *testing.B) {
				b.SetBytes(int64(8 * n))
				for i := 0; i < b.N; i++ {
					im.exp(x, z)
				}
			})
		}
		b.Run(fmt.Sprintf("math.Exp/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(8 * n))
			for i := 0; i < b.N; i++ {
				for j := range x {
					z[j] = float32(math.Exp(float64(x[j])))
				}
			}
		})
	}
}

func TestTanhAccuracy(t *testing.T) {
	for _, im := range implementations() {
		t.Run(im.name, func(t *testing.T) {
			for _, n := range []int{1, 3, 7, 8, 9, 15, 16, 17, 1000, 4097} {
				x := make([]float32, n)
				for i := range x {
					x[i] = -12 + 24*float32(i)/float32(max(1, n-1))
				}
				// and a dense band around zero where cancellation would show
				if n == 4097 {
					for i := range x[:1024] {
						x[i] = -1e-3 + 2e-3*float32(i)/1023
					}
				}
				z := make([]float32, n)
				im.tanh(x, z)
				for i := range x {
					want := math.Tanh(float64(x[i]))
					if math.Abs(float64(z[i])-want) > 2e-6*math.Abs(want)+1e-7 {
						t.Fatalf("n=%d: tanh(%v) = %v, want %v", n, x[i], z[i], want)
					}
				}
			}
			z := make([]float32, 3)
			im.tanh([]float32{0, float32(math.Inf(1)), float32(math.Inf(-1))}, z)
			if z[0] != 0 || z[1] != 1 || z[2] != -1 {
				t.Errorf("special values: %v", z)
			}
		})
	}
}

func TestLogAccuracy(t *testing.T) {
	for _, im := range implementations() {
		t.Run(im.name, func(t *testing.T) {
			for _, n := range []int{1, 3, 7, 8, 9, 15, 16, 17, 1000, 4097} {
				x := make([]float32, n)
				for i := range x {
					x[i] = float32(math.Exp(-85 + 170*float64(i)/float64(max(1, n-1))))
				}
				if n == 4097 {
					for i := range x[:1024] { // dense around 1, where ln is small
						x[i] = 0.9 + 0.2*float32(i)/1023
					}
				}
				z := make([]float32, n)
				im.log(x, z)
				for i := range x {
					want := math.Log(float64(x[i]))
					if math.Abs(float64(z[i])-want) > 2e-6*math.Abs(want)+1e-6 {
						t.Fatalf("n=%d: log(%v) = %v, want %v", n, x[i], z[i], want)
					}
				}
			}
			z := make([]float32, 4)
			im.log([]float32{0, -1, float32(math.Inf(1)), 1}, z)
			if !math.IsInf(float64(z[0]), -1) || !math.IsNaN(float64(z[1])) || !math.IsInf(float64(z[2]), 1) || z[3] != 0 {
				t.Errorf("special values: %v", z)
			}
		})
	}
}

func TestGELUAgainstFloat64(t *testing.T) {
	x := make([]float32, 5000)
	for i := range x {
		x[i] = -8 + 16*float32(i)/4999
	}
	z := make([]float32, len(x))
	g := make([]float32, len(x))
	GELU(x, z)
	GELUGrad(x, g)
	for i, v := range x {
		fv := float64(v)
		u := geluC * (fv + 0.044715*fv*fv*fv)
		want := 0.5 * fv * (1 + math.Tanh(u))
		if !closeEnough(float64(z[i]), want, 1e-5) {
			t.Fatalf("GELU(%v) = %v, want %v", v, z[i], want)
		}
		th := math.Tanh(u)
		wantG := 0.5*(1+th) + 0.5*fv*(1-th*th)*geluC*(1+3*0.044715*fv*fv)
		if !closeEnough(float64(g[i]), wantG, 1e-5) {
			t.Fatalf("GELUGrad(%v) = %v, want %v", v, g[i], wantG)
		}
	}
}

func BenchmarkTanh(b *testing.B) {
	rng := rand.New(rand.NewPCG(15, 16))
	for _, n := range []int{4096, 1 << 20} {
		x := randSlice(rng, n)
		z := make([]float32, n)
		for _, im := range implementations() {
			b.Run(fmt.Sprintf("%s/n=%d", im.name, n), func(b *testing.B) {
				b.SetBytes(int64(8 * n))
				for i := 0; i < b.N; i++ {
					im.tanh(x, z)
				}
			})
		}
		b.Run(fmt.Sprintf("math.Tanh/n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(8 * n))
			for i := 0; i < b.N; i++ {
				for j := range x {
					z[j] = float32(math.Tanh(float64(x[j])))
				}
			}
		})
	}
}

// BenchmarkExpAliasing runs the exp kernel with the output at the same
// page offset as the input (as page-aligned buffers are) and shifted by
// 16 floats. On Intel cores a store and a later load whose addresses
// agree in the low 12 bits stall the load (4K aliasing); a large gap
// between the two rows says the kernels are fine and the allocator has
// to colour its buffers.
func BenchmarkExpAliasing(b *testing.B) {
	rng := rand.New(rand.NewPCG(15, 16))
	const n = 4096
	x := make([]float32, n+2048)
	z := make([]float32, n+2048)
	copy(x, randSlice(rng, n))
	// same low 12 address bits for x[i] and z[i]
	xo := (4096 - int(uintptr(unsafe.Pointer(&x[0]))&4095)) / 4
	zo := (4096 - int(uintptr(unsafe.Pointer(&z[0]))&4095)) / 4
	xa, za := x[xo:xo+n], z[zo:zo+n]
	for _, im := range implementations() {
		b.Run(im.name+"/aligned", func(b *testing.B) {
			b.SetBytes(8 * n)
			for i := 0; i < b.N; i++ {
				im.exp(xa, za)
			}
		})
		b.Run(im.name+"/offset16", func(b *testing.B) {
			zb := z[zo+16 : zo+16+n]
			b.SetBytes(8 * n)
			for i := 0; i < b.N; i++ {
				im.exp(xa, zb)
			}
		})
		b.Run(im.name+"/offset512", func(b *testing.B) {
			zb := z[zo+512 : zo+512+n]
			b.SetBytes(8 * n)
			for i := 0; i < b.N; i++ {
				im.exp(xa, zb)
			}
		})
	}
}

func TestSqrtAndAdamStep(t *testing.T) {
	for _, im := range implementations() {
		x := make([]float32, 1001)
		for i := range x {
			x[i] = float32(i) * 0.37
		}
		z := make([]float32, len(x))
		im.sqrt(x, z)
		for i := range x {
			if want := float32(math.Sqrt(float64(x[i]))); z[i] != want {
				t.Fatalf("%s: sqrt(%v) = %v, want %v", im.name, x[i], z[i], want)
			}
		}
	}
	rng := rand.New(rand.NewPCG(3, 4))
	n := 5000
	w, g, m, v := randSlice(rng, n), randSlice(rng, n), randSlice(rng, n), randSlice(rng, n)
	for i := range v {
		v[i] = float32(math.Abs(float64(v[i])))
	}
	w2, m2, v2 := append([]float32(nil), w...), append([]float32(nil), m...), append([]float32(nil), v...)
	const step, b1, b2, eps, bc2 = 0.01, 0.9, 0.999, 1e-8, 0.5
	AdamStep(w, g, m, v, step, b1, b2, eps, bc2)
	for j := range w2 {
		gj := g[j]
		m2[j] = b1*m2[j] + (1-b1)*gj
		v2[j] = b2*v2[j] + (1-b2)*gj*gj
		denom := float32(math.Sqrt(float64(v2[j]/bc2))) + eps
		w2[j] -= step * m2[j] / denom
		if !closeEnough(float64(w[j]), float64(w2[j]), 1e-5) || !closeEnough(float64(m[j]), float64(m2[j]), 1e-6) || !closeEnough(float64(v[j]), float64(v2[j]), 1e-6) {
			t.Fatalf("AdamStep mismatch at %d: w %v vs %v, m %v vs %v, v %v vs %v", j, w[j], w2[j], m[j], m2[j], v[j], v2[j])
		}
	}
}
