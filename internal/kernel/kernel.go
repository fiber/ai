// Package kernel provides the low-level float32 compute kernels that the
// tensor and BLAS layers are built on.
//
// Every kernel exists as a portable Go implementation and, where available,
// as hand-written SIMD assembly: AVX2/FMA and AVX-512 on amd64, NEON on
// arm64. The best implementation is selected once at package init based on
// CPU features. Before a SIMD implementation is activated it is verified
// against the generic one on random data, so a broken kernel can never
// silently produce wrong results – it is dropped and reported in Warnings.
//
// The environment variable FIBERAI_KERNEL can select an implementation
// ("generic", "avx2", "avx512", "neon") among those the CPU supports; this
// is mainly useful for benchmarking and debugging. Setting
// FIBERAI_KERNEL_FORCE=1 additionally bypasses CPU feature detection.
//
// All kernels operate on contiguous slices and are not safe for overlapping
// input/output slices unless stated otherwise (z may alias x or y for the
// element-wise kernels).
package kernel

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

// BinaryFunc computes z[i] = x[i] op y[i]. All slices must have equal length.
type BinaryFunc func(x, y, z []float32)

// ScalarFunc computes z[i] = x[i] op s.
type ScalarFunc func(x []float32, s float32, z []float32)

// GemmFunc is the GEMM micro-kernel: C[MR×NR] += A[MR×k] · B[k×NR].
//
// a points to a packed panel in k-major order (k rows of MR values),
// b points to a packed panel in k-major order (k rows of NR values),
// c points to row-major C with leading dimension ldc (in elements).
type GemmFunc func(k int, a, b, c *float32, ldc int)

// Active kernels. They are assigned at init and may be re-pointed by tests.
var (
	Add       BinaryFunc // z = x + y
	Sub       BinaryFunc // z = x - y
	Mul       BinaryFunc // z = x * y
	Div       BinaryFunc // z = x / y
	Maximum   BinaryFunc // z = max(x, y)
	AddScalar ScalarFunc // z = x + s
	Scale     ScalarFunc // z = x * s
	MaxScalar ScalarFunc // z = max(x, s)

	// Axpy computes y += alpha * x.
	Axpy func(alpha float32, x, y []float32)
	// Dot returns the inner product of x and y.
	Dot func(x, y []float32) float32
	// Sum returns the sum of all elements.
	Sum func(x []float32) float32
	// Max returns the largest element. x must not be empty.
	Max func(x []float32) float32
	// Exp computes z[i] = exp(x[i]) with a vectorised polynomial
	// approximation (relative error ≈ 1 ulp; inputs are clamped to the
	// representable range, so there is no Inf/NaN for large |x|).
	Exp func(x, z []float32)
	// Tanh computes z[i] = tanh(x[i]) by a rational approximation
	// (relative error ≈ 2e-6 or better over the whole range).
	Tanh func(x, z []float32)
	// Log computes z[i] = ln(x[i]) (Cephes logf; -Inf at 0, NaN below 0,
	// denormals flushed to the smallest normal).
	Log func(x, z []float32)
	// Sqrt computes z[i] = sqrt(x[i]).
	Sqrt func(x, z []float32)

	// Gemm is the active GEMM micro-kernel with tile size MR×NR.
	Gemm   GemmFunc
	MR, NR int
	// GemmZero is Gemm with C overwritten instead of accumulated (β = 0),
	// so a product's first K block needs neither a cleared output nor the
	// read of it. nil when the back-end has no such variant; k must be > 0.
	GemmZero GemmFunc
	// GemmBegin and GemmEnd bracket a run of Gemm calls on one goroutine.
	// They are no-ops except for coprocessor back-ends (AMX) that must
	// enable per-thread state; callers keep the goroutine on its thread
	// between them (the AMX hooks lock it themselves).
	GemmBegin, GemmEnd func()
	// GemmHooks reports whether GemmBegin/GemmEnd do anything; drivers skip
	// the calls (and the deferred End) when they do not.
	GemmHooks bool
	// GemmHints carries the back-end's preferences for the blocked driver;
	// zero fields mean "use the architecture default".
	GemmHints Hints

	// Impl names the active implementation ("generic", "avx2", "avx512", "neon").
	Impl string
	// Warnings lists SIMD implementations that were available but failed
	// verification and were therefore disabled.
	Warnings []string
)

// Hints are a back-end's preferences for the GEMM driver.
type Hints struct {
	KC             int // K block length
	TasksPerWorker int // compute-grid granularity
	Workers        int // cap on goroutines (a coprocessor shared per cluster gains nothing from more)
}

// impl bundles one complete implementation.
type impl struct {
	name string

	add, sub, mul, div, maximum BinaryFunc
	addScalar, scale, maxScalar ScalarFunc
	axpy                        func(alpha float32, x, y []float32)
	dot                         func(x, y []float32) float32
	sum, max                    func(x []float32) float32
	exp, tanh, log, sqrt        func(x, z []float32)
	gemm, gemmZero              GemmFunc
	mr, nr                      int
	gemmBegin, gemmEnd          func() // nil: nothing to do
	hints                       Hints
}

func noop() {}

func (i *impl) beginGemm() {
	if i.gemmBegin != nil {
		i.gemmBegin()
	}
}

func (i *impl) endGemm() {
	if i.gemmEnd != nil {
		i.gemmEnd()
	}
}

func use(i *impl) {
	Add, Sub, Mul, Div, Maximum = i.add, i.sub, i.mul, i.div, i.maximum
	AddScalar, Scale, MaxScalar = i.addScalar, i.scale, i.maxScalar
	Axpy, Dot, Sum, Max = i.axpy, i.dot, i.sum, i.max
	Exp, Tanh, Log, Sqrt = i.exp, i.tanh, i.log, i.sqrt
	Gemm, MR, NR = i.gemm, i.mr, i.nr
	GemmZero = i.gemmZero
	GemmBegin, GemmEnd = noop, noop
	GemmHooks = i.gemmBegin != nil
	if GemmHooks {
		GemmBegin, GemmEnd = i.gemmBegin, i.gemmEnd
	}
	GemmHints = i.hints
	Impl = i.name
}

func init() {
	use(&generic)
	want := os.Getenv("FIBERAI_KERNEL")
	if want == "generic" {
		return
	}
	pool := candidates() // best first
	if want != "" && os.Getenv("FIBERAI_KERNEL_FORCE") == "1" {
		// Skip CPU feature detection. Only for emulators that hide features
		// from CPUID (e.g. Rosetta 2); an unsupported instruction is fatal.
		pool = allImpls()
	}
	for _, c := range pool {
		if want != "" && c.name != want {
			continue
		}
		if err := verify(c); err != nil {
			Warnings = append(Warnings, fmt.Sprintf("kernel %s disabled: %v", c.name, err))
			continue
		}
		use(c)
		return
	}
}

// Available lists the implementations usable on this CPU, best first,
// always ending with "generic".
func Available() []string {
	names := make([]string, 0, 4)
	for _, c := range candidates() {
		names = append(names, c.name)
	}
	return append(names, "generic")
}

// verify compares an implementation against the generic one on random data
// with lengths that exercise all vector and scalar tail paths.
func verify(c *impl) error {
	rng := rand.New(rand.NewPCG(7, 11))
	randSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*4 - 2
		}
		return s
	}
	const tol = 1e-4
	close := func(a, b float32) bool {
		d := math.Abs(float64(a - b))
		return d <= tol*(1+math.Abs(float64(b)))
	}
	closeSlices := func(a, b []float32) bool {
		for i := range a {
			if !close(a[i], b[i]) {
				return false
			}
		}
		return true
	}

	for _, n := range []int{0, 1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 129, 257} {
		x, y := randSlice(n), randSlice(n)
		for i := range y { // keep divisions well conditioned
			if y[i] > -0.25 && y[i] < 0.25 {
				y[i] = 0.5
			}
		}
		want, got := make([]float32, n), make([]float32, n)
		bin := []struct {
			name string
			g, c BinaryFunc
		}{{"add", generic.add, c.add}, {"sub", generic.sub, c.sub}, {"mul", generic.mul, c.mul}, {"div", generic.div, c.div}, {"maximum", generic.maximum, c.maximum}}
		for _, b := range bin {
			b.g(x, y, want)
			b.c(x, y, got)
			if !closeSlices(got, want) {
				return fmt.Errorf("%s mismatch at n=%d", b.name, n)
			}
		}
		sc := []struct {
			name string
			g, c ScalarFunc
		}{{"addScalar", generic.addScalar, c.addScalar}, {"scale", generic.scale, c.scale}, {"maxScalar", generic.maxScalar, c.maxScalar}}
		for _, s := range sc {
			s.g(x, 0.75, want)
			s.c(x, 0.75, got)
			if !closeSlices(got, want) {
				return fmt.Errorf("%s mismatch at n=%d", s.name, n)
			}
		}
		copy(want, y)
		copy(got, y)
		generic.axpy(1.5, x, want)
		c.axpy(1.5, x, got)
		if !closeSlices(got, want) {
			return fmt.Errorf("axpy mismatch at n=%d", n)
		}
		if a, b := c.dot(x, y), generic.dot(x, y); !close(a, b) {
			return fmt.Errorf("dot mismatch at n=%d: %v vs %v", n, a, b)
		}
		if a, b := c.sum(x), generic.sum(x); !close(a, b) {
			return fmt.Errorf("sum mismatch at n=%d: %v vs %v", n, a, b)
		}
		if n > 0 {
			if a, b := c.max(x), generic.max(x); a != b {
				return fmt.Errorf("max mismatch at n=%d: %v vs %v", n, a, b)
			}
		}
		xe := make([]float32, n) // exp over the whole clamped range
		for i := range xe {
			xe[i] = rng.Float32()*200 - 100
		}
		generic.exp(xe, want)
		c.exp(xe, got)
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > 2e-6*math.Abs(float64(want[i])) {
				return fmt.Errorf("exp mismatch at n=%d i=%d: %v vs %v", n, i, got[i], want[i])
			}
		}
		for i := range xe {
			xe[i] = rng.Float32()*20 - 10
		}
		generic.tanh(xe, want)
		c.tanh(xe, got)
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > 2e-6*math.Abs(float64(want[i]))+1e-7 {
				return fmt.Errorf("tanh mismatch at n=%d i=%d: %v vs %v", n, i, got[i], want[i])
			}
		}
		for i := range xe {
			xe[i] = float32(math.Exp(float64(rng.Float32()*80 - 40)))
		}
		generic.log(xe, want)
		c.log(xe, got)
		for i := range got {
			if math.Abs(float64(got[i]-want[i])) > 2e-6*math.Abs(float64(want[i]))+1e-6 {
				return fmt.Errorf("log mismatch at n=%d i=%d: %v vs %v", n, i, got[i], want[i])
			}
		}
		generic.sqrt(xe, want)
		c.sqrt(xe, got)
		for i := range got {
			if got[i] != want[i] {
				return fmt.Errorf("sqrt mismatch at n=%d i=%d: %v vs %v", n, i, got[i], want[i])
			}
		}
	}

	// GEMM micro-kernel: compare against a naive reference on the packed layout.
	for _, k := range []int{0, 1, 2, 3, 4, 5, 8, 9, 17} {
		mr, nr := c.mr, c.nr
		a, b := randSlice(k*mr), randSlice(k*nr)
		ldc := nr + 3
		cgot := randSlice(mr * ldc)
		cwant := append([]float32(nil), cgot...)
		prod := make([]float32, mr*nr) // the bare tile A·B, for the overwriting variant
		for p := 0; p < k; p++ {
			for i := 0; i < mr; i++ {
				for j := 0; j < nr; j++ {
					cwant[i*ldc+j] += a[p*mr+i] * b[p*nr+j]
					prod[i*nr+j] += a[p*mr+i] * b[p*nr+j]
				}
			}
		}
		var ap, bp *float32
		if k > 0 {
			ap, bp = &a[0], &b[0]
		}
		c.beginGemm()
		c.gemm(k, ap, bp, &cgot[0], ldc)
		c.endGemm()
		if !closeSlices(cgot, cwant) {
			return fmt.Errorf("gemm micro-kernel mismatch at k=%d", k)
		}
		if c.gemmZero != nil && k > 0 {
			zgot := randSlice(mr * ldc) // garbage that must be overwritten, padding kept
			zwant := append([]float32(nil), zgot...)
			for i := 0; i < mr; i++ {
				copy(zwant[i*ldc:i*ldc+nr], prod[i*nr:i*nr+nr])
			}
			c.beginGemm()
			c.gemmZero(k, ap, bp, &zgot[0], ldc)
			c.endGemm()
			if !closeSlices(zgot, zwant) {
				return fmt.Errorf("gemm-zero micro-kernel mismatch at k=%d", k)
			}
		}
	}
	return nil
}

// wrappers turn pointer-based assembly routines into slice-based kernels
// with length validation.

func checkLen2(x, y []float32) int {
	if len(x) != len(y) {
		panic(fmt.Sprintf("kernel: length mismatch %d vs %d", len(x), len(y)))
	}
	return len(x)
}

func checkLen3(x, y, z []float32) int {
	if len(x) != len(z) || len(y) != len(z) {
		panic(fmt.Sprintf("kernel: length mismatch %d, %d vs %d", len(x), len(y), len(z)))
	}
	return len(z)
}

func wrapBinary(f func(x, y, z *float32, n int)) BinaryFunc {
	return func(x, y, z []float32) {
		n := checkLen3(x, y, z)
		if n == 0 {
			return
		}
		f(&x[0], &y[0], &z[0], n)
	}
}

func wrapScalar(f func(x, z *float32, s float32, n int)) ScalarFunc {
	return func(x []float32, s float32, z []float32) {
		n := checkLen2(x, z)
		if n == 0 {
			return
		}
		f(&x[0], &z[0], s, n)
	}
}

func wrapAxpy(f func(x, y *float32, alpha float32, n int)) func(alpha float32, x, y []float32) {
	return func(alpha float32, x, y []float32) {
		n := checkLen2(x, y)
		if n == 0 {
			return
		}
		f(&x[0], &y[0], alpha, n)
	}
}

func wrapDot(f func(x, y *float32, n int) float32) func(x, y []float32) float32 {
	return func(x, y []float32) float32 {
		n := checkLen2(x, y)
		if n == 0 {
			return 0
		}
		return f(&x[0], &y[0], n)
	}
}

func wrapSum(f func(x *float32, n int) float32) func(x []float32) float32 {
	return func(x []float32) float32 {
		if len(x) == 0 {
			return 0
		}
		return f(&x[0], len(x))
	}
}

// wrapExp runs the vector routine on the largest multiple of width and
// the portable code on the remainder.
func wrapExp(f func(x, z *float32, n int), width int) func(x, z []float32) {
	return wrapUnary(f, width, genericExp)
}

func wrapUnary(f func(x, z *float32, n int), width int, tail func(x, z []float32)) func(x, z []float32) {
	return func(x, z []float32) {
		n := checkLen2(x, z)
		nv := n &^ (width - 1)
		if nv > 0 {
			f(&x[0], &z[0], nv)
		}
		if nv < n {
			tail(x[nv:], z[nv:])
		}
	}
}

func wrapMax(f func(x *float32, n int) float32) func(x []float32) float32 {
	return func(x []float32) float32 {
		if len(x) == 0 {
			panic("kernel: Max of empty slice")
		}
		return f(&x[0], len(x))
	}
}
