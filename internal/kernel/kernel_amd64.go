//go:build amd64

package kernel

// Implemented in cpu_amd64.s.
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() (eax, edx uint32)

// Implemented in kernel_avx2_amd64.s (AVX2 + FMA).
func addAVX2(x, y, z *float32, n int)
func subAVX2(x, y, z *float32, n int)
func mulAVX2(x, y, z *float32, n int)
func divAVX2(x, y, z *float32, n int)
func maximumAVX2(x, y, z *float32, n int)
func addScalarAVX2(x, z *float32, s float32, n int)
func scaleAVX2(x, z *float32, s float32, n int)
func maxScalarAVX2(x, z *float32, s float32, n int)
func axpyAVX2(x, y *float32, alpha float32, n int)
func dotAVX2(x, y *float32, n int) float32
func dotNormsAVX2(x, y *float32, n int, out *[3]float32)
func sumAVX2(x *float32, n int) float32
func maxAVX2(x *float32, n int) float32
func expAVX2(x, z *float32, n int)                            // n % 8 == 0
func expSumAVX2(x, z *float32, n int, a, b float32) float32   // n % 8 == 0
func expAVX512(x, z *float32, n int)                          // n % 16 == 0
func expSumAVX512(x, z *float32, n int, a, b float32) float32 // n % 16 == 0
func tanhAVX512(x, z *float32, n int)                         // n % 16 == 0
func tanhAVX2(x, z *float32, n int)                           // n % 8 == 0
func logAVX2(x, z *float32, n int)                            // n % 8 == 0
func sqrtAVX2(x, z *float32, n int)                           // n % 8 == 0
func gemmAVX2(k int, a, b, c *float32, ldc int)
func gemmZeroAVX2(k int, a, b, c *float32, ldc int)
func gemmAVX2Body(k int, a, b, c *float32, ldc int)

// Implemented in kernel_avx512_amd64.s (AVX-512F).
func gemmAVX512(k int, a, b, c *float32, ldc int)
func gemmZeroAVX512(k int, a, b, c *float32, ldc int)
func gemmAVX512Body(k int, a, b, c *float32, ldc int)
func gemmAVX512x14(k int, a, b, c *float32, ldc int)
func gemmZeroAVX512x14(k int, a, b, c *float32, ldc int)
func gemmAVX512x14Body(k int, a, b, c *float32, ldc int)

var avx2 = impl{
	name:      "avx2",
	add:       wrapBinary(addAVX2),
	sub:       wrapBinary(subAVX2),
	mul:       wrapBinary(mulAVX2),
	div:       wrapBinary(divAVX2),
	maximum:   wrapBinary(maximumAVX2),
	addScalar: wrapScalar(addScalarAVX2),
	scale:     wrapScalar(scaleAVX2),
	maxScalar: wrapScalar(maxScalarAVX2),
	axpy:      wrapAxpy(axpyAVX2),
	dot:       wrapDot(dotAVX2),
	dotNorms:  wrapDotNorms(dotNormsAVX2),
	sum:       wrapSum(sumAVX2),
	max:       wrapMax(maxAVX2),
	exp:       wrapExp(expAVX2, 8),
	expSum:    wrapExpSum(expSumAVX2, 8),
	tanh:      wrapUnary(tanhAVX2, 8, genericTanh),
	log:       wrapUnary(logAVX2, 8, genericLog),
	sqrt:      wrapUnary(sqrtAVX2, 8, genericSqrt),
	gemm:      gemmAVX2,
	gemmZero:  gemmZeroAVX2,
	mr:        6,
	nr:        16,
}

// The memory-bound element-wise primitives stay AVX2 (the bandwidth is
// the limit, not the lanes); the AVX-512 implementation swaps in the wider
// GEMM micro-kernel and its own exp, exp-sum and tanh, whose AVX2 versions
// are bound by constant loads (T-043).
var avx512 = func() impl {
	i := avx2
	i.name = "avx512"
	i.gemm, i.gemmZero = gemmAVX512x14, gemmZeroAVX512x14
	i.mr, i.nr = 14, 32
	i.exp = wrapExp(expAVX512, 16)
	i.expSum = wrapExpSum(expSumAVX512, 16)
	i.tanh = wrapUnary(tanhAVX512, 16, genericTanh)
	return i
}()

// avx512x12 is the same implementation with the earlier 12×32 micro-kernel,
// kept selectable (FIBERAI_KERNEL=avx512x12) for comparison. On a Xeon
// Gold 6130 the 14×32 tile is 5–9 % faster on one core and 8–10 % on
// sixteen (1 132 → 1 242 GFLOPS at n=1024).
var avx512x12 = func() impl {
	i := avx512
	i.name = "avx512x12"
	i.gemm, i.gemmZero = gemmAVX512, gemmZeroAVX512
	i.mr, i.nr = 12, 32
	return i
}()

// allImpls lists every implementation compiled for this architecture,
// regardless of CPU support.
func allImpls() []*impl { return []*impl{&avx512, &avx512x12, &avx2} }

func candidates() []*impl {
	_, _, ecx1, _ := cpuid(1, 0)
	osxsave := ecx1&(1<<27) != 0
	avx := ecx1&(1<<28) != 0
	fma := ecx1&(1<<12) != 0
	if !(osxsave && avx && fma) {
		return nil
	}
	xcr0, _ := xgetbv()
	if xcr0&0x6 != 0x6 { // OS must save XMM and YMM state
		return nil
	}
	_, ebx7, _, _ := cpuid(7, 0)
	hasAVX2 := ebx7&(1<<5) != 0
	hasAVX512F := ebx7&(1<<16) != 0 && xcr0&0xE0 == 0xE0 // opmask + ZMM state

	var out []*impl
	if hasAVX2 && hasAVX512F {
		out = append(out, &avx512, &avx512x12)
	}
	if hasAVX2 {
		out = append(out, &avx2)
	}
	return out
}
