//go:build arm64

package kernel

// NEON (Advanced SIMD) is mandatory on AArch64, so it is always available.

func addNEON(x, y, z *float32, n int)
func subNEON(x, y, z *float32, n int)
func mulNEON(x, y, z *float32, n int)
func divNEON(x, y, z *float32, n int)
func maximumNEON(x, y, z *float32, n int)
func addScalarNEON(x, z *float32, s float32, n int)
func scaleNEON(x, z *float32, s float32, n int)
func maxScalarNEON(x, z *float32, s float32, n int)
func axpyNEON(x, y *float32, alpha float32, n int)
func dotNEON(x, y *float32, n int) float32
func sumNEON(x *float32, n int) float32
func maxNEON(x *float32, n int) float32
func expNEON(x, z *float32, n int)  // n % 4 == 0
func tanhNEON(x, z *float32, n int) // n % 4 == 0
func logNEON(x, z *float32, n int)  // n % 4 == 0
func gemmNEON(k int, a, b, c *float32, ldc int)

var neon = impl{
	name:      "neon",
	add:       wrapBinary(addNEON),
	sub:       wrapBinary(subNEON),
	mul:       wrapBinary(mulNEON),
	div:       wrapBinary(divNEON),
	maximum:   wrapBinary(maximumNEON),
	addScalar: wrapScalar(addScalarNEON),
	scale:     wrapScalar(scaleNEON),
	maxScalar: wrapScalar(maxScalarNEON),
	axpy:      wrapAxpy(axpyNEON),
	dot:       wrapDot(dotNEON),
	sum:       wrapSum(sumNEON),
	max:       wrapMax(maxNEON),
	exp:       wrapExp(expNEON, 4),
	tanh:      wrapUnary(tanhNEON, 4, genericTanh),
	log:       wrapUnary(logNEON, 4, genericLog),
	gemm:      gemmNEON,
	mr:        8,
	nr:        12,
}

func candidates() []*impl { return []*impl{&neon} }
func allImpls() []*impl   { return []*impl{&neon} }
