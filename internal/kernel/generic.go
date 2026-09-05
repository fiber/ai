package kernel

import (
	"math"
	"unsafe"
)

// generic is the portable pure-Go implementation. It is always available and
// serves as the reference for verifying the SIMD implementations.
var generic = impl{
	name:      "generic",
	add:       genericAdd,
	sub:       genericSub,
	mul:       genericMul,
	div:       genericDiv,
	maximum:   genericMaximum,
	addScalar: genericAddScalar,
	scale:     genericScale,
	maxScalar: genericMaxScalar,
	axpy:      genericAxpy,
	dot:       genericDot,
	sum:       genericSum,
	max:       genericMax,
	exp:       genericExp,
	gemm:      genericGemm,
	mr:        genericMR,
	nr:        genericNR,
}

func genericAdd(x, y, z []float32) {
	n := checkLen3(x, y, z)
	x, y, z = x[:n], y[:n], z[:n]
	for i := range z {
		z[i] = x[i] + y[i]
	}
}

func genericSub(x, y, z []float32) {
	n := checkLen3(x, y, z)
	x, y, z = x[:n], y[:n], z[:n]
	for i := range z {
		z[i] = x[i] - y[i]
	}
}

func genericMul(x, y, z []float32) {
	n := checkLen3(x, y, z)
	x, y, z = x[:n], y[:n], z[:n]
	for i := range z {
		z[i] = x[i] * y[i]
	}
}

func genericDiv(x, y, z []float32) {
	n := checkLen3(x, y, z)
	x, y, z = x[:n], y[:n], z[:n]
	for i := range z {
		z[i] = x[i] / y[i]
	}
}

func genericMaximum(x, y, z []float32) {
	n := checkLen3(x, y, z)
	x, y, z = x[:n], y[:n], z[:n]
	for i := range z {
		a, b := x[i], y[i]
		if b > a {
			a = b
		}
		z[i] = a
	}
}

func genericAddScalar(x []float32, s float32, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		z[i] = x[i] + s
	}
}

func genericScale(x []float32, s float32, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		z[i] = x[i] * s
	}
}

func genericMaxScalar(x []float32, s float32, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		a := x[i]
		if s > a {
			a = s
		}
		z[i] = a
	}
}

func genericAxpy(alpha float32, x, y []float32) {
	n := checkLen2(x, y)
	x, y = x[:n], y[:n]
	for i := range y {
		y[i] += alpha * x[i]
	}
}

func genericDot(x, y []float32) float32 {
	n := checkLen2(x, y)
	x, y = x[:n], y[:n]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += x[i] * y[i]
		s1 += x[i+1] * y[i+1]
		s2 += x[i+2] * y[i+2]
		s3 += x[i+3] * y[i+3]
	}
	for ; i < n; i++ {
		s0 += x[i] * y[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func genericSum(x []float32) float32 {
	n := len(x)
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += x[i]
		s1 += x[i+1]
		s2 += x[i+2]
		s3 += x[i+3]
	}
	for ; i < n; i++ {
		s0 += x[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func genericMax(x []float32) float32 {
	if len(x) == 0 {
		panic("kernel: Max of empty slice")
	}
	m := x[0]
	for _, v := range x[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

const (
	genericMR = 4
	genericNR = 8
)

// genericGemm is the portable 4×8 micro-kernel: C[4×8] += A[4×k] · B[k×8].
func genericGemm(k int, a, b, c *float32, ldc int) {
	if k == 0 {
		return
	}
	ap := unsafe.Slice(a, k*genericMR)
	bp := unsafe.Slice(b, k*genericNR)
	var acc [genericMR][genericNR]float32
	for p := 0; p < k; p++ {
		ar := ap[p*genericMR : p*genericMR+genericMR : p*genericMR+genericMR]
		br := bp[p*genericNR : p*genericNR+genericNR : p*genericNR+genericNR]
		for i := 0; i < genericMR; i++ {
			ai := ar[i]
			row := &acc[i]
			row[0] += ai * br[0]
			row[1] += ai * br[1]
			row[2] += ai * br[2]
			row[3] += ai * br[3]
			row[4] += ai * br[4]
			row[5] += ai * br[5]
			row[6] += ai * br[6]
			row[7] += ai * br[7]
		}
	}
	cp := unsafe.Slice(c, (genericMR-1)*ldc+genericNR)
	for i := 0; i < genericMR; i++ {
		row := cp[i*ldc : i*ldc+genericNR : i*ldc+genericNR]
		for j := range row {
			row[j] += acc[i][j]
		}
	}
}

// Constants of the exp approximation (shared with the assembly kernels):
// x is clamped, split as x = n·ln2 + r with |r| <= ln2/2 (Cody–Waite, n
// obtained by the round-to-nearest "magic number" trick) and e^r is a
// degree-6 minimax polynomial (Cephes expf). The result is p·2^n, formed by
// adding n to the exponent bits of p.
const (
	expLog2e = 1.44269504088896341
	expMagic = 12582912.0 // 1.5·2^23: adding it rounds to the nearest integer in the low mantissa bits
	expLn2Hi = 0.693359375
	expLn2Lo = -2.12194440e-4
	expHi    = 88.3762626647949 // above: result would overflow
	expLo    = -87.0            // below: p·2^n could leave the normal range (the exponent-add trick needs normals)
	expC0    = 1.9875691500e-4
	expC1    = 1.3981999507e-3
	expC2    = 8.3334519073e-3
	expC3    = 4.1665795894e-2
	expC4    = 1.6666665459e-1
	expC5    = 5.0000001201e-1
)

// genericExp is the portable implementation of Exp, ~5× faster than
// math.Exp per element and bit-compatible with the SIMD kernels up to FMA
// contraction.
func genericExp(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	const magicBits = 0x4B400000
	for i, v := range x {
		v = min(max(v, expLo), expHi)
		t := v*expLog2e + expMagic
		nf := t - expMagic
		r := v - nf*expLn2Hi
		r -= nf * expLn2Lo
		p := float32(expC0)
		p = p*r + expC1
		p = p*r + expC2
		p = p*r + expC3
		p = p*r + expC4
		p = p*r + expC5
		p = p*r*r + r + 1
		e := int32(math.Float32bits(t)) - magicBits
		z[i] = math.Float32frombits(math.Float32bits(p) + uint32(e<<23))
	}
}
