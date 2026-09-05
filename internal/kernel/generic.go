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
	gemmZero:  genericGemmZero,
	exp:       genericExp,
	tanh:      genericTanh,
	log:       genericLog,
	sqrt:      genericSqrt,
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
func genericGemm(k int, a, b, c *float32, ldc int) { genericGemmAcc(k, a, b, c, ldc, true) }

// genericGemmZero writes the tile instead of accumulating it (β = 0).
func genericGemmZero(k int, a, b, c *float32, ldc int) { genericGemmAcc(k, a, b, c, ldc, false) }

func genericGemmAcc(k int, a, b, c *float32, ldc int, accumulate bool) {
	if k == 0 && accumulate {
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
		if accumulate {
			for j := range row {
				row[j] += acc[i][j]
			}
		} else {
			copy(row, acc[i][:])
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

// genericTanh is Eigen's float tanh: after clamping to ±7.9988 (where the
// approximation reaches exactly ±1), an odd degree-13 polynomial over an
// even degree-6 one in x; below |x| < 4e-4 the result is x itself. The
// SIMD kernels evaluate the same expression lane-wise.
func genericTanh(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	const (
		clamp = 7.99881172180175781 // the rational is exactly 1 here
		tiny  = 0.0004
		a1    = 4.89352455891786e-03
		a3    = 6.37261928875436e-04
		a5    = 1.48572235717979e-05
		a7    = 5.12229709037114e-08
		a9    = -8.60467152213735e-11
		a11   = 2.00018790482477e-13
		a13   = -2.76076847742355e-16
		b0    = 4.89352518554385e-03
		b2    = 2.26843463243900e-03
		b4    = 1.18534705686654e-04
		b6    = 1.19825839466702e-06
	)
	for i, v := range x {
		switch {
		case v > -tiny && v < tiny:
			z[i] = v
			continue
		case v >= 9: // tanh is exactly 1 in float32 from here on
			z[i] = 1
			continue
		case v <= -9:
			z[i] = -1
			continue
		}
		v = min(max(v, -clamp), clamp)
		v2 := v * v
		p := float32(a13)
		p = p*v2 + a11
		p = p*v2 + a9
		p = p*v2 + a7
		p = p*v2 + a5
		p = p*v2 + a3
		p = p*v2 + a1
		p *= v
		q := float32(b6)
		q = q*v2 + b4
		q = q*v2 + b2
		q = q*v2 + b0
		z[i] = p / q
	}
}

// genericLog is Cephes' logf: split x into mantissa m ∈ [√½, √2) and
// exponent e, ln x = polynomial(m-1) + e·ln 2. The SIMD kernels evaluate
// the same expression lane-wise.
func genericLog(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	const (
		minNormal = 1.17549435e-38
		sqrtHalf  = 0.707106781186547524
		ln2lo     = -2.12194440e-4
		ln2hi     = 0.693359375
	)
	for i, v := range x {
		switch {
		case v == 0:
			z[i] = float32(math.Inf(-1))
			continue
		case v < 0 || v != v:
			z[i] = float32(math.NaN())
			continue
		case math.IsInf(float64(v), 1):
			z[i] = v
			continue
		}
		v = max(v, minNormal)
		bits := math.Float32bits(v)
		e := float32(int32(bits>>23) - 126)
		m := math.Float32frombits(bits&0x007fffff | 0x3f000000) // [0.5, 1)
		if m < sqrtHalf {
			e--
			m = m + m - 1
		} else {
			m--
		}
		zz := m * m
		y := float32(7.0376836292e-2)
		y = y*m - 1.1514610310e-1
		y = y*m + 1.1676998740e-1
		y = y*m - 1.2420140846e-1
		y = y*m + 1.4249322787e-1
		y = y*m - 1.6668057665e-1
		y = y*m + 2.0000714765e-1
		y = y*m - 2.4999993993e-1
		y = y*m + 3.3333331174e-1
		y = y * m * zz
		y += e * ln2lo
		y -= 0.5 * zz
		z[i] = m + y + e*ln2hi
	}
}

func genericSqrt(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i, v := range x {
		z[i] = float32(math.Sqrt(float64(v)))
	}
}
