package kernel

import "math"

// Transcendental and miscellaneous element-wise kernels. These currently
// have Go-only implementations (per-element math calls); they are still
// parallelised by the tensor layer.

// Sqrt computes z[i] = sqrt(x[i]).
func Sqrt(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		z[i] = float32(math.Sqrt(float64(x[i])))
	}
}

// Sigmoid computes z[i] = 1 / (1 + exp(-x[i])) as ½·tanh(x/2) + ½ with
// the vector Tanh; every pass stays within the caller's chunk.
func Sigmoid(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for lo := 0; lo < n; lo += block {
		hi := min(lo+block, n)
		xb, zb := x[lo:hi], z[lo:hi]
		Scale(xb, 0.5, zb)
		Tanh(zb, zb)
		Scale(zb, 0.5, zb)
		AddScalar(zb, 0.5, zb)
	}
}

// block is the sub-range the composed kernels work on so that their
// several passes stay in L1 whatever the caller's slice length.
const block = 4096

// Pow computes z[i] = x[i] ** p.
func Pow(x []float32, p float32, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	switch p {
	case 2:
		for i := range z {
			z[i] = x[i] * x[i]
		}
	case 0.5:
		Sqrt(x, z)
	case -1:
		for i := range z {
			z[i] = 1 / x[i]
		}
	default:
		pp := float64(p)
		for i := range z {
			z[i] = float32(math.Pow(float64(x[i]), pp))
		}
	}
}

// Abs computes z[i] = |x[i]|.
func Abs(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		z[i] = float32(math.Abs(float64(x[i])))
	}
}

// Sign computes z[i] = sign(x[i]) ∈ {-1, 0, 1}.
func Sign(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		switch {
		case x[i] > 0:
			z[i] = 1
		case x[i] < 0:
			z[i] = -1
		default:
			z[i] = 0
		}
	}
}

// GtZeroMask computes z[i] = 1 if x[i] > 0 else 0 (the ReLU derivative).
func GtZeroMask(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		if x[i] > 0 {
			z[i] = 1
		} else {
			z[i] = 0
		}
	}
}

// EqMask computes z[i] = 1 if x[i] == y[i] else 0.
func EqMask(x, y, z []float32) {
	n := checkLen3(x, y, z)
	x, y, z = x[:n], y[:n], z[:n]
	for i := range z {
		if x[i] == y[i] {
			z[i] = 1
		} else {
			z[i] = 0
		}
	}
}

// Clamp computes z[i] = min(max(x[i], lo), hi).
func Clamp(x []float32, lo, hi float32, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		v := x[i]
		if v < lo {
			v = lo
		} else if v > hi {
			v = hi
		}
		z[i] = v
	}
}

const geluC = 0.7978845608028654 // sqrt(2/pi)

// GELU computes the tanh approximation of the Gaussian error linear unit,
// 0.5·x·(1 + tanh(c·(x + 0.044715·x³))), from the vector kernels. x and z
// must not overlap.
func GELU(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for lo := 0; lo < n; lo += block {
		hi := min(lo+block, n)
		xb, zb := x[lo:hi], z[lo:hi]
		geluInner(xb, zb)
		Tanh(zb, zb)
		AddScalar(zb, 1, zb)
		Mul(zb, xb, zb)
		Scale(zb, 0.5, zb)
	}
}

// geluInner writes u = c·(x + 0.044715·x³) into z.
func geluInner(x, z []float32) {
	Mul(x, x, z)
	Mul(z, x, z)
	Scale(z, 0.044715, z)
	Axpy(1, x, z)
	Scale(z, geluC, z)
}

// GELUGrad computes the derivative of GELU. x and z must not overlap.
func GELUGrad(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	var t [block]float32
	for lo := 0; lo < n; lo += block {
		hi := min(lo+block, n)
		xb, zb, tb := x[lo:hi], z[lo:hi], t[:hi-lo]
		geluInner(xb, tb)
		Tanh(tb, tb) // tanh(u)
		for i, v := range xb {
			du := geluC * (1 + 3*0.044715*v*v)
			zb[i] = 0.5*(1+tb[i]) + 0.5*v*(1-tb[i]*tb[i])*du
		}
	}
}
