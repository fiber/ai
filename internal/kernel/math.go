package kernel

import "math"

// Transcendental and miscellaneous element-wise kernels. These currently
// have Go-only implementations (per-element math calls); they are still
// parallelised by the tensor layer.

// Log computes z[i] = ln(x[i]).
func Log(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		z[i] = float32(math.Log(float64(x[i])))
	}
}

// Sqrt computes z[i] = sqrt(x[i]).
func Sqrt(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		z[i] = float32(math.Sqrt(float64(x[i])))
	}
}

// Tanh computes z[i] = tanh(x[i]).
func Tanh(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		z[i] = float32(math.Tanh(float64(x[i])))
	}
}

// Sigmoid computes z[i] = 1 / (1 + exp(-x[i])) using the vector Exp.
func Sigmoid(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	Scale(x, -1, z)
	Exp(z, z)
	for i := range z {
		z[i] = 1 / (1 + z[i])
	}
}

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

// GELU computes the tanh approximation of the Gaussian error linear unit.
func GELU(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		v := float64(x[i])
		z[i] = float32(0.5 * v * (1 + math.Tanh(geluC*(v+0.044715*v*v*v))))
	}
}

// GELUGrad computes the derivative of GELU.
func GELUGrad(x, z []float32) {
	n := checkLen2(x, z)
	x, z = x[:n], z[:n]
	for i := range z {
		v := float64(x[i])
		u := geluC * (v + 0.044715*v*v*v)
		t := math.Tanh(u)
		du := geluC * (1 + 3*0.044715*v*v)
		z[i] = float32(0.5*(1+t) + 0.5*v*(1-t*t)*du)
	}
}
