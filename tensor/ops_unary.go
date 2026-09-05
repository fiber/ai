package tensor

import (
	"github.com/fiber/ai/internal/kernel"
	"github.com/fiber/ai/internal/parallel"
)

// unaryOp applies f chunk-wise to a contiguous tensor, producing a new
// tensor of the same shape. It records nothing; callers wrap it in record.
func unaryOp(x *Tensor, f func(x, z []float32)) *Tensor { return unaryOpChunk(x, minChunk, f) }

// unaryOpMath is unaryOp for expensive per-element functions, which pay
// off parallelising at much smaller sizes.
func unaryOpMath(x *Tensor, f func(x, z []float32)) *Tensor { return unaryOpChunk(x, minChunkMath, f) }

func unaryOpChunk(x *Tensor, chunk int, f func(x, z []float32)) *Tensor {
	out := newTensorUninit(x.shape)
	n := out.size
	if n == 0 {
		return out
	}
	xd := x.values()
	parallel.Range(n, chunk, func(lo, hi int) { f(xd[lo:hi], out.data[lo:hi]) })
	return out
}

// zipMap combines two same-shaped tensors chunk-wise: f(a, b, z).
func zipMap(a, b *Tensor, f func(a, b, z []float32)) *Tensor {
	if !a.shape.Equal(b.shape) {
		fail("zipMap", "internal: shape mismatch %v vs %v", a.shape, b.shape)
	}
	out := newTensorUninit(a.shape)
	n := out.size
	if n == 0 {
		return out
	}
	ad, bd := a.values(), b.values()
	parallel.Range(n, minChunk, func(lo, hi int) { f(ad[lo:hi], bd[lo:hi], out.data[lo:hi]) })
	return out
}

// Neg returns -t.
func (t *Tensor) Neg() *Tensor {
	tc := t.Contiguous()
	out := unaryOp(tc, func(x, z []float32) { kernel.Scale(x, -1, z) })
	return record(out, "Neg", []*Tensor{tc}, func(gy *Tensor) { tc.accumGrad(gy.Neg()) })
}

// Exp returns e^t.
func (t *Tensor) Exp() *Tensor {
	tc := t.Contiguous()
	out := unaryOpMath(tc, kernel.Exp)
	od := out.Detach()
	return record(out, "Exp", []*Tensor{tc}, func(gy *Tensor) { tc.accumGrad(gy.Mul(od)) })
}

// Log returns the natural logarithm of t.
func (t *Tensor) Log() *Tensor {
	tc := t.Contiguous()
	out := unaryOpMath(tc, kernel.Log)
	td := tc.Detach()
	return record(out, "Log", []*Tensor{tc}, func(gy *Tensor) { tc.accumGrad(gy.Div(td)) })
}

// Sqrt returns the element-wise square root.
func (t *Tensor) Sqrt() *Tensor {
	tc := t.Contiguous()
	out := unaryOpMath(tc, kernel.Sqrt)
	od := out.Detach()
	return record(out, "Sqrt", []*Tensor{tc}, func(gy *Tensor) {
		tc.accumGrad(zipMap(gy, od, func(g, y, z []float32) {
			for i := range z {
				z[i] = 0.5 * g[i] / y[i]
			}
		}))
	})
}

// Square returns t².
func (t *Tensor) Square() *Tensor {
	tc := t.Contiguous()
	out := unaryOp(tc, func(x, z []float32) { kernel.Mul(x, x, z) })
	td := tc.Detach()
	return record(out, "Square", []*Tensor{tc}, func(gy *Tensor) {
		tc.accumGrad(zipMap(gy, td, func(g, x, z []float32) {
			for i := range z {
				z[i] = 2 * x[i] * g[i]
			}
		}))
	})
}

// Abs returns |t|. The gradient at 0 is 0.
func (t *Tensor) Abs() *Tensor {
	tc := t.Contiguous()
	out := unaryOp(tc, kernel.Abs)
	td := tc.Detach()
	return record(out, "Abs", []*Tensor{tc}, func(gy *Tensor) {
		tc.accumGrad(zipMap(gy, td, func(g, x, z []float32) {
			kernel.Sign(x, z)
			kernel.Mul(z, g, z)
		}))
	})
}

// Tanh returns the hyperbolic tangent.
func (t *Tensor) Tanh() *Tensor {
	tc := t.Contiguous()
	out := unaryOpMath(tc, kernel.Tanh)
	od := out.Detach()
	return record(out, "Tanh", []*Tensor{tc}, func(gy *Tensor) {
		tc.accumGrad(zipMap(gy, od, func(g, y, z []float32) {
			for i := range z {
				z[i] = g[i] * (1 - y[i]*y[i])
			}
		}))
	})
}

// Sigmoid returns 1 / (1 + e^-t).
func (t *Tensor) Sigmoid() *Tensor {
	tc := t.Contiguous()
	out := unaryOpMath(tc, kernel.Sigmoid)
	od := out.Detach()
	return record(out, "Sigmoid", []*Tensor{tc}, func(gy *Tensor) {
		tc.accumGrad(zipMap(gy, od, func(g, y, z []float32) {
			for i := range z {
				z[i] = g[i] * y[i] * (1 - y[i])
			}
		}))
	})
}

// ReLU returns max(t, 0).
func (t *Tensor) ReLU() *Tensor {
	tc := t.Contiguous()
	out := unaryOp(tc, func(x, z []float32) { kernel.MaxScalar(x, 0, z) })
	td := tc.Detach()
	return record(out, "ReLU", []*Tensor{tc}, func(gy *Tensor) {
		tc.accumGrad(zipMap(gy, td, func(g, x, z []float32) {
			kernel.GtZeroMask(x, z)
			kernel.Mul(z, g, z)
		}))
	})
}

// GELU returns the Gaussian error linear unit (tanh approximation).
func (t *Tensor) GELU() *Tensor {
	tc := t.Contiguous()
	out := unaryOpMath(tc, kernel.GELU)
	td := tc.Detach()
	return record(out, "GELU", []*Tensor{tc}, func(gy *Tensor) {
		tc.accumGrad(zipMap(gy, td, func(g, x, z []float32) {
			kernel.GELUGrad(x, z)
			kernel.Mul(z, g, z)
		}))
	})
}

// Clamp limits every element to [lo, hi]. The gradient is 1 inside the
// range and 0 outside.
func (t *Tensor) Clamp(lo, hi float32) *Tensor {
	tc := t.Contiguous()
	out := unaryOp(tc, func(x, z []float32) { kernel.Clamp(x, lo, hi, z) })
	td := tc.Detach()
	return record(out, "Clamp", []*Tensor{tc}, func(gy *Tensor) {
		tc.accumGrad(zipMap(gy, td, func(g, x, z []float32) {
			for i := range z {
				if x[i] >= lo && x[i] <= hi {
					z[i] = g[i]
				} else {
					z[i] = 0
				}
			}
		}))
	})
}
