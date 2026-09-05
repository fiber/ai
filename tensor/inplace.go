package tensor

import (
	"github.com/fiber/ai/internal/kernel"
	"github.com/fiber/ai/internal/parallel"
	"runtime"
)

// In-place operations modify a tensor's storage directly. They are not
// recorded in the autograd graph and panic when called on a tensor that
// requires grad while grad mode is on; wrap parameter updates in NoGrad.

// Fill sets every element to v.
func (t *Tensor) Fill(v float32) *Tensor {
	t.checkInPlace("Fill")
	if t.IsContiguous() {
		d := t.data[:t.size]
		for i := range d {
			d[i] = v
		}
		return t
	}
	assign(t, Full(v, t.shape...))
	return t
}

// Zero sets every element to 0.
func (t *Tensor) Zero() *Tensor { return t.Fill(0) }

// CopyFrom copies u's elements into t. Shapes must match.
func (t *Tensor) CopyFrom(u *Tensor) *Tensor {
	t.checkInPlace("CopyFrom")
	if !t.shape.Equal(u.shape) {
		fail("CopyFrom", "shape mismatch %v vs %v", t.shape, u.shape)
	}
	assign(t, u)
	return t
}

func (t *Tensor) inplaceBinary(op string, u *Tensor, vec kernel.BinaryFunc, sc kernel.ScalarFunc, commutative bool, f func(a, b float32) float32) *Tensor {
	t.checkInPlace(op)
	if !t.IsContiguous() {
		fail(op, "in-place operation requires a contiguous tensor")
	}
	if shape := broadcastShapes(op, t.shape, u.shape); !shape.Equal(t.shape) {
		fail(op, "cannot broadcast %v into %v in place", u.shape, t.shape)
	}
	binaryInto(t, t, u, vec, sc, commutative, f)
	return t
}

// AddInPlace computes t += u (u broadcastable to t).
func (t *Tensor) AddInPlace(u *Tensor) *Tensor {
	return t.inplaceBinary("AddInPlace", u, kernel.Add, kernel.AddScalar, true, func(a, b float32) float32 { return a + b })
}

// SubInPlace computes t -= u.
func (t *Tensor) SubInPlace(u *Tensor) *Tensor {
	return t.inplaceBinary("SubInPlace", u, kernel.Sub, nil, false, func(a, b float32) float32 { return a - b })
}

// MulInPlace computes t *= u.
func (t *Tensor) MulInPlace(u *Tensor) *Tensor {
	return t.inplaceBinary("MulInPlace", u, kernel.Mul, kernel.Scale, true, func(a, b float32) float32 { return a * b })
}

// DivInPlace computes t /= u.
func (t *Tensor) DivInPlace(u *Tensor) *Tensor {
	return t.inplaceBinary("DivInPlace", u, kernel.Div, nil, false, func(a, b float32) float32 { return a / b })
}

// MulScalarInPlace computes t *= s.
func (t *Tensor) MulScalarInPlace(s float32) *Tensor {
	t.checkInPlace("MulScalarInPlace")
	d := t.values()
	parallel.Range(len(d), minChunk, func(lo, hi int) { kernel.Scale(d[lo:hi], s, d[lo:hi]) })
	if !t.IsContiguous() {
		assign(t, wrap(d, t.shape))
	}
	return t
}

// AddScaledInPlace computes t += alpha * u (the BLAS axpy). Shapes must
// match; u may be strided.
func (t *Tensor) AddScaledInPlace(u *Tensor, alpha float32) *Tensor {
	t.checkInPlace("AddScaledInPlace")
	if !t.shape.Equal(u.shape) {
		fail("AddScaledInPlace", "shape mismatch %v vs %v", t.shape, u.shape)
	}
	if !t.IsContiguous() {
		fail("AddScaledInPlace", "in-place operation requires a contiguous tensor")
	}
	td, ud := t.data[:t.size], u.values()
	parallel.Range(t.size, minChunk, func(lo, hi int) { kernel.Axpy(alpha, ud[lo:hi], td[lo:hi]) })
	runtime.KeepAlive(u)
	return t
}
