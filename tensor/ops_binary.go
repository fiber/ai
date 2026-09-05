package tensor

import (
	"github.com/fiber/ai/internal/kernel"
	"github.com/fiber/ai/internal/parallel"
	"os"
	"strconv"
)

// minChunk is the smallest number of elements handed to one goroutine by
// SIMD element-wise operations: below ~64K elements the cost of waking
// idle worker threads exceeds the work itself.
var minChunk = 1 << 13 // 8K elements: a 128K-element activation then uses all cores; FIBERAI_MIN_CHUNK overrides

// minChunkMath is the chunk size for transcendental element-wise
// operations (exp, tanh, ...), which cost ~10 ns per element.
const minChunkMath = 1 << 12

// binaryOp evaluates z = x op y with broadcasting into a new tensor.
//
// vec is the SIMD kernel for equal-length slices, sc (optional) computes
// x op scalar, and f is the scalar fallback used by the general strided
// path. commutative allows sc to be used when the scalar is on the left.
func binaryOp(op string, x, y *Tensor, vec kernel.BinaryFunc, sc kernel.ScalarFunc, commutative bool, f func(a, b float32) float32) *Tensor {
	shape := broadcastShapes(op, x.shape, y.shape)
	out := newTensorUninit(shape) // binaryInto writes every element
	binaryInto(out, x, y, vec, sc, commutative, f)
	return out
}

// binaryInto is binaryOp writing into out, which must be contiguous with
// the broadcast shape. out may alias x or y.
func binaryInto(out, x, y *Tensor, vec kernel.BinaryFunc, sc kernel.ScalarFunc, commutative bool, f func(a, b float32) float32) {
	n := out.size
	if n == 0 {
		return
	}
	shape := out.shape
	od := out.data[:n]
	if vec == nil { // mask-style ops without a SIMD kernel
		vec = func(x, y, z []float32) {
			for i := range z {
				z[i] = f(x[i], y[i])
			}
		}
	}

	// identical dense layouts: one flat pass
	if x.shape.Equal(shape) && y.shape.Equal(shape) && x.IsContiguous() && y.IsContiguous() {
		xd, yd := x.data[:n], y.data[:n]
		parallel.Range(n, minChunk, func(lo, hi int) { vec(xd[lo:hi], yd[lo:hi], od[lo:hi]) })
		return
	}
	// tensor op scalar
	if y.size == 1 && x.shape.Equal(shape) && x.IsContiguous() {
		xd, s := x.data[:n], y.data[0]
		if sc != nil {
			parallel.Range(n, minChunk, func(lo, hi int) { sc(xd[lo:hi], s, od[lo:hi]) })
		} else {
			parallel.Range(n, minChunk, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					od[i] = f(xd[i], s)
				}
			})
		}
		return
	}
	// scalar op tensor
	if x.size == 1 && y.shape.Equal(shape) && y.IsContiguous() {
		yd, s := y.data[:n], x.data[0]
		if sc != nil && commutative {
			parallel.Range(n, minChunk, func(lo, hi int) { sc(yd[lo:hi], s, od[lo:hi]) })
		} else {
			parallel.Range(n, minChunk, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					od[i] = f(s, yd[i])
				}
			})
		}
		return
	}

	// general strided broadcast, one row (last dimension) at a time
	bx, by := broadcastStrides(x, shape), broadcastStrides(y, shape)
	nd := len(shape)
	last := shape[nd-1]
	xs, ys := bx[nd-1], by[nd-1]
	rows := n / last
	xd, yd := x.data, y.data
	parallel.Range(rows, max(1, minChunk/last), func(lo, hi int) {
		walkRows(shape, [][]int{bx, by}, lo, hi, func(row int, offs []int) {
			orow := od[row*last : (row+1)*last]
			xo, yo := offs[0], offs[1]
			switch {
			case xs == 1 && ys == 1:
				vec(xd[xo:xo+last], yd[yo:yo+last], orow)
			case xs == 1 && ys == 0 && sc != nil:
				sc(xd[xo:xo+last], yd[yo], orow)
			case xs == 0 && ys == 1 && sc != nil && commutative:
				sc(yd[yo:yo+last], xd[xo], orow)
			default:
				for j := range orow {
					orow[j] = f(xd[xo], yd[yo])
					xo += xs
					yo += ys
				}
			}
		})
	})
}

// sumTo reduces g (a gradient of a broadcast result) back to shape by
// summing over the broadcast dimensions.
func sumTo(g *Tensor, shape Shape) *Tensor {
	if g.shape.Equal(shape) {
		return g
	}
	lead := len(g.shape) - len(shape)
	var dims []int
	for d := 0; d < lead; d++ {
		dims = append(dims, d)
	}
	for d, size := range shape {
		if size == 1 && g.shape[lead+d] != 1 {
			dims = append(dims, lead+d)
		}
	}
	r := reduceDims(g, dims, true, reduceSum)
	return r.Reshape(shape...)
}

// Add returns t + u with broadcasting.
func (t *Tensor) Add(u *Tensor) *Tensor {
	out := binaryOp("Add", t, u, kernel.Add, kernel.AddScalar, true, func(a, b float32) float32 { return a + b })
	return record(out, "Add", []*Tensor{t, u}, func(gy *Tensor) {
		t.accumGrad(sumTo(gy, t.shape))
		u.accumGrad(sumTo(gy, u.shape))
	})
}

// Sub returns t - u with broadcasting.
func (t *Tensor) Sub(u *Tensor) *Tensor {
	out := binaryOp("Sub", t, u, kernel.Sub, nil, false, func(a, b float32) float32 { return a - b })
	return record(out, "Sub", []*Tensor{t, u}, func(gy *Tensor) {
		t.accumGrad(sumTo(gy, t.shape))
		if u.requiresGrad {
			u.accumGrad(sumTo(gy, u.shape).Neg())
		}
	})
}

// Mul returns the element-wise product t * u with broadcasting.
func (t *Tensor) Mul(u *Tensor) *Tensor {
	out := binaryOp("Mul", t, u, kernel.Mul, kernel.Scale, true, func(a, b float32) float32 { return a * b })
	td, ud := t.saved(), u.saved()
	return record(out, "Mul", []*Tensor{t, u}, func(gy *Tensor) {
		if t.requiresGrad {
			t.accumGrad(sumTo(gy.Mul(ud), t.shape))
		}
		if u.requiresGrad {
			u.accumGrad(sumTo(gy.Mul(td), u.shape))
		}
	})
}

// Div returns the element-wise quotient t / u with broadcasting.
func (t *Tensor) Div(u *Tensor) *Tensor {
	out := binaryOp("Div", t, u, kernel.Div, nil, false, func(a, b float32) float32 { return a / b })
	ud, od := u.saved(), out.saved()
	return record(out, "Div", []*Tensor{t, u}, func(gy *Tensor) {
		if t.requiresGrad {
			t.accumGrad(sumTo(gy.Div(ud), t.shape))
		}
		if u.requiresGrad { // d/du (t/u) = -t/u² = -out/u
			u.accumGrad(sumTo(gy.Mul(od).Div(ud).Neg(), u.shape))
		}
	})
}

// Maximum returns the element-wise maximum of t and u with broadcasting.
// Where both are equal the gradient flows to t.
func (t *Tensor) Maximum(u *Tensor) *Tensor {
	out := binaryOp("Maximum", t, u, kernel.Maximum, kernel.MaxScalar, true, func(a, b float32) float32 { return max(a, b) })
	td, ud := t.saved(), u.saved()
	return record(out, "Maximum", []*Tensor{t, u}, func(gy *Tensor) {
		if t.requiresGrad {
			mask := binaryOp("Maximum", td, ud, nil, nil, false, func(a, b float32) float32 {
				if a >= b {
					return 1
				}
				return 0
			})
			t.accumGrad(sumTo(gy.Mul(mask), t.shape))
		}
		if u.requiresGrad {
			mask := binaryOp("Maximum", td, ud, nil, nil, false, func(a, b float32) float32 {
				if b > a {
					return 1
				}
				return 0
			})
			u.accumGrad(sumTo(gy.Mul(mask), u.shape))
		}
	})
}

// Minimum returns the element-wise minimum of t and u with broadcasting.
func (t *Tensor) Minimum(u *Tensor) *Tensor {
	return t.Neg().Maximum(u.Neg()).Neg()
}

// Pow returns t raised element-wise to the power p.
func (t *Tensor) Pow(p float32) *Tensor {
	tc := t.Contiguous()
	out := unaryOpMath(tc, func(x, z []float32) { kernel.Pow(x, p, z) })
	td := tc.saved()
	return record(out, "Pow", []*Tensor{tc}, func(gy *Tensor) {
		// d/dx x^p = p·x^(p-1)
		tc.accumGrad(zipMap(gy, td, func(g, x, z []float32) {
			kernel.Pow(x, p-1, z)
			for i := range z {
				z[i] *= p * g[i]
			}
		}))
	})
}

// AddScalar returns t + s.
func (t *Tensor) AddScalar(s float32) *Tensor {
	tc := t.Contiguous()
	out := unaryOp(tc, func(x, z []float32) { kernel.AddScalar(x, s, z) })
	return record(out, "AddScalar", []*Tensor{tc}, func(gy *Tensor) { tc.accumGrad(gy) })
}

// SubScalar returns t - s.
func (t *Tensor) SubScalar(s float32) *Tensor { return t.AddScalar(-s) }

// MulScalar returns t * s.
func (t *Tensor) MulScalar(s float32) *Tensor {
	tc := t.Contiguous()
	out := unaryOp(tc, func(x, z []float32) { kernel.Scale(x, s, z) })
	return record(out, "MulScalar", []*Tensor{tc}, func(gy *Tensor) { tc.accumGrad(gy.MulScalar(s)) })
}

// DivScalar returns t / s.
func (t *Tensor) DivScalar(s float32) *Tensor { return t.MulScalar(1 / s) }

func init() {
	if v, err := strconv.Atoi(os.Getenv("FIBERAI_MIN_CHUNK")); err == nil && v > 0 {
		minChunk = v
	}
}
