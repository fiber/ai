package tensor

import (
	"fmt"
	"strings"
)

// Shape is the list of dimension sizes of a tensor. A scalar has an empty
// shape.
type Shape []int

// Size returns the number of elements described by the shape.
func (s Shape) Size() int {
	n := 1
	for _, d := range s {
		n *= d
	}
	return n
}

// Equal reports whether two shapes are identical.
func (s Shape) Equal(o Shape) bool {
	if len(s) != len(o) {
		return false
	}
	for i := range s {
		if s[i] != o[i] {
			return false
		}
	}
	return true
}

func (s Shape) String() string {
	var b strings.Builder
	b.WriteByte('[')
	for i, d := range s {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%d", d)
	}
	b.WriteByte(']')
	return b.String()
}

func (s Shape) clone() Shape { return append(Shape(nil), s...) }

func checkShape(op string, shape Shape) {
	for _, d := range shape {
		if d < 0 {
			fail(op, "negative dimension in shape %v", shape)
		}
	}
}

// contiguousStrides returns the row-major strides for shape.
func contiguousStrides(shape Shape) []int {
	strides := make([]int, len(shape))
	stride := 1
	for d := len(shape) - 1; d >= 0; d-- {
		strides[d] = stride
		stride *= shape[d]
	}
	return strides
}

// isContiguous reports whether the strides describe a dense row-major
// layout for shape (size-1 dimensions may have any stride).
func isContiguous(shape Shape, strides []int) bool {
	expected := 1
	for d := len(shape) - 1; d >= 0; d-- {
		if shape[d] != 1 && strides[d] != expected {
			return false
		}
		expected *= shape[d]
	}
	return true
}

// normDim converts a possibly negative dimension index into [0, n).
func normDim(op string, dim, n int) int {
	d := dim
	if d < 0 {
		d += n
	}
	if d < 0 || d >= n {
		fail(op, "dimension %d out of range for %d-D tensor", dim, n)
	}
	return d
}

// broadcastShapes returns the NumPy broadcast of a and b.
func broadcastShapes(op string, a, b Shape) Shape {
	if a.Equal(b) {
		return a.clone()
	}
	n := max(len(a), len(b))
	out := make(Shape, n)
	for i := 1; i <= n; i++ {
		da, db := 1, 1
		if i <= len(a) {
			da = a[len(a)-i]
		}
		if i <= len(b) {
			db = b[len(b)-i]
		}
		switch {
		case da == db, db == 1:
			out[n-i] = da
		case da == 1:
			out[n-i] = db
		default:
			fail(op, "shapes %v and %v are not broadcastable", a, b)
		}
	}
	return out
}

// broadcastStrides returns t's strides aligned to the (broader) shape,
// with 0 for broadcast dimensions.
func broadcastStrides(t *Tensor, shape Shape) []int {
	out := make([]int, len(shape))
	off := len(shape) - len(t.shape)
	for d, size := range t.shape {
		if size != 1 {
			out[off+d] = t.strides[d]
		}
	}
	return out
}
