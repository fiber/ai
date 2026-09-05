package tensor

import (
	"math"

	"github.com/fiber/ai/internal/kernel"
	"github.com/fiber/ai/internal/parallel"
)

// Tensor is an n-dimensional float32 array. The zero value is not usable;
// create tensors with the constructors in this package.
type Tensor struct {
	data    []float32 // data[0] is the first element of this (possibly strided) view
	shape   Shape
	strides []int // element strides per dimension
	size    int   // cached shape.Size()

	requiresGrad bool
	retainGrad   bool
	grad         *Tensor
	node         *node
}

// newTensor allocates a zero-filled contiguous tensor.
func newTensor(shape Shape) *Tensor {
	shape = shape.clone()
	return &Tensor{
		data:    make([]float32, shape.Size()),
		shape:   shape,
		strides: contiguousStrides(shape),
		size:    shape.Size(),
	}
}

// wrap builds a contiguous tensor over data without copying.
func wrap(data []float32, shape Shape) *Tensor {
	return &Tensor{data: data, shape: shape, strides: contiguousStrides(shape), size: shape.Size()}
}

// view creates a tensor sharing t's storage with new shape and strides.
func view(data []float32, shape Shape, strides []int) *Tensor {
	return &Tensor{data: data, shape: shape, strides: strides, size: shape.Size()}
}

// New returns a tensor of the given shape holding a copy of data.
// The number of elements must match the shape.
func New(data []float32, shape ...int) *Tensor {
	checkShape("New", shape)
	if len(data) != Shape(shape).Size() {
		fail("New", "%d elements do not fit shape %v", len(data), Shape(shape))
	}
	return wrap(append([]float32(nil), data...), Shape(shape).clone())
}

// FromSlice returns a tensor that uses data as its storage without copying.
// Modifying data afterwards modifies the tensor.
func FromSlice(data []float32, shape ...int) *Tensor {
	checkShape("FromSlice", shape)
	if len(data) != Shape(shape).Size() {
		fail("FromSlice", "%d elements do not fit shape %v", len(data), Shape(shape))
	}
	return wrap(data, Shape(shape).clone())
}

// Zeros returns a zero-filled tensor.
func Zeros(shape ...int) *Tensor {
	checkShape("Zeros", shape)
	return newTensor(shape)
}

// Ones returns a tensor filled with 1.
func Ones(shape ...int) *Tensor { return Full(1, shape...) }

// Full returns a tensor filled with v.
func Full(v float32, shape ...int) *Tensor {
	checkShape("Full", shape)
	t := newTensor(shape)
	for i := range t.data {
		t.data[i] = v
	}
	return t
}

// Scalar returns a 0-dimensional tensor holding v.
func Scalar(v float32) *Tensor { return wrap([]float32{v}, Shape{}) }

// ZerosLike returns a zero tensor with t's shape.
func ZerosLike(t *Tensor) *Tensor { return newTensor(t.shape) }

// OnesLike returns a tensor of ones with t's shape.
func OnesLike(t *Tensor) *Tensor { return Full(1, t.shape...) }

// Eye returns the n×n identity matrix.
func Eye(n int) *Tensor {
	t := Zeros(n, n)
	for i := 0; i < n; i++ {
		t.data[i*n+i] = 1
	}
	return t
}

// Arange returns the 1-D tensor [start, start+step, ...) below stop.
func Arange(start, stop, step float32) *Tensor {
	if step == 0 {
		fail("Arange", "step must not be zero")
	}
	n := int(math.Ceil(float64((stop - start) / step)))
	if n < 0 {
		n = 0
	}
	t := Zeros(n)
	for i := range t.data {
		t.data[i] = start + float32(i)*step
	}
	return t
}

// Linspace returns n evenly spaced values from start to stop inclusive.
func Linspace(start, stop float32, n int) *Tensor {
	t := Zeros(n)
	if n == 1 {
		t.data[0] = start
		return t
	}
	for i := range t.data {
		t.data[i] = start + (stop-start)*float32(i)/float32(n-1)
	}
	return t
}

// OneHot returns a [len(indices), classes] tensor with a 1 at each index.
func OneHot(indices []int, classes int) *Tensor {
	t := Zeros(len(indices), classes)
	for i, c := range indices {
		if c < 0 || c >= classes {
			fail("OneHot", "index %d out of range [0, %d)", c, classes)
		}
		t.data[i*classes+c] = 1
	}
	return t
}

// Shape returns a copy of the tensor's shape.
func (t *Tensor) Shape() Shape { return t.shape.clone() }

// Dims returns the number of dimensions.
func (t *Tensor) Dims() int { return len(t.shape) }

// Size returns the total number of elements.
func (t *Tensor) Size() int { return t.size }

// Dim returns the size of dimension d (negative d counts from the end).
func (t *Tensor) Dim(d int) int { return t.shape[normDim("Dim", d, len(t.shape))] }

// Strides returns a copy of the element strides.
func (t *Tensor) Strides() []int { return append([]int(nil), t.strides...) }

// IsContiguous reports whether the tensor is stored densely in row-major
// order.
func (t *Tensor) IsContiguous() bool { return isContiguous(t.shape, t.strides) }

// Data returns the tensor's elements in row-major order. For a contiguous
// tensor this is the backing slice itself (writes are visible in the
// tensor); otherwise it is a copy.
func (t *Tensor) Data() []float32 {
	if t.IsContiguous() {
		return t.data[:t.size]
	}
	return t.Contiguous().data
}

// Float32s always returns a fresh copy of the elements in row-major order.
func (t *Tensor) Float32s() []float32 {
	return append([]float32(nil), t.Data()...)
}

// offset returns the storage offset of the element at idx.
func (t *Tensor) offset(op string, idx []int) int {
	if len(idx) != len(t.shape) {
		fail(op, "got %d indices for %d-D tensor", len(idx), len(t.shape))
	}
	off := 0
	for d, i := range idx {
		if i < 0 {
			i += t.shape[d]
		}
		if i < 0 || i >= t.shape[d] {
			fail(op, "index %d out of range for dimension %d of size %d", idx[d], d, t.shape[d])
		}
		off += i * t.strides[d]
	}
	return off
}

// At returns the element at the given index (one per dimension).
func (t *Tensor) At(idx ...int) float32 { return t.data[t.offset("At", idx)] }

// Set assigns v to the element at idx. It does not participate in autograd.
func (t *Tensor) Set(v float32, idx ...int) {
	t.checkInPlace("Set")
	t.data[t.offset("Set", idx)] = v
}

// Item returns the value of a single-element tensor.
func (t *Tensor) Item() float32 {
	if t.size != 1 {
		fail("Item", "tensor has %d elements, want 1", t.size)
	}
	return t.data[0]
}

// Contiguous returns t itself if it is contiguous, otherwise a dense copy.
// The result is differentiable.
func (t *Tensor) Contiguous() *Tensor {
	if t.IsContiguous() {
		return t
	}
	out := newTensor(t.shape)
	copyStrided(out.data, t)
	return record(out, "Contiguous", []*Tensor{t}, func(gy *Tensor) { t.accumGrad(gy) })
}

// Clone returns a dense copy of t. The copy is differentiable (gradients
// flow back to t); use Detach for a graph-free copy.
func (t *Tensor) Clone() *Tensor {
	out := newTensor(t.shape)
	copyStrided(out.data, t)
	return record(out, "Clone", []*Tensor{t}, func(gy *Tensor) { t.accumGrad(gy) })
}

// copyStrided writes t's elements in row-major order into dst.
func copyStrided(dst []float32, t *Tensor) {
	if t.size == 0 {
		return
	}
	if t.IsContiguous() {
		copy(dst, t.data[:t.size])
		return
	}
	nd := len(t.shape)
	last := t.shape[nd-1]
	ls := t.strides[nd-1]
	if nd >= 2 && ls != 1 && t.strides[nd-2] == 1 && last >= 8 && t.shape[nd-2] >= 8 {
		copyTransposed(dst, t)
		return
	}
	rows := t.size / last
	parallel.Range(rows, max(1, 4096/last), func(lo, hi int) {
		walkRows(t.shape, [][]int{t.strides}, lo, hi, func(row int, offs []int) {
			d := dst[row*last : (row+1)*last]
			s := offs[0]
			if ls == 1 {
				copy(d, t.data[s:s+last])
				return
			}
			for j := range d {
				d[j] = t.data[s]
				s += ls
			}
		})
	})
}

// copyTransposed handles the transposed-matrix layout (unit stride along
// the second-to-last dimension) with cache-blocked 32×32 tiles, for every
// leading index. Reading down a column of the source touches the same
// cache lines for 32 consecutive rows instead of once per row.
func copyTransposed(dst []float32, t *Tensor) {
	const bs = 32
	nd := len(t.shape)
	rows, cols := t.shape[nd-2], t.shape[nd-1]
	cs := t.strides[nd-1] // source column stride (rows are unit stride)
	outer := t.size / (rows * cols)
	lead := view(t.data, t.shape[:nd-2], t.strides[:nd-2])
	offs := make([]int, outer)
	walkRows(append(lead.shape.clone(), 1), [][]int{append(append([]int(nil), lead.strides...), 0)}, 0, outer, func(o int, off []int) {
		offs[o] = off[0]
	})
	rowBlocks := (rows + bs - 1) / bs
	parallel.For(outer*rowBlocks, func(task int) {
		o, rb := task/rowBlocks, task%rowBlocks
		src := t.data[offs[o]:]
		d := dst[o*rows*cols : (o+1)*rows*cols]
		i0, i1 := rb*bs, min(rb*bs+bs, rows)
		for j0 := 0; j0 < cols; j0 += bs {
			j1 := min(j0+bs, cols)
			for j := j0; j < j1; j++ { // source column j is contiguous over i
				col := src[j*cs+i0 : j*cs+i1]
				for ii, v := range col {
					d[(i0+ii)*cols+j] = v
				}
			}
		}
	})
}

// walkRows visits rows [lo, hi) of a tensor with the given shape in
// row-major order, where a row is one run along the last dimension. For
// each row it calls fn with the row index and the element offset of the
// row start for every operand (strides[i] are operand i's strides aligned
// to shape). A 0-D shape has exactly one row.
func walkRows(shape Shape, strides [][]int, lo, hi int, fn func(row int, offs []int)) {
	nd := len(shape)
	offs := make([]int, len(strides))
	if nd <= 1 {
		if lo == 0 && hi > 0 {
			fn(0, offs)
		}
		return
	}
	outer := shape[:nd-1]
	idx := make([]int, nd-1)
	// position the counters at row lo
	rem := lo
	for d := nd - 2; d >= 0; d-- {
		idx[d] = rem % outer[d]
		rem /= outer[d]
		for i, st := range strides {
			offs[i] += idx[d] * st[d]
		}
	}
	for row := lo; row < hi; row++ {
		fn(row, offs)
		for d := nd - 2; d >= 0; d-- {
			idx[d]++
			for i, st := range strides {
				offs[i] += st[d]
			}
			if idx[d] < outer[d] {
				break
			}
			for i, st := range strides {
				offs[i] -= st[d] * outer[d]
			}
			idx[d] = 0
		}
	}
}

// Equal reports whether u has the same shape and exactly the same values.
func (t *Tensor) Equal(u *Tensor) bool {
	if !t.shape.Equal(u.shape) {
		return false
	}
	a, b := t.Data(), u.Data()
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// AllClose reports whether u has the same shape and all elements satisfy
// |t-u| <= atol + rtol*|u|.
func (t *Tensor) AllClose(u *Tensor, rtol, atol float64) bool {
	if !t.shape.Equal(u.shape) {
		return false
	}
	a, b := t.Data(), u.Data()
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		if math.IsNaN(x) || math.IsNaN(y) || math.Abs(x-y) > atol+rtol*math.Abs(y) {
			return false
		}
	}
	return true
}

// SetThreads limits the number of goroutines used by tensor operations.
// n <= 0 resets to runtime.GOMAXPROCS(0).
func SetThreads(n int) { parallel.SetWorkers(n) }

// Threads returns the current goroutine limit for tensor operations.
func Threads() int { return parallel.Workers() }

// Backend returns the name of the active SIMD kernel implementation
// ("neon", "avx512", "avx2" or "generic").
func Backend() string { return kernel.Impl }

// BackendWarnings lists SIMD implementations that were disabled at start-up
// because they failed self-verification. Normally empty.
func BackendWarnings() []string { return append([]string(nil), kernel.Warnings...) }
