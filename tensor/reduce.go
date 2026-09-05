package tensor

import (
	"sync"

	"github.com/fiber/ai/internal/kernel"
	"github.com/fiber/ai/internal/parallel"
)

type reduceKind int

const (
	reduceSum reduceKind = iota
	reduceMax
)

// normDims validates and sorts reduction dimensions; empty means all.
func normDims(op string, dims []int, nd int) []int {
	if len(dims) == 0 {
		out := make([]int, nd)
		for i := range out {
			out[i] = i
		}
		return out
	}
	out := make([]int, 0, len(dims))
	seen := make([]bool, nd)
	for _, d := range dims {
		d = normDim(op, d, nd)
		if seen[d] {
			fail(op, "dimension %d given twice", d)
		}
		seen[d] = true
		out = append(out, d)
	}
	// ascending order
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

// reduceDims reduces t over dims (ascending, normalised). With keep the
// reduced dimensions stay as size 1; otherwise they are removed.
func reduceDims(t *Tensor, dims []int, keep bool, kind reduceKind) *Tensor {
	r := t.Contiguous().Detach()
	if len(dims) == len(t.shape) && t.size > 0 {
		r = reduceAll(r, kind)
	} else {
		// Highest dimension first: its rows are contiguous, so the bulk of
		// the data is consumed by the fast horizontal kernels and the
		// remaining strided reductions see far less data. Dimensions are
		// kept (size 1) during the loop so indices stay valid.
		for i := len(dims) - 1; i >= 0; i-- {
			r = reduceOne(r, dims[i], kind)
		}
	}
	if keep {
		return r
	}
	shape := make(Shape, 0, len(t.shape))
	drop := make([]bool, len(t.shape))
	for _, d := range dims {
		drop[d] = true
	}
	for d, s := range t.shape {
		if !drop[d] {
			shape = append(shape, s)
		}
	}
	return view(r, r.data, shape, contiguousStrides(shape))
}

// reduceAll reduces every element of a contiguous, non-empty tensor into a
// tensor of all-1 shape with one flat parallel pass.
func reduceAll(t *Tensor, kind reduceKind) *Tensor {
	shape := make(Shape, len(t.shape))
	for i := range shape {
		shape[i] = 1
	}
	out := newTensor(shape)
	xd := t.data[:t.size]
	var mu sync.Mutex
	first := true
	parallel.Range(t.size, minChunk, func(lo, hi int) {
		var v float32
		switch kind {
		case reduceSum:
			v = kernel.Sum(xd[lo:hi])
		case reduceMax:
			v = kernel.Max(xd[lo:hi])
		}
		mu.Lock()
		switch {
		case first:
			out.data[0], first = v, false
		case kind == reduceSum:
			out.data[0] += v
		case v > out.data[0]:
			out.data[0] = v
		}
		mu.Unlock()
	})
	return out
}

// reduceOne reduces one dimension of a contiguous tensor, keeping it as 1.
func reduceOne(t *Tensor, dim int, kind reduceKind) *Tensor {
	shape := t.shape.clone()
	d := shape[dim]
	shape[dim] = 1
	if t.size == 0 {
		return newTensor(shape)
	}
	out := newTensorUninit(shape) // every element is written below
	outer, inner := 1, 1
	for i := 0; i < dim; i++ {
		outer *= t.shape[i]
	}
	for i := dim + 1; i < len(t.shape); i++ {
		inner *= t.shape[i]
	}
	xd, od := t.data, out.data

	if inner == 1 {
		// contiguous rows of length d: one horizontal reduction each
		parallel.Range(outer, max(1, minChunk/d), func(lo, hi int) {
			for o := lo; o < hi; o++ {
				row := xd[o*d : (o+1)*d]
				switch kind {
				case reduceSum:
					od[o] = kernel.Sum(row)
				case reduceMax:
					od[o] = kernel.Max(row)
				}
			}
		})
		return out
	}

	// Vertical reduction: d slabs of length inner are folded into one.
	// Rows are read contiguously and the accumulator row stays in cache.
	combine := func(acc, src []float32) {
		switch kind {
		case reduceSum:
			kernel.Add(acc, src, acc)
		case reduceMax:
			kernel.Maximum(acc, src, acc)
		}
	}
	fold := func(acc []float32, base, j0, j1 int) { // acc = fold of rows [j0, j1)
		copy(acc, xd[base+j0*inner:base+j0*inner+inner])
		for j := j0 + 1; j < j1; j++ {
			combine(acc, xd[base+j*inner:base+j*inner+inner])
		}
	}
	if outer >= parallel.Workers() || d*inner < minChunk {
		parallel.Range(outer, max(1, minChunk/(d*inner)), func(lo, hi int) {
			for o := lo; o < hi; o++ {
				fold(od[o*inner:(o+1)*inner], o*d*inner, 0, d)
			}
		})
		return out
	}
	// few outer slabs: split the rows among goroutines, each folding into a
	// private partial, then merge the partials
	for o := 0; o < outer; o++ {
		base := o * d * inner
		rowsPerChunk := max(1, minChunk/inner)
		nChunks := min((d+rowsPerChunk-1)/rowsPerChunk, 4*parallel.Workers())
		rowsPerChunk = (d + nChunks - 1) / nChunks
		nChunks = (d + rowsPerChunk - 1) / rowsPerChunk
		partials := make([][]float32, nChunks)
		parallel.For(nChunks, func(c int) {
			j0 := c * rowsPerChunk
			j1 := min(j0+rowsPerChunk, d)
			acc := make([]float32, inner)
			fold(acc, base, j0, j1)
			partials[c] = acc
		})
		orow := od[o*inner : (o+1)*inner]
		copy(orow, partials[0])
		for _, p := range partials[1:] {
			combine(orow, p)
		}
	}
	return out
}

// keepShape returns t's shape with dims set to 1.
func keepShape(shape Shape, dims []int) Shape {
	s := shape.clone()
	for _, d := range dims {
		s[d] = 1
	}
	return s
}

// Sum reduces over the given dimensions (all if none), removing them.
// Summing everything yields a 0-D tensor.
func (t *Tensor) Sum(dims ...int) *Tensor {
	nd := normDims("Sum", dims, len(t.shape))
	out := reduceDims(t, nd, false, reduceSum)
	kshape := keepShape(t.shape, nd)
	return record(out, "Sum", []*Tensor{t}, func(gy *Tensor) {
		t.accumGrad(gy.Reshape(kshape...).Expand(t.shape...).Contiguous())
	})
}

// Mean reduces over the given dimensions (all if none) by averaging.
func (t *Tensor) Mean(dims ...int) *Tensor {
	nd := normDims("Mean", dims, len(t.shape))
	count := 1
	for _, d := range nd {
		count *= t.shape[d]
	}
	return t.Sum(dims...).DivScalar(float32(count))
}

// Max reduces over the given dimensions (all if none) taking the maximum.
// Ties share the gradient equally.
func (t *Tensor) Max(dims ...int) *Tensor {
	nd := normDims("Max", dims, len(t.shape))
	for _, d := range nd {
		if t.shape[d] == 0 {
			fail("Max", "cannot reduce empty dimension %d of shape %v", d, t.shape)
		}
	}
	out := reduceDims(t, nd, false, reduceMax)
	kshape := keepShape(t.shape, nd)
	td, od := t.Detach(), out.Detach()
	return record(out, "Max", []*Tensor{t}, func(gy *Tensor) {
		mask := binaryOp("Max", td, od.Reshape(kshape...), nil, nil, false, func(a, b float32) float32 {
			if a == b {
				return 1
			}
			return 0
		})
		count := reduceDims(mask, nd, true, reduceSum)
		t.accumGrad(mask.Mul(gy.Reshape(kshape...).Div(count)))
	})
}

// Min reduces over the given dimensions (all if none) taking the minimum.
func (t *Tensor) Min(dims ...int) *Tensor { return t.Neg().Max(dims...).Neg() }

// Var returns the (population) variance over the given dimensions.
func (t *Tensor) Var(dims ...int) *Tensor {
	nd := normDims("Var", dims, len(t.shape))
	kshape := keepShape(t.shape, nd)
	mean := t.Mean(dims...).Reshape(kshape...)
	return t.Sub(mean).Square().Mean(dims...)
}

// Std returns the (population) standard deviation over the given dimensions.
func (t *Tensor) Std(dims ...int) *Tensor { return t.Var(dims...).Sqrt() }

// Argmax returns, for every position of the other dimensions, the index of
// the maximum along dim. The result has the reduced shape (dim removed)
// flattened in row-major order. It does not participate in autograd.
func (t *Tensor) Argmax(dim int) []int {
	dim = normDim("Argmax", dim, len(t.shape))
	tc := t.Contiguous()
	d := tc.shape[dim]
	if d == 0 {
		fail("Argmax", "cannot reduce empty dimension %d of shape %v", dim, t.shape)
	}
	outer, inner := 1, 1
	for i := 0; i < dim; i++ {
		outer *= tc.shape[i]
	}
	for i := dim + 1; i < len(tc.shape); i++ {
		inner *= tc.shape[i]
	}
	out := make([]int, outer*inner)
	for o := 0; o < outer; o++ {
		for i := 0; i < inner; i++ {
			base := o*d*inner + i
			best, bi := tc.data[base], 0
			for j := 1; j < d; j++ {
				if v := tc.data[base+j*inner]; v > best {
					best, bi = v, j
				}
			}
			out[o*inner+i] = bi
		}
	}
	return out
}
