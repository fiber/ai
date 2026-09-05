package tensor

// Reshape returns a tensor with the same elements and a new shape. One
// dimension may be -1 and is inferred. Contiguous tensors are viewed
// without copying; others are copied.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	newShape := Shape(shape).clone()
	infer := -1
	known := 1
	for d, s := range newShape {
		switch {
		case s == -1:
			if infer >= 0 {
				fail("Reshape", "only one dimension may be -1 in %v", Shape(shape))
			}
			infer = d
		case s < 0:
			fail("Reshape", "negative dimension in %v", Shape(shape))
		default:
			known *= s
		}
	}
	if infer >= 0 {
		if known == 0 || t.size%known != 0 {
			fail("Reshape", "cannot infer dimension: %v -> %v", t.shape, Shape(shape))
		}
		newShape[infer] = t.size / known
	}
	if newShape.Size() != t.size {
		fail("Reshape", "cannot reshape %v (%d elements) to %v", t.shape, t.size, newShape)
	}
	src := t
	if !t.IsContiguous() {
		src = t.Contiguous()
	}
	out := view(src.data, newShape, contiguousStrides(newShape))
	old := src.shape
	return record(out, "Reshape", []*Tensor{src}, func(gy *Tensor) { src.accumGrad(gy.Reshape(old...)) })
}

// Flatten returns a 1-D view (or copy) of all elements.
func (t *Tensor) Flatten() *Tensor { return t.Reshape(-1) }

// T returns the transpose of a 2-D tensor as a view.
func (t *Tensor) T() *Tensor {
	if len(t.shape) != 2 {
		fail("T", "expected a 2-D tensor, got shape %v", t.shape)
	}
	return t.Transpose(0, 1)
}

// Transpose swaps two dimensions. The result is a view.
func (t *Tensor) Transpose(d0, d1 int) *Tensor {
	nd := len(t.shape)
	d0 = normDim("Transpose", d0, nd)
	d1 = normDim("Transpose", d1, nd)
	shape := t.shape.clone()
	strides := append([]int(nil), t.strides...)
	shape[d0], shape[d1] = shape[d1], shape[d0]
	strides[d0], strides[d1] = strides[d1], strides[d0]
	out := view(t.data, shape, strides)
	return record(out, "Transpose", []*Tensor{t}, func(gy *Tensor) { t.accumGrad(gy.Transpose(d0, d1)) })
}

// Permute reorders the dimensions: result dimension i is input dimension
// dims[i]. The result is a view.
func (t *Tensor) Permute(dims ...int) *Tensor {
	nd := len(t.shape)
	if len(dims) != nd {
		fail("Permute", "got %d dimensions for %d-D tensor", len(dims), nd)
	}
	shape := make(Shape, nd)
	strides := make([]int, nd)
	used := make([]bool, nd)
	inverse := make([]int, nd)
	for i, d := range dims {
		d = normDim("Permute", d, nd)
		if used[d] {
			fail("Permute", "dimension %d repeated in %v", d, dims)
		}
		used[d] = true
		shape[i], strides[i] = t.shape[d], t.strides[d]
		inverse[d] = i
	}
	out := view(t.data, shape, strides)
	return record(out, "Permute", []*Tensor{t}, func(gy *Tensor) { t.accumGrad(gy.Permute(inverse...)) })
}

// Squeeze removes dimensions of size 1 (the given ones, or all if none).
func (t *Tensor) Squeeze(dims ...int) *Tensor {
	nd := len(t.shape)
	drop := make([]bool, nd)
	if len(dims) == 0 {
		for d, s := range t.shape {
			drop[d] = s == 1
		}
	} else {
		for _, d := range dims {
			d = normDim("Squeeze", d, nd)
			if t.shape[d] != 1 {
				fail("Squeeze", "dimension %d of shape %v has size %d, not 1", d, t.shape, t.shape[d])
			}
			drop[d] = true
		}
	}
	shape := make(Shape, 0, nd)
	strides := make([]int, 0, nd)
	for d := range t.shape {
		if !drop[d] {
			shape = append(shape, t.shape[d])
			strides = append(strides, t.strides[d])
		}
	}
	out := view(t.data, shape, strides)
	old := t.shape
	return record(out, "Squeeze", []*Tensor{t}, func(gy *Tensor) { t.accumGrad(gy.Reshape(old...)) })
}

// Unsqueeze inserts a dimension of size 1 at position dim (which may equal
// Dims() to append).
func (t *Tensor) Unsqueeze(dim int) *Tensor {
	nd := len(t.shape)
	if dim < 0 {
		dim += nd + 1
	}
	if dim < 0 || dim > nd {
		fail("Unsqueeze", "dimension %d out of range for %d-D tensor", dim, nd)
	}
	shape := make(Shape, 0, nd+1)
	strides := make([]int, 0, nd+1)
	shape = append(shape, t.shape[:dim]...)
	strides = append(strides, t.strides[:dim]...)
	stride := 1
	if dim < nd {
		stride = t.strides[dim] * t.shape[dim]
	}
	shape = append(shape, 1)
	strides = append(strides, stride)
	shape = append(shape, t.shape[dim:]...)
	strides = append(strides, t.strides[dim:]...)
	out := view(t.data, shape, strides)
	old := t.shape
	return record(out, "Unsqueeze", []*Tensor{t}, func(gy *Tensor) { t.accumGrad(gy.Reshape(old...)) })
}

// Expand broadcasts size-1 dimensions to the given shape without copying
// (the result is a view with zero strides). -1 keeps a dimension's size.
// Leading dimensions may be added.
func (t *Tensor) Expand(shape ...int) *Tensor {
	nd := len(shape)
	if nd < len(t.shape) {
		fail("Expand", "cannot expand %v to fewer dimensions %v", t.shape, Shape(shape))
	}
	newShape := make(Shape, nd)
	strides := make([]int, nd)
	off := nd - len(t.shape)
	for d := 0; d < nd; d++ {
		want := shape[d]
		if d < off {
			if want < 0 {
				fail("Expand", "new leading dimension %d must be given explicitly", d)
			}
			newShape[d] = want
			continue
		}
		have := t.shape[d-off]
		switch {
		case want == -1 || want == have:
			newShape[d], strides[d] = have, t.strides[d-off]
		case have == 1:
			newShape[d], strides[d] = want, 0
		default:
			fail("Expand", "cannot expand %v to %v", t.shape, Shape(shape))
		}
	}
	out := view(t.data, newShape, strides)
	return record(out, "Expand", []*Tensor{t}, func(gy *Tensor) { t.accumGrad(sumTo(gy, t.shape)) })
}

// Narrow returns the view of length elements along dim starting at start.
func (t *Tensor) Narrow(dim, start, length int) *Tensor {
	dim = normDim("Narrow", dim, len(t.shape))
	if start < 0 || length < 0 || start+length > t.shape[dim] {
		fail("Narrow", "range [%d, %d) out of bounds for dimension %d of size %d", start, start+length, dim, t.shape[dim])
	}
	shape := t.shape.clone()
	shape[dim] = length
	data := t.data
	if length > 0 && start > 0 {
		data = t.data[start*t.strides[dim]:]
	}
	out := view(data, shape, append([]int(nil), t.strides...))
	return record(out, "Narrow", []*Tensor{t}, func(gy *Tensor) {
		g := newTensor(t.shape)
		assign(g.Narrow(dim, start, length), gy)
		t.accumGrad(g)
	})
}

// Slice returns the view of elements [start, end) along dim.
func (t *Tensor) Slice(dim, start, end int) *Tensor {
	dim = normDim("Slice", dim, len(t.shape))
	if end < start {
		fail("Slice", "end %d before start %d", end, start)
	}
	return t.Narrow(dim, start, end-start)
}

// Select returns the view at index along dim with that dimension removed.
func (t *Tensor) Select(dim, index int) *Tensor {
	dim = normDim("Select", dim, len(t.shape))
	if index < 0 {
		index += t.shape[dim]
	}
	if index < 0 || index >= t.shape[dim] {
		fail("Select", "index %d out of range for dimension %d of size %d", index, dim, t.shape[dim])
	}
	return t.Narrow(dim, index, 1).Squeeze(dim)
}

// Row returns row i of a tensor as a view of its remaining dimensions.
func (t *Tensor) Row(i int) *Tensor { return t.Select(0, i) }

// Cat concatenates tensors along dim. All other dimensions must match.
func Cat(dim int, tensors ...*Tensor) *Tensor {
	if len(tensors) == 0 {
		fail("Cat", "no tensors given")
	}
	first := tensors[0]
	nd := len(first.shape)
	dim = normDim("Cat", dim, nd)
	shape := first.shape.clone()
	total := 0
	for _, u := range tensors {
		if len(u.shape) != nd {
			fail("Cat", "tensors have different ranks: %v vs %v", first.shape, u.shape)
		}
		for d := range shape {
			if d != dim && u.shape[d] != shape[d] {
				fail("Cat", "shapes %v and %v differ outside dimension %d", first.shape, u.shape, dim)
			}
		}
		total += u.shape[dim]
	}
	shape[dim] = total
	out := newTensor(shape)
	off := 0
	for _, u := range tensors {
		assign(out.Narrow(dim, off, u.shape[dim]), u)
		off += u.shape[dim]
	}
	return record(out, "Cat", tensors, func(gy *Tensor) {
		off := 0
		for _, u := range tensors {
			if u.requiresGrad {
				u.accumGrad(gy.Narrow(dim, off, u.shape[dim]).Contiguous())
			}
			off += u.shape[dim]
		}
	})
}

// Stack joins tensors of identical shape along a new dimension dim.
func Stack(dim int, tensors ...*Tensor) *Tensor {
	if len(tensors) == 0 {
		fail("Stack", "no tensors given")
	}
	parts := make([]*Tensor, len(tensors))
	for i, u := range tensors {
		if !u.shape.Equal(tensors[0].shape) {
			fail("Stack", "shapes %v and %v differ", tensors[0].shape, u.shape)
		}
		parts[i] = u.Unsqueeze(dim)
	}
	return Cat(dim, parts...)
}

// assign copies src into the (possibly strided) destination view dst.
// Shapes must match exactly. Not recorded in the graph.
func assign(dst, src *Tensor) {
	if !dst.shape.Equal(src.shape) {
		fail("assign", "shape mismatch %v vs %v", dst.shape, src.shape)
	}
	if dst.size == 0 {
		return
	}
	if dst.IsContiguous() {
		copyStrided(dst.data[:dst.size], src)
		return
	}
	nd := len(dst.shape)
	last := dst.shape[nd-1]
	ds, ss := dst.strides[nd-1], src.strides[nd-1]
	rows := dst.size / last
	walkRows(dst.shape, [][]int{dst.strides, src.strides}, 0, rows, func(_ int, offs []int) {
		d, s := offs[0], offs[1]
		for j := 0; j < last; j++ {
			dst.data[d] = src.data[s]
			d += ds
			s += ss
		}
	})
}
