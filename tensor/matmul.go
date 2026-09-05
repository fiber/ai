package tensor

import (
	"github.com/fiber/ai/internal/blas"
	"github.com/fiber/ai/internal/parallel"
)

// MatMul returns the matrix product of t and u following NumPy semantics:
//
//   - 2-D × 2-D is the ordinary matrix product,
//   - 1-D operands are promoted to a row (left) or column (right) vector
//     and the added dimension is removed again,
//   - with more dimensions the leading (batch) dimensions are broadcast and
//     a matrix product is computed for every batch entry.
//
// Strided operands (transposed views) are consumed without copying.
func (t *Tensor) MatMul(u *Tensor) *Tensor {
	switch {
	case len(t.shape) == 0 || len(u.shape) == 0:
		fail("MatMul", "operands must be at least 1-D, got %v and %v", t.shape, u.shape)
	case len(t.shape) == 1 && len(u.shape) == 1:
		if t.shape[0] != u.shape[0] {
			fail("MatMul", "shape mismatch %v · %v", t.shape, u.shape)
		}
		return t.Unsqueeze(0).MatMul(u.Unsqueeze(1)).Reshape()
	case len(t.shape) == 1:
		return t.Unsqueeze(0).MatMul(u).Squeeze(-2)
	case len(u.shape) == 1:
		return t.MatMul(u.Unsqueeze(1)).Squeeze(-1)
	case len(t.shape) == 2 && len(u.shape) == 2:
		return matmul2D(t, u)
	}
	return matmulBatched(t, u)
}

func mat(t *Tensor, rowDim, colDim int) blas.Mat {
	return blas.Mat{Data: t.data, Rows: t.shape[rowDim], Cols: t.shape[colDim], RS: t.strides[rowDim], CS: t.strides[colDim]}
}

func matmul2D(x, y *Tensor) *Tensor {
	m, k, n := x.shape[0], x.shape[1], y.shape[1]
	if k != y.shape[0] {
		fail("MatMul", "shape mismatch %v · %v", x.shape, y.shape)
	}
	out := newTensor(Shape{m, n})
	if out.size > 0 && k > 0 {
		blas.Gemm(mat(out, 0, 1), mat(x, 0, 1), mat(y, 0, 1))
	}
	xd, yd := x.Detach(), y.Detach()
	return record(out, "MatMul", []*Tensor{x, y}, func(gy *Tensor) {
		if x.requiresGrad { // dX = dY · Yᵀ
			x.accumGrad(gy.MatMul(yd.T()))
		}
		if y.requiresGrad { // dY = Xᵀ · dY
			y.accumGrad(xd.T().MatMul(gy))
		}
	})
}

func matmulBatched(x, y *Tensor) *Tensor {
	nx, ny := len(x.shape), len(y.shape)
	m, k, n := x.shape[nx-2], x.shape[nx-1], y.shape[ny-1]
	if k != y.shape[ny-2] {
		fail("MatMul", "shape mismatch %v · %v", x.shape, y.shape)
	}
	// [B..., m, k] · [k, n]: fold the batch into M when x is dense
	if ny == 2 && x.IsContiguous() {
		return x.Reshape(-1, k).MatMul(y).Reshape(append(x.shape[:nx-2].clone(), m, n)...)
	}
	batch := broadcastShapes("MatMul", x.shape[:nx-2], y.shape[:ny-2])
	shape := append(batch.clone(), m, n)
	out := newTensor(shape)
	nb := batch.Size()
	if out.size > 0 && k > 0 && nb > 0 {
		// per-batch offsets via broadcast strides over the batch dims
		xb := view(x, x.data, x.shape[:nx-2], x.strides[:nx-2])
		yb := view(y, y.data, y.shape[:ny-2], y.strides[:ny-2])
		sx, sy := broadcastStrides(xb, batch), broadcastStrides(yb, batch)
		xoff, yoff := make([]int, nb), make([]int, nb)
		walkRows(append(batch.clone(), 1), [][]int{append(sx, 0), append(sy, 0)}, 0, nb, func(b int, offs []int) {
			xoff[b], yoff[b] = offs[0], offs[1]
		})
		xm, ym := mat(x, nx-2, nx-1), mat(y, ny-2, ny-1)
		run := func(b int, workers int) {
			a, c := xm, ym
			a.Data, c.Data = x.data[xoff[b]:], y.data[yoff[b]:]
			o := blas.Contiguous(out.data[b*m*n:(b+1)*m*n], m, n)
			blas.GemmWorkers(o, a, c, workers)
		}
		if float64(m)*float64(n)*float64(k) < float64(blas.ParallelThreshold) {
			parallel.For(nb, func(b int) { run(b, 1) }) // many small products: parallel over the batch
		} else {
			for b := 0; b < nb; b++ {
				run(b, parallel.Workers())
			}
		}
	}
	xd, yd := x.Detach(), y.Detach()
	return record(out, "MatMul", []*Tensor{x, y}, func(gy *Tensor) {
		if x.requiresGrad {
			x.accumGrad(sumTo(gy.MatMul(yd.Transpose(-1, -2)), x.shape))
		}
		if y.requiresGrad {
			y.accumGrad(sumTo(xd.Transpose(-1, -2).MatMul(gy), y.shape))
		}
	})
}

// Dot returns the inner product of two 1-D tensors as a 0-D tensor.
func (t *Tensor) Dot(u *Tensor) *Tensor {
	if len(t.shape) != 1 || len(u.shape) != 1 {
		fail("Dot", "expected 1-D tensors, got %v and %v", t.shape, u.shape)
	}
	return t.MatMul(u)
}

// Outer returns the outer product of two 1-D tensors.
func (t *Tensor) Outer(u *Tensor) *Tensor {
	if len(t.shape) != 1 || len(u.shape) != 1 {
		fail("Outer", "expected 1-D tensors, got %v and %v", t.shape, u.shape)
	}
	return t.Unsqueeze(1).MatMul(u.Unsqueeze(0))
}
