package tensor

import (
	"math"

	"github.com/fiber/ai/internal/kernel"
	"github.com/fiber/ai/internal/parallel"
)

// Fused neural-network primitives. Each is a single forward pass over rows
// of the last dimension with a hand-derived backward, which is both faster
// and numerically better than composing them from element-wise ops.

// lastDimRows makes dim the last dimension and returns a contiguous tensor
// whose rows are the reduction axis, plus a function undoing the move.
func lastDimRows(op string, t *Tensor, dim int) (rows *Tensor, undo func(*Tensor) *Tensor) {
	nd := len(t.shape)
	if nd == 0 {
		fail(op, "expected at least a 1-D tensor")
	}
	dim = normDim(op, dim, nd)
	if dim == nd-1 {
		return t.Contiguous(), func(r *Tensor) *Tensor { return r }
	}
	return t.Transpose(dim, nd-1).Contiguous(), func(r *Tensor) *Tensor { return r.Transpose(dim, nd-1) }
}

// forRows calls fn for every row of length n of a contiguous tensor.
func forRows(size, n int, fn func(lo, hi int)) {
	if n == 0 || size == 0 {
		return
	}
	parallel.Range(size/n, max(1, minChunk/n), fn)
}

// Softmax normalises t along dim so that the entries sum to 1.
func (t *Tensor) Softmax(dim int) *Tensor {
	x, undo := lastDimRows("Softmax", t, dim)
	n := x.shape[len(x.shape)-1]
	out := newTensorUninit(x.shape)
	forRows(x.size, n, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row, o := x.data[r*n:(r+1)*n], out.data[r*n:(r+1)*n]
			kernel.AddScalar(row, -kernel.Max(row), o)
			kernel.Exp(o, o)
			kernel.Scale(o, 1/kernel.Sum(o), o)
		}
	})
	od := out.Detach()
	out = record(out, "Softmax", []*Tensor{x}, func(gy *Tensor) {
		// dx = y ⊙ (g − ⟨g, y⟩) per row
		g := gy.Contiguous()
		gx := newTensorUninit(x.shape)
		forRows(x.size, n, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				y, gr, o := od.data[r*n:(r+1)*n], g.data[r*n:(r+1)*n], gx.data[r*n:(r+1)*n]
				dot := kernel.Dot(gr, y)
				kernel.AddScalar(gr, -dot, o)
				kernel.Mul(o, y, o)
			}
		})
		x.accumGrad(gx)
	})
	return undo(out)
}

// LogSoftmax returns log(softmax(t)) along dim, computed stably.
func (t *Tensor) LogSoftmax(dim int) *Tensor {
	x, undo := lastDimRows("LogSoftmax", t, dim)
	n := x.shape[len(x.shape)-1]
	out := newTensorUninit(x.shape)
	forRows(x.size, n, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row, o := x.data[r*n:(r+1)*n], out.data[r*n:(r+1)*n]
			kernel.AddScalar(row, -logSumExp(row), o)
		}
	})
	od := out.Detach()
	out = record(out, "LogSoftmax", []*Tensor{x}, func(gy *Tensor) {
		// dx = g − softmax(x) · Σg per row
		g := gy.Contiguous()
		gx := newTensorUninit(x.shape)
		forRows(x.size, n, func(lo, hi int) {
			for r := lo; r < hi; r++ {
				y, gr, o := od.data[r*n:(r+1)*n], g.data[r*n:(r+1)*n], gx.data[r*n:(r+1)*n]
				s := kernel.Sum(gr)
				kernel.Exp(y, o)
				kernel.Scale(o, -s, o)
				kernel.Add(o, gr, o)
			}
		})
		x.accumGrad(gx)
	})
	return undo(out)
}

func logSumExp(row []float32) float32 {
	mx := kernel.Max(row)
	var s float64
	for _, v := range row {
		s += math.Exp(float64(v - mx))
	}
	return mx + float32(math.Log(s))
}

// MSELoss returns mean((pred − target)²) as a 0-D tensor.
func MSELoss(pred, target *Tensor) *Tensor {
	if !pred.shape.Equal(target.shape) {
		fail("MSELoss", "shape mismatch %v vs %v", pred.shape, target.shape)
	}
	p, tg := pred.Contiguous(), target.Contiguous()
	n := p.size
	if n == 0 {
		fail("MSELoss", "empty tensors")
	}
	diff := newTensorUninit(p.shape)
	kernel.Sub(p.data[:n], tg.data[:n], diff.data)
	out := Scalar(kernel.Dot(diff.data, diff.data) / float32(n))
	return record(out, "MSELoss", []*Tensor{p, tg}, func(gy *Tensor) {
		scale := 2 * gy.data[0] / float32(n)
		g := diff.MulScalar(scale)
		p.accumGrad(g)
		if tg.requiresGrad {
			tg.accumGrad(g.Neg())
		}
	})
}

// CrossEntropy returns the mean negative log-likelihood of integer class
// targets given raw logits of shape [rows, classes]. Softmax is fused in;
// do not apply Softmax or LogSoftmax before it.
func CrossEntropy(logits *Tensor, targets []int) *Tensor {
	if len(logits.shape) != 2 {
		fail("CrossEntropy", "expected logits of shape [rows classes], got %v", logits.shape)
	}
	x := logits.Contiguous()
	m, c := x.shape[0], x.shape[1]
	if len(targets) != m {
		fail("CrossEntropy", "got %d targets for %d rows", len(targets), m)
	}
	for _, t := range targets {
		if t < 0 || t >= c {
			fail("CrossEntropy", "target %d out of range [0, %d)", t, c)
		}
	}
	probs := newTensorUninit(x.shape) // softmax, kept for backward
	losses := make([]float64, m)
	forRows(x.size, c, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row, p := x.data[r*c:(r+1)*c], probs.data[r*c:(r+1)*c]
			lse := logSumExp(row)
			kernel.AddScalar(row, -lse, p)
			kernel.Exp(p, p)
			losses[r] = float64(lse - row[targets[r]])
		}
	})
	var total float64
	for _, l := range losses {
		total += l
	}
	out := Scalar(float32(total / float64(m)))
	return record(out, "CrossEntropy", []*Tensor{x}, func(gy *Tensor) {
		// dlogits = (softmax − onehot) · g/m
		scale := gy.data[0] / float32(m)
		g := probs.MulScalar(scale)
		for r, t := range targets {
			g.data[r*c+t] -= scale
		}
		x.accumGrad(g)
	})
}

// LayerNorm normalises the last dimension of x to zero mean and unit
// variance and applies the affine transform gamma*x̂ + beta. gamma and beta
// have the size of the last dimension. One fused op, forward and backward.
func LayerNorm(x, gamma, beta *Tensor, eps float32) *Tensor {
	nd := len(x.shape)
	if nd == 0 {
		fail("LayerNorm", "expected at least a 1-D input")
	}
	n := x.shape[nd-1]
	if gamma.size != n || beta.size != n {
		fail("LayerNorm", "gamma and beta must have %d elements, got %d and %d", n, gamma.size, beta.size)
	}
	xc, gc, bc := x.Contiguous(), gamma.Contiguous(), beta.Contiguous()
	xhat := newTensorUninit(x.shape) // normalised input, needed by backward
	rows := 0
	if n > 0 {
		rows = xc.size / n
	}
	rstd := make([]float32, rows)
	out := newTensorUninit(x.shape)
	gd, bd := gc.data[:n], bc.data[:n]
	forRows(xc.size, n, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row, xh, o := xc.data[r*n:(r+1)*n], xhat.data[r*n:(r+1)*n], out.data[r*n:(r+1)*n]
			mean := kernel.Sum(row) / float32(n)
			kernel.AddScalar(row, -mean, xh)
			v := kernel.Dot(xh, xh) / float32(n)
			rs := float32(1 / math.Sqrt(float64(v)+float64(eps)))
			rstd[r] = rs
			kernel.Scale(xh, rs, xh)
			kernel.Mul(xh, gd, o)
			kernel.Add(o, bd, o)
		}
	})
	return record(out, "LayerNorm", []*Tensor{xc, gc, bc}, func(gy *Tensor) {
		g := gy.Contiguous()
		if gc.requiresGrad || bc.requiresGrad {
			// dgamma = Σ_rows g ⊙ x̂, dbeta = Σ_rows g
			gg := g.Mul(xhat).Reshape(-1, n).Sum(0)
			gb := g.Reshape(-1, n).Sum(0)
			gc.accumGrad(gg.Reshape(gamma.shape...))
			bc.accumGrad(gb.Reshape(beta.shape...))
		}
		if xc.requiresGrad {
			// dx = rstd · (gy − mean(gy) − x̂ · mean(gy ⊙ x̂)),  gy = g ⊙ γ
			gx := newTensorUninit(x.shape)
			forRows(xc.size, n, func(lo, hi int) {
				buf := make([]float32, n)
				for r := lo; r < hi; r++ {
					gr, xh, o := g.data[r*n:(r+1)*n], xhat.data[r*n:(r+1)*n], gx.data[r*n:(r+1)*n]
					kernel.Mul(gr, gd, buf)
					mgy := kernel.Sum(buf) / float32(n)
					mgyx := kernel.Dot(buf, xh) / float32(n)
					kernel.Scale(xh, -mgyx, o)
					kernel.Add(o, buf, o)
					kernel.AddScalar(o, -mgy, o)
					kernel.Scale(o, rstd[r], o)
				}
			})
			xc.accumGrad(gx)
		}
	})
}
