package tensor

import (
	"runtime"

	"github.com/fiber/ai/internal/blas"
	"github.com/fiber/ai/internal/kernel"
	"github.com/fiber/ai/internal/parallel"
)

// Fused describes work applied to a matrix product's output while the
// output block is still in cache (see MatMulFused): in order, Bias (per
// column, length n), RowScale (per row, length m; before the activation,
// so a normalisation folded into the weights is applied to the
// pre-activation), Act, Mul (element-wise, [m×n]) and Residual (added,
// [m×n]). Nil fields are skipped.
type Fused struct {
	Bias     *Tensor
	Act      Activation
	Mul      *Tensor
	RowScale []float32
	Residual *Tensor
}

// Activation selects the element-wise function of a Fused product.
type Activation int

const (
	NoAct Activation = iota
	ReLUAct
	GELUAct
)

// MatMulFused returns epilogue(x · w) for 2-D x [m×k] and w [k×n] with the
// epilogue applied by the GEMM driver on each finished output block, so
// that bias, activation, a gated product and a residual add cost no
// separate passes over the output. It is an inference path: when a
// gradient is being recorded for any operand it computes the same result
// from the ordinary operations, so autograd sees what it always saw.
func MatMulFused(x, w *Tensor, f Fused) *Tensor {
	if len(x.shape) != 2 || len(w.shape) != 2 {
		fail("MatMulFused", "expected 2-D operands, got %v and %v", x.shape, w.shape)
	}
	m, k, n := x.shape[0], x.shape[1], w.shape[1]
	if k != w.shape[0] {
		fail("MatMulFused", "shape mismatch %v · %v", x.shape, w.shape)
	}
	if f.Bias != nil && f.Bias.size != n {
		fail("MatMulFused", "bias has %d elements for %d columns", f.Bias.size, n)
	}
	if f.RowScale != nil && len(f.RowScale) != m {
		fail("MatMulFused", "row scale has %d elements for %d rows", len(f.RowScale), m)
	}
	for _, t := range []*Tensor{f.Mul, f.Residual} {
		if t != nil && !(len(t.shape) == 2 && t.shape[0] == m && t.shape[1] == n) {
			fail("MatMulFused", "Mul/Residual must be [%d×%d], got %v", m, n, t.shape)
		}
	}
	needGrad := GradEnabled() && (x.requiresGrad || w.requiresGrad ||
		f.Bias != nil && f.Bias.requiresGrad || f.Mul != nil && f.Mul.requiresGrad || f.Residual != nil && f.Residual.requiresGrad)
	if needGrad {
		return fusedComposed(x, w, f)
	}

	out := newTensorUninit(Shape{m, n})
	if out.size == 0 {
		return out
	}
	e := blas.Epilogue{Act: blas.Activation(f.Act)}
	if f.Bias != nil {
		e.Bias = f.Bias.values()
	}
	if f.RowScale != nil {
		e.RowScale = f.RowScale
	}
	mul, res := f.Mul, f.Residual
	if mul != nil && !mul.IsContiguous() {
		mul = mul.Contiguous()
	}
	if res != nil && !res.IsContiguous() {
		res = res.Contiguous()
	}
	if mul != nil {
		e.Mul = mat(mul, 0, 1)
	}
	if res != nil {
		e.Residual = mat(res, 0, 1)
	}
	blas.GemmZeroEpilogue(mat(out, 0, 1), mat(x, 0, 1), mat(w, 0, 1), packedOperand(w), e, parallel.Workers())
	runtime.KeepAlive(x)
	runtime.KeepAlive(w)
	runtime.KeepAlive(f.Bias)
	runtime.KeepAlive(mul)
	runtime.KeepAlive(res)
	return out
}

// fusedComposed is MatMulFused from the ordinary operations, for the
// gradient-recording case and as the reference in tests.
func fusedComposed(x, w *Tensor, f Fused) *Tensor {
	y := x.MatMul(w)
	if f.Bias != nil {
		y = y.Add(f.Bias)
	}
	if f.RowScale != nil {
		y = y.Mul(New(f.RowScale, len(f.RowScale), 1))
	}
	switch f.Act {
	case ReLUAct:
		y = y.ReLU()
	case GELUAct:
		y = y.GELU()
	}
	if f.Mul != nil {
		y = y.Mul(f.Mul)
	}
	if f.Residual != nil {
		y = y.Add(f.Residual)
	}
	return y
}

// RowSumSquares returns, for a 2-D tensor, the sum of squares of each row:
// the statistic a folded RMSNorm needs (one read pass, no write).
func (t *Tensor) RowSumSquares() []float32 {
	if len(t.shape) != 2 {
		fail("RowSumSquares", "expected a 2-D tensor, got %v", t.shape)
	}
	x := t
	if !t.IsContiguous() {
		x = t.Contiguous()
	}
	m, n := x.shape[0], x.shape[1]
	out := make([]float32, m)
	parallel.Range(m, max(1, 4096/max(1, n)), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			row := x.data[i*n : (i+1)*n]
			out[i] = kernel.Dot(row, row)
		}
	})
	runtime.KeepAlive(x)
	return out
}
