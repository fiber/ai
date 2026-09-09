// Package nn provides neural-network building blocks on top of package
// tensor. A Module is anything with a Forward and a parameter list; there
// is no registry or module tree — Sequential is a slice.
package nn

import (
	"math"

	"github.com/fiber/ai/tensor"
)

// Module is a differentiable function with trainable parameters.
type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Params() []*tensor.Tensor
}

// Linear is y = x·W + b with W stored as [in, out] so the forward pass
// needs no transpose.
type Linear struct {
	W, B *tensor.Tensor // B is nil when the layer has no bias
}

// NewLinear creates a layer with W ~ N(0, 2/in) (He initialisation) and
// b = 0.
func NewLinear(in, out int) *Linear {
	std := float32(math.Sqrt(2 / float64(in)))
	return &Linear{
		W: tensor.Randn(in, out).MulScalar(std).Detach().SetRequiresGrad(true),
		B: tensor.Zeros(out).SetRequiresGrad(true),
	}
}

// NewLinearNoBias creates a layer without a bias term.
func NewLinearNoBias(in, out int) *Linear {
	l := NewLinear(in, out)
	l.B = nil
	return l
}

func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, tensor.NoAct)
}

// forward is Forward with an activation folded into the product's
// epilogue (Sequential uses it for a Linear followed by ReLU or GELU).
// The fused path is taken for 2-D inputs; with a gradient being recorded
// it computes the same result from the ordinary operations.
func (l *Linear) forward(x *tensor.Tensor, act tensor.Activation) *tensor.Tensor {
	if x.Dims() == 2 && (l.B != nil || act != tensor.NoAct) {
		return tensor.MatMulFused(x, l.W, tensor.Fused{Bias: l.B, Act: act})
	}
	y := x.MatMul(l.W)
	if l.B != nil {
		h := y
		y = y.Add(l.B)
		h.Release() // no-op when autograd needs it
	}
	switch act {
	case tensor.ReLUAct:
		h := y
		y = y.ReLU()
		h.Release()
	case tensor.GELUAct:
		h := y
		y = y.GELU()
		h.Release()
	}
	return y
}

func (l *Linear) Params() []*tensor.Tensor {
	if l.B == nil {
		return []*tensor.Tensor{l.W}
	}
	return []*tensor.Tensor{l.W, l.B}
}

// Sequential applies modules in order.
type Sequential []Module

func (s Sequential) Forward(x *tensor.Tensor) *tensor.Tensor {
	prev := x
	for i := 0; i < len(s); i++ {
		var y *tensor.Tensor
		// A Linear directly followed by ReLU or GELU: one product with the
		// activation applied in its epilogue instead of a separate pass.
		if l, ok := s[i].(*Linear); ok && i+1 < len(s) {
			switch s[i+1].(type) {
			case ReLU:
				y = l.forward(prev, tensor.ReLUAct)
				i++
			case GELU:
				y = l.forward(prev, tensor.GELUAct)
				i++
			}
		}
		if y == nil {
			y = s[i].Forward(prev)
		}
		if prev != x && y != prev {
			prev.Release() // intermediate result; a no-op when autograd or a view holds it
		}
		prev = y
	}
	return prev
}

func (s Sequential) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, m := range s {
		ps = append(ps, m.Params()...)
	}
	return ps
}

// SetTraining switches every module in s that has a training mode.
func (s Sequential) SetTraining(on bool) {
	for _, m := range s {
		SetTraining(m, on)
	}
}

// Stateless activations.
type (
	ReLU    struct{}
	GELU    struct{}
	Tanh    struct{}
	Sigmoid struct{}
	// Softmax normalises along Dim (default -1 when zero value is used
	// with a 2-D input: Dim 0 would be the batch, so prefer -1).
	Softmax struct{ Dim int }
	// Flatten reshapes [batch, ...] to [batch, features].
	Flatten struct{}
)

func (ReLU) Forward(x *tensor.Tensor) *tensor.Tensor    { return x.ReLU() }
func (GELU) Forward(x *tensor.Tensor) *tensor.Tensor    { return x.GELU() }
func (Tanh) Forward(x *tensor.Tensor) *tensor.Tensor    { return x.Tanh() }
func (Sigmoid) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Sigmoid() }
func (s Softmax) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Softmax(s.Dim)
}
func (Flatten) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Reshape(x.Dim(0), -1) }

func (ReLU) Params() []*tensor.Tensor    { return nil }
func (GELU) Params() []*tensor.Tensor    { return nil }
func (Tanh) Params() []*tensor.Tensor    { return nil }
func (Sigmoid) Params() []*tensor.Tensor { return nil }
func (Softmax) Params() []*tensor.Tensor { return nil }
func (Flatten) Params() []*tensor.Tensor { return nil }

// Dropout zeroes activations with probability P while Training is set and
// rescales the rest by 1/(1-P). In evaluation mode it is the identity.
type Dropout struct {
	P        float32
	Training bool
}

// NewDropout returns a dropout layer in training mode.
func NewDropout(p float32) *Dropout { return &Dropout{P: p, Training: true} }

func (d *Dropout) Forward(x *tensor.Tensor) *tensor.Tensor {
	if !d.Training || d.P <= 0 {
		return x
	}
	return x.Dropout(d.P)
}

func (d *Dropout) Params() []*tensor.Tensor { return nil }

// SetTraining implements the training-mode switch.
func (d *Dropout) SetTraining(on bool) { d.Training = on }

// LayerNorm normalises the last dimension with learnable scale and shift.
type LayerNorm struct {
	G, B *tensor.Tensor
	Eps  float32
}

// NewLayerNorm creates a layer norm over a last dimension of size n.
func NewLayerNorm(n int) *LayerNorm {
	return &LayerNorm{
		G:   tensor.Ones(n).SetRequiresGrad(true),
		B:   tensor.Zeros(n).SetRequiresGrad(true),
		Eps: 1e-5,
	}
}

func (l *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.LayerNorm(x, l.G, l.B, l.Eps)
}

func (l *LayerNorm) Params() []*tensor.Tensor { return []*tensor.Tensor{l.G, l.B} }

// Embedding maps integer ids to learned vectors.
type Embedding struct {
	W *tensor.Tensor // [vocab, dim]
}

// NewEmbedding creates a table of vocab vectors of size dim, N(0, 1).
func NewEmbedding(vocab, dim int) *Embedding {
	return &Embedding{W: tensor.Randn(vocab, dim).SetRequiresGrad(true)}
}

// Lookup returns the [len(ids), dim] embeddings of ids. Gradients flow to
// the table rows (a scatter-add, so repeated ids accumulate).
func (e *Embedding) Lookup(ids []int) *tensor.Tensor { return e.W.Rows(ids) }

// Forward treats the batch of one-hot rows as ids via Argmax; prefer Lookup.
func (e *Embedding) Forward(x *tensor.Tensor) *tensor.Tensor { return x.MatMul(e.W) }

func (e *Embedding) Params() []*tensor.Tensor { return []*tensor.Tensor{e.W} }

// trainer is implemented by modules with a training/evaluation switch.
type trainer interface{ SetTraining(bool) }

// SetTraining switches m (and, for Sequential, its children) between
// training and evaluation mode.
func SetTraining(m Module, on bool) {
	if t, ok := m.(trainer); ok {
		t.SetTraining(on)
	}
}

// NumParams counts the trainable scalars of a module.
func NumParams(m Module) int {
	n := 0
	for _, p := range m.Params() {
		n += p.Size()
	}
	return n
}

// ZeroGrad clears the gradients of all parameters.
func ZeroGrad(m Module) {
	for _, p := range m.Params() {
		p.ZeroGrad()
	}
}
