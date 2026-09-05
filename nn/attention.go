package nn

import (
	"math"

	"github.com/fiber/ai/tensor"
)

// MultiHeadAttention projects its input to Heads sets of queries, keys and
// values of size Dim/Heads each, runs scaled dot-product attention per
// head and projects the concatenated heads back to Dim.
type MultiHeadAttention struct {
	Heads, Dim int
	Q, K, V, O *Linear
	// Mask, if set, is added to the attention scores: tensor.CausalMask
	// for autoregressive models, tensor.PaddingMask for padded batches, or
	// their sum. It broadcasts against [batch, heads, T, S].
	Mask *tensor.Tensor
}

// NewMultiHeadAttention creates the four projections for a model width of
// dim split over heads; dim must be divisible by heads.
func NewMultiHeadAttention(dim, heads int) *MultiHeadAttention {
	if dim%heads != 0 {
		panic("nn: attention width must be divisible by the number of heads")
	}
	m := &MultiHeadAttention{Heads: heads, Dim: dim,
		Q: NewLinearNoBias(dim, dim), K: NewLinearNoBias(dim, dim), V: NewLinearNoBias(dim, dim), O: NewLinear(dim, dim)}
	// Xavier-style scale for the projections
	s := float32(1 / math.Sqrt(float64(dim)))
	tensor.NoGrad(func() {
		for _, l := range []*Linear{m.Q, m.K, m.V, m.O} {
			l.W.MulScalarInPlace(s * float32(math.Sqrt(3)))
		}
	})
	return m
}

// Forward is self-attention over x of shape [batch, T, Dim].
func (m *MultiHeadAttention) Forward(x *tensor.Tensor) *tensor.Tensor { return m.Cross(x, x) }

// Cross attends from x ([batch, T, Dim]) to context ([batch, S, Dim]):
// queries come from x, keys and values from context.
func (m *MultiHeadAttention) Cross(x, context *tensor.Tensor) *tensor.Tensor {
	if x.Dims() != 3 || context.Dims() != 3 {
		panic("nn: attention expects [batch, tokens, dim] inputs")
	}
	b, t, s := x.Dim(0), x.Dim(1), context.Dim(1)
	hd := m.Dim / m.Heads
	split := func(y *tensor.Tensor, n int) *tensor.Tensor {
		return y.Reshape(b, n, m.Heads, hd).Permute(0, 2, 1, 3) // [batch, heads, n, hd]
	}
	q := split(m.Q.Forward(x), t)
	k := split(m.K.Forward(context), s)
	v := split(m.V.Forward(context), s)
	out := tensor.Attention(q, k, v, m.Mask) // [batch, heads, t, hd]
	merged := out.Permute(0, 2, 1, 3).Reshape(b, t, m.Dim)
	return m.O.Forward(merged)
}

// Params returns the projection weights (and the output bias).
func (m *MultiHeadAttention) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range []*Linear{m.Q, m.K, m.V, m.O} {
		ps = append(ps, l.Params()...)
	}
	return ps
}

// RMSNorm scales the last dimension by its root mean square and a learned
// gain, the normalisation of Gemma- and Llama-class models.
type RMSNorm struct {
	G   *tensor.Tensor
	Eps float32
}

// NewRMSNorm creates an RMSNorm over a last dimension of size n.
func NewRMSNorm(n int) *RMSNorm {
	return &RMSNorm{G: tensor.Ones(n).SetRequiresGrad(true), Eps: 1e-6}
}

func (r *RMSNorm) Forward(x *tensor.Tensor) *tensor.Tensor { return tensor.RMSNorm(x, r.G, r.Eps) }
func (r *RMSNorm) Params() []*tensor.Tensor                { return []*tensor.Tensor{r.G} }
