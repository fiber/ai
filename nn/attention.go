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
	// RoPEBase, if non-zero, rotates queries and keys by their positions
	// before the scores are computed (tensor.RoPE). A score then depends
	// on the distance between two tokens rather than on where they sit in
	// the window, which is what current decoders do; the alternative is a
	// learned vector per slot added to the input, which says nothing
	// about a position the model never saw. Gemma uses 1e6 for its global
	// layers and 1e4 for its local ones.
	RoPEBase float64
	// PosOffset is the position of the first query; keys always start at
	// zero. Decoding one token at a time against cached keys sets it.
	PosOffset int
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
	if m.RoPEBase != 0 {
		q = tensor.RoPE(q, m.RoPEBase, seq(m.PosOffset, t))
		k = tensor.RoPE(k, m.RoPEBase, seq(0, s))
	}
	out := tensor.Attention(q, k, v, m.Mask) // [batch, heads, t, hd]
	merged := out.Permute(0, 2, 1, 3).Reshape(b, t, m.Dim)
	return m.O.Forward(merged)
}

// KVCache holds the keys and values a decoder has already computed, so
// that generating a token does not recompute the whole prefix. In causal
// attention nothing that comes later changes an earlier token's key or
// value, which is what makes the reuse exact rather than an
// approximation.
//
// A cache belongs to one sequence, not to the module: the same weights
// serve many sequences at once, and per-sequence state hidden inside a
// shared module is a data race waiting to be written. Create one per
// sequence per layer and pass it to Step.
type KVCache struct {
	K, V *tensor.Tensor // [batch, heads, held, headDim], nil until the first Step
	pos  int            // tokens ever appended, which Trim does not undo
}

// Len is the number of tokens the cache holds. After Trim it is smaller
// than Pos.
func (c *KVCache) Len() int {
	if c == nil || c.K == nil {
		return 0
	}
	return c.K.Dim(2)
}

// Pos is the position the next token will occupy: how many have been
// appended, whether or not they are still held. Rotary positions are
// applied at Pos rather than at Len, so trimming the cache does not move
// the tokens that remain.
func (c *KVCache) Pos() int {
	if c == nil {
		return 0
	}
	return c.pos
}

// Reset empties the cache so it can serve another sequence.
func (c *KVCache) Reset() { c.K, c.V, c.pos = nil, nil, 0 }

// Trim keeps the most recent keep tokens and drops the rest, which is
// what bounds the memory of a long generation: the cache grows by one
// row per token for ever otherwise.
//
// Dropping the oldest keys is sound with rotary positions because a
// score depends on the distance between two tokens, not on their
// absolute places: a query at position 400 attending to a key kept from
// position 300 sees the same rotation it would in a window that started
// at 273. It is not sound without them, where the model was told where
// each token sits and the remaining keys would suddenly be at the wrong
// places.
func (c *KVCache) Trim(keep int) {
	n := c.Len()
	if keep >= n || keep < 0 {
		return
	}
	c.K = c.K.Slice(2, n-keep, n).Contiguous()
	c.V = c.V.Slice(2, n-keep, n).Contiguous()
}

// Step attends from x, which holds the tokens not yet seen
// ([batch, new, Dim]), to those tokens and everything in the cache. Only
// the new tokens are projected; their keys and values are appended to
// the cache for the next call.
//
// It takes any number of new tokens, so a prompt can be ingested in one
// call and generation then proceed one token at a time.
//
// No mask is applied: every key in the cache precedes every new query by
// construction, and the new tokens attend to each other in order only if
// there is one of them. Pass a prompt whole and then decode singly, as a
// runtime does.
func (m *MultiHeadAttention) Step(x *tensor.Tensor, c *KVCache) *tensor.Tensor {
	if x.Dims() != 3 {
		panic("nn: attention expects [batch, tokens, dim] inputs")
	}
	b, t := x.Dim(0), x.Dim(1)
	hd := m.Dim / m.Heads
	split := func(y *tensor.Tensor, n int) *tensor.Tensor {
		return y.Reshape(b, n, m.Heads, hd).Permute(0, 2, 1, 3)
	}
	seen := c.Pos()
	q := split(m.Q.Forward(x), t)
	k := split(m.K.Forward(x), t)
	v := split(m.V.Forward(x), t)
	if m.RoPEBase != 0 {
		// The cached keys were rotated at their own positions when they
		// were stored, so the new ones continue from where they stopped.
		q = tensor.RoPE(q, m.RoPEBase, seq(seen, t))
		k = tensor.RoPE(k, m.RoPEBase, seq(seen, t))
	}
	c.pos += t
	if c.K == nil {
		c.K, c.V = k.Contiguous(), v.Contiguous()
	} else {
		c.K = tensor.Cat(2, c.K, k).Contiguous()
		c.V = tensor.Cat(2, c.V, v).Contiguous()
	}
	out := tensor.Attention(q, c.K, c.V, nil) // [batch, heads, t, hd]
	merged := out.Permute(0, 2, 1, 3).Reshape(b, t, m.Dim)
	return m.O.Forward(merged)
}

// seq returns offset, offset+1, ... offset+n-1.
func seq(offset, n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = offset + i
	}
	return p
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
