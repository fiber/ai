package tensor

import "math"

// Attention building blocks. Masks are additive: 0 where attention is
// allowed and a large negative number where it is not, so that softmax
// gives those positions zero weight. Adding a mask broadcasts over batch
// and heads and needs no dedicated kernel.

const maskNeg = -1e9

// CausalMask returns an [n×n] additive mask that lets position i attend
// to positions ≤ i only.
func CausalMask(n int) *Tensor {
	m := newTensor(Shape{n, n})
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			m.data[i*n+j] = maskNeg
		}
	}
	return m
}

// PaddingMask returns a [len(lengths)×1×1×n] additive mask that hides the
// positions at or beyond each sequence's length, for batches padded to n.
func PaddingMask(lengths []int, n int) *Tensor {
	b := len(lengths)
	m := newTensor(Shape{b, 1, 1, n})
	for i, l := range lengths {
		for j := max(0, l); j < n; j++ {
			m.data[i*n+j] = maskNeg
		}
	}
	return m
}

// Attention is scaled dot-product attention: softmax(q·kᵀ/√d + mask)·v.
// q is [..., T, D], k and v are [..., S, D] with matching leading
// dimensions (typically batch and heads); mask is nil or broadcastable
// to [..., T, S]. The result is [..., T, D].
func Attention(q, k, v, mask *Tensor) *Tensor {
	nd := len(q.shape)
	if nd < 2 || len(k.shape) != nd || len(v.shape) != nd {
		fail("Attention", "q, k, v must have the same rank ≥ 2, got %v %v %v", q.shape, k.shape, v.shape)
	}
	d := q.shape[nd-1]
	if k.shape[nd-1] != d || v.shape[nd-2] != k.shape[nd-2] {
		fail("Attention", "shapes do not match: q %v, k %v, v %v", q.shape, k.shape, v.shape)
	}
	scores := q.MatMul(k.Transpose(-2, -1)).MulScalar(float32(1 / math.Sqrt(float64(d))))
	if mask != nil {
		scores = scores.Add(mask)
	}
	return scores.Softmax(-1).MatMul(v)
}

// RMSNorm normalises the last dimension by its root mean square,
// x / √(mean(x²) + eps), and scales by g (size of the last dimension):
// the normalisation Gemma- and Llama-class models use.
func RMSNorm(x, g *Tensor, eps float32) *Tensor {
	nd := len(x.shape)
	if nd == 0 {
		fail("RMSNorm", "expected at least a 1-D input")
	}
	n := x.shape[nd-1]
	if g.size != n {
		fail("RMSNorm", "g must have %d elements, got %d", n, g.size)
	}
	rms := x.Square().Mean(-1).AddScalar(eps).Sqrt().Unsqueeze(-1)
	return x.Div(rms).Mul(g)
}
