package tensor

import (
	"math"
	"runtime"

	"github.com/fiber/ai/internal/parallel"
)

// RoPE applies rotary position embeddings to the last dimension of a
// contiguous tensor shaped [..., T, D], with D even. positions gives the
// absolute position of each of the T slots (len(positions) == T). It uses
// the Hugging Face rotate-half convention: the dimension is split in two
// halves, element i pairs with element i+D/2, and the pair is rotated by
// the angle position · base^(−2i/D). base is the RoPE theta (1e6 for
// Gemma's global layers, 1e4 for its local layers).
//
// The rotation is orthogonal, so the backward pass rotates the gradient by
// the negative angle.
func RoPE(x *Tensor, base float64, positions []int) *Tensor {
	nd := len(x.shape)
	if nd < 2 {
		fail("RoPE", "expected at least a 2-D input, got %v", x.shape)
	}
	T, D := x.shape[nd-2], x.shape[nd-1]
	if D%2 != 0 {
		fail("RoPE", "head dimension %d must be even", D)
	}
	if len(positions) != T {
		fail("RoPE", "got %d positions for %d slots", len(positions), T)
	}
	if !x.IsContiguous() {
		x = x.Contiguous()
	}
	half := D / 2
	// cos/sin tables per (position, frequency): [T, half].
	cos := make([]float32, T*half)
	sin := make([]float32, T*half)
	invFreq := make([]float64, half)
	for i := 0; i < half; i++ {
		invFreq[i] = math.Pow(base, -float64(2*i)/float64(D))
	}
	for t := 0; t < T; t++ {
		p := float64(positions[t])
		for i := 0; i < half; i++ {
			a := p * invFreq[i]
			cos[t*half+i] = float32(math.Cos(a))
			sin[t*half+i] = float32(math.Sin(a))
		}
	}
	rows := x.size / D
	out := newTensorUninit(x.shape)
	rotate(out.data, x.data, cos, sin, rows, T, D, false)
	res := record(out, "RoPE", []*Tensor{x}, func(gy *Tensor) {
		g := newTensorUninit(x.shape)
		rotate(g.data, gy.values(), cos, sin, rows, T, D, true)
		x.accumGrad(g)
	})
	runtime.KeepAlive(x)
	return res
}

// rotate applies the rotary rotation (or its transpose when inverse) from
// src into dst. rows is the number of D-length vectors; each row's position
// index is (row % T).
func rotate(dst, src, cos, sin []float32, rows, T, D int, inverse bool) {
	half := D / 2
	parallel.Range(rows, 1<<12/D+1, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			t := r % T
			c := cos[t*half : t*half+half]
			s := sin[t*half : t*half+half]
			so := src[r*D : r*D+D]
			do := dst[r*D : r*D+D]
			for i := 0; i < half; i++ {
				x1, x2 := so[i], so[i+half]
				if inverse {
					do[i] = x1*c[i] + x2*s[i]
					do[i+half] = -x1*s[i] + x2*c[i]
				} else {
					do[i] = x1*c[i] - x2*s[i]
					do[i+half] = x1*s[i] + x2*c[i]
				}
			}
		}
	})
}
