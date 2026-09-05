package tensor

import (
	"math"

	"github.com/fiber/ai/internal/parallel"
)

// Convolutions via im2col: every output position's receptive field is
// laid out as a column, so the convolution becomes one matrix product per
// batch element and runs on the GEMM path.

// convOut is the output length of a convolution over n with kernel k.
func convOut(n, k, stride, pad int) int {
	o := (n+2*pad-k)/stride + 1
	if o < 0 {
		return 0
	}
	return o
}

// Im2Col lays the kh×kw receptive fields of x [N,C,H,W] out as columns:
// the result is [C·kh·kw, N·Ho·Wo] where Ho and Wo are the output sizes
// for the given stride and zero padding, all images side by side so that
// a convolution is a single matrix product for the whole batch. Its
// backward scatters gradients back into x, adding where fields overlap.
func Im2Col(x *Tensor, kh, kw, stride, pad int) *Tensor {
	if len(x.shape) != 4 {
		fail("Im2Col", "expected [N C H W], got %v", x.shape)
	}
	if kh <= 0 || kw <= 0 || stride <= 0 || pad < 0 {
		fail("Im2Col", "kernel %d×%d, stride %d, pad %d are not valid", kh, kw, stride, pad)
	}
	xc := x.Contiguous()
	n, c, h, w := xc.shape[0], xc.shape[1], xc.shape[2], xc.shape[3]
	ho, wo := convOut(h, kh, stride, pad), convOut(w, kw, stride, pad)
	if ho == 0 || wo == 0 {
		fail("Im2Col", "kernel %d×%d does not fit input %d×%d with pad %d", kh, kw, h, w, pad)
	}
	rows, cols := c*kh*kw, ho*wo
	out := newTensorUninit(Shape{rows, n * cols})
	xd := xc.values()
	parallel.For(n*rows, func(task int) {
		r, b := task/n, task%n
		ci, ki, kj := r/(kh*kw), (r/kw)%kh, r%kw
		src := xd[(b*c+ci)*h*w:]
		dst := out.data[r*n*cols+b*cols:]
		// with stride 1 the valid part of every output row is one contiguous
		// run of the input row: a copy, with zeros on either side
		xlo, xhi := 0, wo
		if stride == 1 {
			xlo, xhi = max(0, pad-kj), min(wo, w+pad-kj)
		}
		for oy := 0; oy < ho; oy++ {
			iy := oy*stride - pad + ki
			row := dst[oy*wo : (oy+1)*wo]
			if iy < 0 || iy >= h {
				clear(row)
				continue
			}
			if stride == 1 {
				clear(row[:xlo])
				copy(row[xlo:xhi], src[iy*w+xlo-pad+kj:])
				clear(row[xhi:])
				continue
			}
			for ox := 0; ox < wo; ox++ {
				ix := ox*stride - pad + kj
				if ix < 0 || ix >= w {
					row[ox] = 0
				} else {
					row[ox] = src[iy*w+ix]
				}
			}
		}
	})
	return record(out, "Im2Col", []*Tensor{xc}, func(gy *Tensor) {
		g := gy.Contiguous()
		gx := newTensor(xc.shape)          // zero, then scatter-add
		parallel.For(n*c, func(task int) { // one task per input channel image: no write conflicts
			b, ci := task/c, task%c
			dst := gx.data[(b*c+ci)*h*w:]
			for ki := 0; ki < kh; ki++ {
				for kj := 0; kj < kw; kj++ {
					r := (ci*kh+ki)*kw + kj
					src := g.data[r*n*cols+b*cols:]
					for oy := 0; oy < ho; oy++ {
						iy := oy*stride - pad + ki
						if iy < 0 || iy >= h {
							continue
						}
						for ox := 0; ox < wo; ox++ {
							ix := ox*stride - pad + kj
							if ix >= 0 && ix < w {
								dst[iy*w+ix] += src[oy*wo+ox]
							}
						}
					}
				}
			}
		})
		xc.accumGrad(gx)
	})
}

// Conv2D convolves x [N,C,H,W] with w [O,C,kh,kw] and adds b [O] (nil
// for none); stride and zero padding apply to both directions. The
// result is [N,O,Ho,Wo]; without a bias it is a view with the batch and
// channel dimensions swapped in memory, which every operation accepts.
func Conv2D(x, w, b *Tensor, stride, pad int) *Tensor {
	if len(x.shape) != 4 || len(w.shape) != 4 {
		fail("Conv2D", "expected x [N C H W] and w [O C kh kw], got %v and %v", x.shape, w.shape)
	}
	if x.shape[1] != w.shape[1] {
		fail("Conv2D", "input has %d channels, filters expect %d", x.shape[1], w.shape[1])
	}
	o, kh, kw := w.shape[0], w.shape[2], w.shape[3]
	if b != nil && b.size != o {
		fail("Conv2D", "bias must have %d elements, got %d", o, b.size)
	}
	col := Im2Col(x, kh, kw, stride, pad) // [C·kh·kw, N·Ho·Wo]
	ho, wo := convOut(x.shape[2], kh, stride, pad), convOut(x.shape[3], kw, stride, pad)
	prod := w.Reshape(o, -1).MatMul(col) // one product for the whole batch: [O, N·Ho·Wo]
	col.Release()                        // no-op while a graph holds it
	out := prod.Reshape(o, x.shape[0], ho, wo).Permute(1, 0, 2, 3)
	if b != nil {
		biased := out.Add(b.Reshape(1, o, 1, 1))
		prod.Release()
		return biased
	}
	return out
}

// Conv1D convolves x [N,C,L] with w [O,C,k] plus bias b [O] or nil:
// Conv2D over a height of one. The result is [N,O,Lo].
func Conv1D(x, w, b *Tensor, stride, pad int) *Tensor {
	if len(x.shape) != 3 || len(w.shape) != 3 {
		fail("Conv1D", "expected x [N C L] and w [O C k], got %v and %v", x.shape, w.shape)
	}
	x4 := x.Reshape(x.shape[0], x.shape[1], 1, x.shape[2])
	w4 := w.Reshape(w.shape[0], w.shape[1], 1, w.shape[2])
	out := conv2DWithPad(x4, w4, b, stride, 0, pad)
	return out.Reshape(out.shape[0], out.shape[1], out.shape[3])
}

// conv2DWithPad is Conv2D with separate vertical and horizontal padding;
// Im2Col pads both the same, so a one-row input with padH = 0 is handled
// by padding only the width through a widened view.
func conv2DWithPad(x, w, b *Tensor, stride, padH, padW int) *Tensor {
	if padH == padW {
		return Conv2D(x, w, b, stride, padH)
	}
	if padH != 0 {
		fail("Conv1D", "internal: asymmetric padding only along the width")
	}
	if padW == 0 {
		return Conv2D(x, w, b, stride, 0)
	}
	// pad the width explicitly, then convolve without padding
	n, c, h, wd := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	padded := Cat(3, Zeros(n, c, h, padW), x, Zeros(n, c, h, padW))
	_ = wd
	return Conv2D(padded, w, b, stride, 0)
}

// MaxPool2D takes the maximum of every k×k window of x [N,C,H,W] with the
// given stride (k for non-overlapping windows). Gradients flow to the
// position of each window's maximum.
func MaxPool2D(x *Tensor, k, stride int) *Tensor {
	if len(x.shape) != 4 {
		fail("MaxPool2D", "expected [N C H W], got %v", x.shape)
	}
	if k <= 0 || stride <= 0 {
		fail("MaxPool2D", "window %d and stride %d must be positive", k, stride)
	}
	xc := x.Contiguous()
	n, c, h, w := xc.shape[0], xc.shape[1], xc.shape[2], xc.shape[3]
	ho, wo := convOut(h, k, stride, 0), convOut(w, k, stride, 0)
	if ho == 0 || wo == 0 {
		fail("MaxPool2D", "window %d does not fit input %d×%d", k, h, w)
	}
	out := newTensorUninit(Shape{n, c, ho, wo})
	argmax := make([]int32, out.size)
	xd := xc.values()
	parallel.For(n*c, func(img int) {
		src := xd[img*h*w:]
		for oy := 0; oy < ho; oy++ {
			for ox := 0; ox < wo; ox++ {
				best, bi := float32(math.Inf(-1)), 0
				for ky := 0; ky < k; ky++ {
					for kx := 0; kx < k; kx++ {
						i := (oy*stride+ky)*w + ox*stride + kx
						if v := src[i]; v > best {
							best, bi = v, i
						}
					}
				}
				o := (img*ho+oy)*wo + ox
				out.data[o] = best
				argmax[o] = int32(bi)
			}
		}
	})
	return record(out, "MaxPool2D", []*Tensor{xc}, func(gy *Tensor) {
		g := gy.Contiguous()
		gx := newTensor(xc.shape)
		parallel.For(n*c, func(img int) {
			base := img * h * w
			for o := img * ho * wo; o < (img+1)*ho*wo; o++ {
				gx.data[base+int(argmax[o])] += g.data[o]
			}
		})
		xc.accumGrad(gx)
	})
}
