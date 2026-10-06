package gemma

import "math"

// Bicubic resampling with antialiasing on 8-bit RGB, following Pillow's
// Resample.c step for step: a separable filter, horizontal pass first,
// coefficients in 22-bit fixed point, the intermediate image rounded to
// 8 bits. PyTorch's native uint8 path computes the same thing on x86, so
// the image processor the reference uses produces these pixels.

// precisionBits is Pillow's fixed-point precision for 8-bit images.
const precisionBits = 32 - 8 - 2

// bicubic is Pillow's cubic convolution kernel with a = −0.5.
func bicubic(x float64) float64 {
	const a = -0.5
	if x < 0 {
		x = -x
	}
	if x < 1 {
		return ((a+2)*x-(a+3))*x*x + 1
	}
	if x < 2 {
		return (((x-5)*x+8)*x - 4) * a
	}
	return 0
}

// resampleCoeffs computes, for every output position, the first input
// index, the number of inputs and their fixed-point weights. When
// shrinking the kernel is widened by the scale, which is the antialiasing.
func resampleCoeffs(inSize, outSize int) (bounds []int, kk []int32, ksize int) {
	const support = 2.0
	scale := float64(inSize) / float64(outSize)
	filterscale := max(scale, 1)
	sup := support * filterscale
	ksize = int(math.Ceil(sup))*2 + 1
	bounds = make([]int, 2*outSize)
	kk = make([]int32, outSize*ksize)
	w := make([]float64, ksize)
	for xx := range outSize {
		center := (float64(xx) + 0.5) * scale
		ss := 1 / filterscale
		xmin := int(center - sup + 0.5)
		if xmin < 0 {
			xmin = 0
		}
		xmax := int(center + sup + 0.5)
		if xmax > inSize {
			xmax = inSize
		}
		xmax -= xmin
		var ww float64
		for x := range xmax {
			w[x] = bicubic((float64(x+xmin) - center + 0.5) * ss)
			ww += w[x]
		}
		for x := range xmax {
			v := w[x]
			if ww != 0 {
				v /= ww
			}
			// Pillow rounds half away from zero when converting to fixed point.
			if v < 0 {
				kk[xx*ksize+x] = int32(-0.5 + v*(1<<precisionBits))
			} else {
				kk[xx*ksize+x] = int32(0.5 + v*(1<<precisionBits))
			}
		}
		bounds[2*xx], bounds[2*xx+1] = xmin, xmax
	}
	return bounds, kk, ksize
}

// clip8 turns a fixed-point accumulator into a pixel value.
func clip8(v int32) uint8 {
	v >>= precisionBits
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// resizeRGB resizes an interleaved 8-bit RGB image of w×h to nw×nh.
func resizeRGB(src []uint8, w, h, nw, nh int) []uint8 {
	if w == nw && h == nh {
		return src
	}
	cur, cw, ch := src, w, h
	bv, kv, ksv := resampleCoeffs(h, nh)
	if nw != w {
		bh, kh, ksh := resampleCoeffs(w, nw)
		// Only the input rows the vertical pass reads are resampled
		// horizontally, and the vertical bounds are shifted to match.
		first := bv[0]
		last := bv[2*(nh-1)] + bv[2*(nh-1)+1]
		if nh == h {
			first, last = 0, h
		}
		rows := last - first
		tmp := make([]uint8, rows*nw*3)
		for y := range rows {
			in := cur[(y+first)*cw*3:]
			out := tmp[y*nw*3:]
			for x := range nw {
				xmin, xn := bh[2*x], bh[2*x+1]
				k := kh[x*ksh:]
				r, g, b := int32(1<<(precisionBits-1)), int32(1<<(precisionBits-1)), int32(1<<(precisionBits-1))
				for i := range xn {
					p := in[(xmin+i)*3:]
					r += int32(p[0]) * k[i]
					g += int32(p[1]) * k[i]
					b += int32(p[2]) * k[i]
				}
				out[x*3], out[x*3+1], out[x*3+2] = clip8(r), clip8(g), clip8(b)
			}
		}
		if nh != h {
			for i := 0; i < nh; i++ {
				bv[2*i] -= first
			}
		}
		cur, cw, ch = tmp, nw, rows
	}
	if nh == h {
		return cur
	}
	out := make([]uint8, nh*nw*3)
	for y := range nh {
		ymin, yn := bv[2*y], bv[2*y+1]
		k := kv[y*ksv:]
		for x := range cw {
			r, g, b := int32(1<<(precisionBits-1)), int32(1<<(precisionBits-1)), int32(1<<(precisionBits-1))
			for i := range yn {
				p := cur[((ymin+i)*cw+x)*3:]
				r += int32(p[0]) * k[i]
				g += int32(p[1]) * k[i]
				b += int32(p[2]) * k[i]
			}
			o := out[(y*nw+x)*3:]
			o[0], o[1], o[2] = clip8(r), clip8(g), clip8(b)
		}
	}
	_ = ch
	return out
}
