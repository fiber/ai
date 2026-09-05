//go:build !arm64

package blas

func packRows4(dst, src *float32, rs, pb, mr int) {
	d := unsafeSlice(dst, pb*mr)
	s := unsafeSlice(src, 3*rs+pb)
	for r := 0; r < 4; r++ {
		row := s[r*rs : r*rs+pb]
		for p, v := range row {
			d[p*mr+r] = v
		}
	}
}
