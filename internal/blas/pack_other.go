//go:build !arm64

package blas

// Without an assembly transpose the plain streaming loop in packA is
// faster; packRows4 is never called here.
const packTranspose = false

func packRows4(dst, src *float32, rs, pb, mr int) { panic("blas: packRows4 without assembly") }
