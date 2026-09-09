//go:build !arm64 && !amd64

package blas

// Without an assembly transpose the plain streaming loop in packA is
// faster; packRows is never asked to take rows here.
const packTranspose = false

const packWidth = 1

func packRows(dst, src *float32, rs, pb, width, rows int) int { return 0 }
