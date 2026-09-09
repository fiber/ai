package blas

import "github.com/fiber/ai/internal/kernel"

// packTranspose selects the register-transpose packing of row-contiguous
// blocks (packRows8, AVX2). Off when the kernel package runs generic
// code, which is also the case on a CPU without AVX.
var packTranspose = kernel.Impl != "generic"

// packWidth is the number of source rows one packRows call transposes.
const packWidth = 8

// packRows8 transposes rows [0, rows) (1 ≤ rows ≤ 8) of a row-contiguous
// block (row stride rs floats, pb columns, pb % 8 == 0) into a k-major
// panel with row stride width: dst[p*width + r] = src[r*rs + p]. Rows
// beyond rows are neither read nor written (masked stores). Implemented
// in pack_amd64.s with 8×8 AVX2 register transposes.
//
//go:noescape
func packRows8(dst, src *float32, rs, pb, width, rows int)

// packRows transposes up to packWidth rows and returns how many it took.
func packRows(dst, src *float32, rs, pb, width, rows int) int {
	packRows8(dst, src, rs, pb, width, rows)
	return rows
}
