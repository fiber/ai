package blas

// packTranspose selects the 4×4 register-transpose packing of
// row-contiguous blocks, which has an assembly implementation on arm64.
const packTranspose = true

// packWidth is the number of source rows one packRows call transposes.
const packWidth = 4

// packRows transposes a full group of 4 rows and returns 4; it declines
// smaller groups (returns 0), which the caller packs with the scalar loop.
func packRows(dst, src *float32, rs, pb, width, rows int) int {
	if rows < packWidth {
		return 0
	}
	packRows4(dst, src, rs, pb, width)
	return packWidth
}

// packRows4 transposes 4 contiguous rows of A (row stride rs floats, pb
// columns, pb % 4 == 0) into a k-major panel with row stride mr:
// dst[p*mr + r] = src[r*rs + p]. Implemented in pack_arm64.s with 4×4
// NEON register transposes.
//
//go:noescape
func packRows4(dst, src *float32, rs, pb, mr int)
