package blas

// packRows4 transposes 4 contiguous rows of A (row stride rs floats, pb
// columns, pb % 4 == 0) into a k-major panel with row stride mr:
// dst[p*mr + r] = src[r*rs + p]. Implemented in pack_arm64.s with 4×4
// NEON register transposes.
//
//go:noescape
func packRows4(dst, src *float32, rs, pb, mr int)
