package blas

import "unsafe"

func unsafeSlice(p *float32, n int) []float32 { return unsafe.Slice(p, n) }
