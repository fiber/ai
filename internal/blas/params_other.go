//go:build !amd64 && !arm64

package blas

const (
	defaultKC = 256
	defaultMC = 128
	defaultNC = 4096
)
