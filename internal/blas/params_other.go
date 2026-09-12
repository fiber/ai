//go:build !amd64 && !arm64

package blas

const (
	defaultKC = 256
	defaultMC = 128
	defaultNC = 4096
)

// Unmeasured architecture: take the conservative limit, which gives up
// the least parallelism. See SmallLimit.
const defaultSmallLimit = 96 * 96 * 96
