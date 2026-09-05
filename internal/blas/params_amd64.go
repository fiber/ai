//go:build amd64

package blas

// Conservative defaults for 32–48 KiB L1d and 0.25–2 MiB L2 (typical x86):
// B panel 16×256×4 = 16 KiB, A block 96×256×4 = 96 KiB, B block 4 MiB.
const (
	defaultKC = 256
	defaultMC = 96 // multiple of both 6 (AVX2 MR) and 12 (AVX-512 MR)
	defaultNC = 4096
)
