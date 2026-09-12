//go:build arm64

package blas

// Tuned on Apple M2 Pro (128 KiB L1d, 16 MiB shared L2): a KC=512 B panel
// (12×512×4 = 24 KiB) and A panel (8×512×4 = 16 KiB) both sit in L1, the
// packed B block (512×4096×4 = 8 MiB) in L2.
const (
	defaultKC = 512
	defaultMC = 128
	defaultNC = 4096
)

// One thread drives most of the AMX unit — 1 182 of 2 118 GFLOPS at
// n=1024 on an M2 Pro — so the small path gives up little parallelism
// and its saving on fixed cost reaches much further up. See SmallLimit.
const defaultSmallLimit = 160 * 160 * 160
