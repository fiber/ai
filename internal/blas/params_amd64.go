//go:build amd64

package blas

// Tuned on a Skylake-SP Xeon Gold 6130 (32 KiB L1d, 1 MiB L2): fewer,
// deeper K blocks won the sweep (KC 256/384/512 → 924/982/999 GFLOPS at
// n=1024, 16 workers) and MC barely mattered once A is packed once per
// block. The AVX-512 B panel is 32×512×4 = 64 KiB (L2), the AVX2 panel
// 32 KiB.
const (
	defaultKC = 512
	defaultMC = 112 // rounded down to a multiple of MR at use: 112 for the 14×32 AVX-512 tile, 108 for the 6×16 AVX2 tile
	defaultNC = 4096
)
