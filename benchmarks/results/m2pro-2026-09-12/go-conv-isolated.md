## Convolution and attention measured alone (`-only conv`, `-only attention`)

In a full `go run ./cmd/bench` the convolution case follows the
element-wise section, which leaves 404 MiB retained in the mapped pool
and forces some 3 900 collections. The convolution then measures 184
GFLOPS. Run on its own it measures 334 to 352. Attention is not
sensitive in the same way — 991 in the suite, 1 001 to 1 006 alone —
because it allocates far less per iteration.

| run | conv [32×64×56×56] · 64 filters 3×3, pad 1 |
|---|---:|
| in the full suite | 40.13 ms, 184.4 GFLOPS |
| alone, run 1 | 21.25 ms, 348.2 GFLOPS |
| alone, run 2 | 22.15 ms, 334.0 GFLOPS |
| alone, run 3 | 21.02 ms, 351.9 GFLOPS |
| alone, off-heap storage disabled | 26.18 ms, 282.6 GFLOPS |

| run | attention [8×8×512×64] | long sequence [1×8×2048×64] |
|---|---:|---:|
| in the full suite | 4.33 ms, 991.2 | 9.02 ms, 952.5 |
| alone, run 1 | 4.27 ms, 1 005.5 | 8.96 ms, 958.4 |
| alone, run 2 | 4.29 ms, 1 000.9 | 9.00 ms, 954.8 |

BENCHMARKS.md quotes the isolated convolution figure and says so. The
ordering effect is worth its own spec: either the suite resets the pool
between sections, or every case that allocates heavily is measured alone.
