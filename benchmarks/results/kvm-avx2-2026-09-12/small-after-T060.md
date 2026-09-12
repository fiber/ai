## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 6.1 µs | 10.8 |
| 64² | 18.1 µs | 29.0 |
| 96² | 53.1 µs | 33.3 |
| 128² | 43.5 µs | 96.5 |
| 160² | 59.2 µs | 138.3 |
| 192² | 86.0 µs | 164.6 |
| 256² | 161.1 µs | 208.3 |
| [64×24]·[24×16] | 4.2 µs | 11.6 |
| [64×16]·[16×3] | 4.4 µs | 1.4 |
| [256×24]·[24×16] | 12.3 µs | 15.9 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 52.7 µs | 1.21M |
| forward + backward + Adam step | 230.8 µs | 277.3K |

packed operands: hits 4345, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries


allocator: mapped hits 40411, misses 12, retained 1 MiB, pinned 0 MiB; GC cycles 40, forced 0
