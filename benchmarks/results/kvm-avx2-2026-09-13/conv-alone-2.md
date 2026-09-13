## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 84.84 ms | 87.2 |

packed operands: hits 0, misses 11, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries


allocator: mapped hits 23, misses 12, retained 392 MiB, pinned 0 MiB; GC cycles 2, forced 1
