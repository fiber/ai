## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, workers 6, backend amx, Go go1.26.2

### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 20.32 ms | 364.1 |

packed operands: hits 0, misses 45, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries


allocator: mapped hits 125, misses 12, retained 336 MiB, pinned 0 MiB; GC cycles 6, forced 5
