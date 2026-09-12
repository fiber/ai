## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 34.77 ms | 123.5 |
| same with causal mask | 36.25 ms | 118.5 |
| [1×8×2048×64] long sequence | 60.19 ms | 142.7 |


allocator: mapped hits 54, misses 9, retained 37 MiB, pinned 0 MiB; GC cycles 1, forced 0
