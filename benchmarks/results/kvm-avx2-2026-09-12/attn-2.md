## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 26.39 ms | 162.8 |
| same with causal mask | 24.81 ms | 173.1 |
| [1×8×2048×64] long sequence | 51.19 ms | 167.8 |


allocator: mapped hits 71, misses 9, retained 37 MiB, pinned 0 MiB; GC cycles 1, forced 0
