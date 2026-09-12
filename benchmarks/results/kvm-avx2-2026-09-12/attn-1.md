## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 23.20 ms | 185.1 |
| same with causal mask | 23.70 ms | 181.2 |
| [1×8×2048×64] long sequence | 43.39 ms | 198.0 |


allocator: mapped hits 79, misses 9, retained 37 MiB, pinned 0 MiB; GC cycles 1, forced 0
