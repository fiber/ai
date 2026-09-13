## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 38.9 | 77.1 |
| 256 | 55.5 | 162.8 |
| 512 | 42.5 | 225.3 |
| 1024 | 36.4 | 232.8 |
| 2048 | 52.1 | 240.1 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 9.8 | 10.9 |
| [8×4096]·[4096×4096] | 83.7 | 23.8 |
| [64×1024]·[1024×1024] | 214.0 | 178.2 |
| [256×768]·[768×3072] | 310.2 | 205.2 |
| [512×512]·[512×512] | 279.8 | 227.9 |
| [1024×1024]·[1024×1024] | 292.0 | 240.2 |
| [1024×1024]·[1024×1024]ᵀ (view) | 252.9 |

packed operands: hits 3415, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries


allocator: mapped hits 38171, misses 24, retained 195 MiB, pinned 0 MiB; GC cycles 5, forced 0
### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 5.1 µs | 12.8 |
| 64² | 18.7 µs | 28.1 |
| 96² | 48.1 µs | 36.8 |
| 128² | 37.0 µs | 113.5 |
| 160² | 55.8 µs | 146.8 |
| 192² | 81.5 µs | 173.7 |
| 256² | 245.5 µs | 136.7 |
| [64×24]·[24×16] | 5.6 µs | 8.8 |
| [64×16]·[16×3] | 3.7 µs | 1.7 |
| [256×24]·[24×16] | 12.5 µs | 15.7 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 55.2 µs | 1.16M |
| forward + backward + Adam step | 242.2 µs | 264.2K |

packed operands: hits 4145, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries


allocator: mapped hits 56567, misses 12, retained 1 MiB, pinned 0 MiB; GC cycles 51, forced 0
### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 34.2 µs | 23.0 |
| x + y, result released | 64K | 13.8 µs | 56.8 |
| x * y | 64K | 27.7 µs | 28.4 |
| x + row (broadcast) | 64K | 27.6 µs | 19.0 |
| x * 2.5 | 64K | 25.5 µs | 20.5 |
| exp(x) | 64K | 31.3 µs | 16.8 |
| exp(x), result released | 64K | 25.1 µs | 20.9 |
| tanh(x) | 64K | 32.5 µs | 16.2 |
| tanh(x), result released | 64K | 29.2 µs | 18.0 |
| sigmoid(x) | 64K | 48.0 µs | 10.9 |
| gelu(x) | 64K | 59.9 µs | 8.8 |
| gelu(x), result released | 64K | 40.8 µs | 12.8 |
| relu(x) | 64K | 28.6 µs | 18.3 |
| x + y | 1M | 343.5 µs | 36.6 |
| x + y, result released | 1M | 118.5 µs | 106.2 |
| x * y | 1M | 315.7 µs | 39.9 |
| x + row (broadcast) | 1M | 314.2 µs | 26.7 |
| x * 2.5 | 1M | 303.1 µs | 27.7 |
| exp(x) | 1M | 361.1 µs | 23.2 |
| exp(x), result released | 1M | 238.7 µs | 35.1 |
| tanh(x) | 1M | 409.5 µs | 20.5 |
| tanh(x), result released | 1M | 287.5 µs | 29.2 |
| sigmoid(x) | 1M | 570.0 µs | 14.7 |
| gelu(x) | 1M | 772.0 µs | 10.9 |
| gelu(x), result released | 1M | 676.8 µs | 12.4 |
| relu(x) | 1M | 283.4 µs | 29.6 |
| x + y | 16M | 7.91 ms | 25.5 |
| x + y, result released | 16M | 6.52 ms | 30.9 |
| x * y | 16M | 7.82 ms | 25.7 |
| x + row (broadcast) | 16M | 6.26 ms | 21.4 |
| x * 2.5 | 16M | 6.06 ms | 22.1 |
| exp(x) | 16M | 6.79 ms | 19.8 |
| exp(x), result released | 16M | 5.62 ms | 23.9 |
| tanh(x) | 16M | 7.55 ms | 17.8 |
| tanh(x), result released | 16M | 6.46 ms | 20.8 |
| sigmoid(x) | 16M | 10.71 ms | 12.5 |
| gelu(x) | 16M | 14.83 ms | 9.1 |
| gelu(x), result released | 16M | 14.37 ms | 9.3 |
| relu(x) | 16M | 5.87 ms | 22.9 |


allocator: mapped hits 450219, misses 1094, retained 448 MiB, pinned 0 MiB; GC cycles 1043, forced 1042
### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 1.91 ms | 35.1 |
| sum(dim=0) | [4096×4096] | 2.16 ms | 31.1 |
| sum(dim=1) | [4096×4096] | 1.86 ms | 36.0 |
| max(dim=1) | [4096×4096] | 2.01 ms | 33.3 |
| softmax(dim=1) | [4096×4096] | 9.36 ms | 14.3 |
| layernorm | [4096×4096] | 7.01 ms | 19.2 |
| transpose+contiguous | [4096×4096] | 18.45 ms | 7.3 |


allocator: mapped hits 272, misses 6, retained 0 MiB, pinned 0 MiB; GC cycles 81, forced 79
### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 22.62 ms | 189.9 |
| same with causal mask | 24.29 ms | 176.9 |
| [1×8×2048×64] long sequence | 41.87 ms | 205.2 |


allocator: mapped hits 101, misses 9, retained 37 MiB, pinned 0 MiB; GC cycles 1, forced 0
### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 97.43 ms | 75.9 |

packed operands: hits 0, misses 9, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries


allocator: mapped hits 17, misses 12, retained 448 MiB, pinned 0 MiB; GC cycles 2, forced 1
### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.58 ms | 162.1K |
| forward + backward | 7.72 ms | 33.1K |
| forward + backward + Adam step | 7.34 ms | 34.9K |

packed operands: hits 1475, misses 472, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 4994, misses 353, retained 115 MiB, pinned 0 MiB; GC cycles 6, forced 5
### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 0, misses 0, retained 0 MiB, pinned 0 MiB; GC cycles 1, forced 0
