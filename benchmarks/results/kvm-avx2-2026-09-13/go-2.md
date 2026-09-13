## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 23.9 | 51.6 |
| 256 | 48.5 | 163.3 |
| 512 | 48.5 | 209.1 |
| 1024 | 49.0 | 255.6 |
| 2048 | 46.4 | 269.9 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 10.1 | 8.3 |
| [8×4096]·[4096×4096] | 75.8 | 26.2 |
| [64×1024]·[1024×1024] | 267.4 | 121.4 |
| [256×768]·[768×3072] | 269.1 | 198.2 |
| [512×512]·[512×512] | 266.9 | 230.3 |
| [1024×1024]·[1024×1024] | 233.9 | 260.7 |
| [1024×1024]·[1024×1024]ᵀ (view) | 246.4 |

packed operands: hits 3663, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries


allocator: mapped hits 28621, misses 25, retained 184 MiB, pinned 0 MiB; GC cycles 4, forced 0
### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 4.7 µs | 13.9 |
| 64² | 16.8 µs | 31.2 |
| 96² | 50.3 µs | 35.2 |
| 128² | 39.8 µs | 105.3 |
| 160² | 54.9 µs | 149.2 |
| 192² | 92.1 µs | 153.7 |
| 256² | 160.4 µs | 209.1 |
| [64×24]·[24×16] | 5.4 µs | 9.2 |
| [64×16]·[16×3] | 4.5 µs | 1.4 |
| [256×24]·[24×16] | 11.9 µs | 16.5 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 78.2 µs | 818.3K |
| forward + backward + Adam step | 249.2 µs | 256.8K |

packed operands: hits 5714, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries


allocator: mapped hits 55950, misses 12, retained 1 MiB, pinned 0 MiB; GC cycles 52, forced 0
### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 35.7 µs | 22.0 |
| x + y, result released | 64K | 13.7 µs | 57.6 |
| x * y | 64K | 30.0 µs | 26.2 |
| x + row (broadcast) | 64K | 30.7 µs | 17.1 |
| x * 2.5 | 64K | 27.0 µs | 19.4 |
| exp(x) | 64K | 31.6 µs | 16.6 |
| exp(x), result released | 64K | 23.1 µs | 22.7 |
| tanh(x) | 64K | 34.7 µs | 15.1 |
| tanh(x), result released | 64K | 24.8 µs | 21.1 |
| sigmoid(x) | 64K | 46.7 µs | 11.2 |
| gelu(x) | 64K | 58.2 µs | 9.0 |
| gelu(x), result released | 64K | 39.8 µs | 13.2 |
| relu(x) | 64K | 26.2 µs | 20.0 |
| x + y | 1M | 362.8 µs | 34.7 |
| x + y, result released | 1M | 122.8 µs | 102.5 |
| x * y | 1M | 330.4 µs | 38.1 |
| x + row (broadcast) | 1M | 297.7 µs | 28.2 |
| x * 2.5 | 1M | 299.2 µs | 28.0 |
| exp(x) | 1M | 466.8 µs | 18.0 |
| exp(x), result released | 1M | 236.9 µs | 35.4 |
| tanh(x) | 1M | 397.8 µs | 21.1 |
| tanh(x), result released | 1M | 302.4 µs | 27.7 |
| sigmoid(x) | 1M | 573.5 µs | 14.6 |
| gelu(x) | 1M | 816.5 µs | 10.3 |
| gelu(x), result released | 1M | 718.2 µs | 11.7 |
| relu(x) | 1M | 299.7 µs | 28.0 |
| x + y | 16M | 10.19 ms | 19.8 |
| x + y, result released | 16M | 8.71 ms | 23.1 |
| x * y | 16M | 8.89 ms | 22.7 |
| x + row (broadcast) | 16M | 6.76 ms | 19.8 |
| x * 2.5 | 16M | 6.10 ms | 22.0 |
| exp(x) | 16M | 6.72 ms | 20.0 |
| exp(x), result released | 16M | 5.58 ms | 24.1 |
| tanh(x) | 16M | 7.15 ms | 18.8 |
| tanh(x), result released | 16M | 6.51 ms | 20.6 |
| sigmoid(x) | 16M | 10.75 ms | 12.5 |
| gelu(x) | 16M | 14.11 ms | 9.5 |
| gelu(x), result released | 16M | 12.75 ms | 10.5 |
| relu(x) | 16M | 6.27 ms | 21.4 |


allocator: mapped hits 446283, misses 1095, retained 384 MiB, pinned 0 MiB; GC cycles 990, forced 989
### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.45 ms | 27.4 |
| sum(dim=0) | [4096×4096] | 2.67 ms | 25.1 |
| sum(dim=1) | [4096×4096] | 2.18 ms | 30.7 |
| max(dim=1) | [4096×4096] | 2.11 ms | 31.9 |
| softmax(dim=1) | [4096×4096] | 11.58 ms | 11.6 |
| layernorm | [4096×4096] | 8.59 ms | 15.6 |
| transpose+contiguous | [4096×4096] | 21.53 ms | 6.2 |


allocator: mapped hits 222, misses 6, retained 0 MiB, pinned 0 MiB; GC cycles 67, forced 65
### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 27.39 ms | 156.8 |
| same with causal mask | 29.15 ms | 147.3 |
| [1×8×2048×64] long sequence | 50.25 ms | 170.9 |


allocator: mapped hits 88, misses 9, retained 37 MiB, pinned 0 MiB; GC cycles 1, forced 0
### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 118.41 ms | 62.5 |

packed operands: hits 0, misses 8, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries


allocator: mapped hits 14, misses 12, retained 252 MiB, pinned 0 MiB; GC cycles 1, forced 0
### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.79 ms | 143.0K |
| forward + backward | 8.46 ms | 30.3K |
| forward + backward + Adam step | 8.48 ms | 30.2K |

packed operands: hits 1294, misses 422, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 4390, misses 353, retained 17 MiB, pinned 0 MiB; GC cycles 5, forced 4
### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 0, misses 0, retained 0 MiB, pinned 0 MiB; GC cycles 1, forced 0
