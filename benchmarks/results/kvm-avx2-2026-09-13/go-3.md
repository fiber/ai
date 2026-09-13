## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 38.2 | 72.5 |
| 256 | 43.9 | 156.9 |
| 512 | 45.2 | 216.2 |
| 1024 | 42.7 | 229.7 |
| 2048 | 48.5 | 233.8 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 9.9 | 9.2 |
| [8×4096]·[4096×4096] | 89.7 | 24.3 |
| [64×1024]·[1024×1024] | 259.4 | 155.8 |
| [256×768]·[768×3072] | 278.7 | 205.3 |
| [512×512]·[512×512] | 237.0 | 213.2 |
| [1024×1024]·[1024×1024] | 295.1 | 213.6 |
| [1024×1024]·[1024×1024]ᵀ (view) | 316.6 |

packed operands: hits 3673, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries


allocator: mapped hits 36471, misses 24, retained 197 MiB, pinned 0 MiB; GC cycles 5, forced 0
### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 4.7 µs | 13.9 |
| 64² | 17.9 µs | 29.3 |
| 96² | 48.4 µs | 36.6 |
| 128² | 39.9 µs | 105.1 |
| 160² | 68.8 µs | 119.1 |
| 192² | 95.1 µs | 148.8 |
| 256² | 332.8 µs | 100.8 |
| [64×24]·[24×16] | 5.9 µs | 8.3 |
| [64×16]·[16×3] | 4.7 µs | 1.3 |
| [256×24]·[24×16] | 11.8 µs | 16.7 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 67.1 µs | 954.4K |
| forward + backward + Adam step | 217.9 µs | 293.8K |

packed operands: hits 3315, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries


allocator: mapped hits 49999, misses 12, retained 1 MiB, pinned 0 MiB; GC cycles 51, forced 0
### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 29.9 µs | 26.3 |
| x + y, result released | 64K | 14.0 µs | 56.2 |
| x * y | 64K | 29.2 µs | 26.9 |
| x + row (broadcast) | 64K | 29.9 µs | 17.5 |
| x * 2.5 | 64K | 24.9 µs | 21.1 |
| exp(x) | 64K | 31.0 µs | 16.9 |
| exp(x), result released | 64K | 25.1 µs | 20.9 |
| tanh(x) | 64K | 33.1 µs | 15.8 |
| tanh(x), result released | 64K | 31.2 µs | 16.8 |
| sigmoid(x) | 64K | 46.2 µs | 11.3 |
| gelu(x) | 64K | 55.7 µs | 9.4 |
| gelu(x), result released | 64K | 45.1 µs | 11.6 |
| relu(x) | 64K | 26.3 µs | 19.9 |
| x + y | 1M | 362.5 µs | 34.7 |
| x + y, result released | 1M | 126.4 µs | 99.5 |
| x * y | 1M | 346.9 µs | 36.3 |
| x + row (broadcast) | 1M | 293.1 µs | 28.6 |
| x * 2.5 | 1M | 294.4 µs | 28.5 |
| exp(x) | 1M | 348.1 µs | 24.1 |
| exp(x), result released | 1M | 243.6 µs | 34.4 |
| tanh(x) | 1M | 415.3 µs | 20.2 |
| tanh(x), result released | 1M | 287.4 µs | 29.2 |
| sigmoid(x) | 1M | 658.8 µs | 12.7 |
| gelu(x) | 1M | 815.7 µs | 10.3 |
| gelu(x), result released | 1M | 654.6 µs | 12.8 |
| relu(x) | 1M | 285.1 µs | 29.4 |
| x + y | 16M | 7.91 ms | 25.5 |
| x + y, result released | 16M | 6.73 ms | 29.9 |
| x * y | 16M | 8.25 ms | 24.4 |
| x + row (broadcast) | 16M | 6.35 ms | 21.1 |
| x * 2.5 | 16M | 6.17 ms | 21.8 |
| exp(x) | 16M | 6.57 ms | 20.4 |
| exp(x), result released | 16M | 5.51 ms | 24.3 |
| tanh(x) | 16M | 6.71 ms | 20.0 |
| tanh(x), result released | 16M | 5.74 ms | 23.4 |
| sigmoid(x) | 16M | 11.53 ms | 11.6 |
| gelu(x) | 16M | 14.54 ms | 9.2 |
| gelu(x), result released | 16M | 13.48 ms | 10.0 |
| relu(x) | 16M | 6.28 ms | 21.4 |


allocator: mapped hits 446215, misses 1094, retained 448 MiB, pinned 0 MiB; GC cycles 1024, forced 1023
### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.48 ms | 27.0 |
| sum(dim=0) | [4096×4096] | 2.36 ms | 28.5 |
| sum(dim=1) | [4096×4096] | 2.07 ms | 32.4 |
| max(dim=1) | [4096×4096] | 1.90 ms | 35.3 |
| softmax(dim=1) | [4096×4096] | 8.59 ms | 15.6 |
| layernorm | [4096×4096] | 8.17 ms | 16.4 |
| transpose+contiguous | [4096×4096] | 19.88 ms | 6.8 |


allocator: mapped hits 256, misses 6, retained 192 MiB, pinned 0 MiB; GC cycles 67, forced 65
### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 22.78 ms | 188.6 |
| same with causal mask | 22.09 ms | 194.4 |
| [1×8×2048×64] long sequence | 44.81 ms | 191.7 |


allocator: mapped hits 102, misses 9, retained 37 MiB, pinned 0 MiB; GC cycles 1, forced 0
### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 94.08 ms | 78.6 |

packed operands: hits 0, misses 10, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries


allocator: mapped hits 20, misses 12, retained 420 MiB, pinned 0 MiB; GC cycles 2, forced 1
### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.37 ms | 186.9K |
| forward + backward | 7.55 ms | 33.9K |
| forward + backward + Adam step | 7.30 ms | 35.1K |

packed operands: hits 1660, misses 482, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 5254, misses 353, retained 86 MiB, pinned 0 MiB; GC cycles 6, forced 5
### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 0, misses 0, retained 0 MiB, pinned 0 MiB; GC cycles 1, forced 0
