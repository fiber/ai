## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 46.4 | 33.6 |
| 256 | 41.7 | 152.5 |
| 512 | 46.0 | 199.7 |
| 1024 | 52.1 | 229.1 |
| 2048 | 47.7 | 288.1 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 10.1 | 10.8 |
| [8×4096]·[4096×4096] | 88.1 | 21.8 |
| [64×1024]·[1024×1024] | 193.2 | 164.4 |
| [256×768]·[768×3072] | 255.0 | 221.0 |
| [512×512]·[512×512] | 246.1 | 248.2 |
| [1024×1024]·[1024×1024] | 283.4 | 267.3 |
| [1024×1024]·[1024×1024]ᵀ (view) | 259.5 |

packed operands: hits 2415, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries

### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 5.5 µs | 11.8 |
| 64² | 18.4 µs | 28.5 |
| 96² | 62.7 µs | 28.2 |
| 128² | 119.4 µs | 35.1 |
| 160² | 196.2 µs | 41.8 |
| 192² | 94.1 µs | 150.5 |
| 256² | 155.2 µs | 216.2 |
| [64×24]·[24×16] | 4.8 µs | 10.3 |
| [64×16]·[16×3] | 4.6 µs | 1.3 |
| [256×24]·[24×16] | 12.5 µs | 15.8 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 59.4 µs | 1.08M |
| forward + backward + Adam step | 232.9 µs | 274.7K |

packed operands: hits 4509, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 98.3 µs | 8.0 |
| x + y, result released | 64K | 11.8 µs | 66.7 |
| x * y | 64K | 31.8 µs | 24.7 |
| x + row (broadcast) | 64K | 29.6 µs | 17.7 |
| x * 2.5 | 64K | 28.3 µs | 18.5 |
| exp(x) | 64K | 32.8 µs | 16.0 |
| exp(x), result released | 64K | 23.0 µs | 22.8 |
| tanh(x) | 64K | 35.9 µs | 14.6 |
| tanh(x), result released | 64K | 26.2 µs | 20.0 |
| sigmoid(x) | 64K | 57.8 µs | 9.1 |
| gelu(x) | 64K | 67.0 µs | 7.8 |
| gelu(x), result released | 64K | 54.1 µs | 9.7 |
| relu(x) | 64K | 31.0 µs | 16.9 |
| x + y | 1M | 558.4 µs | 22.5 |
| x + y, result released | 1M | 164.0 µs | 76.7 |
| x * y | 1M | 336.4 µs | 37.4 |
| x + row (broadcast) | 1M | 316.1 µs | 26.5 |
| x * 2.5 | 1M | 298.8 µs | 28.1 |
| exp(x) | 1M | 385.9 µs | 21.7 |
| exp(x), result released | 1M | 324.9 µs | 25.8 |
| tanh(x) | 1M | 462.8 µs | 18.1 |
| tanh(x), result released | 1M | 412.3 µs | 20.3 |
| sigmoid(x) | 1M | 653.5 µs | 12.8 |
| gelu(x) | 1M | 824.3 µs | 10.2 |
| gelu(x), result released | 1M | 638.1 µs | 13.1 |
| relu(x) | 1M | 321.5 µs | 26.1 |
| x + y | 16M | 8.26 ms | 24.4 |
| x + y, result released | 16M | 5.98 ms | 33.7 |
| x * y | 16M | 7.81 ms | 25.8 |
| x + row (broadcast) | 16M | 6.39 ms | 21.0 |
| x * 2.5 | 16M | 6.27 ms | 21.4 |
| exp(x) | 16M | 6.34 ms | 21.2 |
| exp(x), result released | 16M | 4.90 ms | 27.4 |
| tanh(x) | 16M | 6.59 ms | 20.4 |
| tanh(x), result released | 16M | 5.65 ms | 23.8 |
| sigmoid(x) | 16M | 10.58 ms | 12.7 |
| gelu(x) | 16M | 13.68 ms | 9.8 |
| gelu(x), result released | 16M | 12.78 ms | 10.5 |
| relu(x) | 16M | 5.84 ms | 23.0 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 1.69 ms | 39.7 |
| sum(dim=0) | [4096×4096] | 2.15 ms | 31.2 |
| sum(dim=1) | [4096×4096] | 1.77 ms | 37.8 |
| max(dim=1) | [4096×4096] | 1.89 ms | 35.5 |
| softmax(dim=1) | [4096×4096] | 9.51 ms | 14.1 |
| layernorm | [4096×4096] | 7.98 ms | 16.8 |
| transpose+contiguous | [4096×4096] | 21.57 ms | 6.2 |

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 24.51 ms | 175.3 |
| same with causal mask | 25.61 ms | 167.7 |
| [1×8×2048×64] long sequence | 43.80 ms | 196.1 |

### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 100.68 ms | 73.5 |

packed operands: hits 0, misses 8, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.83 ms | 139.9K |
| forward + backward | 9.24 ms | 27.7K |
| forward + backward + Adam step | 7.85 ms | 32.6K |

packed operands: hits 996, misses 342, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries

### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 365002, misses 1563, retained 490 MiB, pinned 0 MiB; GC cycles 833, forced 799
