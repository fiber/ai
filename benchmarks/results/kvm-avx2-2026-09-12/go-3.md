## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 48.7 | 44.2 |
| 256 | 52.2 | 175.0 |
| 512 | 53.2 | 234.7 |
| 1024 | 52.9 | 265.0 |
| 2048 | 47.4 | 255.4 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 9.8 | 9.3 |
| [8×4096]·[4096×4096] | 82.1 | 18.2 |
| [64×1024]·[1024×1024] | 205.7 | 144.2 |
| [256×768]·[768×3072] | 239.6 | 156.8 |
| [512×512]·[512×512] | 122.2 | 87.7 |
| [1024×1024]·[1024×1024] | 227.2 | 226.7 |
| [1024×1024]·[1024×1024]ᵀ (view) | 289.5 |

packed operands: hits 2121, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries

### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 5.4 µs | 12.2 |
| 64² | 17.5 µs | 29.9 |
| 96² | 58.3 µs | 30.3 |
| 128² | 88.8 µs | 47.2 |
| 160² | 166.8 µs | 49.1 |
| 192² | 101.7 µs | 139.1 |
| 256² | 174.2 µs | 192.6 |
| [64×24]·[24×16] | 4.8 µs | 10.3 |
| [64×16]·[16×3] | 4.5 µs | 1.4 |
| [256×24]·[24×16] | 11.6 µs | 17.0 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 50.7 µs | 1.26M |
| forward + backward + Adam step | 211.9 µs | 302.0K |

packed operands: hits 4017, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 97.6 µs | 8.1 |
| x + y, result released | 64K | 12.2 µs | 64.3 |
| x * y | 64K | 36.7 µs | 21.4 |
| x + row (broadcast) | 64K | 32.0 µs | 16.4 |
| x * 2.5 | 64K | 29.3 µs | 17.9 |
| exp(x) | 64K | 32.0 µs | 16.4 |
| exp(x), result released | 64K | 21.4 µs | 24.6 |
| tanh(x) | 64K | 37.9 µs | 13.8 |
| tanh(x), result released | 64K | 27.0 µs | 19.4 |
| sigmoid(x) | 64K | 44.4 µs | 11.8 |
| gelu(x) | 64K | 53.3 µs | 9.8 |
| gelu(x), result released | 64K | 36.8 µs | 14.2 |
| relu(x) | 64K | 28.2 µs | 18.6 |
| x + y | 1M | 580.5 µs | 21.7 |
| x + y, result released | 1M | 133.4 µs | 94.3 |
| x * y | 1M | 361.9 µs | 34.8 |
| x + row (broadcast) | 1M | 323.0 µs | 26.0 |
| x * 2.5 | 1M | 298.2 µs | 28.1 |
| exp(x) | 1M | 377.6 µs | 22.2 |
| exp(x), result released | 1M | 243.5 µs | 34.5 |
| tanh(x) | 1M | 423.5 µs | 19.8 |
| tanh(x), result released | 1M | 280.5 µs | 29.9 |
| sigmoid(x) | 1M | 635.3 µs | 13.2 |
| gelu(x) | 1M | 810.8 µs | 10.3 |
| gelu(x), result released | 1M | 637.1 µs | 13.2 |
| relu(x) | 1M | 302.4 µs | 27.7 |
| x + y | 16M | 9.13 ms | 22.1 |
| x + y, result released | 16M | 6.12 ms | 32.9 |
| x * y | 16M | 8.55 ms | 23.6 |
| x + row (broadcast) | 16M | 6.43 ms | 20.9 |
| x * 2.5 | 16M | 6.55 ms | 20.5 |
| exp(x) | 16M | 6.62 ms | 20.3 |
| exp(x), result released | 16M | 5.58 ms | 24.0 |
| tanh(x) | 16M | 6.94 ms | 19.3 |
| tanh(x), result released | 16M | 5.61 ms | 23.9 |
| sigmoid(x) | 16M | 10.77 ms | 12.5 |
| gelu(x) | 16M | 13.61 ms | 9.9 |
| gelu(x), result released | 16M | 12.26 ms | 10.9 |
| relu(x) | 16M | 5.90 ms | 22.8 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 1.63 ms | 41.1 |
| sum(dim=0) | [4096×4096] | 2.48 ms | 27.1 |
| sum(dim=1) | [4096×4096] | 1.91 ms | 35.2 |
| max(dim=1) | [4096×4096] | 2.48 ms | 27.0 |
| softmax(dim=1) | [4096×4096] | 10.16 ms | 13.2 |
| layernorm | [4096×4096] | 7.13 ms | 18.8 |
| transpose+contiguous | [4096×4096] | 19.37 ms | 6.9 |

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 22.54 ms | 190.5 |
| same with causal mask | 22.55 ms | 190.5 |
| [1×8×2048×64] long sequence | 41.32 ms | 207.9 |

### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 108.41 ms | 68.2 |

packed operands: hits 0, misses 8, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.55 ms | 164.9K |
| forward + backward | 10.21 ms | 25.1K |
| forward + backward + Adam step | 8.38 ms | 30.5K |

packed operands: hits 1111, misses 316, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries

### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 379260, misses 1567, retained 324 MiB, pinned 0 MiB; GC cycles 827, forced 791
