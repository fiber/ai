## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, workers 6, backend neon, Go go1.26.2

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 85.5 | 86.3 |
| 256 | 95.5 | 407.7 |
| 512 | 100.4 | 535.2 |
| 1024 | 100.8 | 531.8 |
| 2048 | 101.1 | 552.1 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 14.1 | 13.7 |
| [8×4096]·[4096×4096] | 364.4 | 56.6 |
| [64×1024]·[1024×1024] | 501.9 | 411.1 |
| [256×768]·[768×3072] | 546.4 | 498.3 |
| [512×512]·[512×512] | 561.1 | 532.6 |
| [1024×1024]·[1024×1024] | 544.3 | 540.5 |
| [1024×1024]·[1024×1024]ᵀ (view) | 552.9 |

packed operands: hits 6003, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries

### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 1.9 µs | 34.5 |
| 64² | 8.7 µs | 60.1 |
| 96² | 21.6 µs | 82.1 |
| 128² | 49.0 µs | 85.6 |
| 160² | 92.8 µs | 88.2 |
| 192² | 37.6 µs | 376.5 |
| 256² | 80.9 µs | 414.8 |
| [64×24]·[24×16] | 1.9 µs | 25.5 |
| [64×16]·[16×3] | 1.1 µs | 5.4 |
| [256×24]·[24×16] | 5.8 µs | 33.8 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 17.2 µs | 3.73M |
| forward + backward + Adam step | 45.2 µs | 1.42M |

packed operands: hits 8652, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 6.1 µs | 129.4 |
| x + y, result released | 64K | 4.6 µs | 170.6 |
| x * y | 64K | 6.2 µs | 127.3 |
| x + row (broadcast) | 64K | 8.3 µs | 63.4 |
| x * 2.5 | 64K | 5.5 µs | 95.0 |
| exp(x) | 64K | 9.7 µs | 54.3 |
| exp(x), result released | 64K | 8.3 µs | 63.3 |
| tanh(x) | 64K | 10.1 µs | 51.9 |
| tanh(x), result released | 64K | 9.1 µs | 57.7 |
| sigmoid(x) | 64K | 15.8 µs | 33.3 |
| gelu(x) | 64K | 25.0 µs | 21.0 |
| gelu(x), result released | 64K | 20.5 µs | 25.6 |
| relu(x) | 64K | 5.5 µs | 95.4 |
| x + y | 1M | 49.6 µs | 253.5 |
| x + y, result released | 1M | 37.7 µs | 333.5 |
| x * y | 1M | 49.9 µs | 252.1 |
| x + row (broadcast) | 1M | 67.6 µs | 124.1 |
| x * 2.5 | 1M | 46.1 µs | 181.8 |
| exp(x) | 1M | 100.2 µs | 83.7 |
| exp(x), result released | 1M | 96.5 µs | 86.9 |
| tanh(x) | 1M | 102.8 µs | 81.6 |
| tanh(x), result released | 1M | 92.1 µs | 91.1 |
| sigmoid(x) | 1M | 162.2 µs | 51.7 |
| gelu(x) | 1M | 230.9 µs | 36.3 |
| gelu(x), result released | 1M | 218.8 µs | 38.3 |
| relu(x) | 1M | 47.0 µs | 178.4 |
| x + y | 16M | 1.55 ms | 129.9 |
| x + y, result released | 16M | 1.32 ms | 152.7 |
| x * y | 16M | 1.54 ms | 131.1 |
| x + row (broadcast) | 16M | 1.54 ms | 87.1 |
| x * 2.5 | 16M | 995.6 µs | 134.8 |
| exp(x) | 16M | 1.59 ms | 84.5 |
| exp(x), result released | 16M | 1.38 ms | 97.6 |
| tanh(x) | 16M | 1.58 ms | 84.9 |
| tanh(x), result released | 16M | 1.51 ms | 89.1 |
| sigmoid(x) | 16M | 2.54 ms | 52.8 |
| gelu(x) | 16M | 3.72 ms | 36.1 |
| gelu(x), result released | 16M | 3.67 ms | 36.6 |
| relu(x) | 16M | 962.3 µs | 139.5 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 552.2 µs | 121.5 |
| sum(dim=0) | [4096×4096] | 613.4 µs | 109.4 |
| sum(dim=1) | [4096×4096] | 536.4 µs | 125.1 |
| max(dim=1) | [4096×4096] | 543.5 µs | 123.5 |
| softmax(dim=1) | [4096×4096] | 2.24 ms | 60.0 |
| layernorm | [4096×4096] | 1.76 ms | 76.3 |
| transpose+contiguous | [4096×4096] | 12.51 ms | 10.7 |

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 10.63 ms | 404.2 |
| same with causal mask | 10.83 ms | 396.7 |
| [1×8×2048×64] long sequence | 20.29 ms | 423.4 |

### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 38.65 ms | 191.4 |

packed operands: hits 0, misses 20, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 733.6 µs | 349.0K |
| forward + backward | 2.50 ms | 102.4K |
| forward + backward + Adam step | 2.60 ms | 98.3K |

packed operands: hits 2752, misses 1108, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries

### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 1372890, misses 1495, retained 352 MiB, pinned 0 MiB; GC cycles 4144, forced 4043
