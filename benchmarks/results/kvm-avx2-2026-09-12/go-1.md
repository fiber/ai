## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, workers 6, backend avx2, Go go1.27.1

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 44.2 | 47.0 |
| 256 | 48.5 | 179.6 |
| 512 | 47.4 | 240.9 |
| 1024 | 48.8 | 248.2 |
| 2048 | 41.9 | 238.6 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 10.6 | 8.9 |
| [8×4096]·[4096×4096] | 85.3 | 12.4 |
| [64×1024]·[1024×1024] | 167.4 | 106.8 |
| [256×768]·[768×3072] | 230.6 | 167.1 |
| [512×512]·[512×512] | 213.0 | 207.3 |
| [1024×1024]·[1024×1024] | 266.0 | 278.5 |
| [1024×1024]·[1024×1024]ᵀ (view) | 276.0 |

packed operands: hits 2184, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries

### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 5.2 µs | 12.6 |
| 64² | 16.8 µs | 31.3 |
| 96² | 46.3 µs | 38.2 |
| 128² | 81.6 µs | 51.4 |
| 160² | 171.4 µs | 47.8 |
| 192² | 109.7 µs | 129.0 |
| 256² | 182.3 µs | 184.1 |
| [64×24]·[24×16] | 4.5 µs | 10.8 |
| [64×16]·[16×3] | 4.1 µs | 1.5 |
| [256×24]·[24×16] | 12.3 µs | 16.0 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 69.0 µs | 927.4K |
| forward + backward + Adam step | 284.6 µs | 224.9K |

packed operands: hits 3840, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 71.3 µs | 11.0 |
| x + y, result released | 64K | 19.2 µs | 40.9 |
| x * y | 64K | 41.4 µs | 19.0 |
| x + row (broadcast) | 64K | 33.9 µs | 15.4 |
| x * 2.5 | 64K | 31.5 µs | 16.7 |
| exp(x) | 64K | 39.0 µs | 13.4 |
| exp(x), result released | 64K | 27.4 µs | 19.1 |
| tanh(x) | 64K | 42.3 µs | 12.4 |
| tanh(x), result released | 64K | 27.3 µs | 19.2 |
| sigmoid(x) | 64K | 54.7 µs | 9.6 |
| gelu(x) | 64K | 60.0 µs | 8.7 |
| gelu(x), result released | 64K | 48.3 µs | 10.9 |
| relu(x) | 64K | 31.6 µs | 16.6 |
| x + y | 1M | 444.2 µs | 28.3 |
| x + y, result released | 1M | 136.3 µs | 92.3 |
| x * y | 1M | 334.2 µs | 37.7 |
| x + row (broadcast) | 1M | 308.7 µs | 27.2 |
| x * 2.5 | 1M | 316.2 µs | 26.5 |
| exp(x) | 1M | 485.5 µs | 17.3 |
| exp(x), result released | 1M | 348.1 µs | 24.1 |
| tanh(x) | 1M | 450.8 µs | 18.6 |
| tanh(x), result released | 1M | 304.0 µs | 27.6 |
| sigmoid(x) | 1M | 632.7 µs | 13.3 |
| gelu(x) | 1M | 824.9 µs | 10.2 |
| gelu(x), result released | 1M | 663.6 µs | 12.6 |
| relu(x) | 1M | 303.5 µs | 27.6 |
| x + y | 16M | 9.11 ms | 22.1 |
| x + y, result released | 16M | 6.85 ms | 29.4 |
| x * y | 16M | 8.24 ms | 24.4 |
| x + row (broadcast) | 16M | 6.54 ms | 20.5 |
| x * 2.5 | 16M | 6.43 ms | 20.9 |
| exp(x) | 16M | 6.21 ms | 21.6 |
| exp(x), result released | 16M | 5.11 ms | 26.3 |
| tanh(x) | 16M | 7.13 ms | 18.8 |
| tanh(x), result released | 16M | 5.75 ms | 23.3 |
| sigmoid(x) | 16M | 10.53 ms | 12.7 |
| gelu(x) | 16M | 12.57 ms | 10.7 |
| gelu(x), result released | 16M | 11.47 ms | 11.7 |
| relu(x) | 16M | 6.01 ms | 22.3 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 1.86 ms | 36.0 |
| sum(dim=0) | [4096×4096] | 2.38 ms | 28.2 |
| sum(dim=1) | [4096×4096] | 1.92 ms | 35.0 |
| max(dim=1) | [4096×4096] | 1.85 ms | 36.3 |
| softmax(dim=1) | [4096×4096] | 9.38 ms | 14.3 |
| layernorm | [4096×4096] | 7.59 ms | 17.7 |
| transpose+contiguous | [4096×4096] | 22.51 ms | 6.0 |

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 27.56 ms | 155.8 |
| same with causal mask | 25.68 ms | 167.3 |
| [1×8×2048×64] long sequence | 46.39 ms | 185.2 |

### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 117.16 ms | 63.1 |

packed operands: hits 0, misses 7, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.77 ms | 144.8K |
| forward + backward | 13.30 ms | 19.2K |
| forward + backward + Adam step | 10.50 ms | 24.4K |

packed operands: hits 953, misses 250, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries

### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 328884, misses 1614, retained 347 MiB, pinned 0 MiB; GC cycles 813, forced 778
