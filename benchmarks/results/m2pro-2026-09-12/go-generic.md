## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, workers 6, backend generic, Go go1.26.2

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 7.7 | 7.7 |
| 256 | 7.8 | 40.6 |
| 512 | 7.8 | 43.7 |
| 1024 | 7.8 | 43.9 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 12.0 | 11.5 |
| [8×4096]·[4096×4096] | 41.2 | 23.4 |
| [64×1024]·[1024×1024] | 41.5 | 41.2 |
| [256×768]·[768×3072] | 44.2 | 43.2 |
| [512×512]·[512×512] | 43.0 | 43.6 |
| [1024×1024]·[1024×1024] | 44.5 | 44.2 |
| [1024×1024]·[1024×1024]ᵀ (view) | 43.2 |

packed operands: hits 743, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries

### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 10.0 µs | 6.5 |
| 64² | 70.2 µs | 7.5 |
| 96² | 231.8 µs | 7.6 |
| 128² | 547.8 µs | 7.7 |
| 160² | 1.06 ms | 7.8 |
| 192² | 360.8 µs | 39.2 |
| 256² | 858.5 µs | 39.1 |
| [64×24]·[24×16] | 7.8 µs | 6.3 |
| [64×16]·[16×3] | 3.3 µs | 1.9 |
| [256×24]·[24×16] | 29.1 µs | 6.7 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 38.9 µs | 1.64M |
| forward + backward + Adam step | 107.7 µs | 594.1K |

packed operands: hits 815, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 9.3 µs | 85.0 |
| x + y, result released | 64K | 7.6 µs | 103.2 |
| x * y | 64K | 9.2 µs | 85.5 |
| x + row (broadcast) | 64K | 11.8 µs | 44.2 |
| x * 2.5 | 64K | 8.5 µs | 61.4 |
| exp(x) | 64K | 36.6 µs | 14.3 |
| exp(x), result released | 64K | 32.3 µs | 16.2 |
| tanh(x) | 64K | 75.1 µs | 7.0 |
| tanh(x), result released | 64K | 78.1 µs | 6.7 |
| sigmoid(x) | 64K | 105.9 µs | 5.0 |
| gelu(x) | 64K | 116.4 µs | 4.5 |
| gelu(x), result released | 64K | 117.4 µs | 4.5 |
| relu(x) | 64K | 62.6 µs | 8.4 |
| x + y | 1M | 79.3 µs | 158.7 |
| x + y, result released | 1M | 66.1 µs | 190.4 |
| x * y | 1M | 74.7 µs | 168.5 |
| x + row (broadcast) | 1M | 100.4 µs | 83.5 |
| x * 2.5 | 1M | 70.7 µs | 118.6 |
| exp(x) | 1M | 443.6 µs | 18.9 |
| exp(x), result released | 1M | 435.7 µs | 19.3 |
| tanh(x) | 1M | 1.19 ms | 7.0 |
| tanh(x), result released | 1M | 1.16 ms | 7.3 |
| sigmoid(x) | 1M | 1.45 ms | 5.8 |
| gelu(x) | 1M | 1.65 ms | 5.1 |
| gelu(x), result released | 1M | 1.63 ms | 5.1 |
| relu(x) | 1M | 825.8 µs | 10.2 |
| x + y | 16M | 1.61 ms | 125.0 |
| x + y, result released | 16M | 1.39 ms | 144.6 |
| x * y | 16M | 1.58 ms | 127.8 |
| x + row (broadcast) | 16M | 1.71 ms | 78.7 |
| x * 2.5 | 16M | 1.19 ms | 113.3 |
| exp(x) | 16M | 7.16 ms | 18.8 |
| exp(x), result released | 16M | 7.13 ms | 18.8 |
| tanh(x) | 16M | 19.14 ms | 7.0 |
| tanh(x), result released | 16M | 18.89 ms | 7.1 |
| sigmoid(x) | 16M | 22.22 ms | 6.0 |
| gelu(x) | 16M | 26.27 ms | 5.1 |
| gelu(x), result released | 16M | 26.52 ms | 5.1 |
| relu(x) | 16M | 13.71 ms | 9.8 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 742.4 µs | 90.4 |
| sum(dim=0) | [4096×4096] | 1.18 ms | 56.7 |
| sum(dim=1) | [4096×4096] | 732.8 µs | 91.6 |
| max(dim=1) | [4096×4096] | 3.65 ms | 18.4 |
| softmax(dim=1) | [4096×4096] | 15.49 ms | 8.7 |
| layernorm | [4096×4096] | 5.90 ms | 22.7 |
| transpose+contiguous | [4096×4096] | 12.78 ms | 10.5 |

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 114.06 ms | 37.7 |
| same with causal mask | 115.62 ms | 37.1 |
| [1×8×2048×64] long sequence | 226.01 ms | 38.0 |

### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 204.91 ms | 36.1 |

packed operands: hits 0, misses 5, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 9.18 ms | 27.9K |
| forward + backward | 22.06 ms | 11.6K |
| forward + backward + Adam step | 22.77 ms | 11.2K |

packed operands: hits 252, misses 136, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries

### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 538031, misses 1590, retained 413 MiB, pinned 0 MiB; GC cycles 1985, forced 1954
