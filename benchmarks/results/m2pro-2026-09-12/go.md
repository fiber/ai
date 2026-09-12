## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, workers 6, backend amx, Go go1.26.2

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 541.5 | 552.7 |
| 256 | 747.4 | 974.5 |
| 512 | 1094.0 | 1504.9 |
| 1024 | 1182.4 | 2118.0 |
| 2048 | 1035.9 | 2225.2 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 13.9 | 14.2 |
| [8×4096]·[4096×4096] | 272.0 | 71.7 |
| [64×1024]·[1024×1024] | 1620.6 | 949.5 |
| [256×768]·[768×3072] | 2506.9 | 1704.5 |
| [512×512]·[512×512] | 1732.2 | 1540.8 |
| [1024×1024]·[1024×1024] | 2465.2 | 2156.5 |
| [1024×1024]·[1024×1024]ᵀ (view) | 2469.3 |

packed operands: hits 17031, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries

### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 1.0 µs | 62.5 |
| 64² | 2.4 µs | 216.3 |
| 96² | 4.8 µs | 370.9 |
| 128² | 7.5 µs | 557.9 |
| 160² | 12.2 µs | 671.2 |
| 192² | 19.1 µs | 742.8 |
| 256² | 30.4 µs | 1105.4 |
| [64×24]·[24×16] | 2.0 µs | 25.0 |
| [64×16]·[16×3] | 1.5 µs | 4.0 |
| [256×24]·[24×16] | 5.4 µs | 36.3 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 20.8 µs | 3.08M |
| forward + backward + Adam step | 55.1 µs | 1.16M |

packed operands: hits 23060, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 6.2 µs | 127.4 |
| x + y, result released | 64K | 4.6 µs | 171.9 |
| x * y | 64K | 6.6 µs | 118.5 |
| x + row (broadcast) | 64K | 8.4 µs | 62.2 |
| x * 2.5 | 64K | 5.7 µs | 91.3 |
| exp(x) | 64K | 9.6 µs | 54.8 |
| exp(x), result released | 64K | 8.5 µs | 61.8 |
| tanh(x) | 64K | 9.8 µs | 53.3 |
| tanh(x), result released | 64K | 9.0 µs | 58.2 |
| sigmoid(x) | 64K | 16.1 µs | 32.6 |
| gelu(x) | 64K | 25.6 µs | 20.5 |
| gelu(x), result released | 64K | 20.6 µs | 25.5 |
| relu(x) | 64K | 5.5 µs | 95.4 |
| x + y | 1M | 51.4 µs | 244.6 |
| x + y, result released | 1M | 37.2 µs | 337.8 |
| x * y | 1M | 53.3 µs | 236.3 |
| x + row (broadcast) | 1M | 74.7 µs | 112.4 |
| x * 2.5 | 1M | 47.5 µs | 176.6 |
| exp(x) | 1M | 97.9 µs | 85.7 |
| exp(x), result released | 1M | 86.2 µs | 97.3 |
| tanh(x) | 1M | 107.1 µs | 78.3 |
| tanh(x), result released | 1M | 90.0 µs | 93.2 |
| sigmoid(x) | 1M | 162.6 µs | 51.6 |
| gelu(x) | 1M | 233.2 µs | 36.0 |
| gelu(x), result released | 1M | 223.3 µs | 37.6 |
| relu(x) | 1M | 48.0 µs | 174.6 |
| x + y | 16M | 1.62 ms | 123.9 |
| x + y, result released | 16M | 1.37 ms | 147.2 |
| x * y | 16M | 1.60 ms | 125.8 |
| x + row (broadcast) | 16M | 1.52 ms | 88.4 |
| x * 2.5 | 16M | 1.05 ms | 127.7 |
| exp(x) | 16M | 1.60 ms | 83.8 |
| exp(x), result released | 16M | 1.42 ms | 94.6 |
| tanh(x) | 16M | 1.67 ms | 80.5 |
| tanh(x), result released | 16M | 1.50 ms | 89.6 |
| sigmoid(x) | 16M | 2.61 ms | 51.4 |
| gelu(x) | 16M | 3.98 ms | 33.8 |
| gelu(x), result released | 16M | 3.70 ms | 36.2 |
| relu(x) | 16M | 1.01 ms | 133.0 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 583.6 µs | 115.0 |
| sum(dim=0) | [4096×4096] | 638.3 µs | 105.1 |
| sum(dim=1) | [4096×4096] | 543.3 µs | 123.5 |
| max(dim=1) | [4096×4096] | 537.0 µs | 125.0 |
| softmax(dim=1) | [4096×4096] | 2.32 ms | 57.8 |
| layernorm | [4096×4096] | 1.81 ms | 74.0 |
| transpose+contiguous | [4096×4096] | 12.60 ms | 10.6 |

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 4.33 ms | 991.2 |
| same with causal mask | 5.16 ms | 832.0 |
| [1×8×2048×64] long sequence | 9.02 ms | 952.5 |

### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 40.13 ms | 184.4 |

packed operands: hits 0, misses 19, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 309.5 µs | 827.0K |
| forward + backward | 1.25 ms | 205.4K |
| forward + backward + Adam step | 1.44 ms | 177.9K |

packed operands: hits 6212, misses 2108, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries

### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 1731092, misses 1518, retained 404 MiB, pinned 0 MiB; GC cycles 4071, forced 3926
