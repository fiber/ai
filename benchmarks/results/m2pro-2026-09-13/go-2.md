## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, workers 6, backend amx, Go go1.26.2

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 554.8 | 548.3 |
| 256 | 728.4 | 972.8 |
| 512 | 1092.1 | 1545.3 |
| 1024 | 1190.8 | 2139.6 |
| 2048 | 1035.6 | 2260.1 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 14.1 | 14.0 |
| [8×4096]·[4096×4096] | 294.9 | 73.4 |
| [64×1024]·[1024×1024] | 1584.6 | 974.1 |
| [256×768]·[768×3072] | 2544.9 | 1711.4 |
| [512×512]·[512×512] | 1746.6 | 1563.5 |
| [1024×1024]·[1024×1024] | 2457.4 | 2135.2 |
| [1024×1024]·[1024×1024]ᵀ (view) | 2474.7 |

packed operands: hits 22478, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries


allocator: mapped hits 341061, misses 25, retained 120 MiB, pinned 0 MiB; GC cycles 16, forced 0
### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 1.0 µs | 62.8 |
| 64² | 2.5 µs | 209.1 |
| 96² | 5.0 µs | 352.8 |
| 128² | 7.5 µs | 556.7 |
| 160² | 12.1 µs | 677.3 |
| 192² | 19.0 µs | 746.8 |
| 256² | 31.0 µs | 1083.0 |
| [64×24]·[24×16] | 2.0 µs | 25.0 |
| [64×16]·[16×3] | 1.6 µs | 4.0 |
| [256×24]·[24×16] | 5.5 µs | 35.4 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 21.2 µs | 3.02M |
| forward + backward + Adam step | 55.5 µs | 1.15M |

packed operands: hits 30391, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries


allocator: mapped hits 278900, misses 12, retained 1 MiB, pinned 0 MiB; GC cycles 233, forced 0
### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 5.8 µs | 135.2 |
| x + y, result released | 64K | 4.6 µs | 172.7 |
| x * y | 64K | 6.0 µs | 130.2 |
| x + row (broadcast) | 64K | 8.3 µs | 62.9 |
| x * 2.5 | 64K | 5.6 µs | 94.0 |
| exp(x) | 64K | 9.2 µs | 56.9 |
| exp(x), result released | 64K | 8.4 µs | 62.2 |
| tanh(x) | 64K | 9.6 µs | 54.4 |
| tanh(x), result released | 64K | 8.9 µs | 58.7 |
| sigmoid(x) | 64K | 15.9 µs | 33.0 |
| gelu(x) | 64K | 24.9 µs | 21.1 |
| gelu(x), result released | 64K | 20.3 µs | 25.9 |
| relu(x) | 64K | 5.4 µs | 96.6 |
| x + y | 1M | 48.1 µs | 261.4 |
| x + y, result released | 1M | 37.9 µs | 332.0 |
| x * y | 1M | 47.5 µs | 264.7 |
| x + row (broadcast) | 1M | 67.8 µs | 123.7 |
| x * 2.5 | 1M | 45.9 µs | 182.6 |
| exp(x) | 1M | 95.0 µs | 88.3 |
| exp(x), result released | 1M | 96.1 µs | 87.3 |
| tanh(x) | 1M | 102.4 µs | 81.9 |
| tanh(x), result released | 1M | 92.1 µs | 91.1 |
| sigmoid(x) | 1M | 168.3 µs | 49.8 |
| gelu(x) | 1M | 238.1 µs | 35.2 |
| gelu(x), result released | 1M | 215.1 µs | 39.0 |
| relu(x) | 1M | 46.1 µs | 182.0 |
| x + y | 16M | 1.61 ms | 125.4 |
| x + y, result released | 16M | 1.33 ms | 151.8 |
| x * y | 16M | 1.62 ms | 124.2 |
| x + row (broadcast) | 16M | 1.53 ms | 87.8 |
| x * 2.5 | 16M | 1.01 ms | 132.5 |
| exp(x) | 16M | 1.53 ms | 87.9 |
| exp(x), result released | 16M | 1.36 ms | 98.4 |
| tanh(x) | 16M | 1.60 ms | 83.7 |
| tanh(x), result released | 16M | 1.53 ms | 87.5 |
| sigmoid(x) | 16M | 2.72 ms | 49.3 |
| gelu(x) | 16M | 3.97 ms | 33.8 |
| gelu(x), result released | 16M | 3.79 ms | 35.4 |
| relu(x) | 16M | 1.01 ms | 132.8 |


allocator: mapped hits 1695080, misses 1094, retained 320 MiB, pinned 0 MiB; GC cycles 5078, forced 5076
### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 546.8 µs | 122.7 |
| sum(dim=0) | [4096×4096] | 610.0 µs | 110.0 |
| sum(dim=1) | [4096×4096] | 534.6 µs | 125.5 |
| max(dim=1) | [4096×4096] | 538.6 µs | 124.6 |
| softmax(dim=1) | [4096×4096] | 2.27 ms | 59.1 |
| layernorm | [4096×4096] | 1.86 ms | 72.1 |
| transpose+contiguous | [4096×4096] | 12.59 ms | 10.7 |


allocator: mapped hits 960, misses 6, retained 128 MiB, pinned 0 MiB; GC cycles 365, forced 359
### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 4.34 ms | 989.4 |
| same with causal mask | 4.63 ms | 928.3 |
| [1×8×2048×64] long sequence | 9.01 ms | 953.4 |


allocator: mapped hits 523, misses 9, retained 37 MiB, pinned 0 MiB; GC cycles 1, forced 0
### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 20.26 ms | 365.2 |

packed operands: hits 0, misses 45, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries


allocator: mapped hits 125, misses 12, retained 336 MiB, pinned 0 MiB; GC cycles 6, forced 5
### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 300.0 µs | 853.3K |
| forward + backward | 1.18 ms | 216.1K |
| forward + backward + Adam step | 1.43 ms | 179.1K |

packed operands: hits 8520, misses 2822, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 31254, misses 353, retained 120 MiB, pinned 0 MiB; GC cycles 34, forced 33
### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 0, misses 0, retained 0 MiB, pinned 0 MiB; GC cycles 1, forced 0
