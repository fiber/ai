## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, workers 6, backend amx, Go go1.26.2

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 555.2 | 544.9 |
| 256 | 737.8 | 948.9 |
| 512 | 1080.1 | 1550.6 |
| 1024 | 1177.7 | 2154.1 |
| 2048 | 1028.8 | 2264.2 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 14.3 | 13.9 |
| [8×4096]·[4096×4096] | 305.1 | 73.6 |
| [64×1024]·[1024×1024] | 1593.5 | 971.0 |
| [256×768]·[768×3072] | 2509.3 | 1716.1 |
| [512×512]·[512×512] | 1710.3 | 1555.3 |
| [1024×1024]·[1024×1024] | 2459.4 | 2163.6 |
| [1024×1024]·[1024×1024]ᵀ (view) | 2523.0 |

packed operands: hits 22609, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries


allocator: mapped hits 339818, misses 25, retained 120 MiB, pinned 0 MiB; GC cycles 16, forced 0
### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 1.0 µs | 62.4 |
| 64² | 2.5 µs | 208.4 |
| 96² | 5.1 µs | 344.5 |
| 128² | 7.7 µs | 546.2 |
| 160² | 12.0 µs | 681.6 |
| 192² | 18.7 µs | 757.0 |
| 256² | 29.3 µs | 1146.8 |
| [64×24]·[24×16] | 2.0 µs | 25.0 |
| [64×16]·[16×3] | 1.6 µs | 3.9 |
| [256×24]·[24×16] | 5.6 µs | 34.9 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 21.1 µs | 3.03M |
| forward + backward + Adam step | 55.9 µs | 1.14M |

packed operands: hits 31775, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries


allocator: mapped hits 280742, misses 12, retained 1 MiB, pinned 0 MiB; GC cycles 231, forced 0
### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 6.0 µs | 130.0 |
| x + y, result released | 64K | 4.5 µs | 175.7 |
| x * y | 64K | 6.1 µs | 129.8 |
| x + row (broadcast) | 64K | 8.2 µs | 63.8 |
| x * 2.5 | 64K | 5.5 µs | 95.0 |
| exp(x) | 64K | 9.3 µs | 56.6 |
| exp(x), result released | 64K | 8.5 µs | 61.9 |
| tanh(x) | 64K | 9.9 µs | 53.1 |
| tanh(x), result released | 64K | 8.9 µs | 59.0 |
| sigmoid(x) | 64K | 15.6 µs | 33.7 |
| gelu(x) | 64K | 24.8 µs | 21.2 |
| gelu(x), result released | 64K | 19.8 µs | 26.5 |
| relu(x) | 64K | 5.6 µs | 94.5 |
| x + y | 1M | 47.6 µs | 264.2 |
| x + y, result released | 1M | 37.4 µs | 336.6 |
| x * y | 1M | 48.7 µs | 258.2 |
| x + row (broadcast) | 1M | 66.8 µs | 125.6 |
| x * 2.5 | 1M | 46.1 µs | 181.9 |
| exp(x) | 1M | 97.5 µs | 86.1 |
| exp(x), result released | 1M | 81.9 µs | 102.4 |
| tanh(x) | 1M | 101.8 µs | 82.4 |
| tanh(x), result released | 1M | 92.7 µs | 90.5 |
| sigmoid(x) | 1M | 185.0 µs | 45.3 |
| gelu(x) | 1M | 236.2 µs | 35.5 |
| gelu(x), result released | 1M | 220.6 µs | 38.0 |
| relu(x) | 1M | 46.6 µs | 180.1 |
| x + y | 16M | 1.56 ms | 129.4 |
| x + y, result released | 16M | 1.33 ms | 151.6 |
| x * y | 16M | 1.52 ms | 132.2 |
| x + row (broadcast) | 16M | 1.47 ms | 91.2 |
| x * 2.5 | 16M | 964.1 µs | 139.2 |
| exp(x) | 16M | 1.53 ms | 87.7 |
| exp(x), result released | 16M | 1.40 ms | 95.5 |
| tanh(x) | 16M | 1.60 ms | 83.8 |
| tanh(x), result released | 16M | 1.52 ms | 88.5 |
| sigmoid(x) | 16M | 2.54 ms | 52.9 |
| gelu(x) | 16M | 3.67 ms | 36.6 |
| gelu(x), result released | 16M | 3.67 ms | 36.6 |
| relu(x) | 16M | 973.6 µs | 137.9 |


allocator: mapped hits 1700424, misses 1094, retained 384 MiB, pinned 0 MiB; GC cycles 5153, forced 5151
### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 550.8 µs | 121.8 |
| sum(dim=0) | [4096×4096] | 615.9 µs | 109.0 |
| sum(dim=1) | [4096×4096] | 536.7 µs | 125.0 |
| max(dim=1) | [4096×4096] | 533.9 µs | 125.7 |
| softmax(dim=1) | [4096×4096] | 2.31 ms | 58.2 |
| layernorm | [4096×4096] | 1.79 ms | 75.1 |
| transpose+contiguous | [4096×4096] | 12.40 ms | 10.8 |


allocator: mapped hits 979, misses 6, retained 64 MiB, pinned 0 MiB; GC cycles 358, forced 352
### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 4.35 ms | 988.3 |
| same with causal mask | 4.66 ms | 922.4 |
| [1×8×2048×64] long sequence | 8.87 ms | 968.1 |


allocator: mapped hits 521, misses 9, retained 37 MiB, pinned 0 MiB; GC cycles 1, forced 0
### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 20.26 ms | 365.3 |

packed operands: hits 0, misses 45, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries


allocator: mapped hits 125, misses 12, retained 336 MiB, pinned 0 MiB; GC cycles 6, forced 5
### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 303.7 µs | 842.9K |
| forward + backward | 1.21 ms | 212.2K |
| forward + backward + Adam step | 1.42 ms | 180.1K |

packed operands: hits 8544, misses 2820, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 31278, misses 353, retained 126 MiB, pinned 0 MiB; GC cycles 34, forced 33
### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 0, misses 0, retained 0 MiB, pinned 0 MiB; GC cycles 1, forced 0
