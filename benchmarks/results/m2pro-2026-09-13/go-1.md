## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, workers 6, backend amx, Go go1.26.2

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 564.0 | 551.7 |
| 256 | 743.7 | 953.3 |
| 512 | 1094.6 | 1528.1 |
| 1024 | 1192.7 | 1930.4 |
| 2048 | 1045.9 | 2280.3 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 14.2 | 14.3 |
| [8×4096]·[4096×4096] | 296.9 | 74.3 |
| [64×1024]·[1024×1024] | 1622.2 | 971.8 |
| [256×768]·[768×3072] | 2508.9 | 1757.8 |
| [512×512]·[512×512] | 1698.9 | 1510.5 |
| [1024×1024]·[1024×1024] | 2490.1 | 2115.7 |
| [1024×1024]·[1024×1024]ᵀ (view) | 2511.2 |

packed operands: hits 22702, misses 14, packs 7, evictions 0, invalidations 0, refusals 0, held 4 MiB in 1 entries


allocator: mapped hits 340895, misses 25, retained 184 MiB, pinned 0 MiB; GC cycles 16, forced 0
### Small products (fresh operands, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 1.0 µs | 65.4 |
| 64² | 2.4 µs | 215.6 |
| 96² | 4.8 µs | 366.6 |
| 128² | 7.6 µs | 554.3 |
| 160² | 12.0 µs | 680.5 |
| 192² | 18.6 µs | 762.7 |
| 256² | 32.3 µs | 1037.3 |
| [64×24]·[24×16] | 1.9 µs | 26.0 |
| [64×16]·[16×3] | 1.5 µs | 4.0 |
| [256×24]·[24×16] | 5.5 µs | 36.0 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 20.8 µs | 3.08M |
| forward + backward + Adam step | 54.6 µs | 1.17M |

packed operands: hits 29454, misses 2, packs 1, evictions 0, invalidations 0, refusals 0, held 0 MiB in 1 entries


allocator: mapped hits 278898, misses 12, retained 1 MiB, pinned 0 MiB; GC cycles 241, forced 0
### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 5.9 µs | 133.2 |
| x + y, result released | 64K | 4.6 µs | 171.9 |
| x * y | 64K | 6.1 µs | 128.4 |
| x + row (broadcast) | 64K | 8.2 µs | 64.1 |
| x * 2.5 | 64K | 5.5 µs | 95.1 |
| exp(x) | 64K | 9.4 µs | 55.9 |
| exp(x), result released | 64K | 8.4 µs | 62.5 |
| tanh(x) | 64K | 9.6 µs | 54.8 |
| tanh(x), result released | 64K | 9.0 µs | 58.2 |
| sigmoid(x) | 64K | 15.7 µs | 33.4 |
| gelu(x) | 64K | 25.1 µs | 20.9 |
| gelu(x), result released | 64K | 19.8 µs | 26.5 |
| relu(x) | 64K | 5.4 µs | 96.6 |
| x + y | 1M | 48.0 µs | 262.1 |
| x + y, result released | 1M | 37.0 µs | 339.9 |
| x * y | 1M | 48.6 µs | 258.9 |
| x + row (broadcast) | 1M | 67.9 µs | 123.6 |
| x * 2.5 | 1M | 46.3 µs | 181.2 |
| exp(x) | 1M | 97.7 µs | 85.9 |
| exp(x), result released | 1M | 83.2 µs | 100.8 |
| tanh(x) | 1M | 101.4 µs | 82.8 |
| tanh(x), result released | 1M | 91.3 µs | 91.9 |
| sigmoid(x) | 1M | 156.5 µs | 53.6 |
| gelu(x) | 1M | 231.8 µs | 36.2 |
| gelu(x), result released | 1M | 217.6 µs | 38.6 |
| relu(x) | 1M | 46.3 µs | 181.3 |
| x + y | 16M | 1.57 ms | 127.9 |
| x + y, result released | 16M | 1.32 ms | 152.9 |
| x * y | 16M | 1.57 ms | 128.2 |
| x + row (broadcast) | 16M | 1.50 ms | 89.3 |
| x * 2.5 | 16M | 987.5 µs | 135.9 |
| exp(x) | 16M | 1.55 ms | 86.5 |
| exp(x), result released | 16M | 1.37 ms | 97.6 |
| tanh(x) | 16M | 1.64 ms | 82.0 |
| tanh(x), result released | 16M | 1.50 ms | 89.7 |
| sigmoid(x) | 16M | 2.55 ms | 52.7 |
| gelu(x) | 16M | 3.91 ms | 34.3 |
| gelu(x), result released | 16M | 3.63 ms | 37.0 |
| relu(x) | 16M | 993.6 µs | 135.1 |


allocator: mapped hits 1698464, misses 1094, retained 448 MiB, pinned 0 MiB; GC cycles 5103, forced 5101
### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 553.5 µs | 121.3 |
| sum(dim=0) | [4096×4096] | 615.8 µs | 109.0 |
| sum(dim=1) | [4096×4096] | 535.2 µs | 125.4 |
| max(dim=1) | [4096×4096] | 534.4 µs | 125.6 |
| softmax(dim=1) | [4096×4096] | 2.34 ms | 57.3 |
| layernorm | [4096×4096] | 1.89 ms | 71.1 |
| transpose+contiguous | [4096×4096] | 12.43 ms | 10.8 |


allocator: mapped hits 950, misses 6, retained 0 MiB, pinned 0 MiB; GC cycles 370, forced 364
### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 4.35 ms | 987.1 |
| same with causal mask | 4.67 ms | 919.5 |
| [1×8×2048×64] long sequence | 9.15 ms | 938.9 |


allocator: mapped hits 520, misses 9, retained 37 MiB, pinned 0 MiB; GC cycles 1, forced 0
### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 20.08 ms | 368.5 |

packed operands: hits 0, misses 45, packs 0, evictions 0, invalidations 0, refusals 0, held 0 MiB in 0 entries


allocator: mapped hits 125, misses 12, retained 336 MiB, pinned 0 MiB; GC cycles 6, forced 5
### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 306.7 µs | 834.8K |
| forward + backward | 1.19 ms | 215.4K |
| forward + backward + Adam step | 1.40 ms | 182.3K |

packed operands: hits 8625, misses 2826, packs 3, evictions 0, invalidations 0, refusals 0, held 3 MiB in 3 entries


allocator: mapped hits 31410, misses 353, retained 109 MiB, pinned 0 MiB; GC cycles 34, forced 33
### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m


allocator: mapped hits 0, misses 0, retained 0 MiB, pinned 0 MiB; GC cycles 1, forced 0
