## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, backend amx, Go go1.26.2
### Matrix multiply (float32, GFLOPS; result released)
| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 358.8 | 368.7 |
| 256 | 749.6 | 916.4 |
| 512 | 1089.9 | 1509.4 |
| 1024 | 1150.1 | 2134.7 |
| 2048 | 1024.6 | 2269.1 |
| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 13.4 |
| [8×4096]·[4096×4096] | 74.9 |
| [64×1024]·[1024×1024] | 999.0 |
| [256×768]·[768×3072] | 1794.6 |
| [1024×1024]·[1024×1024]ᵀ (view) | 2094.7 |
### Element-wise (all threads)
| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 7.0 µs | 112.3 |
| x + y, result released | 64K | 4.8 µs | 163.6 |
| x * y | 64K | 7.1 µs | 110.9 |
| x + row (broadcast) | 64K | 8.0 µs | 65.5 |
| x * 2.5 | 64K | 6.2 µs | 84.6 |
| exp(x) | 64K | 11.2 µs | 46.8 |
| tanh(x) | 64K | 12.1 µs | 43.4 |
| sigmoid(x) | 64K | 16.4 µs | 32.0 |
| gelu(x) | 64K | 24.8 µs | 21.2 |
| relu(x) | 64K | 6.4 µs | 81.8 |
| x + y | 1M | 59.9 µs | 210.2 |
| x + y, result released | 1M | 43.3 µs | 290.4 |
| x * y | 1M | 57.9 µs | 217.5 |
| x + row (broadcast) | 1M | 75.5 µs | 111.2 |
| x * 2.5 | 1M | 55.3 µs | 151.7 |
| exp(x) | 1M | 111.2 µs | 75.4 |
| tanh(x) | 1M | 123.1 µs | 68.1 |
| sigmoid(x) | 1M | 168.8 µs | 49.7 |
| gelu(x) | 1M | 243.8 µs | 34.4 |
| relu(x) | 1M | 54.0 µs | 155.3 |
| x + y | 16M | 1.74 ms | 116.0 |
| x + y, result released | 16M | 1.51 ms | 133.5 |
| x * y | 16M | 1.74 ms | 115.7 |
| x + row (broadcast) | 16M | 1.56 ms | 86.0 |
| x * 2.5 | 16M | 1.12 ms | 120.3 |
| exp(x) | 16M | 1.60 ms | 83.7 |
| tanh(x) | 16M | 1.79 ms | 75.0 |
| sigmoid(x) | 16M | 2.57 ms | 52.1 |
| gelu(x) | 16M | 3.37 ms | 39.8 |
| relu(x) | 16M | 1.07 ms | 125.5 |
### Reductions (all threads)
| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 615.6 µs | 109.0 |
| sum(dim=0) | [4096×4096] | 744.3 µs | 90.2 |
| sum(dim=1) | [4096×4096] | 579.5 µs | 115.8 |
| max(dim=1) | [4096×4096] | 574.8 µs | 116.7 |
| softmax(dim=1) | [4096×4096] | 2.22 ms | 60.4 |
| layernorm | [4096×4096] | 1.85 ms | 72.4 |
| transpose+contiguous | [4096×4096] | 10.86 ms | 12.4 |
### Attention (all threads, NoGrad, result released)
| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 4.54 ms | 946.7 |
| same with causal mask | 4.88 ms | 879.7 |
### Convolution (all threads, NoGrad, result released)
| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 21.98 ms | 336.7 |
### MLP 784→512→512→10, batch 256 (all threads)
| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 337.6 µs | 758.3K |
| forward + backward | 1.43 ms | 179.2K |
| forward + backward + Adam step | 1.73 ms | 148.2K |
### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)
| shape | time / batch | sentences/s |
|---|---:|---:|
| 32 × 65 tokens, dim 768 | 364.91 ms | 88 |
allocator: mapped hits 1023400, misses 2318, retained 428 MiB, pinned 11 MiB; GC cycles 3541, forced 3471
