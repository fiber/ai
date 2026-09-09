## fiber/ai — linux/amd64, 6 CPUs, GOMAXPROCS 6, backend avx2, Go go1.26.2

KVM guest, 6 vCPU (AVX2, no AVX-512), 9 September 2026, commit 0ceb7eb.

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 25.2 | 52.4 |
| 256 | 49.6 | 139.2 |
| 512 | 41.9 | 179.8 |
| 1024 | 54.1 | 233.7 |
| 2048 | 55.8 | 269.6 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 7.8 | 8.1 |
| [8×4096]·[4096×4096] | 84.2 | 19.5 |
| [64×1024]·[1024×1024] | 197.2 | 117.8 |
| [256×768]·[768×3072] | 220.3 | 190.7 |
| [512×512]·[512×512] | 246.7 | 224.4 |
| [1024×1024]·[1024×1024] | 278.1 | 252.5 |
| [1024×1024]·[1024×1024]ᵀ (view) | 278.8 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 788.1 µs | 1.0 |
| x + y, result released | 64K | 14.9 µs | 52.9 |
| x * y | 64K | 42.4 µs | 18.5 |
| x + row (broadcast) | 64K | 29.7 µs | 17.6 |
| x * 2.5 | 64K | 27.7 µs | 18.9 |
| exp(x) | 64K | 33.6 µs | 15.6 |
| exp(x), result released | 64K | 20.5 µs | 25.6 |
| tanh(x) | 64K | 32.9 µs | 15.9 |
| tanh(x), result released | 64K | 25.7 µs | 20.4 |
| sigmoid(x) | 64K | 45.5 µs | 11.5 |
| gelu(x) | 64K | 56.4 µs | 9.3 |
| gelu(x), result released | 64K | 42.3 µs | 12.4 |
| relu(x) | 64K | 26.9 µs | 19.5 |
| x + y | 1M | 771.1 µs | 16.3 |
| x + y, result released | 1M | 163.1 µs | 77.1 |
| x * y | 1M | 360.0 µs | 35.0 |
| x + row (broadcast) | 1M | 314.0 µs | 26.7 |
| x * 2.5 | 1M | 325.2 µs | 25.8 |
| exp(x) | 1M | 387.7 µs | 21.6 |
| exp(x), result released | 1M | 252.6 µs | 33.2 |
| tanh(x) | 1M | 421.6 µs | 19.9 |
| tanh(x), result released | 1M | 396.6 µs | 21.2 |
| sigmoid(x) | 1M | 678.4 µs | 12.4 |
| gelu(x) | 1M | 909.4 µs | 9.2 |
| gelu(x), result released | 1M | 670.0 µs | 12.5 |
| relu(x) | 1M | 286.1 µs | 29.3 |
| x + y | 16M | 8.70 ms | 23.1 |
| x + y, result released | 16M | 6.14 ms | 32.8 |
| x * y | 16M | 8.16 ms | 24.7 |
| x + row (broadcast) | 16M | 6.26 ms | 21.4 |
| x * 2.5 | 16M | 6.18 ms | 21.7 |
| exp(x) | 16M | 6.35 ms | 21.1 |
| exp(x), result released | 16M | 5.07 ms | 26.5 |
| tanh(x) | 16M | 7.27 ms | 18.5 |
| tanh(x), result released | 16M | 6.37 ms | 21.1 |
| sigmoid(x) | 16M | 10.74 ms | 12.5 |
| gelu(x) | 16M | 13.41 ms | 10.0 |
| gelu(x), result released | 16M | 12.63 ms | 10.6 |
| relu(x) | 16M | 5.82 ms | 23.1 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 1.92 ms | 35.0 |
| sum(dim=0) | [4096×4096] | 2.10 ms | 32.0 |
| sum(dim=1) | [4096×4096] | 1.67 ms | 40.3 |
| max(dim=1) | [4096×4096] | 2.03 ms | 33.0 |
| softmax(dim=1) | [4096×4096] | 9.62 ms | 14.0 |
| layernorm | [4096×4096] | 10.96 ms | 12.2 |
| transpose+contiguous | [4096×4096] | 74.26 ms | 1.8 |

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 28.63 ms | 150.0 |
| same with causal mask | 24.07 ms | 178.5 |
| [1×8×2048×64] long sequence | 45.89 ms | 187.2 |

### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 101.55 ms | 72.9 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.84 ms | 138.8K |
| forward + backward | 9.24 ms | 27.7K |
| forward + backward + Adam step | 11.29 ms | 22.7K |

### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

| shape | time / batch | sentences/s |
|---|---:|---:|
| 32 × 65 tokens, dim 768 | 1.89 s | 17 |

allocator: mapped hits 325252, misses 1881, retained 428 MiB, pinned 11 MiB; GC cycles 734, forced 722
