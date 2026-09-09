## fiber/ai — darwin/arm64, 10 CPUs, GOMAXPROCS 10, backend amx, Go go1.27.0

Apple M4 MacBook Air (4P + 6E, fanless), 9 September 2026, commit 0ceb7eb.

### Matrix multiply (float32, GFLOPS; result released)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 459.5 | 559.3 |
| 256 | 945.0 | 1203.2 |
| 512 | 1349.0 | 1621.2 |
| 1024 | 1417.2 | 1685.0 |
| 2048 | 1302.3 | 1715.9 |

| shape | GFLOPS, B packed once (cache) | GFLOPS, B packed per call |
|---|---:|---:|
| [1×4096]·[4096×4096] | 24.7 | 24.8 |
| [8×4096]·[4096×4096] | 268.6 | 97.6 |
| [64×1024]·[1024×1024] | 1300.6 | 969.4 |
| [256×768]·[768×3072] | 1575.7 | 1350.3 |
| [512×512]·[512×512] | 1441.0 | 1407.9 |
| [1024×1024]·[1024×1024] | 1587.3 | 1550.2 |
| [1024×1024]·[1024×1024]ᵀ (view) | 1595.1 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 7.9 µs | 99.6 |
| x + y, result released | 64K | 4.3 µs | 181.1 |
| exp(x) | 64K | 8.2 µs | 63.6 |
| tanh(x) | 64K | 9.4 µs | 55.8 |
| relu(x) | 64K | 7.2 µs | 73.3 |
| x + y | 1M | 80.5 µs | 156.2 |
| x + y, result released | 1M | 51.1 µs | 246.2 |
| x * 2.5 | 1M | 71.2 µs | 117.8 |
| exp(x) | 1M | 101.2 µs | 82.9 |
| exp(x), result released | 1M | 86.6 µs | 96.9 |
| tanh(x) | 1M | 105.2 µs | 79.8 |
| tanh(x), result released | 1M | 94.1 µs | 89.1 |
| gelu(x), result released | 1M | 205.9 µs | 40.7 |
| relu(x) | 1M | 72.0 µs | 116.4 |
| x + y | 16M | 2.44 ms | 82.5 |
| x + y, result released | 16M | 2.42 ms | 83.4 |
| x * 2.5 | 16M | 1.91 ms | 70.3 |
| exp(x) | 16M | 1.91 ms | 70.2 |
| exp(x), result released | 16M | 1.86 ms | 72.1 |
| tanh(x) | 16M | 2.01 ms | 66.7 |
| relu(x) | 16M | 2.04 ms | 65.8 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 595.4 µs | 112.7 |
| sum(dim=0) | [4096×4096] | 793.3 µs | 84.6 |
| sum(dim=1) | [4096×4096] | 665.8 µs | 100.8 |
| max(dim=1) | [4096×4096] | 710.7 µs | 94.4 |
| softmax(dim=1) | [4096×4096] | 2.73 ms | 49.2 |
| layernorm | [4096×4096] | 2.64 ms | 50.9 |
| transpose+contiguous | [4096×4096] | 11.47 ms | 11.7 |

### Attention (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 4.67 ms | 920.5 |
| same with causal mask | 4.94 ms | 870.2 |
| [1×8×2048×64] long sequence | 9.60 ms | 895.1 |

### Convolution (all threads, NoGrad, result released)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 20.70 ms | 357.4 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 384.6 µs | 665.6K |
| forward + backward | 1.26 ms | 202.4K |
| forward + backward + Adam step | 1.57 ms | 163.3K |

### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)

| shape | time / batch | sentences/s |
|---|---:|---:|
| 32 × 65 tokens, dim 768 | 386.42 ms | 83 |
