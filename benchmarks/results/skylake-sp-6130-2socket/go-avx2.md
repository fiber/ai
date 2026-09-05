## fiber/ai — linux/amd64, 64 CPUs, backend avx2, Go go1.27.1

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 18.3 | 17.8 |
| 256 | 27.4 | 44.0 |
| 512 | 55.7 | 61.7 |
| 1024 | 68.1 | 191.0 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 12.0 |
| [8×4096]·[4096×4096] | 11.3 |
| [64×1024]·[1024×1024] | 27.0 |
| [256×768]·[768×3072] | 93.0 |
| [1024×1024]·[1024×1024]ᵀ (view) | 140.3 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 216.7 µs | 3.6 |
| x * y | 64K | 269.4 µs | 2.9 |
| x + row (broadcast) | 64K | 260.5 µs | 2.0 |
| x * 2.5 | 64K | 263.0 µs | 2.0 |
| exp(x) | 64K | 291.6 µs | 1.8 |
| tanh(x) | 64K | 845.8 µs | 0.6 |
| relu(x) | 64K | 250.2 µs | 2.1 |
| x + y | 1M | 2.05 ms | 6.1 |
| x * y | 1M | 2.15 ms | 5.8 |
| x + row (broadcast) | 1M | 1.89 ms | 4.4 |
| x * 2.5 | 1M | 1.94 ms | 4.3 |
| exp(x) | 1M | 2.13 ms | 3.9 |
| tanh(x) | 1M | 3.60 ms | 2.3 |
| relu(x) | 1M | 1.76 ms | 4.8 |
| x + y | 16M | 29.75 ms | 6.8 |
| x * y | 16M | 29.81 ms | 6.8 |
| x + row (broadcast) | 16M | 24.39 ms | 5.5 |
| x * 2.5 | 16M | 24.72 ms | 5.4 |
| exp(x) | 16M | 27.59 ms | 4.9 |
| tanh(x) | 16M | 33.93 ms | 4.0 |
| relu(x) | 16M | 23.93 ms | 5.6 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 1.57 ms | 42.8 |
| sum(dim=0) | [4096×4096] | 2.77 ms | 24.2 |
| sum(dim=1) | [4096×4096] | 1.60 ms | 41.9 |
| max(dim=1) | [4096×4096] | 1.59 ms | 42.1 |
| softmax(dim=1) | [4096×4096] | 25.95 ms | 5.2 |
| layernorm | [4096×4096] | 44.22 ms | 3.0 |
| transpose+contiguous | [4096×4096] | 33.91 ms | 4.0 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 8.27 ms | 31.0K |
| forward + backward | 24.62 ms | 10.4K |
| forward + backward + Adam step | 28.17 ms | 9.1K |

