## fiber/ai — linux/amd64, 64 CPUs, backend avx512, Go go1.27.1

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 22.8 | 23.0 |
| 256 | 38.0 | 41.9 |
| 512 | 75.4 | 80.0 |
| 1024 | 112.5 | 161.9 |
| 2048 | 119.5 | 510.8 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 8.0 |
| [8×4096]·[4096×4096] | 19.7 |
| [64×1024]·[1024×1024] | 84.4 |
| [256×768]·[768×3072] | 92.7 |
| [1024×1024]·[1024×1024]ᵀ (view) | 220.8 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 226.6 µs | 3.5 |
| x * y | 64K | 224.5 µs | 3.5 |
| x + row (broadcast) | 64K | 195.3 µs | 2.7 |
| x * 2.5 | 64K | 211.4 µs | 2.5 |
| exp(x) | 64K | 303.0 µs | 1.7 |
| tanh(x) | 64K | 892.8 µs | 0.6 |
| relu(x) | 64K | 194.4 µs | 2.7 |
| x + y | 1M | 2.17 ms | 5.8 |
| x * y | 1M | 2.11 ms | 6.0 |
| x + row (broadcast) | 1M | 2.01 ms | 4.2 |
| x * 2.5 | 1M | 2.06 ms | 4.1 |
| exp(x) | 1M | 2.30 ms | 3.7 |
| tanh(x) | 1M | 4.83 ms | 1.7 |
| relu(x) | 1M | 2.06 ms | 4.1 |
| x + y | 16M | 25.36 ms | 7.9 |
| x * y | 16M | 26.56 ms | 7.6 |
| x + row (broadcast) | 16M | 25.42 ms | 5.3 |
| x * 2.5 | 16M | 24.59 ms | 5.5 |
| exp(x) | 16M | 25.27 ms | 5.3 |
| tanh(x) | 16M | 33.00 ms | 4.1 |
| relu(x) | 16M | 25.42 ms | 5.3 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.98 ms | 22.5 |
| sum(dim=0) | [4096×4096] | 4.21 ms | 15.9 |
| sum(dim=1) | [4096×4096] | 2.99 ms | 22.4 |
| max(dim=1) | [4096×4096] | 2.93 ms | 22.9 |
| softmax(dim=1) | [4096×4096] | 27.22 ms | 4.9 |
| layernorm | [4096×4096] | 48.15 ms | 2.8 |
| transpose+contiguous | [4096×4096] | 34.02 ms | 3.9 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 7.80 ms | 32.8K |
| forward + backward | 25.63 ms | 10.0K |
| forward + backward + Adam step | 24.81 ms | 10.3K |

