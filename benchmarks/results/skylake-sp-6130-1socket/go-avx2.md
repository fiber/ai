## fiber/ai — linux/amd64, 32 CPUs, backend avx2, Go go1.27.1

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 21.8 | 23.3 |
| 256 | 31.0 | 82.0 |
| 512 | 65.9 | 240.6 |
| 1024 | 73.8 | 439.4 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 13.0 |
| [8×4096]·[4096×4096] | 36.2 |
| [64×1024]·[1024×1024] | 99.8 |
| [256×768]·[768×3072] | 344.0 |
| [1024×1024]·[1024×1024]ᵀ (view) | 478.5 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 204.2 µs | 3.9 |
| x * y | 64K | 213.3 µs | 3.7 |
| x + row (broadcast) | 64K | 133.5 µs | 3.9 |
| x * 2.5 | 64K | 200.9 µs | 2.6 |
| exp(x) | 64K | 217.4 µs | 2.4 |
| tanh(x) | 64K | 403.4 µs | 1.3 |
| relu(x) | 64K | 171.8 µs | 3.1 |
| x + y | 1M | 1.45 ms | 8.7 |
| x * y | 1M | 1.47 ms | 8.6 |
| x + row (broadcast) | 1M | 1.22 ms | 6.9 |
| x * 2.5 | 1M | 1.19 ms | 7.1 |
| exp(x) | 1M | 1.44 ms | 5.8 |
| tanh(x) | 1M | 3.92 ms | 2.1 |
| relu(x) | 1M | 1.33 ms | 6.3 |
| x + y | 16M | 22.81 ms | 8.8 |
| x * y | 16M | 23.83 ms | 8.5 |
| x + row (broadcast) | 16M | 19.84 ms | 6.8 |
| x * 2.5 | 16M | 19.82 ms | 6.8 |
| exp(x) | 16M | 19.77 ms | 6.8 |
| tanh(x) | 16M | 56.94 ms | 2.4 |
| relu(x) | 16M | 19.74 ms | 6.8 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.39 ms | 28.1 |
| sum(dim=0) | [4096×4096] | 2.68 ms | 25.0 |
| sum(dim=1) | [4096×4096] | 2.42 ms | 27.7 |
| max(dim=1) | [4096×4096] | 2.39 ms | 28.0 |
| softmax(dim=1) | [4096×4096] | 20.33 ms | 6.6 |
| layernorm | [4096×4096] | 37.15 ms | 3.6 |
| transpose+contiguous | [4096×4096] | 25.07 ms | 5.4 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 3.21 ms | 79.7K |
| forward + backward | 16.40 ms | 15.6K |
| forward + backward + Adam step | 18.05 ms | 14.2K |

