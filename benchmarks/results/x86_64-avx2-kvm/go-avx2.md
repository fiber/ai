## fiber/ai — linux/amd64, 6 CPUs, backend avx2, Go go1.26.2

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 26.0 | 28.1 |
| 256 | 32.0 | 52.6 |
| 512 | 37.1 | 115.7 |
| 1024 | 36.7 | 147.5 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 8.7 |
| [8×4096]·[4096×4096] | 16.5 |
| [64×1024]·[1024×1024] | 60.6 |
| [256×768]·[768×3072] | 120.4 |
| [1024×1024]·[1024×1024]ᵀ (view) | 163.1 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 160.0 µs | 4.9 |
| x * y | 64K | 190.8 µs | 4.1 |
| x + row (broadcast) | 64K | 128.2 µs | 4.1 |
| x * 2.5 | 64K | 148.7 µs | 3.5 |
| exp(x) | 64K | 201.9 µs | 2.6 |
| tanh(x) | 64K | 891.4 µs | 0.6 |
| relu(x) | 64K | 161.7 µs | 3.2 |
| x + y | 1M | 1.75 ms | 7.2 |
| x * y | 1M | 1.83 ms | 6.9 |
| x + row (broadcast) | 1M | 1.55 ms | 5.4 |
| x * 2.5 | 1M | 2.17 ms | 3.9 |
| exp(x) | 1M | 2.85 ms | 2.9 |
| tanh(x) | 1M | 13.46 ms | 0.6 |
| relu(x) | 1M | 1.86 ms | 4.5 |
| x + y | 16M | 20.70 ms | 9.7 |
| x * y | 16M | 21.45 ms | 9.4 |
| x + row (broadcast) | 16M | 20.23 ms | 6.6 |
| x * 2.5 | 16M | 22.69 ms | 5.9 |
| exp(x) | 16M | 20.15 ms | 6.7 |
| tanh(x) | 16M | 151.75 ms | 0.9 |
| relu(x) | 16M | 19.97 ms | 6.7 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.29 ms | 29.3 |
| sum(dim=0) | [4096×4096] | 2.65 ms | 25.3 |
| sum(dim=1) | [4096×4096] | 2.18 ms | 30.8 |
| max(dim=1) | [4096×4096] | 2.38 ms | 28.2 |
| softmax(dim=1) | [4096×4096] | 27.89 ms | 4.8 |
| layernorm | [4096×4096] | 44.05 ms | 3.0 |
| transpose+contiguous | [4096×4096] | 91.61 ms | 1.5 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 5.81 ms | 44.0K |
| forward + backward | 17.88 ms | 14.3K |
| forward + backward + Adam step | 20.35 ms | 12.6K |

