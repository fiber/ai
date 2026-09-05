## fiber/ai — linux/amd64, 6 CPUs, backend generic, Go go1.26.2

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 1.7 | 1.8 |
| 256 | 1.9 | 7.2 |
| 512 | 2.0 | 7.9 |
| 1024 | 2.1 | 9.5 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 3.8 |
| [8×4096]·[4096×4096] | 5.0 |
| [64×1024]·[1024×1024] | 7.3 |
| [256×768]·[768×3072] | 8.4 |
| [1024×1024]·[1024×1024]ᵀ (view) | 9.6 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 149.4 µs | 5.3 |
| x * y | 64K | 153.5 µs | 5.1 |
| x + row (broadcast) | 64K | 156.2 µs | 3.4 |
| x * 2.5 | 64K | 152.3 µs | 3.4 |
| exp(x) | 64K | 476.6 µs | 1.1 |
| tanh(x) | 64K | 846.5 µs | 0.6 |
| relu(x) | 64K | 685.1 µs | 0.8 |
| x + y | 1M | 1.64 ms | 7.7 |
| x * y | 1M | 1.64 ms | 7.7 |
| x + row (broadcast) | 1M | 1.49 ms | 5.6 |
| x * 2.5 | 1M | 1.55 ms | 5.4 |
| exp(x) | 1M | 5.10 ms | 1.6 |
| tanh(x) | 1M | 10.17 ms | 0.8 |
| relu(x) | 1M | 3.06 ms | 2.7 |
| x + y | 16M | 23.38 ms | 8.6 |
| x * y | 16M | 23.28 ms | 8.6 |
| x + row (broadcast) | 16M | 20.91 ms | 6.4 |
| x * 2.5 | 16M | 20.20 ms | 6.6 |
| exp(x) | 16M | 70.93 ms | 1.9 |
| tanh(x) | 16M | 155.13 ms | 0.9 |
| relu(x) | 16M | 39.11 ms | 3.4 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.75 ms | 24.4 |
| sum(dim=0) | [4096×4096] | 3.89 ms | 17.2 |
| sum(dim=1) | [4096×4096] | 2.75 ms | 24.4 |
| max(dim=1) | [4096×4096] | 4.44 ms | 15.1 |
| softmax(dim=1) | [4096×4096] | 87.04 ms | 1.5 |
| layernorm | [4096×4096] | 53.45 ms | 2.5 |
| transpose+contiguous | [4096×4096] | 88.45 ms | 1.5 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 43.96 ms | 5.8K |
| forward + backward | 106.72 ms | 2.4K |
| forward + backward + Adam step | 112.86 ms | 2.3K |

