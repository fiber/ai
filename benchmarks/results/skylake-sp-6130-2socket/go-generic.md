## fiber/ai — linux/amd64, 64 CPUs, backend generic, Go go1.27.1

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 3.3 | 3.4 |
| 256 | 3.8 | 20.8 |
| 512 | 4.2 | 33.9 |
| 1024 | 4.4 | 58.2 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 6.5 |
| [8×4096]·[4096×4096] | 8.2 |
| [64×1024]·[1024×1024] | 25.9 |
| [256×768]·[768×3072] | 35.5 |
| [1024×1024]·[1024×1024]ᵀ (view) | 41.7 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 261.8 µs | 3.0 |
| x * y | 64K | 331.4 µs | 2.4 |
| x + row (broadcast) | 64K | 335.3 µs | 1.6 |
| x * 2.5 | 64K | 287.2 µs | 1.8 |
| exp(x) | 64K | 562.6 µs | 0.9 |
| tanh(x) | 64K | 848.8 µs | 0.6 |
| relu(x) | 64K | 936.9 µs | 0.6 |
| x + y | 1M | 2.07 ms | 6.1 |
| x * y | 1M | 2.13 ms | 5.9 |
| x + row (broadcast) | 1M | 2.03 ms | 4.1 |
| x * 2.5 | 1M | 1.94 ms | 4.3 |
| exp(x) | 1M | 3.24 ms | 2.6 |
| tanh(x) | 1M | 4.08 ms | 2.1 |
| relu(x) | 1M | 3.24 ms | 2.6 |
| x + y | 16M | 34.25 ms | 5.9 |
| x * y | 16M | 29.37 ms | 6.9 |
| x + row (broadcast) | 16M | 26.17 ms | 5.1 |
| x * 2.5 | 16M | 25.59 ms | 5.2 |
| exp(x) | 16M | 27.45 ms | 4.9 |
| tanh(x) | 16M | 33.40 ms | 4.0 |
| relu(x) | 16M | 27.73 ms | 4.8 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.82 ms | 23.8 |
| sum(dim=0) | [4096×4096] | 5.72 ms | 11.7 |
| sum(dim=1) | [4096×4096] | 2.81 ms | 23.9 |
| max(dim=1) | [4096×4096] | 2.79 ms | 24.1 |
| softmax(dim=1) | [4096×4096] | 29.87 ms | 4.5 |
| layernorm | [4096×4096] | 45.17 ms | 3.0 |
| transpose+contiguous | [4096×4096] | 33.67 ms | 4.0 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 23.65 ms | 10.8K |
| forward + backward | 52.55 ms | 4.9K |
| forward + backward + Adam step | 52.67 ms | 4.9K |

