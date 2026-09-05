## fiber/ai — linux/amd64, 32 CPUs, backend generic, Go go1.27.1

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 3.5 | 3.6 |
| 256 | 3.9 | 29.7 |
| 512 | 4.3 | 36.5 |
| 1024 | 4.3 | 43.8 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 10.5 |
| [8×4096]·[4096×4096] | 17.0 |
| [64×1024]·[1024×1024] | 30.9 |
| [256×768]·[768×3072] | 46.9 |
| [1024×1024]·[1024×1024]ᵀ (view) | 45.1 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 228.4 µs | 3.4 |
| x * y | 64K | 280.2 µs | 2.8 |
| x + row (broadcast) | 64K | 273.9 µs | 1.9 |
| x * 2.5 | 64K | 211.1 µs | 2.5 |
| exp(x) | 64K | 240.8 µs | 2.2 |
| tanh(x) | 64K | 424.3 µs | 1.2 |
| relu(x) | 64K | 889.9 µs | 0.6 |
| x + y | 1M | 1.64 ms | 7.7 |
| x * y | 1M | 1.68 ms | 7.5 |
| x + row (broadcast) | 1M | 1.58 ms | 5.3 |
| x * 2.5 | 1M | 1.36 ms | 6.2 |
| exp(x) | 1M | 1.33 ms | 6.3 |
| tanh(x) | 1M | 3.25 ms | 2.6 |
| relu(x) | 1M | 876.9 µs | 9.6 |
| x + y | 16M | 22.91 ms | 8.8 |
| x * y | 16M | 23.98 ms | 8.4 |
| x + row (broadcast) | 16M | 19.79 ms | 6.8 |
| x * 2.5 | 16M | 19.79 ms | 6.8 |
| exp(x) | 16M | 25.32 ms | 5.3 |
| tanh(x) | 16M | 57.23 ms | 2.3 |
| relu(x) | 16M | 20.58 ms | 6.5 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.43 ms | 27.6 |
| sum(dim=0) | [4096×4096] | 2.87 ms | 23.4 |
| sum(dim=1) | [4096×4096] | 2.45 ms | 27.4 |
| max(dim=1) | [4096×4096] | 2.43 ms | 27.6 |
| softmax(dim=1) | [4096×4096] | 28.97 ms | 4.6 |
| layernorm | [4096×4096] | 37.30 ms | 3.6 |
| transpose+contiguous | [4096×4096] | 24.80 ms | 5.4 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 12.31 ms | 20.8K |
| forward + backward | 31.59 ms | 8.1K |
| forward + backward + Adam step | 32.10 ms | 8.0K |

