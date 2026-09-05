## fiber/ai — linux/amd64, 32 CPUs, backend avx512, Go go1.27.1

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 29.3 | 30.6 |
| 256 | 43.9 | 80.8 |
| 512 | 98.6 | 244.5 |
| 1024 | 122.0 | 530.4 |
| 2048 | 119.9 | 732.0 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 13.0 |
| [8×4096]·[4096×4096] | 37.0 |
| [64×1024]·[1024×1024] | 113.4 |
| [256×768]·[768×3072] | 322.7 |
| [1024×1024]·[1024×1024]ᵀ (view) | 515.3 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 194.1 µs | 4.1 |
| x * y | 64K | 214.4 µs | 3.7 |
| x + row (broadcast) | 64K | 132.4 µs | 4.0 |
| x * 2.5 | 64K | 224.9 µs | 2.3 |
| exp(x) | 64K | 220.8 µs | 2.4 |
| tanh(x) | 64K | 419.0 µs | 1.3 |
| relu(x) | 64K | 209.3 µs | 2.5 |
| x + y | 1M | 1.49 ms | 8.4 |
| x * y | 1M | 1.35 ms | 9.3 |
| x + row (broadcast) | 1M | 1.27 ms | 6.6 |
| x * 2.5 | 1M | 1.17 ms | 7.2 |
| exp(x) | 1M | 1.84 ms | 4.6 |
| tanh(x) | 1M | 4.53 ms | 1.9 |
| relu(x) | 1M | 1.40 ms | 6.0 |
| x + y | 16M | 22.79 ms | 8.8 |
| x * y | 16M | 23.85 ms | 8.4 |
| x + row (broadcast) | 16M | 19.87 ms | 6.8 |
| x * 2.5 | 16M | 19.76 ms | 6.8 |
| exp(x) | 16M | 19.95 ms | 6.7 |
| tanh(x) | 16M | 57.64 ms | 2.3 |
| relu(x) | 16M | 19.75 ms | 6.8 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.40 ms | 27.9 |
| sum(dim=0) | [4096×4096] | 2.68 ms | 25.0 |
| sum(dim=1) | [4096×4096] | 2.42 ms | 27.7 |
| max(dim=1) | [4096×4096] | 2.40 ms | 28.0 |
| softmax(dim=1) | [4096×4096] | 20.87 ms | 6.4 |
| layernorm | [4096×4096] | 36.07 ms | 3.7 |
| transpose+contiguous | [4096×4096] | 25.45 ms | 5.3 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 4.32 ms | 59.2K |
| forward + backward | 18.06 ms | 14.2K |
| forward + backward + Adam step | 18.28 ms | 14.0K |

