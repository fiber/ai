## fiber/ai — linux/amd64, 6 CPUs, backend avx2, Go go1.26.2

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 24.1 | 27.2 |
| 256 | 33.4 | 49.5 |
| 512 | 35.5 | 105.1 |
| 1024 | 37.3 | 141.0 |
| 2048 | 35.7 | 191.0 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 8.7 |
| [8×4096]·[4096×4096] | 17.7 |
| [64×1024]·[1024×1024] | 60.5 |
| [256×768]·[768×3072] | 128.9 |
| [1024×1024]·[1024×1024]ᵀ (view) | 157.5 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 158.7 µs | 5.0 |
| x * y | 64K | 157.5 µs | 5.0 |
| x + row (broadcast) | 64K | 114.7 µs | 4.6 |
| x * 2.5 | 64K | 158.7 µs | 3.3 |
| exp(x) | 64K | 200.6 µs | 2.6 |
| tanh(x) | 64K | 1.07 ms | 0.5 |
| relu(x) | 64K | 132.0 µs | 4.0 |
| x + y | 1M | 1.64 ms | 7.7 |
| x * y | 1M | 1.60 ms | 7.9 |
| x + row (broadcast) | 1M | 1.47 ms | 5.7 |
| x * 2.5 | 1M | 1.44 ms | 5.8 |
| exp(x) | 1M | 1.64 ms | 5.1 |
| tanh(x) | 1M | 11.36 ms | 0.7 |
| relu(x) | 1M | 1.67 ms | 5.0 |
| x + y | 16M | 24.19 ms | 8.3 |
| x * y | 16M | 20.96 ms | 9.6 |
| x + row (broadcast) | 16M | 19.64 ms | 6.8 |
| x * 2.5 | 16M | 19.15 ms | 7.0 |
| exp(x) | 16M | 19.26 ms | 7.0 |
| tanh(x) | 16M | 144.41 ms | 0.9 |
| relu(x) | 16M | 20.62 ms | 6.5 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.09 ms | 32.1 |
| sum(dim=0) | [4096×4096] | 2.42 ms | 27.7 |
| sum(dim=1) | [4096×4096] | 2.13 ms | 31.5 |
| max(dim=1) | [4096×4096] | 2.16 ms | 31.0 |
| softmax(dim=1) | [4096×4096] | 23.53 ms | 5.7 |
| layernorm | [4096×4096] | 38.69 ms | 3.5 |
| transpose+contiguous | [4096×4096] | 86.12 ms | 1.6 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 6.15 ms | 41.6K |
| forward + backward | 18.21 ms | 14.1K |
| forward + backward + Adam step | 19.15 ms | 13.4K |

