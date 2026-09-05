## fiber/ai — darwin/arm64, 10 CPUs, backend generic, Go go1.26.2

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 7.6 | 7.5 |
| 256 | 7.8 | 37.7 |
| 512 | 7.9 | 47.0 |
| 1024 | 7.9 | 48.6 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 12.4 |
| [8×4096]·[4096×4096] | 19.3 |
| [64×1024]·[1024×1024] | 38.4 |
| [256×768]·[768×3072] | 46.8 |
| [1024×1024]·[1024×1024]ᵀ (view) | 49.7 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 41.2 µs | 19.1 |
| x * y | 64K | 41.8 µs | 18.8 |
| x + row (broadcast) | 64K | 41.8 µs | 12.5 |
| x * 2.5 | 64K | 41.1 µs | 12.8 |
| exp(x) | 64K | 83.3 µs | 6.3 |
| tanh(x) | 64K | 199.5 µs | 2.6 |
| relu(x) | 64K | 282.3 µs | 1.9 |
| x + y | 1M | 211.6 µs | 59.5 |
| x * y | 1M | 208.5 µs | 60.3 |
| x + row (broadcast) | 1M | 215.2 µs | 39.0 |
| x * 2.5 | 1M | 201.8 µs | 41.6 |
| exp(x) | 1M | 570.3 µs | 14.7 |
| tanh(x) | 1M | 1.88 ms | 4.5 |
| relu(x) | 1M | 865.2 µs | 9.7 |
| x + y | 16M | 2.11 ms | 95.3 |
| x * y | 16M | 2.09 ms | 96.2 |
| x + row (broadcast) | 16M | 2.13 ms | 63.0 |
| x * 2.5 | 16M | 1.63 ms | 82.2 |
| exp(x) | 16M | 6.51 ms | 20.6 |
| tanh(x) | 16M | 27.08 ms | 5.0 |
| relu(x) | 16M | 10.03 ms | 13.4 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 750.9 µs | 89.4 |
| sum(dim=0) | [4096×4096] | 1.22 ms | 55.2 |
| sum(dim=1) | [4096×4096] | 752.1 µs | 89.2 |
| max(dim=1) | [4096×4096] | 3.18 ms | 21.1 |
| softmax(dim=1) | [4096×4096] | 14.37 ms | 9.3 |
| layernorm | [4096×4096] | 7.29 ms | 18.4 |
| transpose+contiguous | [4096×4096] | 10.92 ms | 12.3 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 9.53 ms | 26.9K |
| forward + backward | 23.55 ms | 10.9K |
| forward + backward + Adam step | 22.40 ms | 11.4K |

