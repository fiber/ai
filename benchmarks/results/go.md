## fiber/ai — darwin/arm64, 10 CPUs, backend neon, Go go1.26.2

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 67.5 | 71.7 |
| 256 | 86.1 | 146.8 |
| 512 | 97.0 | 364.5 |
| 1024 | 97.7 | 525.5 |
| 2048 | 99.8 | 594.7 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 14.6 |
| [8×4096]·[4096×4096] | 42.6 |
| [64×1024]·[1024×1024] | 212.5 |
| [256×768]·[768×3072] | 374.9 |
| [1024×1024]·[1024×1024]ᵀ (view) | 464.9 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 27.7 µs | 28.4 |
| x * y | 64K | 28.6 µs | 27.5 |
| x + row (broadcast) | 64K | 29.3 µs | 17.9 |
| x * 2.5 | 64K | 27.3 µs | 19.2 |
| exp(x) | 64K | 50.4 µs | 10.4 |
| tanh(x) | 64K | 211.3 µs | 2.5 |
| relu(x) | 64K | 26.4 µs | 19.9 |
| x + y | 1M | 172.5 µs | 73.0 |
| x * y | 1M | 169.7 µs | 74.2 |
| x + row (broadcast) | 1M | 178.3 µs | 47.1 |
| x * 2.5 | 1M | 166.0 µs | 50.5 |
| exp(x) | 1M | 225.8 µs | 37.2 |
| tanh(x) | 1M | 2.16 ms | 3.9 |
| relu(x) | 1M | 170.4 µs | 49.2 |
| x + y | 16M | 2.02 ms | 99.6 |
| x * y | 16M | 2.00 ms | 100.7 |
| x + row (broadcast) | 16M | 1.88 ms | 71.5 |
| x * 2.5 | 16M | 1.52 ms | 88.5 |
| exp(x) | 16M | 2.05 ms | 65.4 |
| tanh(x) | 16M | 26.61 ms | 5.0 |
| relu(x) | 16M | 1.50 ms | 89.3 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 614.1 µs | 109.3 |
| sum(dim=0) | [4096×4096] | 652.6 µs | 102.8 |
| sum(dim=1) | [4096×4096] | 602.6 µs | 111.4 |
| max(dim=1) | [4096×4096] | 587.7 µs | 114.2 |
| softmax(dim=1) | [4096×4096] | 2.91 ms | 46.1 |
| layernorm | [4096×4096] | 3.37 ms | 39.8 |
| transpose+contiguous | [4096×4096] | 10.99 ms | 12.2 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.51 ms | 169.5K |
| forward + backward | 4.24 ms | 60.3K |
| forward + backward + Adam step | 4.50 ms | 56.8K |

