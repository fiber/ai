## fiber/ai — darwin/arm64, Apple M4 (4P+6E), backend neon, Go go1.27.0

Run by hand on 2026-09-05 with `go run cmd/bench/main.go`.

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | 1 thread | all threads |
|---:|---:|---:|
| 128 | 91.2 | 93.8 |
| 256 | 107.6 | 189.6 |
| 512 | 117.0 | 429.5 |
| 1024 | 118.6 | 571.6 |
| 2048 | 120.7 | 614.2 |

| shape | GFLOPS (all threads) |
|---|---:|
| [1×4096]·[4096×4096] | 30.4 |
| [8×4096]·[4096×4096] | 72.9 |
| [64×1024]·[1024×1024] | 244.0 |
| [256×768]·[768×3072] | 483.4 |
| [1024×1024]·[1024×1024]ᵀ (view) | 562.0 |

### Element-wise (all threads)

| op | n | time | GB/s |
|---|---:|---:|---:|
| x + y | 64K | 17.8 µs | 44.1 |
| x * y | 64K | 18.1 µs | 43.5 |
| x + row (broadcast) | 64K | 19.3 µs | 27.2 |
| x * 2.5 | 64K | 16.3 µs | 32.2 |
| exp(x) | 64K | 29.2 µs | 17.9 |
| tanh(x) | 64K | 121.2 µs | 4.3 |
| relu(x) | 64K | 15.8 µs | 33.1 |
| x + y | 1M | 127.8 µs | 98.4 |
| x * y | 1M | 128.4 µs | 98.0 |
| x + row (broadcast) | 1M | 135.3 µs | 62.0 |
| x * 2.5 | 1M | 119.5 µs | 70.2 |
| exp(x) | 1M | 177.0 µs | 47.4 |
| tanh(x) | 1M | 1.48 ms | 5.7 |
| relu(x) | 1M | 119.3 µs | 70.3 |
| x + y | 16M | 2.84 ms | 70.8 |
| x * y | 16M | 2.84 ms | 70.8 |
| x + row (broadcast) | 16M | 2.64 ms | 50.9 |
| x * 2.5 | 16M | 2.15 ms | 62.5 |
| exp(x) | 16M | 2.09 ms | 64.3 |
| tanh(x) | 16M | 23.82 ms | 5.6 |
| relu(x) | 16M | 2.17 ms | 62.0 |

### Reductions (all threads)

| op | shape | time | GB/s |
|---|---|---:|---:|
| sum() | [4096×4096] | 561.4 µs | 119.5 |
| sum(dim=0) | [4096×4096] | 621.4 µs | 108.0 |
| sum(dim=1) | [4096×4096] | 571.2 µs | 117.5 |
| max(dim=1) | [4096×4096] | 570.9 µs | 117.6 |
| softmax(dim=1) | [4096×4096] | 2.74 ms | 49.0 |
| layernorm | [4096×4096] | 4.66 ms | 28.8 |
| transpose+contiguous | [4096×4096] | 10.84 ms | 12.4 |

### MLP 784→512→512→10, batch 256 (all threads)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (NoGrad) | 1.35 ms | 189.4K |
| forward + backward | 3.68 ms | 69.5K |
| forward + backward + Adam step | 3.95 ms | 64.7K |
