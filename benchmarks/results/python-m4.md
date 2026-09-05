## Python — darwin/arm64, Apple M4 (4P+6E), NumPy 2.5.3, PyTorch 2.14.0 (4 threads), Python 3.14.6

Run by hand on 2026-09-05 with the script in benchmarks/python. torch BLAS backend: Accelerate (SME on M4).

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 913.3 | 950.2 | 952.8 |
| 256 | 1396.1 | 1408.3 | 1407.3 |
| 512 | 1762.3 | 1765.0 | 1764.3 |
| 1024 | 1878.7 | 1859.8 | 1844.7 |
| 2048 | 1884.3 | 1878.1 | 1881.4 |

| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 34.8 | 33.2 |
| [8×4096]·[4096×4096] | 93.0 | 93.1 |
| [64×1024]·[1024×1024] | 1568.0 | 1485.9 |
| [256×768]·[768×3072] | 1735.1 | 1731.0 |
| [1024×1024]·[1024×1024]ᵀ (view) | 1702.4 | 1247.3 |

### Element-wise

| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---:|---:|---:|---:|---:|
| x + y | 64K | 6.2 µs | 126.0 | 19.9 µs | 39.6 |
| x * y | 64K | 5.5 µs | 143.4 | 15.4 µs | 51.0 |
| x + row (broadcast) | 64K | 7.9 µs | 66.6 | 15.9 µs | 32.9 |
| x * 2.5 | 64K | 4.4 µs | 118.9 | 11.5 µs | 45.8 |
| exp(x) | 64K | 89.4 µs | 5.9 | 73.4 µs | 7.1 |
| tanh(x) | 64K | 68.6 µs | 7.6 | 184.1 µs | 2.8 |
| relu(x) | 64K | 22.3 µs | 23.6 | 19.9 µs | 26.3 |
| x + y | 1M | 118.2 µs | 106.5 | 67.6 µs | 186.3 |
| x * y | 1M | 86.2 µs | 146.0 | 63.4 µs | 198.3 |
| x + row (broadcast) | 1M | 114.2 µs | 73.5 | 61.8 µs | 135.7 |
| x * 2.5 | 1M | 58.2 µs | 144.1 | 61.8 µs | 135.7 |
| exp(x) | 1M | 1.39 ms | 6.0 | 224.0 µs | 37.4 |
| tanh(x) | 1M | 776.9 µs | 10.8 | 692.0 µs | 12.1 |
| relu(x) | 1M | 279.4 µs | 30.0 | 61.8 µs | 135.7 |
| x + y | 16M | 2.23 ms | 90.1 | 2.14 ms | 94.1 |
| x * y | 16M | 2.17 ms | 92.6 | 2.15 ms | 93.8 |
| x + row (broadcast) | 16M | 2.36 ms | 56.8 | 1.32 ms | 101.3 |
| x * 2.5 | 16M | 1.52 ms | 88.5 | 1.39 ms | 96.5 |
| exp(x) | 16M | 24.94 ms | 5.4 | 3.57 ms | 37.6 |
| tanh(x) | 16M | 12.28 ms | 10.9 | 11.18 ms | 12.0 |
| relu(x) | 16M | 4.74 ms | 28.3 | 1.39 ms | 96.6 |

### Reductions

| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---|---:|---:|---:|---:|
| sum() | [4096×4096] | 2.04 ms | 32.8 | 924.1 µs | 72.6 |
| sum(dim=0) | [4096×4096] | 1.24 ms | 54.3 | 2.79 ms | 24.0 |
| sum(dim=1) | [4096×4096] | 2.07 ms | 32.5 | 913.1 µs | 73.5 |
| max(dim=1) | [4096×4096] | 1.03 ms | 64.9 | 2.38 ms | 28.2 |
| softmax(dim=1) | [4096×4096] | – | – | 6.02 ms | 22.3 |
| layernorm | [4096×4096] | – | – | 1.94 ms | 69.2 |
| transpose+contiguous | [4096×4096] | 51.16 ms | 2.6 | 11.18 ms | 12.0 |

### MLP 784→512→512→10, batch 256 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 306.6 µs | 835.0K |
| forward + backward | 1.28 ms | 200.3K |
| forward + backward + Adam step | 2.19 ms | 116.9K |
