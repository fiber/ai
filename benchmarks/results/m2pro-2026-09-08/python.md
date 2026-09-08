## Python — darwin/arm64, NumPy 2.5.3, PyTorch 2.14.0 (6 threads), Python 3.14.7
### Matrix multiply (float32, GFLOPS)
| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 775.4 | 755.2 | 755.1 |
| 256 | 1117.2 | 1122.9 | 1116.4 |
| 512 | 2125.2 | 2133.1 | 2154.5 |
| 1024 | 2636.4 | 2624.6 | 2682.2 |
| 2048 | 2194.7 | 2190.5 | 2226.7 |
| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 11.3 | 11.3 |
| [8×4096]·[4096×4096] | 69.7 | 69.4 |
| [64×1024]·[1024×1024] | 1296.4 | 1290.2 |
| [256×768]·[768×3072] | 2323.5 | 2358.0 |
| [1024×1024]·[1024×1024]ᵀ (view) | 2384.7 | 2376.7 |
### Element-wise
| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---:|---:|---:|---:|---:|
| x + y | 64K | 7.0 µs | 111.8 | 31.1 µs | 25.3 |
| x * y | 64K | 6.6 µs | 119.1 | 30.4 µs | 25.9 |
| x + row (broadcast) | 64K | 9.7 µs | 54.1 | 30.5 µs | 17.2 |
| x * 2.5 | 64K | 4.4 µs | 118.3 | 29.9 µs | 17.5 |
| exp(x) | 64K | 105.1 µs | 5.0 | 45.5 µs | 11.5 |
| tanh(x) | 64K | 61.3 µs | 8.6 | 79.9 µs | 6.6 |
| relu(x) | 64K | 22.4 µs | 23.4 | 29.9 µs | 17.6 |
| x + y | 1M | 101.5 µs | 124.0 | 68.3 µs | 184.1 |
| x * y | 1M | 101.0 µs | 124.6 | 68.3 µs | 184.1 |
| x + row (broadcast) | 1M | 137.6 µs | 61.0 | 66.2 µs | 126.7 |
| x * 2.5 | 1M | 61.2 µs | 137.0 | 68.3 µs | 122.9 |
| exp(x) | 1M | 1.66 ms | 5.0 | 214.6 µs | 39.1 |
| tanh(x) | 1M | 975.4 µs | 8.6 | 719.2 µs | 11.7 |
| relu(x) | 1M | 350.4 µs | 23.9 | 66.8 µs | 125.7 |
| x + y | 16M | 2.28 ms | 88.4 | 1.38 ms | 145.7 |
| x * y | 16M | 2.28 ms | 88.3 | 1.37 ms | 147.1 |
| x + row (broadcast) | 16M | 2.90 ms | 46.3 | 858.5 µs | 156.3 |
| x * 2.5 | 16M | 1.47 ms | 91.5 | 848.0 µs | 158.3 |
| exp(x) | 16M | 26.37 ms | 5.1 | 3.09 ms | 43.4 |
| tanh(x) | 16M | 15.36 ms | 8.7 | 11.04 ms | 12.2 |
| relu(x) | 16M | 5.59 ms | 24.0 | 850.1 µs | 157.9 |
### Reductions
| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---|---:|---:|---:|---:|
| sum() | [4096×4096] | 2.68 ms | 25.1 | 596.0 µs | 112.6 |
| sum(dim=0) | [4096×4096] | 1.29 ms | 51.9 | 2.18 ms | 30.8 |
| sum(dim=1) | [4096×4096] | 2.73 ms | 24.6 | 571.4 µs | 117.4 |
| max(dim=1) | [4096×4096] | 1.16 ms | 58.0 | 1.18 ms | 56.9 |
| softmax(dim=1) | [4096×4096] | – | – | 6.27 ms | 21.4 |
| layernorm | [4096×4096] | – | – | 2.21 ms | 60.7 |
| transpose+contiguous | [4096×4096] | 60.79 ms | 2.2 | 22.43 ms | 6.0 |
### Attention (PyTorch, no_grad)
| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 7.77 ms | 553.0 |
| same with causal mask | 8.24 ms | 521.5 |
### Convolution (PyTorch, no_grad)
| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 23.77 ms | 311.3 |
### MLP 784→512→512→10, batch 256 (PyTorch)
| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 486.9 µs | 525.8K |
| forward + backward | 1.16 ms | 220.7K |
| forward + backward + Adam step | 2.12 ms | 120.8K |
### EmbeddingGemma (sentence-transformers, fp32, 32 sentences ≈ 64 tokens, one batch)
threads 6
| shape | time / batch | sentences/s |
|---|---:|---:|
| 32 × 65 tokens, dim 768 | 362.00 ms | 88 |
