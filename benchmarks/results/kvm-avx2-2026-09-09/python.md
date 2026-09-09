## Python — linux/x86_64, NumPy 2.5.2, PyTorch 2.14.0+cpu (6 threads), Python 3.12.3

Same KVM guest and day as go.md. torch BLAS: MKL, oneDNN on.

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 88.2 | 43.0 | 129.3 |
| 256 | 183.9 | 55.4 | 154.9 |
| 512 | 199.0 | 63.0 | 183.9 |
| 1024 | 219.7 | 58.2 | 121.6 |
| 2048 | 241.2 | 60.2 | 186.5 |

| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 15.4 | 16.6 |
| [8×4096]·[4096×4096] | 36.7 | 40.2 |
| [64×1024]·[1024×1024] | 138.3 | 125.7 |
| [256×768]·[768×3072] | 208.4 | 170.0 |
| [1024×1024]·[1024×1024]ᵀ (view) | 184.2 | 166.2 |

### Element-wise

| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---:|---:|---:|---:|---:|
| x + y | 64K | 48.8 µs | 16.1 | 26.8 µs | 29.3 |
| x * y | 64K | 51.6 µs | 15.2 | 29.2 µs | 26.9 |
| x + row (broadcast) | 64K | 66.9 µs | 7.8 | 34.3 µs | 15.3 |
| x * 2.5 | 64K | 28.9 µs | 18.2 | 40.5 µs | 13.0 |
| exp(x) | 64K | 178.7 µs | 2.9 | 27.0 µs | 19.4 |
| tanh(x) | 64K | 390.3 µs | 1.3 | 79.1 µs | 6.6 |
| relu(x) | 64K | 53.9 µs | 9.7 | 26.3 µs | 19.9 |
| x + y | 1M | 723.7 µs | 17.4 | 140.9 µs | 89.3 |
| x * y | 1M | 629.6 µs | 20.0 | 177.4 µs | 70.9 |
| x + row (broadcast) | 1M | 692.1 µs | 12.1 | 111.5 µs | 75.2 |
| x * 2.5 | 1M | 473.6 µs | 17.7 | 630.3 µs | 13.3 |
| exp(x) | 1M | 3.89 ms | 2.2 | 2.11 ms | 4.0 |
| tanh(x) | 1M | 6.12 ms | 1.4 | 1.22 ms | 6.9 |
| relu(x) | 1M | 937.7 µs | 8.9 | 128.4 µs | 65.3 |
| x + y | 16M | 40.66 ms | 5.0 | 30.92 ms | 6.5 |
| x * y | 16M | 39.90 ms | 5.0 | 27.70 ms | 7.3 |
| x + row (broadcast) | 16M | 41.96 ms | 3.2 | 25.70 ms | 5.2 |
| x * 2.5 | 16M | 32.84 ms | 4.1 | 26.72 ms | 5.0 |
| exp(x) | 16M | 76.66 ms | 1.8 | 32.73 ms | 4.1 |
| tanh(x) | 16M | 123.37 ms | 1.1 | 41.81 ms | 3.2 |
| relu(x) | 16M | 35.38 ms | 3.8 | 25.07 ms | 5.4 |

### Reductions

| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---|---:|---:|---:|---:|
| sum() | [4096×4096] | 9.95 ms | 6.7 | 2.01 ms | 33.3 |
| sum(dim=0) | [4096×4096] | 8.98 ms | 7.5 | 7.30 ms | 9.2 |
| sum(dim=1) | [4096×4096] | 9.89 ms | 6.8 | 3.12 ms | 21.5 |
| max(dim=1) | [4096×4096] | 8.44 ms | 8.0 | 4.82 ms | 13.9 |
| softmax(dim=1) | [4096×4096] | – | – | 34.18 ms | 3.9 |
| layernorm | [4096×4096] | – | – | 25.59 ms | 5.2 |
| transpose+contiguous | [4096×4096] | 215.64 ms | 0.6 | 127.75 ms | 1.1 |

### Attention (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 33.68 ms | 127.5 |
| same with causal mask | 33.67 ms | 127.5 |
| [1×8×2048×64] long sequence | 65.76 ms | 130.6 |

### Convolution (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 56.84 ms | 130.2 |

### MLP 784→512→512→10, batch 256 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 3.27 ms | 78.2K |
| forward + backward | 5.66 ms | 45.2K |
| forward + backward + Adam step | 8.28 ms | 30.9K |

### EmbeddingGemma (sentence-transformers 6.0.1, fp32, 32 sentences ≈ 64 tokens, one batch)

| shape | time / batch | sentences/s |
|---|---:|---:|
| 32 × 65 tokens, dim 768 | 2.40 s | 13 |
