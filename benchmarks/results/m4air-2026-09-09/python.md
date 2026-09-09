## Python — darwin/arm64, NumPy 2.5.3, PyTorch 2.14.0 (4 threads), Python 3.14.7

Same M4 MacBook Air and day as go.md. torch BLAS: Accelerate (SME on the M4).

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 940.9 | 933.3 | 954.0 |
| 256 | 1231.9 | 1282.2 | 1252.1 |
| 512 | 1479.9 | 1473.2 | 1478.6 |
| 1024 | 1628.0 | 1488.8 | 1542.9 |
| 2048 | 1648.3 | 1573.2 | 1631.3 |

| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 35.1 | 35.1 |
| [8×4096]·[4096×4096] | 91.9 | 92.0 |
| [64×1024]·[1024×1024] | 1420.3 | 1369.8 |
| [256×768]·[768×3072] | 1541.9 | 1506.9 |
| [1024×1024]·[1024×1024]ᵀ (view) | 1500.3 | 1493.0 |

### Element-wise

| op | n | NumPy time | PyTorch time |
|---|---:|---:|---:|
| x + y | 64K | 7.2 µs | 12.3 µs |
| exp(x) | 64K | 88.8 µs | 24.1 µs |
| tanh(x) | 64K | 50.1 µs | 54.4 µs |
| relu(x) | 64K | 19.1 µs | 11.4 µs |
| x + y | 1M | 109.0 µs | 58.9 µs |
| x * 2.5 | 1M | 62.2 µs | 63.2 µs |
| exp(x) | 1M | 1.77 ms | 232.7 µs |
| tanh(x) | 1M | 787.2 µs | 721.1 µs |
| relu(x) | 1M | 303.6 µs | 62.2 µs |
| x + y | 16M | 2.20 ms | 2.14 ms |
| x * 2.5 | 16M | 1.65 ms | 1.33 ms |
| exp(x) | 16M | 25.47 ms | 3.90 ms |
| tanh(x) | 16M | 13.25 ms | 12.08 ms |
| relu(x) | 16M | 5.02 ms | 1.38 ms |

### Reductions

| op | shape | NumPy time | PyTorch time |
|---|---|---:|---:|
| sum() | [4096×4096] | 2.25 ms | 905.6 µs |
| sum(dim=0) | [4096×4096] | 1.41 ms | 2.90 ms |
| sum(dim=1) | [4096×4096] | 2.41 ms | 892.3 µs |
| max(dim=1) | [4096×4096] | 1.14 ms | 1.66 ms |
| softmax(dim=1) | [4096×4096] | – | 5.71 ms |
| layernorm | [4096×4096] | – | 2.39 ms |
| transpose+contiguous | [4096×4096] | 50.42 ms | 14.43 ms |

### Attention (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 5.89 ms | 729.2 |
| same with causal mask | 7.82 ms | 549.4 |
| [1×8×2048×64] long sequence | 10.33 ms | 831.7 |

### Convolution (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 22.26 ms | 332.4 |

### MLP 784→512→512→10, batch 256 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 314.0 µs | 815.2K |
| forward + backward | 735.2 µs | 348.2K |
| forward + backward + Adam step | 1.26 ms | 204.0K |

### EmbeddingGemma (sentence-transformers, fp32, 32 sentences ≈ 64 tokens, one batch)

| shape | time / batch | sentences/s |
|---|---:|---:|
| 32 × 65 tokens, dim 768 | 398.37 ms | 80 |
