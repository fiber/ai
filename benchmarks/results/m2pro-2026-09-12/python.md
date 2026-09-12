## Python — darwin/arm64, NumPy 2.0.2, PyTorch 2.8.0 (6 threads), Python 3.9.6

torch BLAS backend: - Build settings: BLAS_INFO=accelerate, BUILD_TYPE=Release, COMMIT_SHA=a1cb3cc05d46d198467bebbb6e8fba50a325d4e7, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_PYTORCH_QNNPACK -DAT_BUILD_ARM_VEC256_WITH_SLEEF -DUSE_XNNPACK -DUSE_PYTORCH_METAL_EXPORT -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DUSE_COREML_DELEGATE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=braced-scalar-init -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wvla-extension -Wsuggest-override -Wnewline-eof -Winconsistent-missing-override -Winconsistent-missing-destructor-override -Wno-pass-failed -Wno-error=old-style-cast -Wconstant-conversion -Qunused-arguments -faligned-new -fno-math-errno -fno-trapping-math -Werror=format -DUSE_MPS -Wno-missing-braces, LAPACK_INFO=accelerate, TORCH_VERSION=2.8.0, USE_CUDA=OFF, USE_CUDNN=OFF, USE_CUSPARSELT=OFF, USE_EIGEN_FOR_BLAS=ON, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF,

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 791.9 | 792.8 | 782.3 |
| 256 | 1127.1 | 1122.3 | 1121.1 |
| 512 | 2144.2 | 2157.6 | 2156.7 |
| 1024 | 2636.2 | 2652.1 | 2687.6 |
| 2048 | 2225.4 | 2225.1 | 2192.4 |

| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 11.3 | 11.2 |
| [8×4096]·[4096×4096] | 70.6 | 69.3 |
| [64×1024]·[1024×1024] | 1287.6 | 1292.1 |
| [256×768]·[768×3072] | 2150.4 | 2327.9 |
| [1024×1024]·[1024×1024]ᵀ (view) | 2419.5 | 2374.9 |

### Small products (PyTorch)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 1.3 µs | 51.7 |
| 64² | 1.7 µs | 305.3 |
| 96² | 3.1 µs | 566.4 |
| 128² | 5.3 µs | 784.8 |
| 160² | 9.2 µs | 889.8 |
| 192² | 14.3 µs | 991.5 |
| 256² | 29.4 µs | 1143.1 |
| [64×24]·[24×16] | 1.3 µs | 37.8 |
| [64×16]·[16×3] | 2.0 µs | 3.1 |
| [256×24]·[24×16] | 2.0 µs | 99.9 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 34.8 µs | 1.84M |
| forward + backward + Adam step | 252.9 µs | 253.1K |

### Element-wise

| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---:|---:|---:|---:|---:|
| x + y | 64K | 6.7 µs | 117.6 | 30.9 µs | 25.4 |
| x * y | 64K | 6.6 µs | 118.5 | 31.2 µs | 25.2 |
| x + row (broadcast) | 64K | 9.8 µs | 53.6 | 31.1 µs | 16.9 |
| x * 2.5 | 64K | 4.6 µs | 113.9 | 32.4 µs | 16.2 |
| exp(x) | 64K | 102.9 µs | 5.1 | 45.6 µs | 11.5 |
| tanh(x) | 64K | 62.6 µs | 8.4 | 78.8 µs | 6.7 |
| relu(x) | 64K | 22.2 µs | 23.6 | 30.7 µs | 17.1 |
| x + y | 1M | 116.2 µs | 108.3 | 68.3 µs | 184.3 |
| x * y | 1M | 99.8 µs | 126.1 | 68.8 µs | 182.8 |
| x + row (broadcast) | 1M | 139.5 µs | 60.1 | 69.2 µs | 121.2 |
| x * 2.5 | 1M | 65.9 µs | 127.3 | 68.2 µs | 123.0 |
| exp(x) | 1M | 1.66 ms | 5.1 | 208.7 µs | 40.2 |
| tanh(x) | 1M | 987.5 µs | 8.5 | 713.4 µs | 11.8 |
| relu(x) | 1M | 350.5 µs | 23.9 | 68.1 µs | 123.1 |
| x + y | 16M | 2.26 ms | 88.9 | 1.39 ms | 144.5 |
| x * y | 16M | 2.25 ms | 89.7 | 1.47 ms | 137.2 |
| x + row (broadcast) | 16M | 2.93 ms | 45.9 | 852.8 µs | 157.4 |
| x * 2.5 | 16M | 1.46 ms | 91.9 | 860.2 µs | 156.0 |
| exp(x) | 16M | 26.49 ms | 5.1 | 3.10 ms | 43.3 |
| tanh(x) | 16M | 15.83 ms | 8.5 | 11.28 ms | 11.9 |
| relu(x) | 16M | 5.73 ms | 23.4 | 852.1 µs | 157.5 |

### Reductions

| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---|---:|---:|---:|---:|
| sum() | [4096×4096] | 2.71 ms | 24.8 | 593.1 µs | 113.2 |
| sum(dim=0) | [4096×4096] | 1.33 ms | 50.5 | 2.23 ms | 30.1 |
| sum(dim=1) | [4096×4096] | 2.78 ms | 24.1 | 575.7 µs | 116.6 |
| max(dim=1) | [4096×4096] | 1.19 ms | 56.6 | 1.22 ms | 55.0 |
| softmax(dim=1) | [4096×4096] | – | – | 4.60 ms | 29.2 |
| layernorm | [4096×4096] | – | – | 2.37 ms | 56.6 |
| transpose+contiguous | [4096×4096] | 61.96 ms | 2.2 | 23.21 ms | 5.8 |

### Attention (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 7.28 ms | 589.8 |
| same with causal mask | 7.26 ms | 591.6 |
| [1×8×2048×64] long sequence | 12.09 ms | 710.7 |

### Convolution (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 25.80 ms | 286.7 |

### MLP 784→512→512→10, batch 256 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 492.8 µs | 519.5K |
| forward + backward | 1.21 ms | 212.0K |
| forward + backward + Adam step | 2.19 ms | 116.9K |

### EmbeddingGemma (sentence-transformers, fp32, 32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m

