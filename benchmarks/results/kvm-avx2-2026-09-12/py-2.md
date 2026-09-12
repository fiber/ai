## Python — linux/x86_64, NumPy 2.5.2, PyTorch 2.14.0+cpu (6 threads), Python 3.12.3

torch BLAS backend: - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=08187d9e0fba026dc8217405802ab5381dc88d90, CUDA_FLAGS= -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS -Xfatbin -compress-all  -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUB_WRAPPED_NAMESPACE=at_cuda_detail -DDISABLE_CUSPARSE_DEPRECATED -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -DC10_NODEPRECATED, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TORCH_VERSION=2.14.0, USE_CUDA=0, USE_CUDNN=OFF, USE_CUSPARSELT=OFF, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_HIPSPARSELT=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF,

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 89.1 | 44.8 | 107.9 |
| 256 | 169.8 | 54.1 | 130.8 |
| 512 | 188.7 | 56.3 | 180.3 |
| 1024 | 211.6 | 55.5 | 185.9 |
| 2048 | 219.4 | 59.7 | 187.9 |

| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 11.7 | 13.1 |
| [8×4096]·[4096×4096] | 35.3 | 43.1 |
| [64×1024]·[1024×1024] | 168.2 | 114.6 |
| [256×768]·[768×3072] | 174.1 | 158.6 |
| [1024×1024]·[1024×1024]ᵀ (view) | 199.2 | 171.3 |

### Small products (PyTorch)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 6.9 µs | 9.5 |
| 64² | 11.9 µs | 44.2 |
| 96² | 19.3 µs | 91.5 |
| 128² | 32.7 µs | 128.3 |
| 160² | 67.0 µs | 122.3 |
| 192² | 103.4 µs | 136.9 |
| 256² | 233.5 µs | 143.7 |
| [64×24]·[24×16] | 5.7 µs | 8.6 |
| [64×16]·[16×3] | 5.2 µs | 1.2 |
| [256×24]·[24×16] | 10.5 µs | 18.7 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 225.7 µs | 283.5K |
| forward + backward + Adam step | 2.78 ms | 23.1K |

### Element-wise

| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---:|---:|---:|---:|---:|
| x + y | 64K | 39.5 µs | 19.9 | 31.2 µs | 25.2 |
| x * y | 64K | 37.7 µs | 20.8 | 26.2 µs | 30.0 |
| x + row (broadcast) | 64K | 50.8 µs | 10.3 | 44.8 µs | 11.7 |
| x * 2.5 | 64K | 29.4 µs | 17.9 | 30.5 µs | 17.2 |
| exp(x) | 64K | 175.9 µs | 3.0 | 27.6 µs | 19.0 |
| tanh(x) | 64K | 369.0 µs | 1.4 | 72.2 µs | 7.3 |
| relu(x) | 64K | 55.5 µs | 9.4 | 20.5 µs | 25.6 |
| x + y | 1M | 817.4 µs | 15.4 | 120.8 µs | 104.2 |
| x * y | 1M | 892.2 µs | 14.1 | 174.6 µs | 72.1 |
| x + row (broadcast) | 1M | 920.3 µs | 9.1 | 144.3 µs | 58.1 |
| x * 2.5 | 1M | 414.5 µs | 20.2 | 175.0 µs | 47.9 |
| exp(x) | 1M | 2.79 ms | 3.0 | 233.8 µs | 35.9 |
| tanh(x) | 1M | 5.79 ms | 1.4 | 999.1 µs | 8.4 |
| relu(x) | 1M | 867.1 µs | 9.7 | 115.0 µs | 72.9 |
| x + y | 16M | 39.57 ms | 5.1 | 24.48 ms | 8.2 |
| x * y | 16M | 37.98 ms | 5.3 | 23.89 ms | 8.4 |
| x + row (broadcast) | 16M | 40.18 ms | 3.3 | 23.01 ms | 5.8 |
| x * 2.5 | 16M | 31.56 ms | 4.3 | 26.03 ms | 5.2 |
| exp(x) | 16M | 70.33 ms | 1.9 | 27.70 ms | 4.8 |
| tanh(x) | 16M | 116.72 ms | 1.1 | 37.77 ms | 3.6 |
| relu(x) | 16M | 33.68 ms | 4.0 | 23.65 ms | 5.7 |

### Reductions

| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---|---:|---:|---:|---:|
| sum() | [4096×4096] | 9.31 ms | 7.2 | 2.42 ms | 27.7 |
| sum(dim=0) | [4096×4096] | 8.35 ms | 8.0 | 4.14 ms | 16.2 |
| sum(dim=1) | [4096×4096] | 9.01 ms | 7.4 | 2.36 ms | 28.4 |
| max(dim=1) | [4096×4096] | 8.06 ms | 8.3 | 5.67 ms | 11.8 |
| softmax(dim=1) | [4096×4096] | – | – | 34.57 ms | 3.9 |
| layernorm | [4096×4096] | – | – | 23.65 ms | 5.7 |
| transpose+contiguous | [4096×4096] | 178.60 ms | 0.8 | 121.73 ms | 1.1 |

### Attention (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 32.82 ms | 130.9 |
| same with causal mask | 30.69 ms | 139.9 |
| [1×8×2048×64] long sequence | 60.42 ms | 142.2 |

### Convolution (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 57.84 ms | 127.9 |

### MLP 784→512→512→10, batch 256 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 2.31 ms | 110.9K |
| forward + backward | 5.68 ms | 45.1K |
| forward + backward + Adam step | 7.27 ms | 35.2K |

### EmbeddingGemma (sentence-transformers, fp32, 32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m

