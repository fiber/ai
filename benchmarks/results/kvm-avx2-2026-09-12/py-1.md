## Python — linux/x86_64, NumPy 2.5.2, PyTorch 2.14.0+cpu (6 threads), Python 3.12.3

torch BLAS backend: - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=08187d9e0fba026dc8217405802ab5381dc88d90, CUDA_FLAGS= -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS -Xfatbin -compress-all  -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUB_WRAPPED_NAMESPACE=at_cuda_detail -DDISABLE_CUSPARSE_DEPRECATED -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -DC10_NODEPRECATED, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TORCH_VERSION=2.14.0, USE_CUDA=0, USE_CUDNN=OFF, USE_CUSPARSELT=OFF, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_HIPSPARSELT=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF,

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 73.7 | 42.7 | 104.3 |
| 256 | 153.9 | 54.8 | 173.3 |
| 512 | 213.2 | 57.4 | 184.7 |
| 1024 | 183.8 | 54.1 | 173.0 |
| 2048 | 166.7 | 54.1 | 190.1 |

| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 12.6 | 11.6 |
| [8×4096]·[4096×4096] | 37.9 | 37.9 |
| [64×1024]·[1024×1024] | 138.4 | 98.5 |
| [256×768]·[768×3072] | 148.4 | 150.5 |
| [1024×1024]·[1024×1024]ᵀ (view) | 173.7 | 155.4 |

### Small products (PyTorch)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 6.3 µs | 10.5 |
| 64² | 14.1 µs | 37.3 |
| 96² | 20.6 µs | 86.0 |
| 128² | 33.2 µs | 126.3 |
| 160² | 62.9 µs | 130.3 |
| 192² | 82.7 µs | 171.1 |
| 256² | 227.6 µs | 147.4 |
| [64×24]·[24×16] | 5.9 µs | 8.4 |
| [64×16]·[16×3] | 5.4 µs | 1.1 |
| [256×24]·[24×16] | 10.3 µs | 19.1 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 277.4 µs | 230.7K |
| forward + backward + Adam step | 2.04 ms | 31.3K |

### Element-wise

| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---:|---:|---:|---:|---:|
| x + y | 64K | 55.0 µs | 14.3 | 29.1 µs | 27.0 |
| x * y | 64K | 54.2 µs | 14.5 | 33.3 µs | 23.6 |
| x + row (broadcast) | 64K | 65.9 µs | 8.0 | 21.2 µs | 24.7 |
| x * 2.5 | 64K | 27.8 µs | 18.8 | 25.5 µs | 20.6 |
| exp(x) | 64K | 151.9 µs | 3.5 | 27.7 µs | 18.9 |
| tanh(x) | 64K | 373.4 µs | 1.4 | 73.3 µs | 7.2 |
| relu(x) | 64K | 49.7 µs | 10.5 | 20.1 µs | 26.1 |
| x + y | 1M | 808.4 µs | 15.6 | 177.3 µs | 71.0 |
| x * y | 1M | 812.0 µs | 15.5 | 175.8 µs | 71.6 |
| x + row (broadcast) | 1M | 986.1 µs | 8.5 | 124.5 µs | 67.4 |
| x * 2.5 | 1M | 451.7 µs | 18.6 | 167.8 µs | 50.0 |
| exp(x) | 1M | 2.68 ms | 3.1 | 233.2 µs | 36.0 |
| tanh(x) | 1M | 5.99 ms | 1.4 | 1.07 ms | 7.8 |
| relu(x) | 1M | 823.6 µs | 10.2 | 118.5 µs | 70.8 |
| x + y | 16M | 37.45 ms | 5.4 | 39.62 ms | 5.1 |
| x * y | 16M | 45.93 ms | 4.4 | 37.42 ms | 5.4 |
| x + row (broadcast) | 16M | 41.57 ms | 3.2 | 30.75 ms | 4.4 |
| x * 2.5 | 16M | 31.85 ms | 4.2 | 24.29 ms | 5.5 |
| exp(x) | 16M | 76.58 ms | 1.8 | 26.48 ms | 5.1 |
| tanh(x) | 16M | 122.09 ms | 1.1 | 39.02 ms | 3.4 |
| relu(x) | 16M | 34.72 ms | 3.9 | 23.40 ms | 5.7 |

### Reductions

| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---|---:|---:|---:|---:|
| sum() | [4096×4096] | 9.59 ms | 7.0 | 1.97 ms | 34.1 |
| sum(dim=0) | [4096×4096] | 8.55 ms | 7.8 | 6.32 ms | 10.6 |
| sum(dim=1) | [4096×4096] | 10.28 ms | 6.5 | 3.44 ms | 19.5 |
| max(dim=1) | [4096×4096] | 8.24 ms | 8.1 | 5.62 ms | 11.9 |
| softmax(dim=1) | [4096×4096] | – | – | 33.55 ms | 4.0 |
| layernorm | [4096×4096] | – | – | 26.63 ms | 5.0 |
| transpose+contiguous | [4096×4096] | 218.92 ms | 0.6 | 133.84 ms | 1.0 |

### Attention (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 37.66 ms | 114.1 |
| same with causal mask | 38.14 ms | 112.6 |
| [1×8×2048×64] long sequence | 66.83 ms | 128.5 |

### Convolution (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 59.89 ms | 123.5 |

### MLP 784→512→512→10, batch 256 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 2.53 ms | 101.1K |
| forward + backward | 5.88 ms | 43.6K |
| forward + backward + Adam step | 8.00 ms | 32.0K |

### EmbeddingGemma (sentence-transformers, fp32, 32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m

