## Python — linux/x86_64, NumPy 2.5.2, PyTorch 2.14.0+cpu (6 threads), Python 3.12.3

torch BLAS backend: - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=08187d9e0fba026dc8217405802ab5381dc88d90, CUDA_FLAGS= -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS -Xfatbin -compress-all  -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUB_WRAPPED_NAMESPACE=at_cuda_detail -DDISABLE_CUSPARSE_DEPRECATED -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -DC10_NODEPRECATED, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TORCH_VERSION=2.14.0, USE_CUDA=0, USE_CUDNN=OFF, USE_CUSPARSELT=OFF, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_HIPSPARSELT=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF,

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 91.9 | 43.0 | 119.9 |
| 256 | 177.5 | 55.9 | 158.0 |
| 512 | 174.6 | 55.7 | 195.5 |
| 1024 | 211.3 | 50.9 | 196.0 |
| 2048 | 161.8 | 39.3 | 180.1 |

| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 11.5 | 9.0 |
| [8×4096]·[4096×4096] | 51.5 | 41.6 |
| [64×1024]·[1024×1024] | 134.8 | 101.5 |
| [256×768]·[768×3072] | 172.5 | 164.2 |
| [1024×1024]·[1024×1024]ᵀ (view) | 170.1 | 150.1 |

### Small products (PyTorch)

| shape | time | GFLOPS |
|---|---:|---:|
| 32² | 7.0 µs | 9.4 |
| 64² | 14.2 µs | 36.8 |
| 96² | 29.9 µs | 59.2 |
| 128² | 33.4 µs | 125.7 |
| 160² | 68.5 µs | 119.6 |
| 192² | 101.6 µs | 139.4 |
| 256² | 212.4 µs | 158.0 |
| [64×24]·[24×16] | 6.5 µs | 7.6 |
| [64×16]·[16×3] | 5.5 µs | 1.1 |
| [256×24]·[24×16] | 9.0 µs | 21.9 |

### Tiny autoencoder 24→16→3→16→24, batch 64 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 262.4 µs | 243.9K |
| forward + backward + Adam step | 2.05 ms | 31.2K |

### Element-wise

| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---:|---:|---:|---:|---:|
| x + y | 64K | 36.0 µs | 21.9 | 26.9 µs | 29.3 |
| x * y | 64K | 34.9 µs | 22.5 | 28.4 µs | 27.7 |
| x + row (broadcast) | 64K | 47.5 µs | 11.0 | 22.6 µs | 23.2 |
| x * 2.5 | 64K | 28.4 µs | 18.5 | 29.7 µs | 17.7 |
| exp(x) | 64K | 171.9 µs | 3.1 | 26.7 µs | 19.6 |
| tanh(x) | 64K | 366.0 µs | 1.4 | 73.4 µs | 7.1 |
| relu(x) | 64K | 51.1 µs | 10.3 | 21.6 µs | 24.3 |
| x + y | 1M | 981.6 µs | 12.8 | 181.7 µs | 69.3 |
| x * y | 1M | 964.2 µs | 13.0 | 197.3 µs | 63.8 |
| x + row (broadcast) | 1M | 984.7 µs | 8.5 | 92.0 µs | 91.2 |
| x * 2.5 | 1M | 408.5 µs | 20.5 | 181.4 µs | 46.2 |
| exp(x) | 1M | 2.52 ms | 3.3 | 244.6 µs | 34.3 |
| tanh(x) | 1M | 5.88 ms | 1.4 | 895.9 µs | 9.4 |
| relu(x) | 1M | 1.00 ms | 8.4 | 91.2 µs | 92.0 |
| x + y | 16M | 39.47 ms | 5.1 | 26.07 ms | 7.7 |
| x * y | 16M | 43.88 ms | 4.6 | 24.03 ms | 8.4 |
| x + row (broadcast) | 16M | 41.71 ms | 3.2 | 24.38 ms | 5.5 |
| x * 2.5 | 16M | 29.85 ms | 4.5 | 24.73 ms | 5.4 |
| exp(x) | 16M | 72.13 ms | 1.9 | 26.27 ms | 5.1 |
| tanh(x) | 16M | 118.22 ms | 1.1 | 40.29 ms | 3.3 |
| relu(x) | 16M | 38.39 ms | 3.5 | 24.35 ms | 5.5 |

### Reductions

| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---|---:|---:|---:|---:|
| sum() | [4096×4096] | 9.42 ms | 7.1 | 1.82 ms | 36.8 |
| sum(dim=0) | [4096×4096] | 8.33 ms | 8.1 | 6.22 ms | 10.8 |
| sum(dim=1) | [4096×4096] | 10.04 ms | 6.7 | 2.79 ms | 24.0 |
| max(dim=1) | [4096×4096] | 9.87 ms | 6.8 | 6.55 ms | 10.2 |
| softmax(dim=1) | [4096×4096] | – | – | 49.31 ms | 2.7 |
| layernorm | [4096×4096] | – | – | 26.73 ms | 5.0 |
| transpose+contiguous | [4096×4096] | 203.77 ms | 0.7 | 123.84 ms | 1.1 |

### Attention (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [8×8×512×64] q·kᵀ, softmax, ·v | 35.30 ms | 121.7 |
| same with causal mask | 38.68 ms | 111.0 |
| [1×8×2048×64] long sequence | 63.28 ms | 135.7 |

### Convolution (PyTorch, no_grad)

| shape | time | GFLOPS |
|---|---:|---:|
| [32×64×56×56] · 64 filters 3×3, pad 1 | 64.73 ms | 114.3 |

### MLP 784→512→512→10, batch 256 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 2.83 ms | 90.3K |
| forward + backward | 5.83 ms | 43.9K |
| forward + backward + Adam step | 8.60 ms | 29.8K |

### EmbeddingGemma (sentence-transformers, fp32, 32 sentences ≈ 64 tokens, one batch)

skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m

