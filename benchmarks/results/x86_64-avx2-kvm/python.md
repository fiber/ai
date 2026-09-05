## Python — linux/x86_64, NumPy 2.5.2, PyTorch 2.14.0+cpu (6 threads), Python 3.12.3

torch BLAS backend: - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=08187d9e0fba026dc8217405802ab5381dc88d90, CUDA_FLAGS= -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS -Xfatbin -compress-all  -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUB_WRAPPED_NAMESPACE=at_cuda_detail -DDISABLE_CUSPARSE_DEPRECATED -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -DC10_NODEPRECATED, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TORCH_VERSION=2.14.0, USE_CUDA=0, USE_CUDNN=OFF, USE_CUSPARSELT=OFF, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_HIPSPARSELT=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF,

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 87.8 | 30.1 | 68.1 |
| 256 | 149.5 | 44.1 | 159.5 |
| 512 | 178.9 | 48.0 | 182.9 |
| 1024 | 145.7 | 53.8 | 169.9 |
| 2048 | 190.7 | 42.0 | 144.7 |

| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 14.2 | 16.0 |
| [8×4096]·[4096×4096] | 51.7 | 47.4 |
| [64×1024]·[1024×1024] | 134.6 | 132.8 |
| [256×768]·[768×3072] | 171.8 | 148.4 |
| [1024×1024]·[1024×1024]ᵀ (view) | 143.6 | 117.0 |

### Element-wise

| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---:|---:|---:|---:|---:|
| x + y | 64K | 40.1 µs | 19.6 | 40.7 µs | 19.3 |
| x * y | 64K | 54.1 µs | 14.5 | 26.6 µs | 29.6 |
| x + row (broadcast) | 64K | 67.0 µs | 7.8 | 23.1 µs | 22.7 |
| x * 2.5 | 64K | 29.6 µs | 17.7 | 30.4 µs | 17.3 |
| exp(x) | 64K | 178.6 µs | 2.9 | 24.5 µs | 21.4 |
| tanh(x) | 64K | 368.3 µs | 1.4 | 80.8 µs | 6.5 |
| relu(x) | 64K | 56.8 µs | 9.2 | 22.6 µs | 23.2 |
| x + y | 1M | 851.9 µs | 14.8 | 270.1 µs | 46.6 |
| x * y | 1M | 761.9 µs | 16.5 | 204.0 µs | 61.7 |
| x + row (broadcast) | 1M | 704.4 µs | 11.9 | 93.2 µs | 90.0 |
| x * 2.5 | 1M | 441.1 µs | 19.0 | 162.2 µs | 51.7 |
| exp(x) | 1M | 3.19 ms | 2.6 | 205.2 µs | 40.9 |
| tanh(x) | 1M | 5.91 ms | 1.4 | 1.06 ms | 7.9 |
| relu(x) | 1M | 834.0 µs | 10.1 | 148.5 µs | 56.5 |
| x + y | 16M | 42.28 ms | 4.8 | 34.96 ms | 5.8 |
| x * y | 16M | 71.86 ms | 2.8 | 27.54 ms | 7.3 |
| x + row (broadcast) | 16M | 37.89 ms | 3.5 | 25.03 ms | 5.4 |
| x * 2.5 | 16M | 32.19 ms | 4.2 | 26.36 ms | 5.1 |
| exp(x) | 16M | 68.63 ms | 2.0 | 35.60 ms | 3.8 |
| tanh(x) | 16M | 119.61 ms | 1.1 | 44.18 ms | 3.0 |
| relu(x) | 16M | 34.80 ms | 3.9 | 26.84 ms | 5.0 |

### Reductions

| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---|---:|---:|---:|---:|
| sum() | [4096×4096] | 9.16 ms | 7.3 | 2.43 ms | 27.7 |
| sum(dim=0) | [4096×4096] | 9.71 ms | 6.9 | 6.51 ms | 10.3 |
| sum(dim=1) | [4096×4096] | 9.92 ms | 6.8 | 1.85 ms | 36.2 |
| max(dim=1) | [4096×4096] | 9.50 ms | 7.1 | 4.41 ms | 15.2 |
| softmax(dim=1) | [4096×4096] | – | – | 45.87 ms | 2.9 |
| layernorm | [4096×4096] | – | – | 30.05 ms | 4.5 |
| transpose+contiguous | [4096×4096] | 187.91 ms | 0.7 | 134.17 ms | 1.0 |

### MLP 784→512→512→10, batch 256 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 2.20 ms | 116.1K |
| forward + backward | 5.66 ms | 45.2K |
| forward + backward + Adam step | 8.24 ms | 31.1K |

