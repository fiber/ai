## Python — linux/x86_64, NumPy 2.5.3, PyTorch 2.14.0+cpu (32 threads), Python 3.12.3

torch BLAS backend: - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=08187d9e0fba026dc8217405802ab5381dc88d90, CUDA_FLAGS= -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS -Xfatbin -compress-all  -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUB_WRAPPED_NAMESPACE=at_cuda_detail -DDISABLE_CUSPARSE_DEPRECATED -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -DC10_NODEPRECATED, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_PYTORCH_QNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, TORCH_VERSION=2.14.0, USE_CUDA=0, USE_CUDNN=OFF, USE_CUSPARSELT=OFF, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_HIPSPARSELT=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF,

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 139.3 | 155.4 | 236.1 |
| 256 | 468.7 | 158.6 | 626.5 |
| 512 | 717.0 | 162.5 | 990.9 |
| 1024 | 1303.0 | 169.4 | 1434.3 |
| 2048 | 881.3 | 173.2 | 779.9 |

| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 10.9 | 11.0 |
| [8×4096]·[4096×4096] | 35.9 | 58.1 |
| [64×1024]·[1024×1024] | 594.7 | 698.7 |
| [256×768]·[768×3072] | 1087.6 | 980.1 |
| [1024×1024]·[1024×1024]ᵀ (view) | 1292.3 | 1034.4 |

### Element-wise

| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---:|---:|---:|---:|---:|
| x + y | 64K | 16.3 µs | 48.1 | 17.8 µs | 44.2 |
| x * y | 64K | 16.2 µs | 48.6 | 15.5 µs | 50.6 |
| x + row (broadcast) | 64K | 27.0 µs | 19.4 | 18.9 µs | 27.7 |
| x * 2.5 | 64K | 7.8 µs | 66.9 | 18.0 µs | 29.2 |
| exp(x) | 64K | 42.1 µs | 12.5 | 19.0 µs | 27.5 |
| tanh(x) | 64K | 26.8 µs | 19.6 | 21.2 µs | 24.8 |
| relu(x) | 64K | 39.9 µs | 13.1 | 14.9 µs | 35.2 |
| x + y | 1M | 505.8 µs | 24.9 | 24.1 µs | 521.4 |
| x * y | 1M | 362.0 µs | 34.8 | 20.9 µs | 601.9 |
| x + row (broadcast) | 1M | 411.2 µs | 20.4 | 22.8 µs | 367.7 |
| x * 2.5 | 1M | 225.7 µs | 37.2 | 22.0 µs | 381.6 |
| exp(x) | 1M | 667.1 µs | 12.6 | 38.3 µs | 219.1 |
| tanh(x) | 1M | 420.7 µs | 19.9 | 54.2 µs | 154.7 |
| relu(x) | 1M | 638.8 µs | 13.1 | 19.2 µs | 437.0 |
| x + y | 16M | 26.66 ms | 7.6 | 17.48 ms | 11.5 |
| x * y | 16M | 26.60 ms | 7.6 | 17.44 ms | 11.5 |
| x + row (broadcast) | 16M | 23.35 ms | 5.7 | 13.67 ms | 9.8 |
| x * 2.5 | 16M | 20.20 ms | 6.6 | 13.71 ms | 9.8 |
| exp(x) | 16M | 24.55 ms | 5.5 | 13.84 ms | 9.7 |
| tanh(x) | 16M | 22.09 ms | 6.1 | 13.93 ms | 9.6 |
| relu(x) | 16M | 24.32 ms | 5.5 | 13.74 ms | 9.8 |

### Reductions

| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---|---:|---:|---:|---:|
| sum() | [4096×4096] | 4.79 ms | 14.0 | 2.42 ms | 27.8 |
| sum(dim=0) | [4096×4096] | 4.80 ms | 14.0 | 6.29 ms | 10.7 |
| sum(dim=1) | [4096×4096] | 4.87 ms | 13.8 | 2.41 ms | 27.9 |
| max(dim=1) | [4096×4096] | 5.03 ms | 13.3 | 2.46 ms | 27.3 |
| softmax(dim=1) | [4096×4096] | – | – | 14.40 ms | 9.3 |
| layernorm | [4096×4096] | – | – | 14.12 ms | 9.5 |
| transpose+contiguous | [4096×4096] | 819.06 ms | 0.2 | 68.53 ms | 2.0 |

### MLP 784→512→512→10, batch 256 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 708.5 µs | 361.3K |
| forward + backward | 2.06 ms | 124.0K |
| forward + backward + Adam step | 3.63 ms | 70.6K |

