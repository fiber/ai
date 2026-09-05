## Python — darwin/arm64, NumPy 2.0.2, PyTorch 2.8.0 (6 threads), Python 3.9.6

torch BLAS backend: - Build settings: BLAS_INFO=accelerate, BUILD_TYPE=Release, COMMIT_SHA=a1cb3cc05d46d198467bebbb6e8fba50a325d4e7, CXX_COMPILER=/usr/bin/c++, CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_PYTORCH_QNNPACK -DAT_BUILD_ARM_VEC256_WITH_SLEEF -DUSE_XNNPACK -DUSE_PYTORCH_METAL_EXPORT -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DUSE_COREML_DELEGATE -O2 -fPIC -DC10_NODEPRECATED -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=braced-scalar-init -Werror=range-loop-construct -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wvla-extension -Wsuggest-override -Wnewline-eof -Winconsistent-missing-override -Winconsistent-missing-destructor-override -Wno-pass-failed -Wno-error=old-style-cast -Wconstant-conversion -Qunused-arguments -faligned-new -fno-math-errno -fno-trapping-math -Werror=format -DUSE_MPS -Wno-missing-braces, LAPACK_INFO=accelerate, TORCH_VERSION=2.8.0, USE_CUDA=OFF, USE_CUDNN=OFF, USE_CUSPARSELT=OFF, USE_EIGEN_FOR_BLAS=ON, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF, USE_XCCL=OFF, USE_XPU=OFF,

### Matrix multiply (float32, GFLOPS)

| n (n×n · n×n) | NumPy | PyTorch 1 thread | PyTorch all threads |
|---:|---:|---:|---:|
| 128 | 754.5 | 796.7 | 783.3 |
| 256 | 1133.7 | 1117.9 | 1133.3 |
| 512 | 2141.6 | 2152.9 | 2130.6 |
| 1024 | 2693.4 | 2689.8 | 2664.9 |
| 2048 | 2241.4 | 2208.6 | 2242.9 |

| shape | NumPy GFLOPS | PyTorch GFLOPS |
|---|---:|---:|
| [1×4096]·[4096×4096] | 11.1 | 11.3 |
| [8×4096]·[4096×4096] | 69.5 | 68.0 |
| [64×1024]·[1024×1024] | 1291.7 | 1275.0 |
| [256×768]·[768×3072] | 2389.2 | 2357.0 |
| [1024×1024]·[1024×1024]ᵀ (view) | 2430.5 | 2397.7 |

### Element-wise

| op | n | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---:|---:|---:|---:|---:|
| x + y | 64K | 7.1 µs | 110.8 | 30.9 µs | 25.5 |
| x * y | 64K | 6.8 µs | 116.4 | 30.8 µs | 25.5 |
| x + row (broadcast) | 64K | 9.6 µs | 54.5 | 31.0 µs | 16.9 |
| x * 2.5 | 64K | 4.5 µs | 116.1 | 31.0 µs | 16.9 |
| exp(x) | 64K | 103.2 µs | 5.1 | 46.4 µs | 11.3 |
| tanh(x) | 64K | 61.7 µs | 8.5 | 78.7 µs | 6.7 |
| relu(x) | 64K | 22.1 µs | 23.7 | 30.7 µs | 17.1 |
| x + y | 1M | 100.3 µs | 125.4 | 71.9 µs | 175.0 |
| x * y | 1M | 98.3 µs | 128.0 | 67.3 µs | 187.1 |
| x + row (broadcast) | 1M | 136.6 µs | 61.4 | 65.2 µs | 128.7 |
| x * 2.5 | 1M | 64.2 µs | 130.7 | 67.2 µs | 124.8 |
| exp(x) | 1M | 1.63 ms | 5.1 | 216.3 µs | 38.8 |
| tanh(x) | 1M | 986.2 µs | 8.5 | 723.0 µs | 11.6 |
| relu(x) | 1M | 347.4 µs | 24.1 | 68.4 µs | 122.6 |
| x + y | 16M | 2.25 ms | 89.4 | 1.37 ms | 146.7 |
| x * y | 16M | 2.24 ms | 90.0 | 1.37 ms | 147.1 |
| x + row (broadcast) | 16M | 2.86 ms | 46.9 | 844.2 µs | 159.0 |
| x * 2.5 | 16M | 1.45 ms | 92.6 | 854.8 µs | 157.0 |
| exp(x) | 16M | 26.28 ms | 5.1 | 3.01 ms | 44.6 |
| tanh(x) | 16M | 15.61 ms | 8.6 | 11.61 ms | 11.6 |
| relu(x) | 16M | 5.48 ms | 24.5 | 852.1 µs | 157.5 |

### Reductions

| op | shape | NumPy time | NumPy GB/s | PyTorch time | PyTorch GB/s |
|---|---|---:|---:|---:|---:|
| sum() | [4096×4096] | 2.76 ms | 24.3 | 593.0 µs | 113.2 |
| sum(dim=0) | [4096×4096] | 1.32 ms | 50.8 | 2.14 ms | 31.3 |
| sum(dim=1) | [4096×4096] | 2.73 ms | 24.6 | 575.5 µs | 116.6 |
| max(dim=1) | [4096×4096] | 1.17 ms | 57.4 | 1.14 ms | 58.8 |
| softmax(dim=1) | [4096×4096] | – | – | 6.47 ms | 20.7 |
| layernorm | [4096×4096] | – | – | 2.35 ms | 57.2 |
| transpose+contiguous | [4096×4096] | 60.20 ms | 2.2 | 22.66 ms | 5.9 |

### MLP 784→512→512→10, batch 256 (PyTorch)

| phase | time / batch | samples/s |
|---|---:|---:|
| forward (no_grad) | 481.1 µs | 532.1K |
| forward + backward | 1.17 ms | 219.3K |
| forward + backward + Adam step | 2.21 ms | 116.1K |

