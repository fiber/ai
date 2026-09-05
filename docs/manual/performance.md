# Performance

## Back-ends

At start-up the kernel layer detects the CPU and picks the best available
implementation: `neon` on arm64, `avx512` or `avx2` on amd64, `generic`
(portable Go) otherwise. Before a SIMD implementation is activated it is
verified against the Go version on random inputs; a failing kernel is
disabled and reported by `tensor.BackendWarnings()`.

| Variable | Effect |
|---|---|
| `FIBERAI_KERNEL=generic\|avx2\|avx512\|neon` | select an implementation among those the CPU supports (benchmarking, debugging) |
| `FIBERAI_KERNEL_FORCE=1` | skip CPU feature detection for the selected implementation — only for emulators such as Rosetta 2 that hide features from CPUID |
| `GOMAXPROCS` | default goroutine limit |

`tensor.SetThreads(n)` limits the goroutines used by tensor operations at
run time; `tensor.Threads()` reads it back.

## What is fast

Numbers from `go run ./cmd/bench`; full tables and the NumPy/PyTorch
comparison in [BENCHMARKS.md](../../BENCHMARKS.md).

- **Matrix products** run at ~87 % of a core's FMA peak (Apple M2 Pro:
  ~100 GFLOPS per core, ~600 GFLOPS on 10 cores; Apple M4: 120 / 614).
  Transposed operands cost nothing — pass `w.T()`, do not materialise it.
- **Element-wise operations** on large tensors run at memory bandwidth.
  `exp` is a vectorised kernel; `Softmax`, `CrossEntropy` and `Sigmoid`
  use it. `Tanh`, `GELU` and `Log` still call `math.*` per element and are
  the slowest operations in the library.
- **Reductions** along the last dimension run at memory bandwidth; along
  other dimensions they fold rows with per-goroutine partial results.

## Advice

1. Prefer one big operation over many small ones. Every operation
   allocates its result; below ~64K elements the fixed cost (allocation,
   goroutine wake-up) dominates.
2. Reuse buffers on hot paths with the in-place family
   (`AddInPlace`, `AddScaledInPlace`, `CopyFrom`) — a fresh 64 MB result
   costs a zero-fill pass and page faults on top of the computation.
3. Keep batch dimensions leading and the reduction dimension last where you
   can; that is the layout the fused kernels are written for.
4. For matrix–vector shapes (`[1×k]·[k×n]`) the library takes a dedicated
   path that is as fast as anything else on the machine; for a handful of
   rows (`[8×k]`) it still packs `B` and is slower than it should be (see
   TODO.md T-005).
5. Large GEMMs on Apple Silicon: Accelerate/NumPy/PyTorch use the AMX or
   SME matrix unit and reach 2–2.7 TFLOPS; NEON tops out around 600 GFLOPS
   on ten cores. That gap is hardware, not code; an SME kernel for M4-class
   chips is planned (T-006).

## Measuring

```sh
go run ./cmd/bench                                   # Markdown tables
go test ./internal/blas -bench Gemm -run x           # GEMM by size and worker count
go test ./internal/kernel -bench . -run x            # individual kernels per implementation
FIBERAI_KERNEL=generic go run ./cmd/bench -quick     # the pure-Go baseline
```
