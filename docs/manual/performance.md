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

- **Matrix products** run at 84–90 % of a core's FMA peak on one core
  (Apple M2 Pro ~100 GFLOPS, M4 121, Xeon Gold 6130 with AVX-512 150–162)
  and scale to ~600 GFLOPS on the ten Apple cores and ~1 150 GFLOPS on a
  16-core Skylake-SP socket (n=1024–2048), which is ahead of NumPy/
  OpenBLAS and PyTorch/MKL at n=2048 on that machine and 20 % behind MKL
  at n=1024. Transposed operands cost nothing — pass `w.T()`, do not
  materialise it.

  On x86 the AVX-512 kernel is chosen where available (Skylake-SP and
  later Xeons, Zen 4/5), otherwise AVX2. `KC=512` and a fine compute grid
  are the tuned defaults; `FIBERAI_BLAS_KC/MC/NC` and `FIBERAI_BLAS_TASKS`
  override them for tuning runs. On a two-socket machine pin the process
  to one socket (`numactl --cpunodebind=0 --membind=0`) until topology-
  aware scheduling exists (TODO T-013); running on both sockets is slower
  than one.
- **Element-wise operations** on large tensors run at memory bandwidth.
  `exp` is a vectorised kernel; `Softmax`, `CrossEntropy` and `Sigmoid`
  use it. `Tanh`, `GELU` and `Log` still call `math.*` per element and are
  the slowest operations in the library.
- **Reductions** along the last dimension run at memory bandwidth; along
  other dimensions they fold rows with per-goroutine partial results.

## Storage reuse (opt-in)

`tensor.SetPoolLimit(bytes)` turns on recycling of freed tensor storage:
when a tensor and all its views become unreachable, a GC cleanup returns
buffers between 4 KiB and 4 MiB to a pool, and the next result of similar
size reuses memory that is still mapped and cache-warm instead of paying
for a zero-filled allocation and fresh page faults. `tensor.PoolStats()`
reports hits, misses and the bytes held.

It is off by default because the measured effect depends on the workload:
on the M2 Pro it halves the time of element-wise operations on 64K–1M
elements (`x + y` 64K: 26 → 12 µs) but slowed a training step of
medium-sized layers by ~10 %, because the retained memory raises the
garbage collector's heap goal and buffers only come back after a GC
cycle. Enable it for inference over many small tensors; leave it off for
training. Regardless of the pool, results that an operation writes
completely are no longer zero-filled.

`Data()` on a contiguous tensor hands out the backing slice and therefore
pins its storage for good — it is never recycled while the program runs.
Use `Float32s()` (copy) or `At` when you only read.

## Advice

1. Prefer one big operation over many small ones. Every operation
   allocates its result; below ~64K elements the fixed cost (allocation,
   goroutine wake-up) dominates.
2. On hot paths the in-place family (`AddInPlace`, `AddScaledInPlace`,
   `CopyFrom`) avoids even the pooled allocation; results of tens of MB
   still cost a pass over memory when the pool has nothing of that size.
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
