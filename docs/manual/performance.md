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

## Heap ballast

Element-wise operations allocate their result. With a small live heap
Go's collector runs every few operations, returns freed results to the
operating system, and the next result faults its pages back in and is
zero-filled — that, not arithmetic, dominated small and medium tensors
(`x + y` on 64K elements: 15 µs, of which ~3 µs compute). The library
therefore keeps a 128 MiB *ballast* allocation alive at start-up. Its
pages are never touched (address space, not resident memory), but the
collector sizes the heap relative to live data, so it runs less often
and freed buffers get reused by the allocator. Measured on the M2 Pro:
`x + y` 64K 14.7 → 9.9 µs, 1M 194 → 129 µs, `relu` 1M 205 → 108 µs, an
MLP training step +11 %; large results (16M elements) and GEMM are
unchanged. The cost is up to roughly the ballast's size of garbage
between collections.

`tensor.SetHeapBallast(bytes)` changes it (0 removes it; the environment
variable `FIBERAI_HEAP_BALLAST` overrides the default before start-up);
`tensor.HeapBallast()` reads it. Raise it for services that churn many
medium-sized tensors, remove it in memory-constrained processes.

## Off-heap results

Go zero-fills every allocation on the allocating goroutine, and fresh
pages fault in one at a time. On a 16-core Xeon a 1M-element result cost
647 µs that way (16M: 10 ms, about 6.5 GB/s) while the addition itself
takes a few tens of microseconds across the cores; `x + y` on 1M elements
was three quarters allocation. Results of 128 KiB and more therefore do
not live on the Go heap: they are mapped with `mmap` (on Linux with a
request for transparent huge pages). Nothing zero-fills them; the kernel
hands out pages on first touch, which happens inside the parallel
kernels, spread over all cores.

Mappings of results that became unreachable come back through a GC
cleanup into a size-classed free list and are reused for the next result
of that size — still mapped, already faulted in, cache-warm. Because
off-heap memory does not count toward Go's heap goal, the library runs a
collection itself when the free list is empty and the outstanding mapped
memory has doubled since the last one (at least 256 MiB), and waits for
its cleanups before mapping more. The free list keeps at most 512 MiB
(least recently used size classes are unmapped first).

`tensor.SetMappedLimit(bytes)` changes that retention limit; a negative
value disables off-heap results altogether (environment:
`FIBERAI_MAPPED_LIMIT`). `tensor.MappedStats()` reports hits, misses,
retained and pinned bytes. Measured on the M2 Pro, where the heap path
was already cheap: `x + y` 1M 114 → 85 µs, 16M 2.25 → 1.69 ms, `relu` 1M
112 → 71 µs, softmax [4096×4096] 2.88 → 2.24 ms, layer norm 3.58 → 2.50
ms, MLP forward+backward 65K → 75K samples/s. On a Xeon Gold 6130 (one
socket, 16 cores), whose allocation cost was fifteen times the M2's:
`x + y` 1M 890 → 496 µs (28.8 µs with `Release()`, PyTorch 24), 64K 194
→ 49 µs (8.8 released, PyTorch 17.8), 16M 22.8 → 13.9 ms (PyTorch 17.5),
layer norm 36 → 18.5 ms, MLP inference 80K → 176K samples/s. See
BENCHMARKS.md.

What the collector cannot give back is cache residency. PyTorch's
reference counting frees a discarded result the moment it is dropped, so
a loop like `z = x + y` keeps writing the same buffer and, for tensors up
to a few megabytes, never leaves the L2/L3 cache: 24 µs for 1M elements
on a Xeon whose memory sustains 25 GB/s (that would be 500 µs from
DRAM). Between two collections we rotate through tens of buffers and pay
the memory round trip. `t.Release()` closes that gap where it matters:
it hands the storage back immediately, and the next result of that size
gets it while it is still in cache. Release is a no-op whenever the
storage might still be needed (a view exists, `Data()` was taken,
autograd recorded the tensor), so library code calls it on every
intermediate it produces (`nn.Linear`, `nn.Sequential`), which pays off
in `NoGrad` inference; call it yourself on discarded results in hot
loops. The tensor must not be used after `Release()`. The other half of
cache residency is that the same core handles the same slice every time:
`parallel.Range` assigns chunks owner-first for that reason (see
[internals.md](internals.md)).

Two things to know. `Data()` on a contiguous tensor hands out the mapped
slice, so its storage is pinned for good (never unmapped, never reused).
And a slice obtained from a mapped tensor is not a Go pointer: it keeps
nothing alive. Inside the library every operation keeps its input tensors
reachable until it is done with their data; user code that holds a slice
from `Data()` is safe because of the pinning, but code that reaches into
`tensor` internals must follow the same rule.

## Heap storage reuse (opt-in, small results)

Below 128 KiB results stay on the Go heap. `tensor.SetPoolLimit(bytes)`
turns on recycling for them: when a tensor and all its views become
unreachable, a GC cleanup returns buffers between 4 KiB and 128 KiB to a
pool, and the next result of similar size reuses memory instead of paying
for a zero-filled allocation. `tensor.PoolStats()` reports hits, misses
and the bytes held.

It is off by default because the measured effect depends on the workload:
on the M2 Pro it halves the time of element-wise operations on 64K–1M
elements (`x + y` 64K: 26 → 12 µs) but slowed a training step of
medium-sized layers by ~10 %, because the retained memory raises the
garbage collector's heap goal and buffers only come back after a GC
cycle. Enable it for inference over many small tensors; leave it off for
training. Regardless of the pool, results that an operation writes
completely are no longer zero-filled.

`Data()` on a contiguous tensor hands out the backing slice and therefore
pins its storage for good; it is never recycled while the program runs.
Use `Float32s()` (copy) or `At` when you only read.

## Advice

1. Prefer one big operation over many small ones. Every operation
   allocates its result; below ~64K elements the fixed cost (allocation,
   goroutine wake-up) dominates.
2. On hot paths the in-place family (`AddInPlace`, `AddScaledInPlace`,
   `CopyFrom`) avoids any allocation; a fresh result of tens of MB costs
   its page faults once, after that the mapping is reused.
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
