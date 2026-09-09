# Performance

## Back-ends

At start-up the kernel layer detects the CPU and picks the best available
implementation: `neon` on arm64, `avx512` or `avx2` on amd64, `generic`
(portable Go) otherwise. Before a SIMD implementation is activated it is
verified against the Go version on random inputs; a failing kernel is
disabled and reported by `tensor.BackendWarnings()`.

| Variable | Effect |
|---|---|
| `FIBERAI_KERNEL=generic\|avx2\|avx512\|neon\|amx` | select an implementation among those the CPU supports (benchmarking, debugging) |
| `FIBERAI_AMX=0` / `=1` | Apple Silicon (macOS): matrix products on the AMX coprocessor, on by default for the M1–M4; `0` switches to NEON, `1` forces it on a newer chip (see below) |
| `FIBERAI_KERNEL_FORCE=1` | skip CPU feature detection for the selected implementation — only for emulators such as Rosetta 2 that hide features from CPUID |
| `GOMAXPROCS` | default goroutine limit; on Linux the default worker count is the number of physical cores in the process's CPU affinity mask (hyperthreads counted once), elsewhere GOMAXPROCS |
| `FIBERAI_WORKERS=n` | override the default worker count |
| `FIBERAI_BLAS_THRESHOLD=n` | multiply-adds below which a matrix product runs on one core (default 1 048 576) |

`tensor.SetThreads(n)` limits the goroutines used by tensor operations at
run time; `tensor.Threads()` reads it back.

### AMX on Apple Silicon

Apple's M1–M4 carry an undocumented matrix coprocessor, AMX, which is
what Accelerate (and therefore NumPy and PyTorch) use for SGEMM. On
those chips fiber/ai runs its GEMM micro-kernel on it by default: a
32×32 tile per call, four 16×16 f32 outer products per k-step,
software-pipelined over four register slots. Measured on the M2 Pro
against PyTorch/Accelerate:

| | fiber/ai NEON | fiber/ai AMX | PyTorch |
|---|---:|---:|---:|
| SGEMM 512², GFLOPS | 514 | 1 409 | 2 131 |
| SGEMM 1024² | 581 | 1 881 | 2 665 |
| SGEMM 2048² | 636 | **2 250** | 2 243 |
| [256×768]·[768×3072] | 500 | 1 562 | 2 357 |
| MLP training step, samples/s | 85 K | **146 K** | 116 K |
| MLP inference | 221 K | **569 K** | 532 K |

SME, the documented matrix extension of the M4, was measured with the
probe under `internal/kernel/smeprobe`: 1.06 TFLOPS on one thread and
1.4 on the cluster against 1.4 and 1.7 through the AMX instructions on
the same chip, and Go's preemption signals cost SME its Z registers
unless signals are masked around each kernel call. fiber/ai therefore
stays on AMX on the M4; the probe and the findings are in spec T-006
for the day a chip needs SME.

The instructions are undocumented and an unsupported chip would fault
with an illegal instruction that the start-up self-test cannot catch, so
the kernel enables itself only on the generations it was verified on
(M1–M4 by brand string; `FIBERAI_AMX=1` forces it on a newer chip,
`FIBERAI_AMX=0` switches back to NEON). AMX state belongs to the
OS thread, so the driver locks each worker goroutine to its thread for
the duration of a task and enables the state there; the element-wise
kernels stay NEON. The units sit with the performance cores, so the
back-end caps GEMM workers at their number. Not available on Linux/arm64.

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
  later Xeons, Zen 4/5), otherwise AVX2. Every back-end has an
  overwriting tile variant, so a product's output is written in its
  first K block rather than cleared and accumulated into. `KC=512` and a fine compute grid
  are the tuned defaults; `FIBERAI_BLAS_KC/MC/NC` and `FIBERAI_BLAS_TASKS`
  override them for tuning runs. On a two-socket machine pin the process
  to one socket (`numactl --cpunodebind=0 --membind=0`) until topology-
  aware scheduling exists (TODO T-013); running on both sockets is slower
  than one.
- **Element-wise operations** on large tensors run at memory bandwidth.
  `exp`, `tanh` and `log` are vectorised kernels (AVX2, NEON, and a Go
  version with the same arithmetic); `Sigmoid` and `GELU` are composed
  from `tanh` and the vector primitives, `Softmax` and `CrossEntropy`
  from `exp`. On the M2 Pro `tanh` over 1M elements went from 2.1 ms
  (`math.Tanh` per element) to ~110 µs, seven times faster than PyTorch
  there; the kernel alone does 0.4 ns per element on one core (NEON on
the M2 Pro and AVX2 on a Skylake-SP core alike, the latter at IPC 1.8).
When measuring such kernels use a time-based `-benchtime` of a second or
more: a run of a few milliseconds ends before the core reaches its turbo
clock and reports three times the real cost.
- **Reductions** along the last dimension run at memory bandwidth; along
  other dimensions they fold rows with per-goroutine partial results.
  `Softmax`, `LogSoftmax`, `CrossEntropy`, `LayerNorm` and, under
  `NoGrad`, `Attention` are fused row operations: layer norm keeps only mean and rstd per row for the
  backward pass and recomputes x̂ into an L1-resident scratch row, so a
  [4096×4096] layer norm reads its input once and writes the output once
  (M2 Pro 1.9 ms against PyTorch's 2.35).

## Packed operands: weights are packed once

The GEMM driver works on operands repacked into panel layout. In
inference the right operand of almost every product is a weight that
does not change between calls, so packing it every time is wasted work:
for a transformer projection [256×768]·[768×3072] it was a third of the
call. The tensor layer therefore keeps a cache of packed right operands.
An operand enters the cache the second time a product sees it (so an
activation used once never does) and stays valid while its storage is
unmodified: every in-place operation and `Set` invalidate it, and a
tensor whose buffer escaped through `Data()` is never cached, because a
caller could write through that slice at any time. Batched products
(attention) and operands under 64K elements are not cached.

What it is worth on the M2 Pro, right operand reused across calls:

| product | B packed per call | B packed once |
|---|---:|---:|
| [8×4096]·[4096×4096] (GFLOPS) | 75 | 250 |
| [64×1024]·[1024×1024] | 992 | 1 582 |
| [256×768]·[768×3072] | 1 806 | 2 594 |
| 1024² | 2 172 | 2 471 |
| MLP forward, batch 256 (samples/s) | 768 K | 839 K |
| EmbeddingGemma, 32 × 65 tokens (sentences/s) | 86 | 98 |

Training does not benefit: the weights change every step, and the
optimisers write them through `Data()`, so they are never cached; the
step costs the same as before within noise.

**Packing itself** is a register transpose on both architectures: a
row-contiguous block (the A operand of every product, a transposed B, the
query rows and probabilities of attention) goes into the k-major panel
through 4×4 NEON transposes on arm64 and 8×8 AVX2 transposes with masked
stores on amd64 (`packRows8`), so a 6-row group, the tail of the 14-row
AVX-512 tile and the whole of the AVX2 one, costs the same as a full
group. Before, amd64 streamed each row with a scalar store every MR
floats, and the pinned Xeon spent a third of the attention benchmark in
those loops (21 % `packA`, 11 % `packBPanel` in the profile).

**Eviction.** Least recently used, with one guard: an entry is only
evicted when it has not been hit within the last pass over the working
set (two lookups per entry). If a new operand cannot be fitted by
evicting such stale entries, it is not inserted and packs per call.
Plain LRU cycles when the working set is slightly larger than the cap,
every entry evicted just before it is needed again, which is what the
Xeon showed with EmbeddingGemma's 420 MB behind a 512 MiB cap and other
operands in front; with the guard, what fits stays.

Knobs and numbers: `tensor.SetPackedCacheLimit(bytes)` caps the memory
held (default 512 MiB; 0 disables and drops every entry),
`FIBERAI_PACK_CACHE=0` disables it at start-up,
`tensor.ResetPackedCache()` drops everything, and
`tensor.PackedCacheReport()` returns hits, misses, packs, evictions,
invalidations (the operand changed), refusals (no evictable space),
bytes and entries; `cmd/bench` prints that line after every section. A
steady-state inference run shows misses of about twice the number of
weights (first sighting, then the pack) and hits for everything after.

## Fused epilogues: bias, activation and gating on the hot block

In inference the work between two matrix products is element-wise: add
a bias, apply an activation, multiply by a gate, add the residual. Each
is a pass over the whole output; for a transformer block that meant
reading and writing a 4–6 MB matrix seven times for three products, and
on a server with a tenth of the M2's memory bandwidth those passes were
half of an EmbeddingGemma forward. `tensor.MatMulFused(x, w,
tensor.Fused{Bias, RowScale, Act, Mul, Residual})` applies them inside
the GEMM driver: once the last K block of an output region has been
accumulated, whichever task finished the region runs the epilogue on it
while it is still in cache. The order is bias, row scale, activation,
element-wise multiply, residual add.

The region is a row block (128 rows) by at least 256 columns. Narrower,
per micro-tile panel, and the vector kernels spend their time on
dispatch instead of data (GELU alone cost 5 % of the forward that way);
the whole row block, and one core runs the block's epilogue while the
others wait for it. The matrix-vector and few-rows paths apply the
epilogue over the finished output instead; correct everywhere, fused
where it matters.

Under gradient recording `MatMulFused` computes the same result from the
ordinary operations, so autograd sees what it always saw; the fusion is
an inference path. `nn.Linear` uses it for its bias and `nn.Sequential`
for a `ReLU` or `GELU` that directly follows a `Linear`, both under
`NoGrad`. `models/gemma` folds the pre-attention and pre-feed-forward
RMSNorms into the weights it loads (RMSNorm(x)·W equals
(1/rms(x)) · (x · diag(g)W)): the encoder computes one sum of squares
per row with `RowSumSquares`, a read-only pass, and passes the factor
as `RowScale`; the gated feed-forward is one call, `gate = h·W'gate`
with `Act: GELUAct, Mul: up`, instead of gate, GELU and product.

M2 Pro, same day, packed-operand cache warm:

| | separate passes | fused |
|---|---:|---:|
| MLP forward, batch 256 (samples/s) | 840 K | 875 K |
| EmbeddingGemma, 32 × 65 tokens (sentences/s) | 94 | 106 |

The MLP gains little: three products of 256 rows are dominated by
per-call overhead, not by the passes between them. The transformer gains
13 % on the M2, whose memory bandwidth was never the bottleneck, and
18 % on the Xeon (49 → 58 sentences/s, PyTorch 37), where it is.

## Attention from the micro-kernel

Without gradients recording, `Attention` and `AttentionScaled` do not go
through the GEMM driver. Per head, K and V are packed once into
micro-kernel panels (heads that share storage, as grouped-query `Expand`
views do, share one packing; the packed operands are held in groups of
at most 64 MB). Each task, a head and a block of query rows, packs its
query rows once and works through them in MR-row groups: the scores of
the group against every key go into L1-sized scratch straight from the
micro-kernel, the softmax runs on them, the probabilities are packed as
the left operand and multiplied with the packed V, and the rows leave
scaled by their 1/Σ, a D-wide pass instead of an S-wide one. The softmax
is three passes: the mask joins the raw scores, one `Max`, and
`kernel.ExpSum`, which applies the scale and the shift, exponentiates
and sums in one pass (NEON and AVX2 assembly). Before, each task ran two
general GEMMs of depth 64 with their own packing (K and V repacked for
every row block, the probabilities packed a second time after the
softmax) and a seven-pass softmax.

M2 Pro, GFLOPS over the two products, PyTorch 2.14 same day:

| shape | before | now | PyTorch |
|---|---:|---:|---:|
| [8×8×512×64] | 947 | 1 110 | 553 |
| same, causal mask | ~900 | 1 070 | 566 |
| [1×8×2048×64], long sequence | – | 1 120 | 652.5 |

What remains on the M2 is arithmetic, not overhead: half the task time
is the AMX products (the score product has depth 64, so every 32×32
tile pays its store after 64 steps), a third the exponential (16.8 M of
them per call at 0.4 ns each on one core), the rest the probability
packing and the row scaling. On the Xeon the same change gives 479 →
533 GFLOPS on the pinned socket and 437 → 747 across both (the new path
carries little memory traffic), against 1 141 for PyTorch; per core it
reaches 33 GFLOPS where the plain GEMM reaches 72, and the AVX2
exponential at about 1.2 ns per element is the first suspect. See
BENCHMARKS.md.

## Denormals and masks

x86 cores process denormal floats (below about 1.2e−38) through
microcode assists that cost around a hundred cycles per operation, and Go
does not set the flush-to-zero flags. Apple's cores handle denormals at
full speed, so a workload can look fine on a Mac and crawl on a Xeon. The
one place this bit fiber/ai was masked attention: a masked score gave
exp(−87) after clamping, the softmax normalisation turned it into a
denormal weight, and the weights·V product ran eleven times slower with a
causal mask than without. The exp kernels now return exactly 0 for inputs
below the clamp, so masked weights are true zeros on every back-end (Xeon Gold 6130, pinned socket: 470 GFLOPS with the causal mask against 479 without, from 39 before the fix). If a
model of your own produces values in that range (activations decaying
towards zero over many steps, tiny variances), expect the same effect and
clamp or rescale.

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
layer norm 36 → 11 ms, MLP inference 80K → 288K samples/s, training step
14K → 64K (PyTorch 71K). See BENCHMARKS.md.

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
loops. Under autograd, `Backward` does the same for the graph's
intermediates once their consumers have run (see
[autograd.md](autograd.md#what-backward-leaves-behind)); a training step
therefore keeps a working set of a few tens of MB instead of rotating
through the whole allocation budget. The tensor must not be used after `Release()`. The other half of
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
   on ten cores. On the M1–M4 fiber/ai uses the same unit by default
   (2.3 TFLOPS at n=2048 on the M2 Pro); an SME kernel for M4-class chips
   is planned (T-006).

## Measuring

```sh
go run ./cmd/bench                                   # Markdown tables
go test ./internal/blas -bench Gemm -run x           # GEMM by size and worker count
go test ./internal/kernel -bench . -run x            # individual kernels per implementation
FIBERAI_KERNEL=generic go run ./cmd/bench -quick     # the pure-Go baseline
```
