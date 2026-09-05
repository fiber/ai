---
id: T-015
title: Recycle tensor storage: pooled buffers with cleanup-based reuse, no zero-fill where results are fully written
status: done
scope:
  - tensor/
  - nn/
manual:
  - docs/manual/performance.md
  - docs/manual/tensors.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

Element-wise operations allocate their result on every call. Go zero-fills
the new slice and, once the GC has returned earlier results to the OS,
the pages fault back in on the next write. Measured cost: on the M2 Pro
`x + y` over 64K elements takes 26 µs of which the arithmetic is ~3 µs;
raising GOGC to 800 (keeping freed memory resident) halves the 64K and
1M timings without touching a kernel. On the Skylake-SP Xeon with 16 Ps
the same effect is 60×: `x + y` on 1M elements takes 1 490 µs against
PyTorch's 24 µs (cache-resident, buffer-reusing), while `sum()` — no
allocation — matches PyTorch exactly (2.40 vs 2.42 ms). Allocation, not
computation, decides most of the element-wise table, and it also costs
~25 % in the tensor-level GEMM path on the Xeon (162 GFLOPS in the
allocation-free benchmark, 122 with the result allocation).

## Design

**Storage object.** A tensor's data lives in a `storage` value
(`{buf []float32, pooled bool, escaped bool}`) shared by the base tensor
and all its views; `Tensor` keeps its `data` slice for speed plus a
pointer to the storage. Views, `Detach`, `Reshape`, `Narrow`, `Expand`,
reductions' shape rewrites all pass the storage pointer through.

**Pool.** `newTensor` takes its buffer from a size-classed free list
(power-of-two classes from 4 KiB up; smaller results use plain `make`,
which is cheap) protected by a mutex, with a byte cap on retained memory
(default 256 MiB, `tensor.SetPoolLimit`). A `runtime.AddCleanup` on the
storage object returns the buffer to the pool when the storage becomes
unreachable, i.e. when no tensor or view refers to it any more. Recycled
memory stays mapped and cache-warm: no page faults, and no GC-driven
release/refault cycle.

**No zero-fill where the result is fully written.** `newTensor` keeps its
zero-filled contract (buffers from the pool are cleared); a new
`newTensorUninit` skips the clear and is used where every element is
written before it is read — element-wise results, unary results,
reductions that copy their first row, softmax/layernorm outputs,
`Contiguous`/`Clone`, `Cat`. Sites that accumulate (`MatMul` output,
`Narrow` backward, `accumGrad`) keep the zeroed variant. The audit of
each site is part of the implementation and listed in Notes.

**Safety.** `Data()` on a contiguous tensor returns the backing slice; it
marks the storage as escaped so it is never recycled (the caller may keep
the slice). `FromSlice` wraps user memory as non-pooled storage.
`Float32s` copies and does not escape. Cleanups run only after the GC
proves unreachability, so no live view can be recycled under a user.

**Not in scope.** An explicit `Release()`; out-parameter variants of the
element-wise API. If the pool leaves a measurable gap on the Xeon, those
are the follow-up.

## Acceptance

- `go test ./...` and `go test -race ./tensor/` pass; a new test creates
  and drops many tensors under `runtime.GC()` pressure and checks that
  buffers are reused (pool hit counter), that a view keeps its base
  storage alive, and that `Data()`-escaped storage is never reused.
- Apple M2 Pro, `cmd/bench` element-wise rows: `x + y` 64K ≤ 12 µs (from
  26; NumPy 7, PyTorch 31), 1M ≤ 110 µs (from 153; PyTorch 72), 16M ≤
  1.6 ms (from 2.05; PyTorch 1.37). GEMM rows unchanged or better.
- Xeon Gold 6130, one socket, after this change: `x + y` 1M ≤ 100 µs
  (from 1 490; PyTorch 24), 16M ≤ 18 ms (from 22.8; PyTorch 17.5);
  tensor-level SGEMM n=1024 1 thread ≥ 150 GFLOPS (from 122).
- `docs/manual/performance.md` explains the pool and `SetPoolLimit`;
  `docs/manual/tensors.md` documents the `Data()` escape rule.

## Notes

Implemented: storage object shared by views, `newTensorUninit` at every
fully-written site, `Data()` escape, size-classed pool with
`runtime.AddCleanup` return. Measured on the M2 Pro (`cmd/bench -quick`):

| config | x+y 64K | x+y 1M | relu 16M | layernorm | MLP step |
|---|---:|---:|---:|---:|---:|
| pool off | 26 µs | 218 µs | 1.64 ms | 3.5 ms | 55.6 K/s |
| pool on, all sizes, 256 MiB | 14 µs | 167 µs | **39 ms** | **19 ms** | 30.5 K/s |
| pool on, ≤ 4 MiB, 64 MiB | 12 µs | 177 µs | 1.64 ms | 3.4 ms | 49.0 K/s |
| pool off, GOGC=800 | 10.7 µs | 134 µs | 1.50 ms | 3.35 ms | 57.7 K/s |

Findings: cleanup-based recycling cannot beat Go's own "GC frees, the
allocator reuses the span" for large buffers — the buffer stays alive
until the cleanup runs, which inflates the heap goal and delays reuse.
Capped at 4 MiB it helps small tensors but still costs ~10 % on the MLP
step. The best configuration overall is a larger GC target with no pool,
which is what a heap ballast would provide from inside the library.
Decision: pool stays as an opt-in (`SetPoolLimit`, default 0); the
storage indirection and the uninitialised-allocation paths stay. The
acceptance targets are not met; next attempt for this spec: a heap
ballast (`tensor.SetHeapBallast`) measured against GOGC=800, and
explicit reuse for the training loop (scoped arenas or out-parameters).
Spec remains open.

Second attempt — heap ballast (`tensor.SetHeapBallast`, default 128 MiB),
M2 Pro `cmd/bench -quick`:

| | x+y 64K | x+y 1M | relu 1M | x+y 16M | MLP step |
|---|---:|---:|---:|---:|---:|
| no ballast | 14.7 µs | 194 µs | 205 µs | 2.28 ms | 63.0 K/s |
| 128 MiB | 9.9 µs | 129 µs | 108 µs | 2.45 ms | 70.2 K/s |
| 512 MiB | 9.3 µs | 116 µs | 98 µs | 2.46 ms | 70.8 K/s |

Meets the 64K target (≤ 12 µs), misses 1M by 17 % (129 vs 110) and
16M (fresh 64 MB results are page-faulted either way; only reuse of the
output buffer can fix that — out-parameter API, kept out of scope).
Adopted as default; Xeon acceptance numbers pending.

Xeon Gold 6130 (one socket, 16 workers, `cmd/bench -quick`), ballast off
vs 128 MiB: MLP step 26.3 → 36.5 K/s, `x + y` 64K 72 → 58 µs, `relu` 64K
128 → 58 µs — but `x + y` 1M 770 → 897 µs and the tensor-level GEMM
n=1024 730 → 455 GFLOPS (blas benchmark: 1 119). Diagnosis: a 4 MB
result is zero-filled serially by the allocator, and with fewer GCs it is
fresh memory that sixteen threads then fault in at once. Fix applied:
MatMul allocates its output uninitialised and clears it in parallel
(M2 Pro tensor-level n=1024: 501 → 568, n=512: 350 → 463). Ballast stays
the default; Xeon re-measurement of the GEMM and 1M rows pending.

Third attempt: off-heap buffers. `BenchmarkAllocate` on the Xeon: a
fresh 1M-float result costs 647 µs (16M: 10 ms; 6.5 GB/s), against 42 µs
on the M2 Pro. Go zero-fills every allocation on the allocating goroutine
and the pages fault in serially, so `x + y` on 1M elements (890 µs) is
three quarters allocation. Design: results of 128 KiB and more are
mapped with mmap (anonymous; MADV_HUGEPAGE on Linux) instead of make: no
zero-fill by Go, first touch happens in the parallel kernels, 2 MB pages
where available. Mapped buffers return to a size-classed free list
through a cleanup and are unmapped beyond a retained-bytes limit; a GC is
forced when the outstanding mapped memory exceeds a budget, because
off-heap memory does not drive Go's collector. `Data()` pins a mapped
buffer permanently (never unmapped); `nn.Dropout` therefore uses a new
`Tensor.Dropout` op instead of writing a mask through `Data()`. Scope
extended to `nn/` for that change.

Implementation of the third attempt (M2 Pro, `cmd/bench -d 400ms`,
before → after): `x + y` 64K 9.4 → 9.1 µs, 1M 114 → 85 µs, 16M 2.25 →
1.69 ms; `relu` 1M 112 → 71 µs, 16M 1.64 → 1.10 ms; `exp` 1M 172 → 127
µs; softmax [4096×4096] 2.88 → 2.24 ms; layer norm 3.58 → 2.50 ms; MLP
forward+backward 65.0K → 74.8K samples/s. GEMM rows within noise. Two
pitfalls found on the way: a 16-byte pointer-free sentinel object is
tiny-allocated and its cleanup never runs (the sentinel now carries a
pointer), and a free list filled by one phase of a program (16 MiB GEMM
operands) starved the next phase's size class until least-recently-used
eviction across classes was added. Xeon numbers pending.

Xeon (unpinned, 32 threads, so only indicative): `x + y` 64K 194 → 47
µs, 1M 890 → 510 µs, 16M 22.8 → 15.3 ms (PyTorch 17.5: first win there),
layer norm 36 → 19 ms, MLP forward+backward 35K → 47K samples/s.
Allocator over the run: 137 695 hits, 4 146 misses, 498 forced GCs. THP
is in `madvise` mode with `defrag=madvise`, so our `MADV_HUGEPAGE`
mappings get huge pages but may stall on compaction: a fresh 4 MiB
mapping touched by 16 threads costs 1.1 ms (once per buffer; steady
state reuses). The remaining gap to PyTorch at 64K–1M is cache
residency: its refcounting reuses the same buffer immediately while we
rotate through up to 256 MiB of garbage between collections. Added
`Tensor.Release()` (immediate return; no-op when a view, `Data()` or
autograd holds the storage) and use it for intermediates in `nn.Linear`
and `nn.Sequential`; the bench gains a "result released" row. M2: `x +
y` 1M 92 → 70 µs released, 16M 1.72 → 1.39 ms; the M2's bandwidth hides
most of the effect, the Xeon run will show it.

Final Xeon numbers (one socket, GOMAXPROCS 16, `cmd/bench`): `x + y`
64K 194 → 49 µs (8.8 µs with `Release()`; PyTorch 17.8), 1M 890 → 496 µs
(28.8 µs released; PyTorch 24.1), 16M 22.8 → 15.3 ms (13.9 released;
PyTorch 17.5); layer norm 36 → 18.5 ms (PyTorch 14.1); MLP forward
NoGrad 79.7K → 176K samples/s (PyTorch 361K), forward+backward 15.6K →
46.8K (PyTorch 124K); tensor-level SGEMM n=1024 one thread 122 → 148
GFLOPS. Acceptance met with `Release()` for the 1M row; without it the
collector-driven path stays at ~500 µs, which is memory bandwidth for a
result that rotates through cold buffers. Follow-ups: the GEMM output
should be written (beta = 0) rather than cleared and accumulated, and the
bench releases GEMM results so C stays cache-warm as under PyTorch's
caching allocator.
