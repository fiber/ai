---
id: T-019
title: Worker pool idle path: rare Gosched, wake-ups only for parked helpers, cheaper small jobs
status: done
scope:
  - internal/parallel/
  - cmd/bench/
  - tensor/
  - internal/kernel/
  - optim/
manual:
  - docs/manual/internals.md
  - docs/manual/performance.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

A CPU profile of the MLP training step on the M2 Pro shows 70 % of all
samples in `runtime.goschedImpl → runtime.lock2 → usleep` and only ~13 %
in kernels: the idle helpers of `internal/parallel` call
`runtime.Gosched()` every 4 096 spin iterations, ten of them at once, and
serialise on the scheduler lock. Another 15 % goes to
`pthread_cond_signal`/`pthread_cond_wait`: every `publish` takes `parkMu`
and broadcasts even when nobody is parked. The step's actual arithmetic
is a fraction of a millisecond per core; the rest is this overhead. On
the 16-core Xeon the contention is worse (forward+backward 47K samples/s
against PyTorch's 124K).

## Design

- Spin loop: check the clock instead of counting loads; `Gosched` at
  most every ~50 µs and only when other goroutines are runnable is not
  observable cheaply, so simply raise the interval to 1<<16 loads and
  bound the total spin by time (200 µs) via `nanotime` sampled every
  1 024 loads.
- `publish`: keep a count of parked helpers; take `parkMu` and
  `Broadcast` only when it is non-zero. The generation bump stays an
  atomic store; a helper decides to park under `parkMu` after
  re-checking the generation, so no wake-up is lost.
- Caller side: the end-of-job poll (20 000 iterations with a `Gosched`
  every 1 024) gets the same treatment.
- Measure with `cmd/bench -only mlp -cpuprofile` before and after; the
  `runtime.*` share of samples must drop below 20 %.
- With fork/join this cheap, the tensor layer's element-wise chunk of
  65 536 elements (which gives a 128K-element MLP activation two
  workers) is revisited: scope extended to `tensor/` for `minChunk`.

Alternatives considered: fewer helpers than GOMAXPROCS (helps the
scheduler, loses cores on the big GEMMs); parking immediately (each MLP
op would then pay a futex wake-up per helper, the situation T-014
fixed).

## Acceptance

- `go test ./internal/parallel/` (including `-race`) passes; nested calls
  and panics behave as before.
- M2 Pro `cmd/bench` MLP forward+backward ≥ 100K samples/s (from 73K;
  PyTorch 219K), training step ≥ 90K (from 68K; PyTorch 116K); GEMM and
  element-wise rows unchanged or better.
- Xeon Gold 6130, one socket: forward+backward ≥ 70K (from 47K; PyTorch
  124K), training step ≥ 60K (from 43K; PyTorch 71K).
- Manual: internals.md describes the idle path, performance.md updates
  the MLP figures.

## Notes

Xeon sweep (one socket): spin 50 µs 30K, 300 µs 49K, 1 ms 50K, 3 ms
53K samples/s forward+backward; Gosched frequency makes no difference.
Element-wise chunk: 65 536 → 48.6K, 32 768 → 59K, 16 384 → 62K, 8 192 →
68.8K, and `x + y` 64K 57 → 32 µs. M2 Pro with 8 192: forward+backward
76K → 96K, no element-wise row worse. Profile on the Xeon: 53 % of core
time is helpers spinning, 21 % GEMM, 3.6 % the scalar Adam loop, 3.5 %
`time.Since` in the spin. The step is bound by the main goroutine's
serial path; the remaining levers are the Adam step (fused SIMD kernel
instead of a scalar loop with float64 sqrt; scope extended to
`internal/kernel/` for a vector `Sqrt` and `optim/`), the working-set
rotation through cold mapped buffers between forced collections (a
smaller budget keeps the rotation in cache at the price of more
collections; `FIBERAI_MAP_BUDGET` added to measure), and fewer, cheaper
synchronisation rounds per small GEMM.

Graph release (tensor scope): `Backward` now counts consumers per
tensor (incremented in `record`, decremented as each consumer's backward
runs) and hands an intermediate's storage and gradient back once the
count reaches zero, dropping its node; the backward closures keep
`saved()` aliases instead of views so the storage is not marked shared.
Default on (`SetReleaseGraph`); reads of a released tensor panic with a
message. M2 Pro NEON training step 84.9K → 92.3K samples/s with the
retained working set down from ~200 MiB to 28 MiB; AMX unchanged at
147K. Packing buffers in `blas` moved from a `sync.Pool` (emptied by
every GC, then re-zeroed by Go) to a persistent free list. Xeon numbers
pending for both.

Final Xeon numbers (one socket, GOMAXPROCS 16, commit cb2a1e9): MLP
forward 169K → 288K samples/s, forward+backward 47K → 65K, training step
42.7K → 64K (PyTorch 361K / 124K / 71K); tensor-level SGEMM n=1024 1 208
and n=2048 1 307 GFLOPS (the kernel's own figures), [256×768]·[768×3072]
1 162 (PyTorch 980); `x + y` 1M released 17 µs (PyTorch 24); layer norm
11.0 ms (PyTorch 14.1). Three x86 regressions found and fixed on the way,
all invisible on the M2 Pro: packing buffers handed out by size instead
of LIFO (cold buffers, 60 µs per GEMM), sixteen workers faulting fresh
output pages concurrently once MatMul stopped clearing its output (3.6×
the page faults), and the Go fallback of the register-transpose packing
(30 % of a forward pass). Acceptance: forward+backward ≥ 70K not quite
reached (65K); training step ≥ 60K met. Remaining gap to PyTorch's 124K
forward+backward is the serial path of small GEMM shapes on 16 cores
(T-005) and per-op overhead.
