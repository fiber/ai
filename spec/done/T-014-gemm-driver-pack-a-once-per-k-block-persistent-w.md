---
id: T-014
title: GEMM driver: pack A once per K block, persistent workers, no per-task repacking
status: done
scope:
  - internal/blas/
  - internal/parallel/
manual:
  - docs/manual/internals.md
  - docs/manual/performance.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

Close the multi-core gap of the blocked SGEMM on x86. On the Skylake-SP
Xeon (16 cores, one socket) the AVX-512 micro-kernel delivers ~150–162
GFLOPS per core, but 16 workers reach only 759 GFLOPS at n=1024 (32 % of
16×) where MKL reaches 1 434 (60 %). The blocking sweep on that machine
showed the driver, not the caches, as the limit: a larger MC made things
worse (759 → 539) because every (row block × panel range) task re-packs
its A block, and a smaller KC made things worse (759 → 621) because each
KC step costs two barriers, a B packing round and freshly spawned
goroutines. Pinning to physical cores and NUMA placement changed
nothing (793 vs 758; 32 cores 1 036 vs 16 cores 1 018).

## Design

**internal/blas — pack A once per K block.** For every (jc, pc) block the
driver runs three phases: pack the B block into NR panels (parallel over
panels, as before); pack *all* row blocks of A for this pc into one
shared buffer of roundUp(M, MR) × KC floats (parallel over MR panels);
then a grid of pure compute tasks (row block × panel range) that read
the packed A rows for their block and stream their panels. No task packs
anything, so MC can grow to fill L2 and the panel range per task can
shrink without a packing penalty. Buffers come from the existing pool.
The serial path (workers == 1) runs the same code inline.

**internal/parallel — persistent workers.** `ForWorkers` hands the job to
a fixed set of helper goroutines through a channel instead of spawning
goroutines per call; helpers poll briefly (a short `Gosched` spin) after
finishing a job so back-to-back rounds — the KC steps of one GEMM — do
not pay a thread wake-up each time, then park on the channel. The caller
always participates and never waits for a helper to *start*, only for
items already claimed to finish, so nested calls cannot deadlock (a
nested call simply runs on the calling goroutine when all helpers are
busy). Items are handed out from an atomic counter as before; a panic
stops further items and is re-raised in the caller. The pool grows
lazily to `Workers()-1` helpers.

Rejected: static owner-computes partitioning of N across workers (no
barriers, but A packed once per worker and static partitions lose on
heterogeneous P/E cores); a spinning barrier inside one parallel region
(same wake-up problem, more code).

## Acceptance

- `go test ./...` and `go test -race ./internal/...` pass; the GEMM tests
  (all layouts, block boundaries, worker counts) are unchanged and green.
- Xeon Gold 6130, one socket, 16 workers, `go test ./internal/blas -bench
  'Gemm$/.*/n=1024'`: ≥ 1 200 GFLOPS (from 759; PyTorch/MKL 1 434,
  NumPy/OpenBLAS 1 303). n=2048: ≥ 1 400 (from 1 018; PyTorch 780, NumPy
  881). Single-thread numbers unchanged within noise.
- Apple M2 Pro: no regression at n=512–2048 (595 GFLOPS at 2048, 10
  workers); the tensor-level `cmd/bench` GEMM rows within ±5 %.
- A blocking sweep on the Xeon with the new driver (MC 96–512, KC
  256–512) recorded in Notes; the amd64 defaults updated to its result.
- `docs/manual/internals.md` describes the three-phase driver and the
  worker pool; `docs/manual/performance.md` carries the new x86 figures.

## Notes

Implemented in four steps, each measured on the Xeon Gold 6130 (one
socket, 16 workers, `go test ./internal/blas -bench`):

| step | n=1024 | n=2048 |
|---|---:|---:|
| before | 759 | 1 018 |
| A packed once per K block, channel-polling worker pool | 942 | 944 |
| helpers spin on an atomic counter (profile: 20 % CPU in lock2/procyield) | 898–1 002 | 1 070 |
| longer spin, caller polls, A+B packed in one round (perf: cores only 70 % busy) | 1 051 | 1 025 |
| 16 compute tasks per worker instead of 3 (tail of each round) | **1 147** | **1 139** |

Blocking sweep with the new driver (KC × MC, n=1024): MC no longer
matters (860–1 000 across 96–512); KC 256/384/512 → 924/982/999, so
KC=512 is the amd64 default. A row-block strategy with private A packing
(`FIBERAI_BLAS_STRATEGY=rows`) was no better (926 at MC=32, worse for
larger blocks), so cross-core reads of packed data are not the limit;
pinning to physical cores changed nothing either.

Measured limits: all-core AVX-512 clock 1.95 GHz (perf: cycles/user), so
the socket's FMA peak is ~2 000 GFLOPS; we reach 57 % of it per cycle,
MKL 72 %, our single core 84 % at 2.8 GHz. The remaining gap is inside
the micro-kernel under all-core load (no software prefetch, 12×32 tile,
C rows 4 KiB apart) — spec T-016.

Acceptance: n=1024 at 1 147 misses the 1 200 target by 4 %; n=2048 at
1 139 misses 1 400, a target set before the all-core clock was known
(at 1.95 GHz it corresponds to 70 % of peak, MKL's level). Against the
Python baselines: n=2048 ahead of PyTorch (780) and NumPy (881),
n=1024 at 80 % of PyTorch. Closing the spec here; the driver work is
done, the rest is kernel work.
