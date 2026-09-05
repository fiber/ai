---
id: T-019
title: Worker pool idle path: rare Gosched, wake-ups only for parked helpers, cheaper small jobs
status: open
scope:
  - internal/parallel/
  - cmd/bench/
  - tensor/
manual:
  - docs/manual/internals.md
  - docs/manual/performance.md
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
