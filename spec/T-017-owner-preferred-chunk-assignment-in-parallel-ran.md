---
id: T-017
title: Owner-preferred chunk assignment in parallel.Range for cache locality of repeated element-wise operations
status: open
scope:
  - internal/parallel/
  - tensor/
  - cmd/bench/
manual:
  - docs/manual/performance.md
  - docs/manual/internals.md
created: 2026-09-05
---

## Goal

Repeated element-wise operations on the same tensors should run from the
cores' private caches, as they do in PyTorch. On the Xeon Gold 6130
(unpinned, 32 threads) `x + y` on 1M elements with the result released
takes 53 µs against PyTorch's 24 µs; 12 MB in 24 µs is L2-class
bandwidth, and 53 µs is roughly what the shared L3 and cross-core snoops
give. `parallel.Range` hands chunks out from an atomic counter, so chunk
c lands on a different goroutine (and core) on every call and the data a
core touched last time is in someone else's L2.

## Design

`job` gets an `owned` mode used by `RangeWorkers`: items are claimed
through a per-item CAS instead of a counter. Worker w (the caller is 0,
helper k is k+1) first claims items w, w+workers, w+2·workers, … — the
same chunks on every call, so the data a core wrote last time is what it
reads now — and then scans for anything unclaimed (work stealing), so a
slow or busy worker cannot stall the round and nested calls stay
deadlock-free. Helpers whose id is not below the job's worker count skip
owned jobs. `For` keeps the counter (GEMM's task grid is tuned for it).
Goroutine-to-core affinity is not enforced; a spinning helper stays on
its P in practice. `cmd/bench` prints GOMAXPROCS next to the CPU count so
a pinned run can be told from an unpinned one.

Alternatives: static partitioning without stealing (deadlock-prone with
nested calls and slow under interference); OS thread pinning through
`LockOSThread` plus sched_setaffinity (heavier, Linux-only, and Go's
scheduler would still migrate other goroutines onto those cores).

## Acceptance

- `go test ./internal/parallel/` covers owned mode: every item exactly
  once for many n/worker combinations, panics propagate, nested calls
  complete.
- Xeon Gold 6130, one socket pinned, `cmd/bench`: `x + y` 1M, result
  released, ≤ 35 µs (from 53; PyTorch baseline 24 µs); 64K released
  stays ≤ 10 µs (PyTorch 18 µs). M2 Pro rows unchanged or better.
- Manual: internals.md describes the claiming scheme, performance.md the
  locality argument next to `Release()`.

## Notes

Implemented. M2 Pro (`cmd/bench -d 300ms`): `x + y` 1M 92 → 76 µs,
released 70 → 59 µs; 16M released 1.39 → 1.40 ms; no row worse. The M2
has enough bandwidth that locality matters little; the Xeon run decides.
`go test -race ./internal/parallel/` takes ~200 s because the spin loops
run under the race detector; it passes.
