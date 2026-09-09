---
id: T-044
title: Result recycling without Release: cheaper forced collections and released exp/tanh rows for the kernel comparison
status: open
scope:
  - cmd/bench/
  - tensor/
manual:
  - docs/manual/performance.md
created: 2026-09-09
---

## Goal

On the pinned Xeon every 1M element-wise bench row costs about 480 µs,
`relu` and `x * 2.5` as much as `tanh`, while `x + y` with the result
released costs 23 µs. On the M2 the same rows show the kernels (relu 53
µs, tanh 120 µs, released add 41 µs). An unreleased 4 MB result is
reclaimed by a forced collection once 256 MiB are outstanding, one per
64 results; on the M2 that collection takes 0.4 ms (gctrace), 6 µs per
result. For the Xeon's figure the collection would have to take about
30 ms, which points at `collectMapped`'s 20 ms wait for the sentinel
cleanup rather than at the collector itself. Two things follow: the
bench needs released rows for the transcendental kernels so the
comparison with PyTorch's MKL vector library is a kernel comparison,
and the recycling path must not wait a fixed 20 ms when cleanups are
late.

## Design

- **Bench**: released variants of `exp(x)`, `tanh(x)` and `gelu(x)` at
  every size (the result released right after the call, like the
  existing `x + y, result released` row); the unreleased rows stay, they
  are what naive code pays.
- **`collectMapped`**: after `runtime.GC()` poll for the sentinel with
  `runtime.Gosched()` in a loop bounded by 2 ms instead of blocking on a
  20 ms timer, and return as soon as the free list of the requested
  class has an entry (the caller passes the class); if nothing arrives
  within the bound, map fresh (huge pages make that cheap) and let the
  next collection reclaim the rest. The decision is refined once the
  Xeon gctrace shows whether the collection or the wait is the cost.

## Acceptance

- Baselines (Xeon pinned, 2026-09-09): fiber/ai 1M rows 481–485 µs for
  every operation, `x + y` released 23 µs; PyTorch (MKL) tanh 1M 54 µs
  and exp 16M 13.8 ms on the Xeon, tanh 1M 719 µs and exp 16M 3.09 ms on
  the M2; fiber/ai M2 tanh 1M 120 µs, relu 53 µs. Targets: Xeon
  unreleased 1M rows within 2× of the released ones; released tanh 1M
  on the Xeon ≤ 60 µs (level with PyTorch) and exp ≤ 50 µs; M2 rows
  unchanged within noise; `go test ./...` on both.
- Manual: performance.md's off-heap section describes the recycling
  cost without `Release()` with the measured figures.

## Notes
