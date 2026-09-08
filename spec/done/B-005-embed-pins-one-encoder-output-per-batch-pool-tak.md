---
id: B-005
title: Embed pins one encoder output per batch: pool() takes Data(), so Release is refused and memory grows with the number of batches
status: done
scope:
  - models/gemma/
manual: none
done: 2026-09-08
created: 2026-09-08
---

## Goal

`cmd/bench -only embed` on the Xeon ended with `pinned 32 MiB` in the
allocator line and the figure grows with every batch: `Model.Embed`
leaks one encoder output per forward pass. A service embedding millions
of log lines would grow without bound. Make repeated `Embed` calls keep
the mapped-storage statistics flat.

## Design

`embedBatch` hands the encoder output to `pool`, which reads it through
`Data()`. `Data()` marks the storage as escaped (the caller may keep the
slice), so the `Release()` that follows is refused and the buffer stays
mapped for the life of the process. `pool` keeps nothing, so the right
call is `Recycle`, which accepts escaped storage from an owner that
knows better; the head output is read with `Float32s()` (a copy) and
released normally. A test embeds the same batch twenty times and checks
that `tensor.MappedStats()` reports no growth in pinned or retained
bytes after the first call.

## Acceptance

- Twenty `Embed` calls: pinned bytes unchanged after the first, retained
  bytes bounded; the parity tests still pass.
- `cmd/bench -only embed` ends with pinned 0 MiB.

## Notes

Fixed: the encoder output and the pooled vector are recycled, the head
output is copied out with Float32s and recycled. Steady state over 20
further Embed calls on the M2 Pro: pinned 0 → 0 bytes, retained bounded
by the mapped limit (the free list fills to its cap with reusable
buffers, which is by design). Parity tests unchanged.

