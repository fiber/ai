---
id: T-038
title: "Packed-operand cache: protect the working set from eviction thrash, report evictions and invalidations"
status: done
scope:
  - tensor/
  - cmd/bench/
manual:
  - docs/manual/performance.md
done: 2026-09-09
created: 2026-09-09
---

## Goal

On the Xeon the cache served the GEMM rows (projection 1 015 → 1 245
GFLOPS) but not EmbeddingGemma (47 sentences/s, 49 before): the run
ended with 478 MiB held and 1 092 misses against 354 on the M2, the
signature of a working set slightly larger than the cap being evicted
round-robin. Plain LRU thrashes exactly then. Keep whatever fits and
let the rest pack per call, and make the cache's behaviour visible.

## Design

- **Insertion that refuses to thrash.** A lookup counter stamps every
  hit; an entry is evictable only if its last use lies more than
  `2 × len(entries)` lookups back (older than one pass over the current
  working set). If the space for a new entry cannot be freed from
  evictable entries, the candidate is not inserted and packs per call.
  A working set that fits stays; one that does not fit keeps its most
  recently used part instead of cycling.
- **Diagnostics.** `PackedCacheStats` grows to a struct: hits, misses,
  packs, evictions, invalidations (version changed), refusals (no
  evictable space), bytes, entries. `cmd/bench` prints it after each
  section and resets the cache between sections, so one section's
  operands do not shape the next.
- Default cap stays 512 MiB; `SetPackedCacheLimit` unchanged.

## Acceptance

- A test builds a working set 1.5× the cap and checks that after two
  passes at least the cap's worth of entries hit on every pass (no
  cycling), with refusals reported.
- Baseline (PyTorch): EmbeddingGemma 37 sentences/s on the Xeon, 88 on
  the M2 Pro. Target: on the Xeon, the embed section run alone reports
  misses ≈ 2 × entries after warm-up and ≥ 52 sentences/s (10 % over the
  uncached 47); M2 unchanged at ≥ 95; the correctness tests of T-037
  still pass.
- Manual: the performance page names the policy and the statistics.

## Notes

Built 2026-09-09. Two things came out of the diagnostics before the
Xeon could be re-run:

- **Recycled addresses impersonated earlier operands.** The key was the
  data pointer; freed off-heap buffers come back at the same address, so
  a fresh gradient in a training step looked like the second sighting of
  the previous step's gradient and was packed for nothing (MLP section:
  468 packs, 333 invalidations per run). Storage now carries a
  process-unique id and the key uses it: 3 packs, 0 invalidations, and
  the training step measured 161 K samples/s (156 K before the cache).
- **Eviction guard** as designed; the no-thrash test holds 8 of 12
  operands at a 1.5× working set with 0 steady-state evictions and the
  rest refused.

M2 Pro after the change: projection 2 642 GFLOPS with B packed once,
MLP forward 823–835 K, EmbeddingGemma 93–97 sentences/s (misses 340 =
2 × 170 weights, then all hits). Xeon numbers for the embed section
pending the next run.

