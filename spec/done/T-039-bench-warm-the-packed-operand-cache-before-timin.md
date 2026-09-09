---
id: T-039
title: "bench: warm the packed-operand cache before timing (two warm-up calls for reused-weight rows)"
status: done
scope:
  - cmd/bench/
manual: none
done: 2026-09-09
created: 2026-09-09
---

## Goal

`timeIt` warms up with one call. With the packed-operand cache an
operand is packed on its second sighting, so the first timed call pays
the packing, and when a call is long relative to the measurement window
(EmbeddingGemma on the Xeon: 700 ms per forward against 700 ms of
measurement) the window holds little else. The Xeon embed row therefore
showed no gain while the statistics showed the cache working (170 hits,
340 misses). Warm reused-weight rows until the cache is populated before
timing.

## Design

`timeIt` gains a variant with a warm-up count; the EmbeddingGemma row
and the "B packed once" GEMM column warm twice (first sighting, pack),
the per-call column and everything else keep one warm-up. Timed calls
then measure the steady state on both machines alike.

## Acceptance

- After the change the embed section's report shows misses equal to
  twice the number of weights and every timed call a hit, on the M2 Pro
  and the Xeon; the "B packed once" column reports 14 misses for 7
  shapes as today.
- No performance impact on the library; this changes measurement only.

## Notes
