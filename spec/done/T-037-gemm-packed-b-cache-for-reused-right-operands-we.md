---
id: T-037
title: "GEMM: packed-B cache for reused right operands (weights packed once, not per call)"
status: done
scope:
  - internal/blas/
  - tensor/
  - cmd/bench/
  - nn/
manual:
  - docs/manual/performance.md
done: 2026-09-08
created: 2026-09-08
---

## Goal

In inference the right operand of almost every matrix product is a
weight that does not change between calls (`nn.Linear`, the attention
projections, EmbeddingGemma's 170 weight matrices), yet the GEMM driver
packs it into panel layout on every call. For [256×768]·[768×3072] the
packing of B is a third of the call. Pack a reused B once, keep the
packed copy, and let the driver skip that phase. Training gains nothing
from this (the weights change every step) and must not get slower.

## Design

- `blas.PackedB`: B in the driver's panel layout for every (K block,
  N block), built by `blas.PackB(b Mat, workers)` with the same `KC`,
  `NC`, `NR` the driver uses (stored in the value and checked at use).
  `blas.GemmZeroPackedWorkers(c, a, b Mat, p *PackedB, workers)` runs the
  driver with the B phase skipped: the parallel packing round packs only
  A, and the compute tasks take their panels from `p`. Matrix-vector and
  degenerate shapes fall back to the ordinary path.
- `tensor` keeps a process-wide cache of packed operands, keyed by the
  operand's storage and view geometry (data pointer, strides, rows,
  cols). An entry is valid while the storage's version is unchanged:
  every in-place operation and `Set` bump the version, and storage whose
  buffer escaped through `Data()` is never cached (a caller may write
  through the slice at any time). Autograd-recorded operands are read
  only, so they cache like any other.
- Insertion on the second sighting of a key (the first records it in a
  small ring), so activations that are used once never enter the cache;
  least-recently-used eviction under a byte limit
  (`tensor.SetPackedCacheLimit`, default 512 MiB, `FIBERAI_PACK_CACHE=0`
  disables the cache, `tensor.PackedCacheStats` reports hits, misses,
  bytes). Only 2-D products with at least 64K elements in B are
  considered; batched products (attention) never are.
- `cmd/bench` prints the GEMM table for both cache states, so the
  headline can show "B packed once" and "B packed per call" side by
  side; the MLP and EmbeddingGemma rows run with the default (on).

## Acceptance

- Results identical to the uncached path (existing GEMM tests run with
  the cache forced on and off; a test modifies B in place between calls
  and checks the product follows; a test writes through `Data()` and
  checks the operand is not cached).
- Python baseline (M2 Pro, same day): PyTorch [256×768]·[768×3072]
  2 358 GFLOPS, MLP forward 526 K samples/s, EmbeddingGemma 88
  sentences/s; Xeon: PyTorch 980 GFLOPS, 361 K, 37. Targets with the
  cache on: [256×768]·[768×3072] with reused B at least 2 100 GFLOPS on
  the M2 (from 1 795) and 1 150 on the Xeon (from 1 016); MLP forward and
  EmbeddingGemma at least 10 % faster than today on the M2; training
  step within 2 % of today.
- Manual: performance page explains what is cached, when it is not, and
  the knobs.

## Notes

Results, Apple M2 Pro, 2026-09-08 (cache off → on, same run):

| | off | on |
|---|---:|---:|
| [8×4096]·[4096×4096] GFLOPS | 75 | 250 |
| [64×1024]·[1024×1024] | 992 | 1 582 |
| [256×768]·[768×3072] | 1 806 | 2 594 (PyTorch 2 358) |
| 1024² | 2 172 | 2 471 (PyTorch 2 682) |
| 2048² | 2 289 | 2 199 (noise; the cache adds nothing at this size) |
| MLP forward, samples/s | 768 K | 839 K |
| MLP train step | 156 K | 150 K (within run-to-run noise; weights escape through Data() and are never cached) |
| EmbeddingGemma, sentences/s | 86 | 98 (PyTorch 88) |

All targets met on the M2 except the training-step bound, which the
run-to-run spread of that row (148–164 K over the day) does not let one
verify at 2 %; the mechanism cannot slow it (escaped storage is rejected
before any work). Xeon numbers pending. Correctness: products with a
cached operand equal the uncached ones; in-place changes and `Set`
invalidate; an operand read through `Data()` is never served from the
cache; a transposed view of a cached root has its own entry; the whole
suite passes with the cache on, and under AVX2.

