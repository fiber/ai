---
id: T-050
title: Small-GEMM path: one call, no rounds, no packing where the kernel reads row-major, chosen by size
status: done
scope:
  - internal/kernel/
  - internal/blas/
  - tensor/
  - cmd/bench/
  - benchmarks/python/
manual:
  - docs/manual/performance.md
done: 2026-09-10
created: 2026-09-09
---

## Goal

Below about 200² our products run at a third to a half of Accelerate's
and MKL's rate and the gap closes smoothly to level at 256² (M2 Pro,
fresh operands: 128² 359 against 764 GFLOPS, 160² 471 against 890, 256²
1 133 against 1 136; Xeon 128² 99 against 236; VM 52 against 129).
Threads are not the reason: one worker gives 379 at 128² on the M2, six
give 370. At that size the arithmetic is three microseconds of an
eleven-microsecond call; the rest is the blocked driver's fixed cost,
two operands packed through the buffer pool, a packing round and a
compute round with their barriers, K-block bookkeeping. MKL and
Accelerate have a separate small-matrix path. So do we now. The shapes
that matter for it are not squares: a netwatch-sized autoencoder
(24 → 16 → 3 → 16 → 24, batch 64) is nothing but products of this size.

## Design

- **`gemmSmall`** in `internal/blas`, taken by the driver entry points
  when m·n·k ≤ `SmallLimit` (default from measurement, environment
  `FIBERAI_BLAS_SMALL`, 0 disables): one call on the calling goroutine.
  B is packed once into NR panels (or taken from the packed-operand
  cache when the caller has it and it is a single K block); A is read
  row-major by `kernel.GemmRM` where the back-end has it (AVX2,
  AVX-512), with the last partial row group copied into a zero-padded
  scratch, and packed into MR panels elsewhere (NEON, AMX, generic).
  Tiles come straight from `GemmZero` (or a cleared tile and `Gemm`),
  edge tiles through a scratch tile; the epilogue, if any, is applied
  over the whole output afterwards. No parallel rounds, no K blocking
  (the depth of a small product fits the panel budget).
- **Kernels that read B in place.** The 128² profile on the M2 was 35 %
  micro-kernel, 45 % packing the two operands and 18 % allocating the
  result, so the small path must not pack what a kernel can read where
  it is. `kernel.GemmRB` (AMX: A packed, B row-major with a row stride,
  a one-register change in the load macro) and `kernel.GemmRMB` (AVX2,
  AVX-512: both operands row-major) with their overwriting variants,
  verified at start-up like the others. The small path then packs only
  A on AMX, nothing on x86, and the ragged last column panel of B (fewer
  than NR columns) goes through a small zero-padded scratch so no kernel
  reads past the matrix.
- Results of 64 KiB and more use the mapped pool (`mapMin` halved), so
  a 128² product's output is recycled instead of allocated and zeroed
  on the heap for every call.
- The gemv and few-rows paths keep precedence for m = 1 and n = 1.
- **Bench**: a "small products" section, squares 32 to 256 and the
  autoencoder shapes, in `cmd/bench` and `bench.py`; a "tiny
  autoencoder train step" row (24 → 16 → 3 → 16 → 24, batch 64, Adam)
  next to the MLP rows on both sides.

## Acceptance

- Products equal the blocked driver's to 1e-4 on shapes with every
  edge case (m, n not multiples of MR/NR, k odd, m·n·k at the limit),
  with and without an epilogue, on the native back-end, generic, and
  AVX2 under Rosetta; `go test ./...` on both machines.
- Baselines (M2 Pro, PyTorch 2.14 on Accelerate, same day): 128²
  764 GFLOPS (ours 359), 160² 890 (471), 192² 993 (751); Xeon (MKL)
  128² 236 (99), 256² 627 (540); VM 128² 129 (52). Targets: M2 128² ≥
  600, 160² ≥ 750; Xeon 128² ≥ 180; VM 128² ≥ 90; the limit set where
  the small path stops winning on the M2, verified on the Xeon; the
  tiny-autoencoder training step measured against PyTorch on both
  machines (no target yet: the first measurement is the baseline).
- Manual: performance.md's GEMM section describes the small path and
  the limit.

## Notes
- M2 Pro results (same day, PyTorch on Accelerate): fresh-operand squares
  before → after → PyTorch: 64² 156 → 217 → 288; 96² 264 → 363 → 555;
  128² 379 → 562 → 777; 160² 526 → 674 → 909; above the limit unchanged
  (192² 718 vs 1 005, 256² 1 131 vs 1 156). Targets: 128² ≥ 600 missed
  by 6 %, 160² ≥ 750 missed by 10 %. Tiny autoencoder 24 → 16 → 3 → 16
  → 24, batch 64: forward 3.06 M samples/s (PyTorch 2.38 M), training
  step **1.18 M against 284 K**, 4.2× ahead: the per-call overhead that
  dominates a small model's step is where Python pays, not where we do.
  The very narrow products ([256×24]·[24×16] 36 vs 92 GFLOPS) stay
  behind: the AMX tile is 32 wide and half of it idles at n = 16; the
  gemv-like [64×16]·[16×3] is ahead (4.0 vs 2.9).
- Limit set to 160³ from the M2 crossover (at 192² the six-worker driver
  wins 732 to 595); the Xeon may want a higher one, to be measured with
  `-only small,gemm,mlp` pinned and `bench.py --only small`.
- The 128² profile after the change would be worth another look on the
  Xeon: the M2's remaining gap is the AMX tile store per k = 128 steps
  and the result recycling, not packing any more.
