---
id: T-028
title: Small GEMM shapes: B read in place for M below the tile, fewer rounds and lower fixed cost for n up to 256
status: done
scope:
  - internal/blas/
  - internal/kernel/
  - cmd/bench/
manual:
  - docs/manual/performance.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

Two shapes where the packed GEMM loses to the libraries: a handful of
rows against a large matrix (batch-1 inference, [8×4096]·[4096×4096]:
NEON 55 GFLOPS against PyTorch's 68 on the M2 Pro, 56 against 58 on the
Xeon) and small square products (n=128: 73 GFLOPS against 236 on the
Xeon, 70 against 783 on the M2 Pro with NEON), where the fixed cost of
packing rounds and the single-thread threshold dominate.

## Design

- **Few rows, B in place.** For m ≤ 8 with row-major B, no packing:
  the output rows for a block of 1 024 columns stay in L1 while each row
  of B streams through once, with one `Axpy` per output row and B row.
  Parallel over column blocks. B is read exactly once; the previous path
  packed all of B (64 MB for 4096²) to multiply eight rows.
- **Small products, more threads.** `ParallelThreshold` decides when a
  product runs on one core; it is measured again on both architectures
  (env knob `FIBERAI_BLAS_THRESHOLD`) and lowered where the M2 Pro shows
  a gain, with the Xeon check in the next round.

## Acceptance

- Tests: the few-rows path matches the packed path for m = 1…8, strided
  and transposed operands, β = 0 and accumulate.
- M2 Pro NEON: [8×4096]·[4096×4096] ≥ 68 GFLOPS (PyTorch 68); n=128 all
  cores ≥ 150 GFLOPS (from 70; PyTorch 783 with AMX — the NEON figure
  cannot reach that, the target is the single-core rate times a useful
  share of cores). No regression on the large shapes.
- Xeon (next round): [8×4096] ≥ 58, n=128 ≥ 150 (PyTorch 58 / 236).

## Notes

Measured on the M2 Pro. The B-in-place path as a composition of `Axpy`
calls is slower, not faster: [8×4096]·[4096×4096] 22 GFLOPS against 54
packed (NEON) and 76 (AMX). With 1 024-column blocks it issues 131 000
kernel calls of 2 KFLOP each and the call cost dominates; it stays in
the code behind `FewRows` (default 0, `FIBERAI_BLAS_FEWROWS=8` to try)
until a register kernel exists that keeps the rows of C in registers
while B streams. The threshold experiment paid off: lowering
`ParallelThreshold` from 4M to 1M multiply-adds takes n=128 from 85 to
173 GFLOPS with NEON (n=256 unchanged within noise, AMX unchanged
because its worker cap already applied). New default 1M; the Xeon check
(n=128 was 73 single-threaded there) is the next round. Acceptance:
n=128 ≥ 150 met on the M2 Pro NEON; the [8×4096] target is not met and
the item stays in TODO as the register kernel.
