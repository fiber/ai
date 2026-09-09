---
id: T-042
title: amd64 register-transpose packing: AVX2 8x8 transposes with masked tails for A panels and transposed B panels
status: done
scope:
  - internal/blas/
manual:
  - docs/manual/performance.md
done: 2026-09-09
created: 2026-09-09
---

## Goal

The pinned Xeon profile of the attention benchmark after T-041 is 37 %
micro-kernel, 21 % `packA`, 15 % `expSumAVX2` and 11 % `packBPanel`: a
third of the time in Go packing loops. On arm64 the A packing of a
row-contiguous block is a 4×4 NEON register transpose (`packRows4`); on
amd64 there is no assembly and `packA` streams each row into the k-major
panel with a scalar store every MR floats, and `packBPanel` does the
same for a transposed B (rows of the source contiguous, panel k-major).
Every GEMM packs its A operand this way on x86 (the same fallback once
cost 30 % of an MLP forward on the Xeon); attention packs q, the
probabilities and Kᵀ so. Give amd64 a register-transpose packing.

## Design

- **`packRows8`** (AVX2 assembly, `pack_amd64.s`): transposes up to 8
  contiguous source rows (row stride `rs` floats, `pb` columns with
  pb % 8 == 0) into a k-major panel with row stride `width`:
  dst[p*width + r] = src[r*rs + p]. Per 8-column block it loads the
  `rows` valid rows (missing rows read as zero), runs the standard 8×8
  transpose (unpack, shuffle, perm2f128) and stores the eight panel rows
  with `VMASKMOVPS` under a lane mask of `rows` lanes, so a 6-row group
  (the tail of MR 14, the whole of MR 6) costs the same as a full one
  and never writes past its rows. AVX2 is enough: the Xeon runs it, and
  the AVX-512 back-end shares the blas packing.
- **Shared Go helper** `packRowsTransposed(panel, data, base, stride,
  pb, width, rows)`: the register-transpose path in `packWidth`-row
  groups (8 on amd64, 4 on arm64 through the existing `packRows4`, which
  handles full groups only) with the scalar code for whatever the
  kernel declines (arm64 groups under 4 rows) and for the pb tail. Used
  by `packA` for row-contiguous A (CS == 1) and by `packBPanel` for a
  transposed B (RS == 1); the plain streaming loop stays for
  architectures without assembly. Behaviour on arm64 is unchanged.
- `packTranspose` becomes true on amd64; `pack_other.go` is built for
  neither arm64 nor amd64.

## Acceptance

- `packA` and `packBPanel` outputs equal the scalar reference for rows
  1–8 and 9–16 (both MR 6 and 14 style groups), widths 6, 8, 14, 32, pb
  in {8, 16, 24, 7, 13, 64, 65}, natively and under Rosetta with
  `GOARCH=amd64` (AVX2 forced); the GEMM, epilogue and attention tests
  pass on both, and `FIBERAI_KERNEL=avx512x12` and `avx2` on the Xeon.
- Baselines (Xeon Gold 6130, pinned socket, 2026-09-09; PyTorch 2.14 /
  MKL same day): attention [8×8×512×64] 533 GFLOPS (PyTorch 1 141), MLP
  forward 415 K samples/s (361 K), SGEMM 1024² tensor level 1 219 (MKL
  1 434), EmbeddingGemma 58 sentences/s (37). Targets: attention ≥ 650,
  MLP forward ≥ 450 K, 1024² ≥ 1 300, EmbeddingGemma ≥ 62; M2 rows
  unchanged (arm64 path untouched).
- Manual: performance.md's packed-operand or attention section says
  what the x86 packing does and what it was worth.

## Notes

- The transposed-B path (`packBPanel`, RS == 1) now uses the register
  transposes on arm64 as well (it was the scalar loop there before); the
  A path on arm64 is unchanged. M2 rows unchanged within noise.
- `packTranspose` is a variable on amd64, off when the kernel package
  runs generic code (no AVX): the pack kernel is AVX1/AVX2 and must not
  run on a CPU without it.
- Xeon measurement pending (targets in Acceptance).
