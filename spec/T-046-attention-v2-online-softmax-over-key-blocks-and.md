---
id: T-046
title: Attention v2: online softmax over key blocks and a micro-kernel that reads its left operand row-major, so nothing is packed per row group
status: open
scope:
  - internal/kernel/
  - internal/blas/
  - tensor/
manual:
  - docs/manual/performance.md
created: 2026-09-09
---

## Goal

After T-041–T-045 the pinned Xeon runs attention [8×8×512×64] at 693
GFLOPS against oneDNN's 1 141 in PyTorch; the M2 (AMX) at 1 110 against
553. Per row group the current driver computes the scores for every key
at once, packs the probabilities into the A-panel layout for the P·V
product, and runs a three-pass softmax over rows the length of the key
dimension. oneDNN's kernel keeps a key block's scores in L1, corrects a
running softmax, and reads the probabilities as they are. Do the same:
a micro-kernel variant that reads its left operand row-major (nothing
packed per row group), and an online softmax over key blocks. The M2's
AMX path keeps the packed layout it needs and is unchanged.

## Design

- **`kernel.GemmRM` / `GemmRMZero`** (`gemmRMAVX512x14`, `gemmRMAVX2`):
  the existing micro-kernels with the A element broadcasts taken from a
  row-major tile (`a` with row stride `lda` floats) instead of a k-major
  panel: rows 0–6 addressed from one base pointer with offsets lda·i,
  rows 7–13 from a second base, both advanced by one float per k step;
  B stays a packed NR-wide panel, C as before. Same FMA count, same
  register budget (28 accumulators, 2 B vectors, broadcast temporaries).
  nil on back-ends without it (NEON, AMX, generic), verified at start-up
  against the generic kernel on the same inputs like `Gemm`.
- **`blas.AttentionBlock`** takes the row-major path when `GemmRM` is
  available: the query rows are copied once into a contiguous [rPad×D]
  scratch (zero rows beyond `rows`); per MR-row group and per key block
  of `attentionKeyBlock` keys (128: 14 × 128 scores are 7 KB) it computes
  the block's scores with `GemmRMZero(D, q rows, lda D, K panels)`, runs
  the online softmax per row (mask on the block slice; block max; new
  running max m'; correction c = exp(a·(m − m')); l = l·c + ExpSum(row,
  a, −a·m'); O row scaled by c when c ≠ 1; columns beyond S in the last
  block set to 0), then accumulates O[MR×DPad] += P[MR×KB] · V[block]
  with `GemmRM(KB, P row-major, lda KBpad, V panel slice)`; the packed V
  of T-041 already holds each key block contiguously per column panel.
  Rows leave scaled by 1/l. Without `GemmRM` the T-041 path runs
  unchanged. The mask callback gains the block's first key.
- The exponential stays `ExpSum` (exact class); a shorter polynomial is
  a follow-up only if the profile after this still shows it.

## Acceptance

- Fused equals composed to 1e-4 relative / 1e-5 absolute on the T-041
  shape matrix (rows not a multiple of MR, keys not a multiple of NR or
  of the key block, D 3/40/64/256, single rows, 1753 keys, all masks,
  grouped-query views, 3-D/4-D/5-D), on AVX2 under Rosetta (the
  row-major path), natively on the M2 (the packed path, unchanged) and
  under `FIBERAI_KERNEL=generic`; `GemmRM` verified at start-up; Gemma
  parity unchanged. `go test ./...` on both machines.
- Baselines (2026-09-09, pinned Xeon socket; PyTorch 2.14 SDPA same
  day): attention [8×8×512×64] 693 GFLOPS (PyTorch 1 141), causal 662
  (1 129), long sequence [1×8×2048×64] 748 (1 141); M2 1 110 / 1 070 /
  1 120 (PyTorch 553 / 566 / 653). Targets: Xeon ≥ 850 unmasked and
  ≥ 900 on the long sequence, where the L1-resident blocks matter most;
  M2 unchanged within noise; EmbeddingGemma on the Xeon ≥ 60.
- Manual: performance.md's attention section describes the block
  structure and the two paths.

## Notes

- Built and verified here on AVX2 under Rosetta (row-major path, full
  shape matrix, start-up verification of `GemmRM`/`GemmRMZero`) and
  natively on the M2 (packed path unchanged: 1 130–1 136 GFLOPS after a
  cool-down; a first reading of 923 right after the test suites was
  thermal). AVX-512 variant assembled here, verified on the Xeon.
- Xeon measurement pending, both back-ends (`avx512` default and
  `FIBERAI_KERNEL=avx2`, which is the deployment case).
