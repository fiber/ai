---
id: T-026
title: Fused attention for inference: per-row-block scores, softmax in cache, no materialised score matrix
status: done
scope:
  - tensor/
  - internal/blas/
  - cmd/bench/
  - examples/
manual:
  - docs/manual/tensors.md
  - docs/manual/performance.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

`tensor.Attention` materialises the [T×S] score matrix per batch and
head: for [8×8×512×64] that is 64 MiB written by the first product, read
and written by the mask and softmax passes, read again by the second
product, and returned to the allocator only at the next collection. The
call runs at 370 GFLOPS where its two products alone would run at about
a terraflop on the AMX. Inference (the syslog embedding model, any
served transformer) needs the fast path; training keeps the composed
one, whose backward autograd already provides. Folds the TODO entry
recorded as T-025 in T-024's notes.

## Design

- `Attention` dispatches: when no gradient is being recorded (`NoGrad`,
  or none of q, k, v requires one) it runs `attentionFused`; otherwise
  the composed path as before.
- `attentionFused`: for every (leading index, block of 64 query rows) a
  task computes the block's scores [64×S] with `blas.GemmZeroWorkers(…, 1)`
  into a per-task scratch (K is read as a strided view, no transpose
  copy), scales, adds the mask rows (a [T×S] mask or a padding mask row,
  both by broadcasting the row), runs the softmax row by row with the
  kernel primitives (`Max`, `AddScalar`, `Exp`, `Sum`, `Scale`), and
  multiplies the [64×S] probabilities with V into the output rows with a
  second single-worker GEMM. Tasks run under `parallel.For`; scratch
  buffers come from the packing-buffer free list. Nothing larger than a
  block ever leaves L2, and no [T×S] tensor is allocated.
- Leading dimensions of any rank; q, k, v may be strided views (head
  splits); the mask may be nil, [T×S] (broadcast over leading dims) or
  [B×1×1×S] (per batch), the two shapes `CausalMask` and `PaddingMask`
  produce, or their sum [B×1×T×S].

## Acceptance

- Fused and composed paths agree to 1e-5 on random inputs with no
  mask, a causal mask, a padding mask and their sum, for 3-D and 4-D
  inputs and for strided (permuted) q/k/v.
- `cmd/bench` attention row [8×8×512×64] under NoGrad: ≥ 800 GFLOPS on
  the M2 Pro (from 370), i.e. within 20 % of the GEMM rate of the two
  products, and no forced collections attributable to it. The PyTorch
  figure for the same shape is measured in the next x86 round (the
  Python bench gains the row); the composed training path is unchanged,
  no impact on other operations.
- tensors.md notes the two paths; performance.md lists attention among
  the fused operations with the figure.

## Notes

Implemented. `Attention` dispatches to `attentionFused` unless a
gradient is being recorded; fused and composed paths agree to 1e-5 for
no mask, [T×S], padding and summed masks, 3-D and permuted inputs.
[8×8×512×64] on the M2 Pro (AMX): 370 → 993 GFLOPS (11.6 → 4.3 ms), with
the causal mask 930; NEON 381. Query rows per task were the lever: 64
rows 702 GFLOPS, 128 rows 934, 256 rows 993 (K and V are repacked per
block, so fewer blocks win as long as the scores stay in L2); the block
is now sized to about 512 KiB of scores, never below 32 rows. No forced
collections in the run (the composed path forced one every four calls).
Acceptance met (≥ 800). The PyTorch figure for this shape goes into the
Python bench in the next x86 round.
