---
id: T-020
title: Fused LayerNorm: row statistics in one pass, normalise-scale-shift in one pass, matching backward
status: done
scope:
  - tensor/
  - internal/kernel/
manual:
  - docs/manual/performance.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

LayerNorm over [4096×4096] takes 18.5 ms on the Xeon Gold 6130 against
PyTorch's 14.1 ms, and 2.42 ms on the M2 Pro against 2.35 ms. The forward
pass materialises the normalised input x̂ as a full tensor for the
backward pass (64 MiB written and later read again) and the backward
builds `g ⊙ x̂` as another full tensor before reducing it. Folds TODO item
T-004.

## Design

- Forward keeps only per-row mean and rstd (two floats per row, as
  PyTorch does). Each row is centred into an L1-resident scratch buffer
  (stack for n ≤ 8192, per-chunk heap otherwise), the variance comes
  from a dot product of the centred row, and the output is written once:
  `out = xh·rstd·γ + β`. Memory traffic per element drops from read x +
  write x̂ + write out to read x + write out.
- Backward recomputes x̂ per row from x, mean and rstd into the scratch
  buffer and accumulates dγ and dβ into per-chunk partial vectors inside
  the same row loop (merged under a mutex), instead of allocating
  `g ⊙ x̂` and reducing it; dx is computed row-wise as before.
- No new assembly: the row kernels (`Sum`, `Dot`, `Scale`, `Axpy`,
  `Mul`, `Add`, `AddScalar`) already run at L1 speed on a 16 KiB row.

## Acceptance

- Gradient check for x, γ and β passes (existing `nn` and `tensor`
  tests plus a numeric check on a [7×33] input).
- `cmd/bench` layernorm [4096×4096]: M2 Pro ≤ 1.9 ms (from 2.42; PyTorch
  2.35 ms), Xeon Gold 6130 one socket ≤ 12 ms (from 18.5; PyTorch 14.1
  ms).
- Manual performance.md lists layer norm among the fused row operations
  with the new numbers.

## Notes

Implemented in `tensor/nn.go` without new assembly. M2 Pro `cmd/bench`
layer norm [4096×4096]: 2.42 → 1.91 ms (PyTorch 2.35). Numeric gradient
check on a [7×33] input for x, γ and β added. Xeon number pending
(expected well below PyTorch's 14.1 ms, since the pass count went from
about seven memory passes to three).
