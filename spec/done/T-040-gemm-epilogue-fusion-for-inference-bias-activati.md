---
id: T-040
title: "GEMM epilogue fusion for inference: bias, activation, gated product, residual and row scale applied on the hot output block"
status: done
scope:
  - internal/blas/
  - tensor/
  - nn/
  - models/gemma/
  - cmd/bench/
manual:
  - docs/manual/performance.md
  - docs/manual/nn-and-optim.md
done: 2026-09-09
created: 2026-09-09
---

## Goal

On the Xeon an EmbeddingGemma batch takes 740 ms of which the matrix
products are about 360; the rest is element-wise work between them
(146 RMSNorms, GELU, the gated product, residual adds), each a full pass
over 4–6 MB through a memory system with a tenth of the M2's bandwidth.
A Gemma feed-forward block is seven passes for three products. Apply
those operations on the product's output block while it is still in
cache, right after the last K block has been accumulated, and remove the
separate passes from the inference paths that dominate: `nn.Linear`
with bias and a following activation, and the Gemma block.

## Design

- **`blas.Epilogue`** carries what is applied to the finished output:
  `Bias []float32` (per column), `Act` (None, ReLU, GELU), `Mul Mat`
  (element-wise multiply, for the gated GELU(gate)·up), `Residual Mat`
  (added), `RowScale []float32` (per row, for a folded RMSNorm). Order:
  bias, row scale, activation, mul, residual (the row scale precedes the
  activation so a folded norm applies to the pre-activation).
  `GemmZeroEpilogue(c, a, b, p *PackedB, e Epilogue, workers)` is the
  entry.
- **Where it runs.** In the shared strategy the compute tasks of a row
  block are grouped so that a group covers at least `epilogueCols`
  (256) columns; on the last K block the task that finishes a group
  applies the epilogue to the group's region (`ib` rows × the group's
  columns) row by row with the existing vector kernels (Add, Scale,
  MaxScalar, GELU, Mul), tracked with one atomic counter per group. The
  region was just written by the micro-kernels and sits in L2. Per
  task (a few panels wide) the kernels were dispatch-bound: GELU alone
  cost 5 % of the EmbeddingGemma forward and the fused encoder was 8 %
  slower than the unfused one; per whole row block one core ran the
  block's epilogue while the others idled and the MLP gained nothing.
  The rows strategy, the matrix-vector paths and the few-rows path
  apply the epilogue over the whole output after the product; correct
  everywhere, fused where it matters.
- **`tensor.MatMulFused(x, w, Fused{...})`**: the inference entry; when a
  gradient is being recorded for any operand it falls back to the
  composed operations, so results and autograd are unchanged. Uses the
  packed-operand cache for `w` like `MatMul`.
- **`nn.Linear.Forward`** fuses the bias under NoGrad; **`nn.Sequential`**
  fuses a `ReLU` or `GELU` that directly follows a `Linear` under NoGrad
  (the inference forward of the MLP benchmark becomes three products
  and nothing else).
- **`models/gemma`:** the pre-norms are folded: RMSNorm(x)·W equals
  rowscale(1/rms(x)) ⊙ (x · (g ⊙ W)), so the loader keeps W' = diag(g)·W
  (cached packed like any weight) and the encoder computes one
  sum-of-squares per row (a read-only pass) and passes `RowScale`; the
  q, k, v products share the scale, as do gate and up. The gated product
  becomes one fused call: `gate = h·W'gate` with `Act: GELU, Mul: up`.
  The post-norms (applied to a product's output before the residual)
  need the whole row and stay as they are. Per layer that removes the
  writes and re-reads of two pre-norms and of the gate, GELU and product
  intermediates: about 40 MB of 90 MB traffic.
- `cmd/bench` needs no new rows: MLP forward and EmbeddingGemma show
  the effect; the MLP row's "forward (NoGrad)" is the fused path.

## Acceptance

- Fused products equal the composed ones to 1e-5 for every epilogue
  combination on shapes covering tails, K > KC (several K blocks, the
  epilogue applied once), the gemv paths (m = 1, n = 1) and the rows
  strategy; gradient-recording calls take the composed path and the
  autograd tests are unchanged; Gemma parity stays at cosine ≥ 0.9999
  on the 64 sentences and the long inputs.
- Baselines (PyTorch, same day): MLP forward 526 K samples/s on the M2,
  361 K on the Xeon; EmbeddingGemma 88 and 37 sentences/s. Today:
  fiber/ai 835 K / 380 K and 96 / 43–47. Targets: MLP forward ≥ 950 K
  (M2) and ≥ 450 K (Xeon); EmbeddingGemma ≥ 105 (M2) and ≥ 55 (Xeon);
  training rows unchanged within noise.
- Manual: performance page describes the fusion and what qualifies;
  nn page notes that `Linear` and `Sequential` fuse under NoGrad.

## Notes

- Results, M2 Pro, same day, cache warm: MLP forward 837–845 K → 871–877 K
  samples/s (target 950 K not met: at 256 rows the three products are
  per-call-overhead-bound, the passes between them were never the
  cost); EmbeddingGemma 92–95 → 105–108 sentences/s (target 105 met).
  Training rows unchanged within noise (forward + backward 188–193 K,
  with Adam 160–164 K). Gemma parity 1.000000 short and long. Xeon
  pending.
- `kernel.GELU` is not alias-safe (it re-reads x after writing z per
  block), so the epilogue copies the row to scratch first; the copy is
  negligible at 256+ columns.
- The `Mul` operand of the gated product is the `up` projection, which
  is computed first and read once in the gate's epilogue; at 2080×1152
  it is not in cache any more, but that is one read instead of the
  former write + read + write + read of gate, GELU and product.
