---
id: T-041
title: Attention from the micro-kernel: K and V packed once per head, probabilities packed once, fused exp-sum softmax
status: done
scope:
  - tensor/
  - internal/blas/
  - internal/kernel/
  - cmd/bench/
  - benchmarks/python/
manual:
  - docs/manual/performance.md
  - docs/manual/nn-and-optim.md
done: 2026-09-09
created: 2026-09-09
---

## Goal

Attention runs at 42 % of the GEMM rate on both machines (M2 Pro 947
against 2 269 GFLOPS, Xeon 479 against 1 156) and on the Xeon 2.3×
behind PyTorch's `scaled_dot_product_attention` (1 091). Each fused
task today hands two general GEMMs to the blocked driver with one
worker: Q·Kᵀ with depth 64 and P·V with 64 columns, each packing both
operands per call. So K and V are repacked for every row block of the
same head, the probabilities are written by the softmax and then packed
a second time, and the softmax itself is seven passes over each score
row. Compute attention from the micro-kernel directly: K and V packed
once per head, the query rows and the probabilities packed once per
task, the softmax in as few passes as the kernels allow, the
normalisation applied to the 64-wide output rows instead of the
512-wide probability rows.

## Design

- **Packing phase.** Per leading index (batch × head) pack Kᵀ into B
  panels (depth D, NR-wide key panels, zero-padded to roundUp(S, NR))
  and V into B panels (depth S, NR-wide column panels over D, padded to
  roundUp(D, NR)); one parallel round over heads × panels using the
  existing `packBPanel` (exported to the tensor layer as needed).
  Memory is 2·nb·S·D floats (16 MB for the bench shape); heads are
  processed in groups that keep the packed operands under a budget
  (64 MB). Leading indices whose k or v offset repeats (grouped-query
  `Expand` views, stride 0 over heads) share one packing.
- **Task = (head, row block)** of `attentionRows(S)` rows. Pack the q
  rows into A panels (depth D) with `packA`, zero rows beyond `rows`;
  compute the score tile [roundUp(rows, MR) × roundUp(S, NR)] in
  per-task scratch with `kernel.GemmZero` per MR×NR tile (clear +
  `Gemm` on back-ends without it), no driver in between. Softmax per
  row over the first S columns: Scale and the mask Add as today, Max,
  AddScalar(−max), then **`kernel.ExpSum`** (new: z = exp(x) and the
  sum of z in one pass; NEON and AVX2 by adding an accumulator to the
  existing exp loops, which the AVX-512 back-end shares, generic Go
  otherwise; inputs below the clamp flush to 0 like `Exp`), keeping
  1/Σ per row aside; padded columns are set to 0. Pack P into A panels
  (depth S) with `packA`, run the micro-kernel over V's panels into an
  out tile [roundUp(rows, MR) × roundUp(D, NR)] in scratch, and copy
  each row into the output scaled by its 1/Σ (`kernel.Scale` into the
  destination): the normalisation costs a D-wide pass instead of an
  S-wide one. `GemmBegin`/`GemmEnd` bracket each task as in the driver.
- **Depth.** The micro-kernel takes the whole depth in one call (D for
  the scores, S for P·V); for S ≤ 2048 the panels stay L2-resident on
  the M2 and, at 1 MB per core, on the Xeon up to about S = 1024. If the
  long-sequence row shows the spill, block the depth at KC with the
  accumulating kernel; recorded as a follow-up if not done here.
- The composed path stays for the shapes the fused path declines and
  for gradient recording; the interface (`Attention`, `AttentionScaled`,
  the masks) does not change.
- **`cmd/bench` and `bench.py`:** one more attention row, [1×8×2048×64]
  (a long sequence with 8 heads: few tasks, long rows), so the depth
  question is measured on both sides.

## Acceptance

- Fused equals composed to 1e-4 relative / 1e-5 absolute on D in {64,
  256, 40, 3}, S in {512, 65, 1753, 7}, T in {512, 65, 33, 1}, with no
  mask, causal, padding and window masks, grouped-query `Expand` views
  of k and v, and one to three leading dimensions; on the native
  back-end, under `FIBERAI_KERNEL=generic`, and on AVX2 under Rosetta.
  The existing attention and Gemma tests pass; Gemma parity stays
  cosine 1.000000 short and ≥ 0.9999 long.
- Baselines (PyTorch SDPA float32, [8×8×512×64], 2026-09-08): M2 Pro
  553 GFLOPS, Xeon Gold 6130 one socket 1 091; fiber/ai today 947 and
  479. Targets: M2 ≥ 1 400, Xeon ≥ 800 (PyTorch may stay ahead on the
  Xeon; BENCHMARKS.md says so if it does); the causal row within 10 %
  of the unmasked one; EmbeddingGemma not slower than 105 sentences/s
  on the M2.
- Manual: performance.md gets an attention paragraph (what is packed
  once, where the time goes, the long-sequence row); the attention
  section of nn-and-optim.md points to it.

## Notes

- M2 Pro results, same day: [8×8×512×64] 947 → 1 103–1 113 GFLOPS,
  causal ~900 → 1 068–1 076, long sequence [1×8×2048×64] 1 115–1 123
  (PyTorch 652.5). The 1 400 target is not met on the M2: the profile
  after the change is half AMX products (depth-64 score tiles pay a
  32×32 store per 64 steps), a third `expSumNEON` (16.8 M exponentials
  per call at 0.4 ns each, compute-bound at the polynomial's op count),
  the rest the probability pack and row scaling; there is no overhead
  left to remove without a cheaper exponential. Removing the
  per-task GEMM driver calls and repacking was worth 17 %. Xeon pending
  (target ≥ 800 from 479).
- `ExpSum` grew the affine form z = exp(a·x + b) so the softmax needs
  no scale or shift pass; the mask is added to the raw scores as
  invScale·mask (the scale is applied afterwards), which is why the
  mask callback receives invScale. Negative or zero scales scale first.
- Per-task scratch is MR rows wide: the block size only balances the
  load (`attentionRows`), 256 rows unless that leaves fewer than four
  tasks per worker.
- Follow-ups if measurement asks for them: depth blocking at KC for
  very long sequences on the Xeon; a strided-store ExpSum writing the
  probabilities in panel layout would remove the pack pass (13 % on
  the M2); using the efficiency cores for the NEON share of the work
  (the AMX cap leaves four idle on the M2 Pro).
