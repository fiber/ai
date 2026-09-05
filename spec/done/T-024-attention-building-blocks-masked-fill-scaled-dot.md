---
id: T-024
title: Attention building blocks: masked fill, scaled dot-product attention, multi-head attention, RMSNorm, causal and padding masks
status: done
scope:
  - tensor/
  - nn/
  - examples/
  - cmd/bench/
manual:
  - docs/manual/nn-and-optim.md
  - docs/manual/tensors.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

Transformers are the models people want to run: EmbeddingGemma for the
syslog project, any sequence model for events. Their one new operation
is attention; everything else (Linear, GELU, LayerNorm, Softmax,
Embedding) exists. This adds attention and the two pieces around it
that Gemma-class models use, RMSNorm and token lookup, so that a
transformer block can be written in fiber/ai. Conv1D/Conv2D (the other
half of TODO item T-009) stay open.

## Design

- **Masks are additive.** `tensor.CausalMask(n)` returns an [n×n] tensor
  with 0 on and below the diagonal and −1e9 above; `tensor.PaddingMask(
  lengths []int, n)` returns [batch×1×1×n] with −1e9 beyond each length.
  Adding them to the scores broadcasts over batch and heads and needs no
  new kernel; softmax turns −1e9 into zero weight. No masked-fill op.
- `tensor.Attention(q, k, v, mask *Tensor) *Tensor`: q [B,H,T,D],
  k and v [B,H,S,D]; scores = q·kᵀ/√D (+ mask), softmax over S, times v →
  [B,H,T,D]. Batched matrix products with strided operands; autograd
  from the constituent operations.
- `nn.MultiHeadAttention{Heads, Dim}` with projections `Q, K, V, O
  *Linear`: `Forward(x)` self-attention with an optional mask set on the
  module (`Mask *tensor.Tensor`), `Cross(x, context)` for encoder–decoder
  use. Head split and merge through `Reshape` and `Permute`, so no copies
  until the products.
- `nn.RMSNorm{G, Eps}` and `tensor.RMSNorm(x, g, eps)`: x / √(mean(x²)+ε)
  · g over the last dimension, composed from existing operations (a
  fused kernel can follow if a profile asks for it).
- `Embedding.Lookup(ids []int)` gathers rows with `Rows`; the existing
  one-hot `Forward` stays.
- `cmd/bench` gains an attention row (batch 8, 8 heads, 512 tokens,
  head dim 64, forward under NoGrad) so the Python comparison can be
  extended with the same shape.

## Acceptance

- `Attention` matches a naive loop implementation to 1e-5 on random
  inputs, with and without a causal mask; padding-masked positions get
  zero weight.
- `MultiHeadAttention` and `RMSNorm` pass a numeric gradient check on
  small shapes; `Lookup` gradients scatter-add like `Rows`.
- A two-layer transformer block built from these modules trains on a
  toy sequence task (predict the next token of a repeating pattern) to
  over 95 % accuracy in the example under `examples/nn/attention`.
- Performance: the attention row runs at ≥ 60 % of the GEMM rate of its
  two products on the M2 Pro (the rest is softmax and the transposes);
  the PyTorch figure for the same shape is to be added to the Python
  bench in the next x86 round. No impact on existing operations.
- Manual: nn-and-optim.md documents the modules, tensors.md the masks
  and `Attention`.

## Notes

Implemented: `tensor.CausalMask`, `PaddingMask`, `Attention`, `RMSNorm`;
`nn.MultiHeadAttention` (self and cross), `nn.RMSNorm`,
`Embedding.Lookup` now via `Rows`. Tests: attention against a naive
loop with and without a padding mask, causal mask hides the future,
numeric gradient checks for RMSNorm and the attention module.
`examples/nn/attention`: a two-layer pre-norm transformer (102 476
parameters) reaches 96.9 % next-token accuracy on repeating patterns
after 400 steps in 4 s on the M2 Pro (accuracy counted from position 6,
before one period has passed the next token is unknowable). Bench row
[8×8×512×64]: 11.6 ms, 370 GFLOPS over the two products, 13.0 ms with
the causal mask. That is below the 60 % criterion against the GEMM rate
of the products (~1 TFLOPS on AMX for these shapes): the score matrix
is 64 MiB per call and the softmax and mask passes stream it three
times, plus a forced collection every four calls. A fused attention
(scores per row block, softmax in registers, never materialising the
full matrix) is the follow-up, recorded in TODO as T-025; the PyTorch
figure for the same shape joins the Python bench in the next x86 round.
