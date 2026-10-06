---
id: T-077
title: EmbeddingGemma 2 text encoder in models/gemma
status: done
scope:
  - models/gemma/
  - docs/manual/models.md
  - docs/manual/README.md
manual:
  - docs/manual/models.md
  - docs/manual/README.md
done: 2026-10-06
created: 2026-10-06
---

## Goal

Google released `google/embeddinggemma-2` on 2026-10-06. `gemma.Load`
runs EmbeddingGemma-300m (a Gemma 3 text encoder) and nothing else; the
new checkpoint has a different `model_type` (`embedding_gemma2`) and a
text tower that differs from the old one in a dozen places, so loading
it today fails or, worse, would produce wrong vectors if the config
check were loosened.

The text path is what nora/norad and tutorial chapters 10 and 18 use,
so it comes first: the same `gemma.Load(dir)` / `Embed` API, the new
checkpoint underneath, vectors matching the reference implementation.

The checkpoint also carries a Gemma 4 vision tower (16 layers, axial
2-D RoPE) and audio tower (12-layer conformer over mel features), each
with its own preprocessing. Those are separate work with their own
specs and are out of scope here; this spec loads none of their weights.

## Design

Reference: `transformers` main (5.18.0.dev0),
`models/embedding_gemma2/modeling_embedding_gemma2.py`, read in full.
The text tower against the 300m encoder, item by item:

- **Config.** `config.json` nests the text model under `text_config`.
  `LoadConfig` reads `model_type`; for `embedding_gemma2` it decodes the
  text config, including `per_layer_config` (full-attention layers use
  `head_dim` 512 and one KV head; sliding layers 256 and two),
  `rope_parameters` per layer type (10 000 sliding, 1 000 000 full),
  `hidden_size_per_layer_input` and `embedding_dim`.
- **RMSNorm is `x̂·w`, not Gemma 3's `x̂·(1+w)`.** The loader stops
  folding the +1 for this variant. Getting this wrong is silent, so a
  test checks the weights look like scales (mean near 1), not offsets.
- **Attention.** Per-layer head dimension and KV-head count, so `wq`,
  `wk`, `wv`, `wo` widths vary by layer. Scaling is 1.0 (no
  `query_pre_attn_scalar`). Values get an unscaled RMSNorm over the head
  dimension (`v_norm`) before attention. RoPE dimension follows the
  layer's head dimension.
- **Sliding window** is an inclusive radius: a sliding layer attends
  where |i − j| ≤ 512, so the bound our `WindowMask` takes is 513. The
  300m model's bound was `sliding_window/2 + 1`; this is not the same
  rule and is tested with an input longer than the window.
- **Per-layer embeddings (PLE).** Once per forward pass: the scaled
  token embeddings times `per_layer_model_projection` (512 → 24·512),
  times hidden⁻⁰·⁵, reshaped to one 512-vector per layer per token and
  RMS-normalised. In every layer, after attention and MLP, a third
  residual block: `x + norm(proj(gelu(gate(x)) ⊙ ple_l))`, then the
  whole layer output is multiplied by the layer's stored `layer_scalar`.
- **Head.** Final RMSNorm, then `embedding_projection` (512 → 768), mean
  pooling including the prompt, L2 normalisation. There is no
  sentence-transformers Dense module any more; the projection is linear,
  so it is applied after pooling, which is equivalent and B·T/B times
  cheaper.
- **Unchanged:** scaled token embeddings, gated GELU-tanh MLP with the
  four sandwich norms, q/k norms, bidirectional attention, the tokenizer
  and the prompt table, batching by length.

Implementation stays in `models/gemma`, behind the same `Load`/`Embed`
API, so a caller switches models by switching directories. The variant
is detected from `model_type`, and the norm folding into the following
products keeps working because it never depended on the +1.

## Acceptance

- Parity against the reference (`transformers` main + sentence-transformers,
  float32, CPU) on the 300m test sentences plus a prompted query and
  document: mean cosine ≥ 0.9995 and minimum ≥ 0.999, the thresholds the
  300m parity test uses.
- A long input (> 513 tokens) matches the reference too, which is the
  test of the window rule.
- `gemma.Load` on the 300m directory still passes its existing parity
  tests unchanged.
- Loading reports a clear error when asked for a modality the package
  does not implement, rather than ignoring image or audio input.
- `docs/manual/models.md` documents the new checkpoint, what is and is
  not loaded, and its throughput on the M2 next to the 300m model.
- No performance impact on existing paths: a new code path for the new
  variant, the 300m encode path untouched.

## Notes

Measured on the M2 Pro with nothing else running. Parity: 64 sentences
mean and minimum cosine 1.000000, prompted query and document 1.000000,
long inputs (879, 1147, 1753 tokens) 1.000000. The 300m tests pass
unchanged. Speed, 32 short sentences, best of five after warm-up:
fiber/ai 394 (300m) and 256 (EmbeddingGemma 2) sentences/s, PyTorch
8 threads 240 and 164. On 421-token documents 13.6 and 9.2 docs/s.

The reference environment is a separate venv (/Users/sven/models/venv-eg2:
torch 2.14, transformers main 5.19.0.dev0, sentence-transformers 6.1,
plus Pillow and torchvision, which the model's processor imports even for
text), so the 300m references stay reproducible with the old one.
`testdata/make_reference_eg2.py` writes the reference files.

Three things found on the way, none of which a reading of the model
code alone would have shown:

- **Grouped-query attention with more than one KV head.** The 300m model
  has one, and `AttentionScaled` expresses GQA by a stride-0 `Expand` of
  that head, which cannot widen two heads to four. The query heads are
  grouped as [B, Hkv, rep, T, D] instead, so query head h reads KV head
  h/rep as the reference's `repeat_kv` does; the fused kernel handles the
  extra leading axis unchanged (checked against materialised heads).
- **A use-after-recycle, B-009's class again.** With one KV head,
  [B,T,1,D] permuted to [B,1,T,D] is already contiguous, so `Contiguous`
  returns a view of the same storage and recycling the source freed what
  attention then read. It showed only in padded batches (single inputs
  matched at 1e-6 through all 24 layers, batched inputs at cosine 0.79),
  because whether the freed buffer is reused in time depends on the
  allocation pattern. The 300m path guards this with `rel`, which refuses
  shared storage; the new path now does the same where no copy happened.
  Found by tracing every layer against reference activations.
- **The window rule matters by 0.4 %.** With the 300m model's
  half-window rule the long inputs come out at 0.993–0.996: plausible and
  wrong. The long-input test was checked against that deliberately wrong
  rule to make sure it fails.

The speed gap to the 300m model is the architecture, not the code: the
text tower does 136 M multiply-adds per token against 101 M, and PyTorch
shows the same ratio (1.46× against our 1.54×). A suspected pack-cache
problem was measured and ruled out: the packed weights need 521 MB, three
matrices did not fit the 512 MB default and were repacked per call, and
raising the limit changed throughput by 0.3 %.

The acceptance criterion about refusing other modalities is met as the
API allows: `Embed` takes text only, so the one way image or audio input
can reach it is a placeholder token pasted into the text, and those are
refused with an error naming the modality.

Out of scope and not started: the vision tower (Gemma 4, 16 layers,
axial 2-D RoPE, 280 soft tokens per image) and the audio tower (12-layer
conformer over mel features), each with its own preprocessing.
