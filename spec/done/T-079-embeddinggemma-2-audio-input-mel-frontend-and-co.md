---
id: T-079
title: EmbeddingGemma 2 audio input: mel frontend and conformer tower
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

T-077 and T-078 run EmbeddingGemma 2's text and image paths. The fourth
modality the checkpoint maps into the same space is audio: a spoken
sentence and its transcript land near each other, and a clip can be
searched with words. This spec adds audio input through the same
`Model`, completing the checkpoint except for video files, which the
model reads as image frames (T-078's path) and which need a video
decoder this repository does not have.

## Design

Reference: `transformers` main, `gemma4/feature_extraction_gemma4.py`,
`gemma4/modeling_gemma4.py` (`Gemma4AudioModel` and its parts),
`audio_utils.py` (window, mel filter bank) and
`embedding_gemma2/processing_embedding_gemma2.py`, read in full.

**Input** is 16 kHz mono samples as `[]float32` in [−1, 1]; a small WAV
reader for PCM16 files (channels averaged) is provided, and other sample
rates are refused rather than resampled with a filter that differs from
the reference's.

**Mel frontend**, in Go: truncate to 30 s (480 000 samples), pad to a
multiple of 128 samples, 160 samples of left padding, frames of 320
samples every 160, periodic Hann window, a 512-point real FFT,
magnitudes (not power) through 128 HTK-scale triangular filters up to
8 kHz, `log(x + 0.001)`, and frames whose last sample is padding zeroed.
The FFT is a radix-2 implementation in the package; no new dependency.

**Audio tower** (a Gemma 4 conformer, 12 layers, width 1024, 8 heads of
128): two stride-2 3×3 convolutions with LayerNorm and ReLU (4× fewer
frames, 25 per second), a projection to 1024, then per layer two
half-weighted feed-forward blocks (SiLU), chunked local self-attention
— each frame attends to itself and the 12 before it, with a
Transformer-XL relative position term, learned per-dimension query
scale and a tanh soft cap of 50 — and a light convolution block (GLU,
causal depthwise convolution of width 5, SiLU). Every projection is a
*clipped linear*: input and output clamped to stored bounds, which the
checkpoint sets to finite values, so they are applied and the norm
cannot be folded into the weights as elsewhere. Output projection to
1536, then `embed_audio`: weight-free RMSNorm and 1536 → 512.

The chunked attention is implemented directly per query rather than by
the reference's block-and-shift construction: for query q and key k in
[q − 12, q] the relative term is q·r(q − k), and every other pair is
masked, which is what the shift produces (derived in the Notes).

**Sequence**: `<|audio>`, one `<|audio|>` per valid soft token, `<audio|>`
in the text, as for images. **API**: `EmbedAudio(clips [][]float32,
opts...)`, and `Input.Audio` filled by `<|audio|>` placeholders in
`EmbedInputs`. The audio tower (300 M parameters, about 1.2 GB in
float32) loads on first use, like the vision tower.

## Acceptance

- Parity against the reference (sentence-transformers 6.1, float32) on
  public-domain speech (LibriVox) of 3 s and 20 s, the same recording
  over 30 s (truncation), a tone, noise, near-silence and a clip shorter
  than one soft token's worth of padding: mean cosine ≥ 0.9995,
  minimum ≥ 0.999.
- The mel features match the reference's within float32 tolerance,
  checked during development and recorded.
- A semantic check without the reference: spoken sentences (synthesised
  with `say` at test time where available, never committed) are each
  nearest to their own transcript among the set.
- A mixed input of text and audio matches the reference.
- Text and image paths and their tests unchanged.
- `docs/manual/models.md` documents audio input, its cost per second of
  audio against PyTorch on the M2, and what is not supported.
- No performance impact on existing paths.

## Notes

Measured on the M2 Pro with nothing else running. Parity on seven clips
(3 s, 20 s and 40 s-truncated LibriVox speech, tone, noise, near-silence,
50 ms): cosine 1.000000 mean and minimum; mixed text and audio 1.000000.
Mel features: frame counts and masks identical, values within 4e-4 (the
reference's numpy 2 FFT is single precision). Each of four sentences
spoken by `say` at test time is nearest to its own transcript. Cost:
20 s clip 0.67 s, 3 s clip 0.18 s; PyTorch (8 threads) 1.20 s and 0.85 s.

**The window was one key too wide.** The first run gave 0.9976–0.9998,
with only the 50 ms clip (one token) exact — the error grew with the
number of frames, which pointed at attention. Gemma 4's sliding-window
mask function allows 0 ≤ q − k < left, where left = context_left − 1 = 12:
twelve keys, the current frame and eleven before it. The relative
position table has thirteen entries and the context window extracted per
block has twenty-four, which is how I had read thirteen keys; the mask
removes the thirteenth. The derivation of the direct per-query form
stands otherwise: for 0 ≤ d = q − k ≤ 12 the shifted term is q·r(d),
d = 12 is masked.

**Pinned memory.** The first multimodal assembly read the token
embedding table with Data(), which marks storage as escaped: 512 MB
(262 144 × 512 × 4 bytes, exactly) pinned for the model's lifetime and
not releasable by Close. The image path from T-078 had the same line.
Rows() looks the tokens up without pinning; both memory tests now report
0 bytes pinned.

**Test data and licences.** The speech is LibriVox (public domain); the
tone, noise and near-silence are synthetic. `say` output is not
committed: Apple's licence for the system voices covers personal,
non-commercial use, not redistribution in a public repository. The
retrieval test synthesises its sentences at run time and skips without
`say`.

Out of scope, as the spec said: video files (the model reads frames at
one per second through the vision tower; there is no decoder here) and
resampling (refused rather than done with a filter that differs from
the reference's).
