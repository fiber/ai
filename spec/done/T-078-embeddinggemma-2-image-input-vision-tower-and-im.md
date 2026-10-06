---
id: T-078
title: EmbeddingGemma 2 image input: vision tower and image preprocessing
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

T-077 runs EmbeddingGemma 2's text tower. The checkpoint maps images into
the same 768-dimensional space, which is what makes it more than a text
model: a photo of a switch rack and the sentence "a rack of network
switches" land near each other, and a mixed input — a product description
with its pictures — becomes one vector. This spec adds image input:
images alone, and text interleaved with images, through the same `Model`.

Video in the reference is frames sampled at one per second and run
through the same vision tower; once images work, frames supplied by the
caller are the same path. Decoding video files is not in scope. Audio is
its own tower and its own spec.

## Design

Reference: `transformers` main, `gemma4/modeling_gemma4.py`
(`Gemma4VisionModel` and its parts), `gemma4/image_processing_gemma4.py`
(the default, torchvision-backed processor) and
`embedding_gemma2/processing_embedding_gemma2.py`, read in full.

**Preprocessing**, in Go, from an `image.Image`:

- Aspect-ratio-preserving size: the largest height and width that are
  multiples of 48 (patch 16 × pooling 3) and fit 280 × 9 = 2520
  patches, with the reference's edge cases for extreme aspect ratios.
- Bicubic resize with antialiasing, on 8-bit pixels, matching the
  reference's resampler. This is the part most likely to differ by a
  rounding step, so it is checked at the pixel level during development
  and the result recorded.
- Rescale to [0, 1], patchify to rows of 16·16·3 values in the
  reference's (row, column, channel) order, and (x, y) patch positions.

**Vision tower** (16 layers, width 768, 12 heads of 64):

- Patch embedding: `2·(p − 0.5)` through a 768 → 768 projection, plus
  learned x and y position embeddings from a [2, 10240, 768] table.
- Encoder layers: the same sandwich norms, q/k RMSNorm, weight-free
  v-norm and unscaled scores as the text tower, a gated GELU MLP to
  3072, and **axial 2-D RoPE**: the first half of each head rotates by
  the patch's x position, the second half by its y position, base 100.
  Attention is bidirectional over all patches of one image.
- Pooling: average over 3×3 patch blocks by position, giving at most
  280 soft tokens, times √768.
- `embed_vision`: weight-free RMSNorm, then 768 → 512 into the text
  model's width.

**Sequence.** An image becomes `<|image>`, one `<|image|>` per soft
token, `<image|>` inside the text sequence; the soft tokens replace the
placeholders' embeddings (unscaled, as in the reference), the text tower
runs over the whole sequence including BOS and EOS, and the result is
mean-pooled and normalised as for text. `encode2` is split so the text
tower can start from embeddings rather than token ids.

**API**: `EmbedImages(images []image.Image, opts...)` for images alone,
and `EmbedInputs(inputs []Input, opts...)` where `Input` holds text with
`<|image|>` placeholders and the images that fill them, in order — the
form the model card shows. `Dim` and prompts work as for text. A
placeholder without an image, or an image without a placeholder, is an
error.

## Acceptance

- Parity against the reference (sentence-transformers 6.1, float32) on a
  set of images that covers portrait, landscape, square, small (upscaled)
  and large (downscaled) inputs, a grayscale image and a natural photo:
  mean cosine ≥ 0.9995, minimum ≥ 0.999, the text tower's thresholds.
  Images are PNG so both sides see the same pixels; JPEG decoding
  differences between Go and libjpeg are measured separately and
  reported, not hidden in the parity numbers.
- A mixed text-and-image input matches the reference on the same
  thresholds.
- A semantic check that does not depend on the reference: synthetic
  images of distinct shapes and colours are each nearest to their own
  description among the set.
- The text-only path and its tests are unchanged.
- `docs/manual/models.md` documents image input, what it costs per image
  on the M2 against PyTorch on the same machine, and what is not
  supported.
- No performance impact on existing paths; the vision tower is a new
  path.

## Notes

Measured on the M2 Pro with nothing else running. Parity on eight
images: cosine 1.000000 mean, 0.999997 minimum (the 40×30 noise image);
the mixed text-and-image input and the prompted image 1.000000 each.
Each of four images is nearest to its own description among four texts.
Cost: 0.91 s per image, against PyTorch (8 threads) 1.06 s with batches
of 8 and 1.27 s one at a time.

**The resize.** The reference uses torchvision's bicubic resize with
antialiasing on 8-bit tensors, which on CPU takes PyTorch's native uint8
path. The Go code is a direct port of Pillow's Resample.c (22-bit fixed
point, horizontal pass first, 8-bit intermediate). Pixel comparison:
sizes, patch counts and positions identical for all eight images, four
images bit-exact, the others differing by 1–2 levels in 0.003–0.86 % of
values, only where the image is enlarged. The reference's pixels differ
from Pillow's resize by exactly the same counts, value for value, so the
Go output is Pillow's, and the difference is PyTorch's ARM uint8 kernel
departing from Pillow; on x86 PyTorch uses an AVX kernel written to
match Pillow. The 0.999997 is that difference.

**JPEG.** Go's decoder and libjpeg disagree slightly; the Apollo 17 photo
decoded by each embeds at cosine 0.99978. The parity images are PNG so
the comparison measures the model, and the decoder effect is a separate,
reported test.

The vision tower is loaded on the first image (`sync.Once`), so a
text-only program never holds its 680 MB. Images run through the tower
one at a time; the text tower batches the resulting sequences as for
text. Batching images through the vision tower, and the alternative soft
token budgets the model card mentions (70 to 1120), are follow-ups.
