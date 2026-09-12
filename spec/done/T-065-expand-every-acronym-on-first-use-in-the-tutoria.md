---
id: T-065
title: Expand every acronym on first use in the tutorial
status: done
scope:
  - docs/tutorial/
manual:
  - none
done: 2026-09-12
created: 2026-09-12
---

## Goal

The tutorial is written for Go developers who never learned NumPy or
PyTorch, and it uses the field's acronyms without ever saying what they
stand for. `ReLU` first appears inside a code block in chapter 5 and is
discussed in the prose beside it without being expanded. `MLP` is used
bare in chapter 8, `GELU` in a code block in chapter 10, `MSELoss` in
chapter 6, `CNN` in a results table. A reader who does not already know
the field cannot look these up from the page; a reader who does know
them does not need the tutorial.

## Design

Expand each on first use, in the prose next to where it appears, in the
fewest words that carry the meaning:

- ReLU, chapter 5 — rectified linear unit
- MSELoss, chapter 6 — mean squared error, and what it averages
- MLP, chapter 8 — multilayer perceptron, with a pointer back to the
  chapters that built one
- CNN, chapter 8 — convolutional neural network
- GELU, chapter 10 — Gaussian error linear unit, and why it is used here
  rather than ReLU
- RMSNorm, chapter 13 — root-mean-square normalisation
- RoPE, chapter 13 — rotary position embedding
- nats, chapter 13 — the natural-logarithm unit of cross-entropy
- KV cache, chapters 13 and 14 — key-value cache

Expand once, at the first occurrence, and use the short form after. A
tutorial that re-expands an acronym in every chapter reads as though it
does not trust the reader to have read the previous one.

## Acceptance

- Every acronym above is expanded at its first occurrence in the
  tutorial, in prose rather than in a code comment.
- No acronym is expanded twice.
- No other text changes: this is a documentation edit and touches no
  code and no numbers.

## Notes

The manual has the same problem with GEMM, SIMD, AMX, NEON, AVX2,
AVX-512 and BPE. It is reference material for a reader who has already
finished the tutorial, so the case is weaker, but the first use of each
in `docs/manual/` deserves the same treatment. Left as a follow-up
rather than widened into this change.
