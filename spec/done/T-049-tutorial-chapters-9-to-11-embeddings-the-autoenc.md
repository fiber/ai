---
id: T-049
title: Tutorial chapters 9 to 11: embeddings, the autoencoder, attention
status: done
scope:
  - docs/tutorial/
  - examples/tutorial/
manual:
  - none
done: 2026-09-09
created: 2026-09-09
---

## Goal

Three chapters on the ideas a Go developer building log and counter
systems still lacks after chapter 8: a model that turns text into
geometry (embeddings, no training loop), a model whose only output is
how unusual its input is (the autoencoder), and a layer that chooses
what to look at (attention, written by hand so its weights can be
read). Each reuses data the repository ships and runs in seconds; the
embedding chapter needs the EmbeddingGemma directory from the models
manual and says so if it is missing.

## Design

- **9 Embeddings** (`examples/tutorial/09-embeddings`): sixteen syslog
  templates embedded with `models/gemma` (`FIBERAI_MODELS` or `-model`);
  a vector is 768 numbers; cosine as the ruler; the most and least
  similar pair; "interface down" against "interface up" (topic, not
  polarity); three plain-language queries with their nearest templates.
- **10 The autoencoder** (`examples/tutorial/10-autoencoder`): per
  minute the counters of all eight airports (in zone, arrivals,
  departures: 24 numbers), 24 → 16 → 3 → 16 → 24, trained on four
  ordinary days; the reconstruction error as the score, the 99.9th
  percentile of training as the threshold; the validation day stays
  quiet, the storm day does not, per hour, with the airport and counter
  that contribute most.
- **11 Attention** (`examples/tutorial/11-attention`): chapter 8's
  forecast (Dublin, eight airports, 120 minutes, 30 ahead) with a
  hand-written attention pooling: each minute a token of ten numbers
  (eight airports, two position features), keys and values from two
  weight matrices, one learned query, softmax over the 120 minutes,
  weighted sum, a small head; the weights of a trained model for one
  window printed as a text bar chart per ten minutes; MAE next to
  chapter 8's models.
- Texts in `docs/tutorial/09-embeddings.md`, `10-autoencoder.md`,
  `11-attention.md`; README rows; "Eight chapters" becomes eleven.

## Acceptance

- Each program runs in a few seconds on a laptop (chapter 9 after a
  one-time model download) and prints what its chapter quotes; chapter
  10 raises no or almost no alarms on the validation day and several
  storm-day hours; chapter 11's attention weights are readable and its
  MAE is in the range of chapter 8's models. `go vet ./...`.
- No performance impact (tutorial and examples only).

## Notes
