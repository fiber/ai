---
id: T-070
title: Tutorial chapter 18: a text classifier on frozen embeddings
status: done
scope:
  - docs/tutorial/
  - examples/tutorial/
  - docs/manual/applications.md
  - docs/manual/getting-started.md
  - README.md
manual:
  - docs/manual/applications.md
done: 2026-09-14
created: 2026-09-14
---
---

## Goal

Chapter 10 shows that EmbeddingGemma puts lines with similar meaning at
nearby points and searches sixteen templates by cosine. It stops short
of the thing a service actually needs: a decision per line. Route this
alert to the network team, that one to security, this ticket to
storage. The production pattern for that is not to train a language
model and not to fine-tune one: embed once with the frozen encoder and
train a small head on the vectors, which takes seconds, fits in the
same process, and can be retrained every time somebody corrects a
label. The tutorial has a language model and no chapter on the pattern
most readers will ship first.

The chapter must also say what the pattern buys over the obvious
alternative. A bag of words with a linear layer classifies lines it
has seen the words of. The embedding head classifies wordings it has
never seen, because the encoder already knows that "link flapping" and
"interface went down" are the same topic. That is a measurable claim.

## Design

A new chapter 18, `examples/tutorial/18-text-classifier`, appended after
the tokenizer chapter; it needs the Gemma weights like chapter 10 and
says so and stops without them.

- **Data.** Generated syslog-style lines in five categories (access,
  network, storage, jobs, hardware), each category written in about
  eight distinct *shapes* with random hosts, users, addresses and
  counts filled in. The split is by shape, not by line: some shapes of
  every category are held out entirely, so the test set contains
  wordings the head never saw. The chapter is explicit that the data
  is generated and that the held-out shapes are the whole test.
- **Two classifiers in the same loop.** A bag-of-words baseline (word
  counts over the training vocabulary into `nn.NewLinear(V, 5)`) and
  the embedding head (`nn.NewLinear(768, 5)` on `gemma.Embed` vectors),
  trained identically with cross-entropy and AdamW for a few seconds.
  Accuracy and `metrics.Confusion` on seen shapes and on held-out
  shapes for both. The baseline is expected to hold on seen shapes and
  fall on unseen ones; the head is expected to hold on both.
- **Nearest centroid as the zero-training alternative**: the mean
  vector per category, classification by cosine. One line of code, and
  the chapter reports where it stands relative to the trained head,
  whichever way it comes out.
- **What the head says about lines from nowhere.** A sixth category
  the head was never trained on (application errors). Report the head's
  confidence (softmax maximum) on those lines against its confidence on
  in-distribution lines, and what a confidence threshold would cost in
  the chapter-12 sense: how many real lines it sends to a human in
  order to catch the foreign ones. A classifier has no "none of the
  above" unless you build one.
- **Serving.** Embedding is the cost; the head is 3 845 numbers. The
  chapter reports lines per second for embedding a batch and the time
  to retrain the head from scratch, and points at chapter 8 for the
  service.

`docs/manual/applications.md` gains a short section on the pattern.

## Acceptance

- `go run ./examples/tutorial/18-text-classifier` trains both
  classifiers and prints accuracy on seen and held-out shapes, the
  confusion matrices, the nearest-centroid result, the confidence
  comparison for the foreign category, and the timings. Without the
  weights it says what is missing and exits 0.
- A test that does not need the weights checks the generator: every
  category has held-out shapes, no held-out shape appears in training,
  and the bag-of-words features are built from the training vocabulary
  only.
- Every figure in the chapter comes from a run of the committed code on
  a named machine with nothing else running.
- Chapter counts read eighteen; the index row and manual section exist.
- No library code changes; no performance impact.

## Notes

Measured on the M2 laptop with nothing else running (`ch18-run`):
bag of words 100 % on seen wordings, 52 % on held-out; embedding head
100 % and 94.5 %; nearest centroid 99.7 % and 92.5 %. Embedding 190
lines/s, either head 0.1 s to train.

The confidence section was first measured against the seen wordings
only, where a 0.90 threshold catches 99 % of foreign lines for free.
Against the held-out wordings the same threshold sends 57 % of
correctly classifiable lines to a person, and the chapter now reports
that column, because it is the one a real feed produces. The bag of
words' failures were traced word by word with a throwaway test before
the prose named them (`from`, `client`, `device`/`var`/`log`, `full`).
