---
id: T-021
title: Tutorial: AI for Go developers with fiber/ai, eight chapters with runnable programs
status: open
scope:
  - docs/tutorial/
  - examples/
  - tensor/
  - nn/
  - optim/
manual:
  - docs/manual/getting-started.md
created: 2026-09-05
---

## Goal

Go developers who never learned NumPy or PyTorch should be able to learn
what "training a model" is, and build one, from fiber/ai alone. A
tutorial of eight chapters in `docs/tutorial/`, each a short text and a
runnable program under `examples/tutorial/`, with the program's real
output quoted in the text. It doubles as the acceptance test of the API:
where a chapter needs a helper that does not exist, the helper is added
rather than the chapter working around it.

## Design

Chapters (one program each, `go run ./examples/tutorial/NN-name`):

1. A tensor is a slice with a shape: New/Zeros/Randn, Shape, At, Data,
   views (T, Reshape, Narrow) share storage, Float32s copies.
2. Broadcasting: `x.Add(row)`, `x.Sub(mean)`, standardising columns;
   why it saves loops and allocations.
3. A gradient without formulas: `y = 3x + 1`, `SetRequiresGrad`,
   `Backward`, reading `Grad()`; what "the slope of the loss" means.
4. Linear regression by hand: loss, a step against the gradient, the
   learning rate, in forty lines with plain tensors.
5. The first classifier: the spiral with `nn.Sequential`, `CrossEntropy`,
   `optim.Adam`, accuracy under `NoGrad`.
6. Anatomy of a training loop: shuffling, mini-batches, epochs, a
   validation split, watching for overfitting, `Dropout`.
7. A model in service: `NoGrad`, `Release`, `SetThreads`, latency per
   request, saving and loading parameters.
8. From toy to network data: chapter 1 of the network workbook (traffic
   forecast with an expectation band) as a worked example on synthetic
   counters.

Helpers the chapters need and that are added on the way: `Tensor.Rows`
(gather rows by index, the natural way to build a mini-batch),
`nn.SaveParams`/`nn.LoadParams` (parameters to and from an `io.Writer`
in a small self-describing binary format), and `tensor.Standardize`
helpers only if chapter 2 shows them to be worth it.

The manual's getting-started page links the tutorial as the place to
start for readers new to the field.

## Acceptance

- Every chapter program runs in under ten seconds on the M2 Pro and its
  quoted output matches what it prints.
- `Rows` and the save/load round trip have tests.
- No performance impact: the chapters use existing operations; the new
  helpers are outside every hot path.
- getting-started.md links the tutorial; `docs/tutorial/README.md` lists
  the chapters.

## Notes
