---
id: T-021
title: Tutorial: AI for Go developers with fiber/ai, eight chapters with runnable programs
status: done
scope:
  - docs/tutorial/
  - examples/
  - tensor/
  - nn/
  - optim/
manual:
  - docs/manual/getting-started.md
done: 2026-09-05
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

Done: eight chapters in `docs/tutorial/`, eight programs under
`examples/tutorial/`, every quoted output taken from a run on the M2
Pro (all under ten seconds; chapter 8 trains on 56 160 examples for 8
epochs in about 4 s). Added on the way: `Tensor.Rows` (gather rows by
index with a scatter-add gradient), `nn.SaveParams`/`nn.LoadParams`
(binary parameter files, shape-checked on load) with tests. Chapter 8
was the honest one: with perfectly weekly synthetic traffic the
"same minute last week" baseline beat the model (24.4 against 28.5
Mbit/s); day-to-day level variation and a trend, both present in any
real series, turn it around (37.9 against 31.9). Chapter 6's dropout
variant does not clearly beat the plain small model at 600 epochs; the
text says so and uses the curves to introduce early stopping.
