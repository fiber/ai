---
id: T-055
title: Chapter 12 explains the PyTorch difference instead of keeping score
status: done
scope:
  - docs/tutorial/
  - docs/manual/applications.md
manual:
  - docs/manual/applications.md
done: 2026-09-12
created: 2026-09-12
---

## Goal

Chapter 12 ended its PyTorch section by keeping score — "PyTorch is the
steadier of the two and holds a tenth of a point on the mean; we are the
faster." Two problems with that. It is the wrong register for this
project, which exists because Go's support for the field was poor and not
to win an argument with PyTorch. And a tenth of a point over three seeds
is not a result at all: a fourth seed erased it.

## Design

Report the accuracies as one number with their ranges, drop the
adjectives, and spend the space on the one difference that does have a
cause: the default initialisation.

`nn.NewConv2D` uses He normal, σ = √(2/fan_in), zero bias.
`torch.nn.Conv2d` defaults to `kaiming_uniform_(a=√5)`, σ =
1/√(3·fan_in), about 2.45 times narrower, with a uniform bias.
`benchmarks/python/mnist.py --init he` already exists to give the PyTorch
model ours, so the claim is one flag away from being checked.

Measured over seeds 12 to 15, five epochs, batch 128, Adam 1e-3, M2 Pro:

| | mean | range |
|---|---:|---|
| fiber/ai, He normal | 98.66% | 98.44 - 98.85 |
| PyTorch, He normal | 98.67% | 98.51 - 98.86 |
| PyTorch, its own default | 98.70% | 98.60 - 98.75 |

Matched, the two libraries agree to three hundredths of a point. The
wider initialisation produces both the best run in the table and the
worst; the narrower one halves the spread, 0.15 against 0.35, and gains
0.03 on the mean.

The manual's application page gets the same treatment.

## Acceptance

- No comparative adjective about speed remains in chapter 12 or in
  docs/manual/applications.md: the tables carry the numbers, the prose
  carries the method and the cause.
- Accuracy is given as a mean with a range over seeds 12 to 15, never as
  a single best run.
- The initialisation claim is reproducible with
  `benchmarks/python/mnist.py --init he` against
  `go run ./examples/mnist -model cnn`; the baseline is PyTorch 2.8 at a
  mean of 98.70% and the matched-initialisation figure is 98.67%.

## Notes

Whether He normal should stay the default for Conv2D is a real question
this raises and does not answer. It is the textbook choice for ReLU and
it found the best single run here; PyTorch's `a=√5` is a historical
accident that nobody defends on the merits but is measurably steadier.
Changing a default that every existing model's reproducibility depends on
needs its own spec and more than one task's evidence.
