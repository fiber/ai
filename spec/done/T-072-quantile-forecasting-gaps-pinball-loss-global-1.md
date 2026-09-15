---
id: T-072
title: Quantile forecasting gaps: pinball loss, global 1-D pooling, parameter snapshots
status: done
scope:
  - tensor/
  - nn/
  - docs/manual/nn-and-optim.md
  - docs/manual/tensors.md
manual:
  - docs/manual/nn-and-optim.md
  - docs/manual/tensors.md
done: 2026-09-15
created: 2026-09-15
---
---

## Goal

An agent building a quantile forecaster on this library had to write
four things by hand before it could start. Three of them are training
mechanics that any forecasting or sequence model needs, and their
absence is not a design decision, just a gap:

- **Pinball loss.** The loss for quantile regression, and the reason a
  forecast can carry a prediction interval instead of a single number.
  `tensor` has MSE and cross-entropy and nothing else, so any
  probabilistic forecast starts with a hand-written loss and a
  hand-written backward.
- **Global 1-D pooling.** `nn` has `Conv1D` and `MaxPool2D`. The usual
  end of a 1-D convolutional encoder — collapse the length axis, keep
  the channels, hand the vector to a `Linear` — has no layer, so it
  cannot sit in a `Sequential`.
- **Snapshot and restore.** Chapter 7 of the tutorial says the model to
  keep is the one from the epoch where validation bottomed out, and the
  library offers no way to hold that model except writing it to disk
  and reading it back. Early stopping is the most common regulariser
  there is and it needs an in-memory copy.

## Design

**`tensor.PinballLoss(pred, target *Tensor, quantiles ...float32)`**, in
`tensor/nn.go` beside `MSELoss`. `pred` has shape `[..., Q]` with at
least two dimensions and `Q == len(quantiles)`; `target` has `pred`'s
shape without the last dimension, so a multi-horizon multi-quantile
forecast `[batch, horizon, Q]` scores against `[batch, horizon]` and the
single-quantile case is `[rows, 1]` against `[rows]`. Every quantile
must be in (0, 1).

The loss of one element is `max(q·d, (q−1)·d)` with `d = y − ŷ`, and the
result is the mean over all elements, so the number is comparable across
different numbers of quantiles. The backward is the subgradient
`∂/∂ŷ = −q` where `d ≥ 0` and `1 − q` otherwise, scaled by `1/(n·Q)`;
the forward stores those coefficients, as `MSELoss` stores its
difference. `d = 0` takes `−q`, which is the convention PyTorch users
get from `torch.max`.

**`nn.GlobalAvgPool1D`** and **`nn.GlobalMaxPool1D`**, in `nn/conv.go`
beside `MaxPool2D`: `[batch, channels, length] → [batch, channels]`,
forwarding to `Mean(2)` and `Max(2)`, which already have backward
passes. Empty structs with `Params() nil`, so they drop into a
`Sequential` between `Conv1D` and `Linear` where `Flatten` would
otherwise force a fixed length.

**`nn.Snapshot`**, in `nn/params.go`:

```go
s := nn.NewSnapshot(model)   // allocates and captures once
s.Capture(model)             // overwrite, no allocation
s.Restore(model)             // copy back in place
```

`Capture` and `Restore` check the parameter count and shapes and return
an error on a mismatch, as `LoadParams` does. `Restore` writes into the
existing tensors under `NoGrad`, so an optimiser already holding them
keeps working — the point being that a training loop can restore the
best parameters and carry on. Gradients and optimiser state are not
part of a snapshot, and the documentation says so: restoring does not
rewind Adam's moments.

## Acceptance

- `PinballLoss` agrees with PyTorch on value and gradient for a
  random `[8, 3]` case with quantiles 0.1/0.5/0.9 to within 1e-6, and
  a Go finite-difference check confirms the subgradient at points away
  from `d = 0`.
- `PinballLoss` with the single quantile 0.5 equals half the mean
  absolute error, checked in a test.
- A test trains a two-quantile forecaster on data with known noise and
  asserts what quantile regression promises: the 0.1 prediction lies
  below the 0.9 prediction, and each covers roughly its share of the
  targets.
- `GlobalAvgPool1D` and `GlobalMaxPool1D` produce the documented shape,
  gradients match finite differences, and a `Sequential` containing one
  of them trains.
- `Snapshot` round-trips parameters bit for bit; `Restore` into a model
  whose optimiser has since stepped leaves the optimiser usable; a
  mismatched model gives an error rather than a corrupt restore.
- No performance impact: three additions, no change to any existing
  operation or kernel.

## Notes

`Snapshot.Capture` needed one more thing than planned. `Float32s`
always returns a fresh copy, so capturing a 3 M-parameter model would
have allocated 12 MB per epoch — the opposite of the point. The
parameters are read with a new `tensor.CopyTo(dst)`, which fills a
slice the caller already has, allocates nothing for a contiguous tensor
and, unlike `Data()`, does not mark the storage escaped. What Capture
allocates is now only the model's own `Params()` slices, one per
module; a test asserts that the count is the same for a model with a
thousand times more parameters.

**The PyTorch leg of the first acceptance criterion did not run.** The
venv with torch lives in the `aitest` account on the AVX2 guest and
this session has a key only for the personal account; installing torch
elsewhere would break the "keep it in aitest" rule. What was checked
instead:

- Value and gradient against an independent float64 implementation of
  the definition, written in Python from the formula rather than from
  the Go code: loss 0.595698059 against 0.595698087 (2.8e-8), maximum
  gradient difference 1.3e-9 over the 24 elements of an [8, 3] case
  with quantiles 0.1/0.5/0.9.
- A Go finite-difference check of the subgradient away from the kink.
- The single quantile 0.5 equalling half the mean absolute error.
- Fitting three quantiles of a standard normal on 4 000 samples:
  −1.2816 / 0 / 1.2816 recovered to within 0.15.

The convention question that only a PyTorch comparison settles is what
happens exactly at d = 0 (this implementation charges −q, which is what
`torch.max` yields) and whether users expect the mean over quantiles or
the sum. Both are documented. The comparison is worth running when the
account question is settled.
