---
id: T-023
title: Data preparation and metrics: counter deltas, windows, time split, standardiser, batches; confusion matrix; class-weighted cross-entropy
status: done
scope:
  - data/
  - metrics/
  - tensor/
  - examples/
  - docs/tutorial/
manual:
  - docs/manual/applications.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

The network workbook's first, second and fourth projects need the same
handful of data steps before any model sees a number: counters turned
into rates with resets handled, sliding windows with time features, a
split by time, standardisation with statistics that are kept, and
mini-batches; afterwards a confusion matrix and per-class precision and
recall to judge a classifier, and class weights in the loss for the
3 000-workstations-40-cameras case. Tutorial chapter 6 wrote the batch
loop by hand; this makes it one line.

## Design

`data` (no assembly, plain Go over slices and tensors):
- `Rates(counter []float64, interval time.Duration, maxRate float64) []float32`:
  per-interval rates from cumulative counters; a negative delta is a
  counter reset and the new value counts from zero; rates above
  `maxRate` (the link speed) are treated as resets too; a missing sample
  is NaN and the caller decides.
- `Windows(series []float32, window, horizon int, at func(t int) []float32)`
  builds `[n × (window+extra)]` inputs and `[n × 1]` targets like tutorial
  chapter 8; `TimeFeatures(t time.Time) []float32` gives the four
  sine/cosine coordinates of time of day and day of week.
- `SplitByTime(x, y *tensor.Tensor, trainFraction float64)` returns the
  leading part for training and the rest for validation, as views.
- `Standardizer` with `Fit(x)`, `Transform(x)`, `Inverse(x)`, exported
  `Mean`/`Std` tensors so they can be saved next to the model; columns
  with zero spread get spread 1.
- `Batches(n, size int, r *rand.Rand) iter.Seq[[]int]` yields shuffled
  index slices for `Tensor.Rows`.

`metrics`:
- `Confusion(pred, truth []int, classes int) Matrix` with `Accuracy()`,
  `Precision(c)`, `Recall(c)`, `F1(c)` and a `String()` table.
- `MAE`, `RMSE` on tensors of equal shape.

`tensor`:
- `CrossEntropyWeighted(logits, targets, weights)`: per-class weights,
  normalised by the total weight of the batch's targets as PyTorch does,
  sharing the fused forward/backward with `CrossEntropy`.

## Acceptance

- Tests: a counter reset and an over-speed delta are handled; windows
  line up with a hand-built example; the standardiser round-trips and
  produces zero-mean unit-spread columns; `Batches` covers every index
  exactly once per epoch; the confusion matrix and its rates match a hand
  count; weighted cross-entropy equals unweighted for unit weights and
  matches a hand computation for others, with a numeric gradient check.
- No performance impact: the data helpers run once per data set, the
  weighted loss shares the existing fused kernel and is not on the
  benchmark path.
- `applications.md` documents both packages; tutorial chapter 6 mentions
  `data.Batches` as the short form of its loop.

## Notes

Implemented as specified: `data` (Rates, TimeFeatures, Windows,
SplitByTime, Standardizer with Fit/Transform/Inverse, Batches as an
iterator), `metrics` (Confusion with Accuracy/Precision/Recall/F1 and a
table, MAE, RMSE) and `tensor.CrossEntropyWeighted` sharing the fused
path with `CrossEntropy` (the plain loss now goes through the same
function with nil weights; the backward scales each row by its weight
over the batch's total weight). Tests as listed in the acceptance,
including a numeric gradient check for the weighted loss.
