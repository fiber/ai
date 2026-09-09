---
id: T-048
title: Tutorial chapter 8: convolutions over time, a 1-D convolutional forecaster on the airspace counters
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

The tutorial has seven chapters and no convolution, although `Conv1D`
and `Conv2D` exist and the benchmark measures them. The convolution
that matters for the framework's users is the one over time: a counter
series, a filter that slides along it, a forecast. Chapter 8 teaches it
on data the repository already ships, the per-minute aircraft counts of
the airspace example (public, ODbL), and ends where chapter 7 and the
airspace example begin: a model whose error on an unusual day is the
signal.

## Design

- `examples/tutorial/08-convolution/main.go`: reads
  `examples/airspace/testdata/counters.csv.gz` (flag `-data`, `-airport`
  default EGLL), builds the per-minute `in_zone` series over the six
  days, windows of 120 minutes predicting the count 30 minutes ahead
  (`data.Windows`), split by the day of the target: four days train,
  the fifth validates, the sixth (the storm) is held out. Two baselines
  (persistence, same minute the day before), then the chapter-6 MLP
  over the flat window and a small 1-D CNN (Conv1D 1→8 width 7, ReLU,
  Conv1D 8→4 width 7, ReLU, Flatten, Linear), trained with the same
  loop; a five-line module in the program reshapes the flat window into
  [batch, 1, 120], which is also the chapter's "write your own module"
  moment. Prints validation MAE in aircraft for all four, the seven
  weights of a learned filter, and the storm-day MAE, where the error of
  every model jumps.
- `docs/tutorial/08-convolution.md`: why a flat window treats minute 17
  and minute 18 as unrelated columns, what a filter is (a tiny model
  applied at every position, the same weights everywhere), how the
  output of one layer becomes the channels of the next, reading a
  learned filter, and the storm day as the bridge to the airspace
  example. Quotes the program's output. README table gains the row;
  "Seven chapters" becomes eight.

## Acceptance

- `go run ./examples/tutorial/08-convolution` runs in a few seconds on
  a laptop and prints the numbers the chapter quotes; the CNN beats the
  flat MLP and both beat the baselines on the validation day; the storm
  day shows a clearly larger error for every model. `go vet ./...`.
- No performance impact (tutorial and example only).

## Notes
