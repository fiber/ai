---
id: T-075
title: Tutorial chapter 19: a forecast with an interval, and the early stopping that gets it
status: done
scope:
  - docs/tutorial/
  - examples/tutorial/
  - docs/manual/applications.md
  - docs/manual/getting-started.md
  - README.md
manual:
  - docs/manual/applications.md
done: 2026-09-15
created: 2026-09-15
---
---

## Goal

Chapter 9 forecasts one number: the aircraft count thirty minutes from
now. Every operational use of that number wants a second thing the
chapter cannot give — how sure the model is. A point forecast is a lie
of precision, and on the storm day it is a confident one.

The library grew four things this week for exactly this kind of work
(`PinballLoss`, `GlobalAvgPool1D`, `Snapshot`, `safetensors.Save`) and
the tutorial demonstrates none of them. They are not four separate
features, they are one workflow: an encoder that does not care how long
the history is, a loss that predicts quantiles, a way to keep the best
epoch, and a file somebody else can read. The chapter is that workflow
on data the reader already knows.

The `data` helpers that are also unused by the tutorial — `Windows`,
`SplitByTime`, `Standardizer` — were considered here and do not fit,
which is worth writing down rather than forcing: `Windows` takes a
single series and this model reads eight channels, `SplitByTime` splits
by fraction and this split is by day with the storm day held out
separately, and `Standardizer` fits a mean and spread per *column*,
which is wrong for a window whose columns are all the same quantity
measured at different minutes. They belong to the leakage chapter
instead, and this chapter explains the `Standardizer` trap in a
sentence, since a reader will otherwise reach for it here.

## Design

A new chapter 19, `examples/tutorial/19-intervals`, on the airspace
counters of chapters 9, 11 and 12, forecasting the same target thirty
minutes ahead so the numbers can be put beside chapter 9's.

- **The model.** `toChannels` → `Conv1D(8→16)` → ReLU →
  `Conv1D(16→16, stride 4)` → ReLU → `GlobalAvgPool1D` →
  `Linear(16, 3)`: one output per quantile, 0.1/0.5/0.9. Pooling rather
  than `Flatten` means the model is not tied to a 120-minute history,
  which the chapter then *tests* by feeding it 240 minutes and
  reporting what happens, rather than asserting the benefit.
- **The loss** is `tensor.PinballLoss` over the three quantiles.
- **Early stopping with `nn.Snapshot`.** Capture whenever the
  validation loss improves, restore at the end, and report which epoch
  was kept against the last one — the difference is the point of the
  mechanism.
- **Evaluation is calibration, not error.** For the validation day and
  the storm day: how often the true count falls inside [q10, q90]
  (should be 80 % if the model is honest), the mean width of that
  interval in aircraft, and the median output's MAE so it can be
  compared with chapter 9's point forecast.
- **Export.** `safetensors.Save` with named tensors, and the chapter
  prints what the file holds.

The storm day is expected to break the calibration, and the chapter
says so with the measured number: an interval learned from four
ordinary days does not know what a disrupted day looks like, which is
the same lesson chapter 12 reached from the other side.

## Acceptance

- `go run ./examples/tutorial/19-intervals` trains, restores the best
  epoch, prints coverage and interval width for both days, the MAE of
  the median output, the long-history check, and the contents of the
  written safetensors file.
- A test asserts what the chapter claims structurally: the three
  outputs come out ordered on the great majority of rows, coverage is
  computed correctly against a hand-checked case, and the model accepts
  a history length it was not trained on.
- Every figure in the chapter comes from one run of the committed code
  on a named machine with nothing else running.
- Chapter counts read nineteen; chapter 18 hands on to 19 and 19 closes
  the tutorial; the index row exists.
- No library code changes; no performance impact.

## Notes

Measured on the M2 laptop with nothing else running, 60 epochs, 17
seconds:

    validation day  coverage 77.8%  band 7.4 ac  median MAE 2.23 ac
    storm day       coverage 52.3%  band 7.5 ac  median MAE 3.27 ac
    kept epoch 2 (0.1456); the last epoch was 0.1691, +16.2%
    240 minutes of history: coverage 72.8%, band 7.6 ac, MAE 2.55 ac
    quantiles crossed on 11 of 2880 rows

Four results worth recording, three of them not what the spec expected.

The validation-day calibration is good (77.8 % against a nominal 80 %)
and the storm day's is not (52.3 %) — as anticipated. What was not
anticipated is that the band does not widen on the storm day in the
aggregate: 7.5 against 7.4 aircraft. It widens where the model can see
the disruption in its inputs (14–15 aircraft during the morning
closure) and stays narrow where the inputs look ordinary (2–3 aircraft
in the evening, 2 % coverage, while the airport worked a backlog). The
chapter makes "narrow and wrong" the lesson rather than "the band is
too narrow".

Early stopping is less heroic than the headline. The validation loss
bottoms out in a noisy plateau over epochs 2 to 8 (0.145 to 0.151) and
epoch 2 is the lowest point of the noise, not a meaningfully better
model than epoch 4. The 16.2 % is real against epoch 60. The chapter
says both.

`GlobalAvgPool1D` makes the model accept any history length, and the
chapter tests that claim instead of asserting it: 240 minutes runs and
is *worse* on every measure, because the learned features are averages
over the length trained on. Length independence is about shapes, not
accuracy, and the manual sentence written when the layer was added
would have been read as a promise of more.

The median output beats chapter 9's point forecast on both days. The
likely cause is that chapter 9 trains `MSELoss`, which fits the mean,
and reports the mean absolute error, whose optimum is the median; the
two models also differ in filter count and pooling, so the chapter
presents this as a reason to match loss to metric, not as a controlled
result.
