# 19. A forecast with an interval

Chapter 9 predicts one number: how many aircraft will be in Dublin's
zone in thirty minutes. Everybody who would act on that number wants a
second one — how sure the model is — and a single number cannot carry
it. A point forecast is a lie of precision, and on a disrupted day it
is a confident lie.

This chapter predicts three numbers instead: the 10th, 50th and 90th
percentile of what the count will be. The middle one is the forecast,
the outer two are a band that should contain the truth eight times out
of ten. Whether it actually does is the chapter's subject, because a
band nobody checked is worse than no band at all.

```
go run ./examples/tutorial/19-intervals
```

Same data as chapters 9, 11 and 12, same split, same horizon, about
seventeen seconds.

## Three outputs and the loss that separates them

Nothing in the model knows what a percentile is. The loss does all of
it:

```go
loss := tensor.PinballLoss(m.Forward(xb), yb, 0.1, 0.5, 0.9)
```

`PinballLoss` charges each output asymmetrically. With `d = target −
prediction`, an output trained at q pays `q·d` when it guessed too low
and `(1−q)·|d|` when it guessed too high. For q = 0.9 guessing too low
costs nine times as much as guessing too high, so that output is pushed
upwards until only one target in ten is still above it — which is the
definition of the 90th percentile. The middle output, q = 0.5, pays the
same either way: that is the mean absolute error, and its minimiser is
the median.

Three columns in, three columns out, one loss. The predictions have
shape `[batch, 3]` and the targets `[batch]`.

## A model that does not care how long the history is

```go
nn.Sequential{
    toChannels{8},
    nn.NewConv1D(8, 16, 7), nn.ReLU{},
    second,                 nn.ReLU{},   // stride 4
    nn.GlobalAvgPool1D{},
    nn.NewLinear(16, 3),
}
```

Two convolutions over the eight airports' last two hours, as in chapter
9, and then the difference: `GlobalAvgPool1D` averages each of the 16
channels over the whole length, leaving 16 numbers per example. Chapter
9 used `Flatten` there, which turns 30 time steps × 8 channels into 240
fixed inputs and welds the model to a 120-minute history. Pooling
throws away *where* something happened and keeps *how much* of it there
was, and the model is then a function of the channel count alone. 2 771
parameters.

## Keeping the best epoch

```
  epoch   1  validation pinball 0.1532  (best so far)
  epoch   2  validation pinball 0.1456  (best so far)
  epoch   3  validation pinball 0.1487
  epoch   4  validation pinball 0.1484
  epoch   5  validation pinball 0.1509
  epoch   8  validation pinball 0.1504
  epoch  20  validation pinball 0.1525
  epoch  30  validation pinball 0.1611
  epoch  60  validation pinball 0.1691

kept epoch 2 (validation pinball 0.1456); the last epoch was 0.1691, +16.2%
```

Chapter 7 said the model to keep is the one from the epoch where the
validation loss bottomed out. `nn.Snapshot` is how you keep it without
going through a file:

```go
best := nn.NewSnapshot(m)
for epoch := range epochs {
    trainOneEpoch()
    if v := valLoss(); v < bestLoss {
        bestLoss = v
        best.Capture(m)     // reuses its buffers, cheap enough every epoch
    }
}
best.Restore(m)             // in place, so the optimiser stays usable
```

Read the curve honestly. The useful training is over within about two
epochs; epochs 2 to 8 are a noisy plateau between 0.145 and 0.151, and
epoch 2 is the lowest point of that noise rather than a meaningfully
better model than epoch 4. What the snapshot definitely buys is not
ending up with epoch 60, which is 16 % worse — the model spends
fifty-eight epochs memorising four days of traffic. Training 60 epochs
at all is a waste; a real loop stops when the loss has not improved for
a while.

## Does the band mean anything?

```
                     coverage     band width     median MAE
validation day          77.8%         7.4 ac        2.23 ac
storm day               52.3%         7.5 ac        3.27 ac

the band is meant to hold 80% of the targets
```

On the validation day, **77.8 % against a nominal 80 %** — the band is
close to honest, and it is 7.4 aircraft wide on a series that peaks at
24. That width is the useful part: it says a forecast of "twelve" means
somewhere between about eight and sixteen, which is what an operator
would have guessed and now does not have to guess.

On the storm day the same band holds the truth **52.3 %** of the time.
It is not wider — 7.5 aircraft, essentially unchanged — it is just
wrong. The model has no way to know that this day is different; it
learned what uncertainty looks like on four ordinary days, and a
disrupted day is not among them. This is chapter 12's lesson arriving
from the other side: there, four days did not contain enough ordinary
behaviour to place a threshold; here they do not contain enough to
calibrate an interval.

The hourly breakdown shows the model is not entirely blind:

```
storm day by hour, widest bands first
  hour     band width   coverage
  10:00       15.6 ac       42%
  11:00       14.5 ac      100%
  09:00       14.0 ac       18%
  18:00        2.9 ac       10%
  17:00        2.4 ac        2%
```

During the morning closure the band doubles to 14–15 aircraft: the
model sees the neighbours' counts collapse, has never seen that pattern,
and hedges. That is the right instinct and it is not enough — at 09:00
it still misses four times out of five. The evening is worse in a
different way: bands of 2–3 aircraft, coverage of 2 %, the model
confidently predicting a quiet evening while the airport worked through
its backlog. **Narrow and wrong is the dangerous combination**, and a
model will produce it whenever the input looks familiar and the world
is not.

## Two things that come with quantile outputs

**They can cross.** The three outputs are three independent columns of
one linear layer and nothing makes them ordered, so on 11 of 2 880 rows
the 10th percentile came out above the 90th. It is rare here and it
happens; if it matters, sort the three outputs before using them, or
predict the middle one and two non-negative widths.

**The median output is a better point forecast than chapter 9's.** Its
mean absolute error is 2.23 aircraft on the validation day and 3.27 on
the storm day, against 2.63 and 3.66 for chapter 9's convolutional
model on the same data. Part of that is early stopping, and part is
that chapter 9 trained with `MSELoss` — which fits the *mean* — and
then measured the mean absolute *error*, whose optimum is the median.
If you are going to be judged on MAE, train the median. The models also
differ in filter count and pooling, so this is a reason to check the
loss against the metric, not a proof that the loss alone did it.

## What the pooling actually bought

The model accepts any history length, so it can be handed four hours
instead of two without retraining:

```
the same trained model on 240 minutes of history instead of 120:
  validation day: coverage 72.8%, band 7.6 ac, median MAE 2.55 ac
```

It runs, and it is worse: coverage falls from 77.8 % to 72.8 %, the
error from 2.23 to 2.55. Nothing is broken — the shapes work — but the
features the model learned are *averages over the length it was trained
on*, and averaging over twice as long dilutes exactly the recent minutes
that carry the forecast. Length independence is a property of the
shapes, not a promise about accuracy. Use it to serve one model on
ragged inputs, not to feed it more history for free.

## Handing the model over

```
wrote forecaster.safetensors:
  conv1.b    F32   [16]
  conv1.w    F32   [16 8 7]
  conv2.b    F32   [16]
  conv2.w    F32   [16 16 7]
  head.b     F32   [3]
  head.w     F32   [16 3]
  metadata: map[horizon:30 quantiles:[0.1 0.5 0.9] scale:mean 8.2181 sd 4.9849 target:EIDW]
```

`safetensors.Save(path, tensors, meta)` writes named tensors that Python
and the rest of the ecosystem can open, where `nn.SaveParams` writes an
unnamed blob only this program can read. The metadata is not decoration:
the model's outputs are in scaled units, and without `mean 8.2181 sd
4.9849` its numbers do not mean aircraft. A checkpoint without the
scaling that produced it is scrap.

## What to remember

- Predict quantiles with `PinballLoss` when a single number would be
  read as more certain than it is.
- **Measure coverage.** A band is a claim about how often it contains
  the truth, and the claim is cheap to check and often wrong.
- Calibration is learned from the days in the training set. On a day
  unlike those, the band is not automatically wider — it can be narrow
  and wrong.
- `nn.Snapshot` keeps the best epoch; `safetensors.Save` writes
  something another stack can read, with the scaling in the metadata.

## What to try

Stop when the validation loss has not improved for ten epochs instead
of running all 60, and see how much time that saves — the snapshot
already holds the model you want. Then add 0.01 and 0.99 to the
quantile list: the outer band should be much wider, and its coverage
should be close to 98 %; check whether four days of training data can
support a claim that strong. And train on five days with the storm in
the training set and validate on an ordinary one, which is the
experiment that answers whether the storm day was unlearnable or merely
unseen.

That is the end of the tutorial. The [manual](../manual/README.md)
covers the API in full, and its [applications
page](../manual/applications.md) collects what these chapters measured
in one place.
