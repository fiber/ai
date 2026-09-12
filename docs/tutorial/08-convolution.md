# 8. Convolutions over time

Chapter 6 fed a model a window of numbers and let it learn. It never
told the model that the numbers were in order: that minute 17 and minute
18 are neighbours, that a rise over the last ten minutes means the same
thing whether it happens at the start of the window or at the end. A
convolution is the layer that knows. This chapter builds one over a
counter series and ends where the airspace example begins.

```
go run ./examples/tutorial/08-convolution
```

The data is what the airspace example ships: for eight European
airports, the number of aircraft inside a 40-mile zone, one value per
minute, six days in January 2025. The task is a forecast: from the last
two hours, how many aircraft will be in Dublin's zone half an hour from
now.

```
EIDW: 6 days of per-minute counts, 24 aircraft in the zone at the busiest minute
each example: the last 120 minutes, predicting the count 30 minutes after the window
5611 examples train (2025-01-19 to 2025-01-22), 1440 validate (2025-01-23), 1440 are the storm day (2025-01-24)
```

The split is by day, as chapter 6 said it must be for anything measured
over time: four days to train, the fifth to validate, and the sixth kept
back entirely. The sixth is 24 January 2025, the day Storm Éowyn closed
Irish and Scottish airports.

## Two baselines first

Before any model, two forecasts that cost nothing. *Persistence* says
the count in thirty minutes is the count now. *Yesterday* says it is
whatever it was at this minute the day before.

```
baselines on the validation day: persistence 3.02, same minute yesterday 2.95 aircraft
```

Three aircraft of error on a series that peaks at twenty-four. Every
model below has to beat that, or it has learned nothing worth its
parameters.

## A flat window, as in chapter 6

The first model is the chapter-6 MLP — a *multilayer perceptron*, the
plain stack of linear layers and activations from chapters 5 and 6:
120 inputs, one per minute,
64 hidden units, one output.

```
flat MLP, EIDW alone (7809 parameters)
epoch  5  train 2.23  validation 2.73 aircraft
epoch 20  train 2.15  validation 3.02 aircraft
```

It gets to 2.7 and drifts back to 3.0, the persistence baseline, while
its training error keeps falling. It is memorising four days. To this
model, minute 17 is a column, minute 18 is another column, and a pattern
it has seen at minutes 10 to 20 is a different pattern when it appears
at minutes 90 to 100.

## A filter is a tiny model applied everywhere

A one-dimensional convolution takes a filter, here seven weights, and
slides it along the series: at every position it multiplies the seven
minutes under it by the seven weights and sums. One filter, seven
parameters, applied at all 120 positions. Its output is a new series of
120 values that says, at each minute, how much the last seven minutes
looked like the filter's pattern. Eight filters make eight such series;
the framework calls them channels. A second layer slides its own filters
over those eight channels at once.

```go
second := nn.NewConv1D(8, 8, 7)
second.Stride = 4
cnn := nn.Sequential{
    toChannels{1},           // [batch, 120] -> [batch, 1, 120]
    nn.NewConv1D(1, 8, 7),   // 8 filters of width 7 over the one series
    nn.ReLU{},
    second,                  // 8 filters over the 8 channels, every 4th minute
    nn.ReLU{},
    nn.Flatten{},
    nn.NewLinear(8*30, 1),
}
```

`Stride = 4` makes the second layer step four minutes at a time, so its
output is 30 positions long instead of 120; the linear layer at the end
reads 8 × 30 numbers. `toChannels` is the first module written in this
tutorial rather than taken from `nn`, and it is the whole recipe: a
`Forward` and a `Params`:

```go
type toChannels struct{ channels int }

func (m toChannels) Forward(x *tensor.Tensor) *tensor.Tensor {
    return x.Reshape(x.Dim(0), m.channels, window)
}
func (toChannels) Params() []*tensor.Tensor { return nil }
```

```
1-D CNN (convolutional neural network), EIDW alone (761 parameters)
epoch  5  train 2.42  validation 2.67 aircraft
epoch 20  train 2.35  validation 2.74 aircraft
```

Ten times fewer parameters than the MLP, a better validation error, and
train and validation stay close: a filter that must work at every
position cannot memorise where in the window something happened.

## Neighbours as channels

Aircraft arriving in Dublin left Manchester and Amsterdam an hour ago.
The second experiment gives both models all eight airports. For the MLP
that is 960 inputs; for the convolution it is eight channels, and the
first layer's filters are now 8 × 7 weights each, one row per airport.

```
flat MLP, all eight airports (61569 parameters)
epoch 20  train 1.59  validation 2.52 aircraft

1-D CNN, all eight airports (1153 parameters)
epoch 20  train 1.61  validation 2.63 aircraft
```

Both improve on the single series by about a fifth of an aircraft, and
they end level with each other, within the noise of a validation day of
1 440 minutes. The convolution gets there with 1 153 parameters against
61 569. That ratio is the point of the layer: it does not know more, it
assumes more, and the assumption (the same pattern means the same thing
at every minute) happens to be true of time.

The learned filters are not readable the way a rule is; their seven
weights on Dublin's own minutes are small differences of neighbouring
values, and each filter also weighs the other airports:

```
  filter 0:   0.17   0.10  -0.09  -0.12   0.10   0.01   0.13   listens most to LFPG among the others
  filter 2:  -0.06  -0.19  -0.20  -0.16  -0.04   0.10  -0.09   listens most to EGCC among the others
```

What is readable is the count of them.

## The storm day

Now the sixth day, which no model has seen. Mean absolute error in
aircraft:

| | validation day | storm day |
|---|---:|---:|
| persistence | 3.02 | 2.56 |
| same minute yesterday | 2.95 | 4.89 |
| flat MLP, alone | 3.02 | 2.78 |
| 1-D CNN, alone | 2.74 | 2.69 |
| flat MLP, eight airports | 2.52 | 3.61 |
| 1-D CNN, eight airports | 2.63 | 3.66 |

Read the table by rows. Persistence gets *better* on the storm day:
Dublin's sky was empty for hours, and "it stays as it is" is exactly
right about an empty sky. The models that look only at Dublin's last two
hours are barely affected for the same reason; the window already shows
the storm. What breaks is anything that expects a normal day: the
yesterday baseline is off by five aircraft, and the two models that read
the neighbours, where Frankfurt and Paris ran a normal Friday, forecast
a normal Dublin and miss by 40 % more than usual.

That is not a failure of the forecast, it is the signal. A model trained
on ordinary days, compared with what is actually happening, measures how
unusual the day is; the size of its error is the alarm. The airspace
example does exactly this, with the movement profile instead of a
convolution, and chapter 7's service is where such a model lives. Where
a forecaster earns its keep is not in being right on the storm day but
in being wrong there by a margin nobody could ignore.

## What to try

Change `-airport` to `EGLL` or `EDDF`: at Heathrow the neighbours help
less, at Frankfurt (which the storm did not reach) the storm-day error
tells you something about Friday traffic instead. Make the filters
wider, or add a third layer, and watch the parameter count and the
validation error together; the interesting models in this family are the
ones that get better without getting bigger.

The same idea in two dimensions, sliding filters across an image instead
of along a series, is [chapter 12](12-mnist.md).
