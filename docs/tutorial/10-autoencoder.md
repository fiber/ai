# 10. The autoencoder: unusual by reconstruction

Chapters 4 to 8 trained models to predict a target somebody had
written down. Most operational data has no target. Nobody labels the
minutes of a day as normal or not; the only thing you have is a lot of
ordinary days and the wish to notice when a day stops being ordinary.
The autoencoder is the model for that, and it needs no labels because
its target is its own input.

```
go run ./examples/tutorial/10-autoencoder
```

Per minute, for the eight airports of the airspace example, three
counters each: aircraft in the zone, arrivals, departures. Twenty-four
numbers that describe a minute of European air traffic.

```
8640 minutes over 6 days, 24 numbers per minute (8 airports × in_zone, arrivals, departures)

autoencoder 24 → 16 → 3 → 16 → 24, 923 parameters, without the clock
epoch 30  mean error train 3.87  validation 4.27
threshold 13.8 (99.9th percentile of the training minutes)
```

## Squeeze and rebuild

The model has two halves. An encoder takes the 24 numbers down to 3; a
decoder takes the 3 back up to 24. It is trained to make the output
equal the input, and the loss is the ordinary mean squared error
between them:

```go
enc := nn.Sequential{nn.NewLinear(24, 16), nn.GELU{}, nn.NewLinear(16, 3)}
dec := nn.Sequential{nn.NewLinear(3, 16), nn.GELU{}, nn.NewLinear(16, 24)}
loss := tensor.MSELoss(dec.Forward(enc.Forward(xb)), xb)
```

Copying 24 numbers through 3 is impossible in general, so the model
learns to copy the minutes it sees in training well, which are ordinary
ones: a quiet night, a morning wave, an afternoon plateau, each with
the eight airports in their usual proportions. Show it a minute that
does not fit any of those and the copy comes out wrong. The size of
that error, the *reconstruction error*, is the model's whole output.
The threshold that turns it into an alarm is read off the training
days: 99.9 % of their minutes reconstruct with an error under 13.8.

## The storm morning that was not there

Storm Éowyn closed Dublin and Edinburgh for the morning of 24 January.
The first model does not see it:

```
storm day by hour        without   with   largest contribution (with the clock)
07:00                   0      4   LFPG arrivals          ####
08:00                   0      1   EIDW in_zone           #
09:00                   3      7   EGLL departures        #######
10:00                   0      4   EIDW in_zone           ####
```

Without the clock, the morning hours raise almost nothing. An empty
Dublin at 09:00 looks, to a model that only sees counts, exactly like
an empty Dublin at 03:00, and it has seen thousands of those. The model
is not wrong; the input does not contain the information. The second
model gets two more numbers per minute, the sine and cosine of the
time of day, which place the minute on a 24-hour clock face:

```go
a := 2 * math.Pi * float64(minute) / 1440
xs = append(xs, float32(math.Sin(a)), float32(math.Cos(a)))
```

Now "empty at 09:00" is a different input from "empty at 03:00", one
the model has never had to reconstruct, and the morning lights up. The
largest contributor per hour names where the error comes from: Dublin's
zone count at 08:00 and 10:00, and Paris arrivals at 07:00, where
flights that would have come from Ireland did not.

Every feature you leave out is a kind of normal the model cannot
recognise. That is the most useful sentence in this chapter.

## Minutes are noisy, hours are not

Even with the clock, the storm day only has 36 minutes over the
threshold against 12 on the validation day; a minute is a small thing.
Averaging the same errors per hour:

```
hour   validation   storm
06:00        5.7     7.5
07:00        6.0    10.0
08:00        7.8     7.2
09:00        7.6    10.6
10:00        5.9    10.4
14:00        7.6    10.8
15:00        5.7     9.5
```

The storm morning is now half again as bad as the validation morning,
hour after hour, and the afternoon, when the closed airports reopened
into a backlog, is worse still. This is why the airspace example
requires an anomaly to persist for several minutes before it alarms,
and why chapter 7's service would report hours, not minutes.

## Four days are not a threshold

One more thing the table shows: the program marks storm hours that are
above the worst hour of the four training days, and the validation day
would earn that mark three times as well (08:00, 13:00, 18:00). Four
days of normal contain only 96 hours; the worst of them is not the
worst normal hour there is. A threshold needs weeks of ordinary data,
and it needs the validation day to stay quiet before the storm day is
allowed to count. The model is cheap; the knowledge of what is normal
is what costs.

## What to try

Take out `departures` and keep only `in_zone`, or add the go-arounds
and holding counters the file also has. Change the bottleneck from 3 to
1 and to 8 and watch the validation error and the storm contrast move
in opposite directions: too narrow and ordinary minutes reconstruct
badly, too wide and the storm does too well. And run the chapter with
`-data` pointing at counters of your own; the program needs only the
columns it names.
