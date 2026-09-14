# 12. A score is not a decision

Chapter 11 built a model that gives every minute a number: how badly it
reconstructs. The storm morning scores higher than the ordinary ones,
which is the result the chapter wanted, and then the chapter admits it
cannot finish the job — *"four days are not a threshold"*.

This chapter finishes it, or rather shows honestly why it cannot be
finished with what we have. That is the more useful outcome: the
arithmetic here is what decides whether a monitoring system survives
contact with the person carrying the pager.

```
go run ./examples/tutorial/12-threshold
```

## There are no labels

Start with the awkward part. Nobody marked which minutes of the storm
day *should* have raised an alarm. There is no file of correct answers,
and there won't be one on your system either — that is the normal
condition for anomaly detection, and it is what separates it from every
earlier chapter.

The temptation is to invent labels: call the storm hours "the ones above
anything in training" and measure against that. Don't. The threshold
would then be derived from the same statistic as the labels, every
threshold would catch every "anomaly", and the numbers would look
perfect while meaning nothing. I wrote that version first and it
reported 100 % recall at every setting, which is how I knew it was
wrong.

What we honestly have is one day believed ordinary and one day known to
have been disrupted. So the question becomes: **how often does a
threshold fire on the quiet day, and how much of the storm does it
see?**

## The distributions

```
hourly reconstruction error
  four training days (96 hours): median 4.86, 90th 6.38, worst 6.92
  the quiet validation day:      median 5.23, worst 7.82
  the storm day:                 median 6.75, worst 11.12
```

Read those three lines carefully, because the whole problem is in them.
The storm day is clearly higher — median 6.75 against 4.86. But **the
quiet day's worst hour (7.82) is above every hour in training (6.92)**,
and the storm day's median (6.75) sits below it. The distributions
overlap. There is no value you can pick that separates them cleanly,
because they are not separable.

Scores are per minute, averaged into hours here. That averaging is the
first and cheapest false-alarm reduction there is: chapter 11 found
minutes too noisy to act on, and sixty of them averaged are much
steadier. Do that before you reach for a better model.

## What each threshold costs

```
threshold        quiet day     storm day    alarms/day    per operator
                   fires         fires      one airport    (8 airports)
  q=0.50   4.86    15 of 24      16 of 24       15.0          120.0
  q=0.75   5.63     9 of 24      14 of 24        9.0           72.0
  q=0.90   6.38     5 of 24      14 of 24        5.0           40.0
  q=0.99   6.87     5 of 24      12 of 24        5.0           40.0
  q=1.00   6.92     3 of 24      12 of 24        3.0           24.0
```

The last column is the one that matters and the one nobody computes
until it is too late. At the strictest possible setting — above *every*
hour the model was trained on — a single airport still raises three
false alarms a day, and there are eight airports on this feed. **Twenty
four alarms a day, on days when nothing happens.**

No one will read those. Within a week the alerts are muted, and the
system is worse than nothing because it now provides false assurance.
Relax the threshold to catch more of the storm and it gets worse: q=0.90
catches two more storm hours and costs 40 alarms a day.

This is why "the model detects the storm" is not a result. The model
does detect the storm. It is still unusable.

## The accuracy trap

If you did have labels and reached for the obvious metric:

```
At q=0.90, scoring the storm day as one long event: precision 74%, recall 58%, accuracy 69%
A detector that never fires at all:                accuracy 50%
  — and if storms came one day in sixty instead of one in six, 98.3%.
```

Here the storm is one day in six, so doing nothing scores 50 %. On a
real system storms are rarer — one day in sixty — and **the same
do-nothing detector scores 98.3 %**. Report accuracy on a rare event and
you are reporting the base rate, not the model.

`metrics.Confusion` gives precision and recall per class for this
reason. Precision answers "when it fires, how often is it right"; recall
answers "of the things worth catching, how many did it catch". They
trade against each other, and which one you weight is a business
decision, not a modelling one: a missed engine failure and a spurious
page have very different costs, and no amount of tuning tells you what
they are.

## Choosing without labels

The practical recipe, which the table above follows:

1. Score a period you believe is ordinary.
2. Put the threshold at a **quantile of that**, not at a value chosen by
   looking at the incident. The incident is one sample; the ordinary
   period is thousands.
3. Convert the false-positive rate into **alarms per day across the
   whole fleet**, and check that a human would tolerate it. This is the
   step that is usually skipped.
4. Only then ask what the threshold catches.

Working in that order stops you from tuning against your one interesting
event, which is the fastest way to build a detector that finds exactly
one thing.

## Why this one cannot be fixed by a better model

```
Four days of ordinary weather is 96 hours: not enough to place a
threshold, whatever the arithmetic above suggests.
```

Ninety-six hours contains one Tuesday morning, one Friday evening, no
bank holiday, no fog, no strike, no runway closure. The quiet day's
worst hour already exceeds everything in training — not because that
hour was strange, but because four days do not contain the range of
ordinary.

A threshold placed on 96 hours is a threshold placed on a sample of
normality that does not include normality. The fix is weeks of data, not
a deeper autoencoder, and knowing which of those two you need is worth
more than either.

## What to try

Set `-epochs 200` and watch the alarm rate barely move: the model is not
the constraint. Then score only `in_zone` instead of all three counters
and see the overlap widen — fewer signals, less separation.

And take the quiet day out of training entirely, then check whether its
worst hour still exceeds the training maximum. If it does, that is your
answer about how much data a threshold needs, measured rather than
argued.

Next: [13. Attention: choosing what to look at](13-attention.md).
