---
id: T-068
title: Tutorial chapter 12: a score is not a decision — choosing a threshold
status: done
scope:
  - docs/tutorial/
  - examples/tutorial/
  - docs/manual/applications.md
  - README.md
  - metrics/
manual:
  - docs/manual/applications.md
done: 2026-09-13
created: 2026-09-13
---

## Goal

Chapter 11 trains an autoencoder that reconstructs ordinary airspace
minutes badly when the airspace is not ordinary, and ends by admitting
the thing it cannot do: "Four days are not a threshold." It produces a
number per minute and stops. Every reader who wants to use it then has
to answer the question the tutorial skipped — at what value does this
number mean *wake someone up*?

That question is most of the work in an operational system, and getting
it wrong is how monitoring dies: a threshold that fires 200 times a day
is switched off within a week, and one that never fires is decoration.
The tutorial has fifteen chapters on making models and none on deciding
what to do with their output.

It also needs saying that accuracy is the wrong measure here and that
the reason is arithmetic, not taste. If 1439 of 1440 minutes are normal,
a model that always says "normal" scores 99.93 %.

## Design

A new chapter 12, `examples/tutorial/12-threshold`, reusing chapter 11's
autoencoder and the same airspace counters so the reader is choosing a
threshold for a model they already understand.

- Score the four training days, the quiet validation day and the storm
  day. Show the distributions rather than a single number.
- **Precision and recall, from `metrics`.** Sweep the threshold and
  print the curve: how many storm hours are caught, how many quiet hours
  are wrongly flagged, at each setting.
- **The alarm budget.** Turn the false-positive rate into alarms per day
  and then into alarms per day across a fleet, which is the number that
  decides whether a system survives contact with the people carrying the
  pager. One alarm per thousand minutes is one per airport per 17 hours,
  and eight airports make that eleven a week.
- **Choosing without labels**, which is the usual case: pick the
  threshold from a quantile of ordinary data, not from the incidents,
  because you have plenty of the former and almost none of the latter.
  Then check what it would have cost on a period known to be quiet.
- Close on the honest limit chapter 11 named: four days do not contain
  enough ordinary behaviour to place a threshold, and the fix is more
  ordinary data rather than a better model.

Placement forces a small renumber: 12 to 15 become 13 to 16. Thresholds
belong immediately after the model that produces the score, and the
language-model chapters are the capstone and should stay last. Four
chapters is a quarter of the churn of the last insert and the procedure
is now known.

## Acceptance

- `go run ./examples/tutorial/12-threshold` prints the score
  distributions, a threshold sweep with precision and recall from
  `metrics`, and the alarm rate each threshold implies.
- A test asserts what the chapter claims: that the storm hours score
  above the quiet day's, that accuracy exceeds 99 % for a model that
  flags nothing, and that the sweep is monotone in the direction it
  should be.
- Every figure quoted in the chapter comes from a run of the committed
  code on a named machine.
- Headings, links, index rows and example paths are consistent after the
  renumber; no link points at a file that no longer exists.
- No performance impact: one example and documentation, no library code
  on any hot path.

## Notes
The first version of the example invented labels — it called the storm
hours "those above anything in training" and measured precision and
recall against them. Every threshold then reported 100 % recall, because
the labels and the thresholds came from the same statistic. That is the
mistake the chapter now warns about, and it is in the chapter because I
made it.

The rewritten example does not pretend to have ground truth. It reports
what a threshold costs on a day believed ordinary and what it sees on a
day known to be disrupted, which is the honest question, and it uses
`metrics.Confusion` only for the one coarse label that can be defended.

The measured result is stronger than the one the spec anticipated. The
detector works — the storm day's median hour scores 6.75 against 4.86 in
training — and is unusable anyway: at the strictest possible threshold,
above every training hour, one airport still fires three times on the
quiet day, which is twenty four alarms a day across the eight on the
feed. The quiet day's worst hour (7.82) already exceeds the training
maximum (6.92), so the distributions overlap and no threshold separates
them.

The accuracy figure also came out differently and better. With the storm
one day in six a do-nothing detector scores 50 %, which is not
dramatic; the chapter extrapolates honestly to one day in sixty, where
the same detector scores 98.3 %.

