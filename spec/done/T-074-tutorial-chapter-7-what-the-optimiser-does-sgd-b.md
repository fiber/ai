---
id: T-074
title: Tutorial chapter 7: what the optimiser does, SGD before Adam
status: done
scope:
  - docs/tutorial/
  - examples/tutorial/
  - docs/manual/nn-and-optim.md
manual:
  - docs/manual/nn-and-optim.md
done: 2026-09-15
created: 2026-09-15
---

## Goal
---

## Goal

`optim.NewSGD` appears nowhere in the tutorial. The reader derives a
gradient step by hand in chapter 4 — subtract the gradient times a
small number — and from chapter 6 on is handed `optim.NewAdam` with no
account of what happened in between. Adam is three ideas stacked on
that hand-written step, and a reader who has not seen the step fail
cannot tell which idea is doing the work, cannot debug a learning rate,
and will carry "use Adam" as superstition rather than as a choice.

This is the largest single gap between the library and the tutorial,
and it is a section rather than a chapter: the loop that shows it is
already written, only the `Step` changes.

## Design

A new section in chapter 7 immediately after "Mini-batches", where the
loop body `ZeroGrad / Backward / Step` has just been shown and `Step`
is the only part not yet explained. `examples/tutorial/07-training-loop`
gains a comparison that trains the same architecture from the same
seed, changing nothing but the optimiser:

- plain SGD at three learning rates, chosen after measurement so that
  one crawls, one works and one is unstable;
- SGD with momentum 0.9 at the middle rate;
- Adam at its usual 1e-3.

Reported per run: training and validation loss at a few epochs and the
epoch at which the validation loss first drops below a fixed mark, so
"faster" is a number rather than an impression. Fewer epochs than the
overfitting comparison, which needs 600; this one is about the first
descent.

The prose gives the three update rules in the order they were invented,
each as one line of arithmetic — the step, the velocity, the
per-parameter scale — and says what each fixes. It names the practical
rule the library's defaults encode (AdamW at 1e-3, momentum 0.9 when
using SGD) and the honest caveat that tuned SGD with momentum still
wins on some problems, so Adam is the default and not the answer.

The existing three overfitting runs keep using Adam and are untouched,
so the chapter's later numbers do not move.

`docs/manual/nn-and-optim.md` gains two sentences under Optimisers on
what to reach for first.

## Acceptance

- `go run ./examples/tutorial/07-training-loop` prints the optimiser
  comparison before the overfitting runs, and the three overfitting
  tables are unchanged from the committed chapter text.
- The learning rates quoted are the ones the committed code uses, and
  every number in the new section comes from one run on a named
  machine with nothing else running.
- The claim that one learning rate is too large is visible in the
  output rather than asserted.
- Chapter 7's existing sections, links and the index row are unchanged
  otherwise.
- No library code changes; no performance impact.

## Notes

Placed after "The validation set" rather than after "Mini-batches" as
the spec said: the comparison is scored on validation loss, and that
term is introduced in the section between the two.

Measured on the M2 laptop with nothing else running, 200 epochs on the
small model, validation loss:

    SGD lr 0.003                     ep10 1.397  ep200 0.451  never < 0.30
    SGD lr 0.05                      ep10 0.821  ep200 0.291  ep 36
    SGD lr 0.5                       NaN throughout
    SGD lr 0.05, momentum 0.9        ep10 0.307  ep200 0.219  ep 7
    Adam lr 0.001                    ep10 1.702  ep200 0.243  ep 145

Two things came out of the measurement rather than out of the plan.
The unstable rate produces `NaN` rather than a merely bad number, which
makes a better lesson than the "oscillates" the spec anticipated: a
`NaN` loss is the one failure that announces itself, and it is almost
always the learning rate. And Adam is the slowest of the three that
work — by a factor of twenty against momentum. On a 321-parameter model
with well-scaled inputs there is nothing for its per-parameter scaling
to fix, so the chapter says that plainly and explains where Adam does
earn its keep, instead of quietly choosing a model where the default
looks good.

`compare` reseeds the global generator for each optimiser, which moved
the initial weights of the three overfitting runs that follow and
changed their numbers. Reseeding after the comparison restores them; a
diff against the committed chapter text confirms the three tables are
identical.
