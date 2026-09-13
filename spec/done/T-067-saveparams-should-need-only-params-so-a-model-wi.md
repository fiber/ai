---
id: T-067
title: SaveParams should need only Params, so a model with its own forward can be saved
status: done
scope:
  - nn/
  - examples/tutorial/
  - docs/tutorial/
manual:
  - docs/manual/nn-and-optim.md
done: 2026-09-13
created: 2026-09-13
---

## Goal

`SaveParams` and `LoadParams` take an `nn.Module`, and `Module` requires
`Forward(*tensor.Tensor) *tensor.Tensor`. Saving never calls `Forward`
— it only calls `Params()` — but the interface demands it anyway, so any
model whose forward pass has a different shape cannot be saved.

That is not a corner case. Chapter 14's language model takes token ids
and a batch layout, `Forward(ids []int, B, T int)`, because that is what
a decoder needs. It has `Params()`, it is an ordinary model, and the
library will not persist it. So the chapter trains for seven minutes and
throws the result away, and a reader who wants to generate twice has to
train twice.

## Design

Widen the parameter, not the model. Saving needs one method:

    type Parameterised interface{ Params() []*tensor.Tensor }

    func SaveParams(w io.Writer, m Parameterised) error
    func LoadParams(r io.Reader, m Parameterised) error

`Module` already satisfies it, so every existing call keeps compiling
and the file format is untouched. This is the narrowest fix available:
the functions asked for more than they used, and now they ask for what
they use.

Chapter 14's example then gets `-save` and `-load`, so seven minutes of
training is spent once, and the chapter says so. Chapter 15 gets the
same, since comparing two generation paths is exactly when you want a
trained model back without waiting for it.

Rejected: a second pair of functions taking `[]*tensor.Tensor`. It would
work and would leave two ways to do one thing with nothing to choose
between them.

## Acceptance

- A model with a non-standard forward — chapter 14's, which takes token
  ids — round-trips through `SaveParams` and `LoadParams` and generates
  identical text from the same seed before and after.
- Every existing caller compiles unchanged and the file format is
  unaltered, verified by loading a file written by the old code.
- `go run ./examples/tutorial/14-language-model -save m.bin` followed by
  `-load m.bin -steps 0` generates without training.
- No performance impact: an interface change on a path that runs once
  per save, and no library code on any hot path.

## Notes
The change is four lines: an interface with one method, and two
signatures widened to take it. Every existing caller compiles unchanged
because `Module` already satisfies it, and the file format is untouched.

Two things surfaced while testing it that were not in the plan.

The example's generation shared its random stream with the training
loop, so a loaded model produced different text from the model that had
been saved — the training loop had drawn from the stream by the time
generation started, and a loaded model had not. Correct behaviour, wrong
arrangement: generation now draws from its own stream, and `-load` with
`-steps 0` reproduces exactly what `-save` produced. Without that the
save feature would have looked broken while working perfectly.

The manual did not document `SaveParams` at all — persistence was
described only in tutorial chapter 8. It has a section now, on the page
where a reader would look for it.

