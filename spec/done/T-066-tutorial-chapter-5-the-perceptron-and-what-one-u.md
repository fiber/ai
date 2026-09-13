---
id: T-066
title: Tutorial chapter 5: the perceptron and what one unit cannot do
status: done
scope:
  - docs/tutorial/
  - examples/tutorial/
  - docs/manual/
  - README.md
manual:
  - docs/manual/getting-started.md
done: 2026-09-13
created: 2026-09-13
---

## Goal

The tutorial jumps from fitting a line to numbers (chapter 4) straight
to a two-layer network on spiral data. The chapter that was 5 asserts
the central idea in one sentence — "three of them in a row would still
be one straight line, because a line of a line is a line" — which is
true and completely unconvincing to a reader meeting it for the first
time.

There is a way to make it undeniable in forty lines, and it also teaches
something a reader will actually use: how to recognise that a model
cannot express the answer, as distinct from being badly tuned. Those two
look identical from the outside — a loss that stops falling — and only
one is fixed by turning knobs.

## Design

A new chapter 5, `examples/tutorial/05-perceptron`:

- One unit, two weights, a bias, a threshold. Trained by Rosenblatt's
  rule, not by gradients: no loss function, nothing differentiated. It
  is worth showing that learning existed before the machinery of
  chapter 3, so gradient descent looks like a choice rather than a law.
- AND separates in six epochs, OR in four. XOR never does, at any
  learning rate, from any initialisation — and the perceptron
  convergence guarantee is why that can be stated rather than tested.
- Then one hidden layer, trained the modern way: one unit still fails
  (loss 0.167), two solve it exactly (loss 0.000), four are no better.
  The jump that matters is one to two, and what makes it work is the
  nonlinearity between the layers, not the extra parameters.
- Closes on the practical point: three failure modes look the same from
  outside, and the tell for this one is that tuning changes nothing.

Placement forces a renumber. The chapter has to come before the spiral
classifier or it teaches nothing, so chapters 5 to 14 become 6 to 15,
along with their example directories, headings, cross-references, "Next"
links and the manual's references. That is churn, and it is cheapest
now: the tutorial will only get longer.

## Acceptance

- `go test ./examples/tutorial/05-perceptron/` asserts the argument: AND
  and OR separate, XOR does not, one hidden unit fails and two succeed.
  If the chapter's claim ever stops being true, the test fails.
- Every chapter heading, index row, "Next" link, cross-reference and
  example path is consistent after the renumber; no link points at a
  file that no longer exists.
- `go build ./...`, `go vet ./...` and `go test ./...` pass.
- No performance impact: documentation and one example, no library code.

## Notes

Written for a reader who is a strong algorithmic thinker and decades
away from their last mathematics course, which is the tutorial's stated
audience. The one piece of notation in the first draft — the decision
rule written as a dot product — was replaced by the line of Go that
implements it. The chapter also lost a table mapping the tutorial's
whole arc, which was commentary about the tutorial rather than teaching.

The renumber caught three classes of stale reference that a careless
pass would have left: the `# N.` headings inside each chapter, the
numbered link labels (`[7. A model in service](08-service.md)`), and
capitalised sentence-opening references, which a substitution for
"chapter N" misses. Worth remembering if chapters are ever inserted
again.
