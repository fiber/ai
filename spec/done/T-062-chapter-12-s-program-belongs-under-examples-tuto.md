---
id: T-062
title: Chapter 12's program belongs under examples/tutorial like every other chapter
status: done
scope:
  - examples/
  - docs/tutorial/
  - docs/manual/applications.md
  - README.md
  - BUGS.md
manual:
  - docs/manual/applications.md
done: 2026-09-12
created: 2026-09-12
---

## Goal

Every tutorial chapter's program lives in `examples/tutorial/NN-name`,
from `01-tensors` to `11-attention`. Chapter 12's lives in
`examples/mnist`, because T-052 treated it as an application rather than
as the chapter it is. The result is that `examples/tutorial` stops at 11
and a reader looking there for chapter 12 does not find it — which is
how this was reported.

Chapters 13 and 14 are coming and would face the same choice, so the
convention should be settled before there are three exceptions instead
of one.

`BUGS.md` also still carries the full description of B-009 although the
bug is fixed and listed in BUGS-FIXED.md. `gate done` removed its
checklist line but not the paragraphs written above it by hand, so the
file that is supposed to list open bugs opens with a closed one.

## Design

`git mv examples/mnist examples/tutorial/12-mnist`, and update the four
places that name the old path: the tutorial index, the chapter's own
run line, the manual's application page, and BUGS.md. The archived specs
under `spec/done/` keep the old path, because they record what was true
when they were written and rewriting history in them would be worse than
a stale path.

The manual keeps its MNIST section on the applications page. The example
is a tutorial chapter, but that page is where a reader looks for "what
can this thing do", and the section links to the chapter.

Remove the B-009 paragraphs from BUGS.md. Nothing is lost: the same
material is in `spec/done/B-009-…`, at more length.

## Acceptance

- `examples/tutorial/` contains one directory per chapter, 01 to 12,
  with no gaps.
- `go test ./...` and `go vet ./...` pass; `go run
  ./examples/tutorial/12-mnist` trains as before.
- No reference to `examples/mnist` remains outside `spec/done/`.
- BUGS.md lists only open bugs.
- No performance impact: this moves files and touches no code path.

## Notes
Reported by reading the repository rather than the documentation:
`examples/tutorial` stopped at 11 and chapter 12's program was
elsewhere. The tutorial index was correct the whole time, which is why
nobody working on it noticed — the index says `go run ./examples/mnist`
and that worked. The convention was broken where only a browser of the
tree would see it.

The B-009 leftover in BUGS.md has the same shape: `gate done` moves the
checklist line it created and cannot know about paragraphs added by
hand above it. Worth remembering when writing a long bug entry — the
prose is not managed, only the line is.

