---
id: B-010
title: README and getting-started still say the tutorial has seven chapters
status: done
scope:
  - README.md
  - docs/manual/getting-started.md
manual:
  - docs/manual/getting-started.md
done: 2026-09-12
created: 2026-09-12
---

## Goal

The tutorial has grown from seven chapters to twelve, and three places
still say seven: the repository tree in README.md, the sentence under it,
and the first line of the getting-started page. The first thing a visitor
reads about the tutorial understates it by five chapters, and the count
is the one number on that page a reader can check in ten seconds.

## Design

Say twelve in all three, mention in README.md that the last chapter
trains on MNIST and compares against PyTorch, since that is the part a
sceptical reader wants to find, and add the MNIST example to the tree's
`examples/` line, which T-052 also left out.

The counts are hand-maintained and will go stale again. Generating them
is not worth a build step for a number that changes twice a year; the
guard is that a chapter is added under a spec, and the spec declares the
manual pages it touches.

## Acceptance

- No occurrence of "seven chapters" or "seven runnable" remains in
  README.md or docs/.
- The count matches the files in docs/tutorial/.

## Notes

Found by the site builder reading the repository as a newcomer would,
which is the reading the README is written for and the one nobody who
works on it performs.
