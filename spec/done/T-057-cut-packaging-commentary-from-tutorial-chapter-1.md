---
id: T-057
title: Cut packaging commentary from tutorial chapter 12
status: done
scope:
  - docs/tutorial/
manual:
  - none
done: 2026-09-12
created: 2026-09-12
---

## Goal

Chapter 12 spent a paragraph explaining why the digits live in a separate
module: bandwidth for the operator of a public server, module size for
people who import `tensor` and never touch MNIST. Both are true and
neither teaches anything. A reader opening the chapter wants to know how
to read someone else's binary format and what a convolution buys; they
did not come for our packaging decisions, which belong in the spec that
made them and in the manual.

A shorter version of the same problem appears at the end of the PyTorch
section, where the chapter recounts a bug the comparison found. That is
repository history.

## Design

Replace the packaging paragraph with the thing it was displacing: what
the idx format actually is, shown as the four lines of `encoding/binary`
that read it, and why that shape — fixed-width binary records with the
geometry in a header — is worth recognising, since most real formats look
like it and nothing like a CSV. Keep one factual sentence that the data
comes from `ai-data` and is verified by checksum, because a reader does
need to know where the bytes came from.

Cut the bug anecdote to nothing and keep the one-line lesson that a
difference is only interesting once you know its cause, which is the
transferable part. Keep the CPU-against-CPU caveat, which a reader
genuinely needs when reading the timings.

## Acceptance

- No paragraph in chapter 12 explains a decision about module layout,
  distribution or repository history.
- The `idx` section shows the header struct and the read, so a reader
  could open a format they have never seen with the same technique.
- The chapter is shorter than before.

## Notes

Caught in review: "that is filler prose that adds no educational value to
the tutorial". Worth applying to the other chapters at some point — the
rationale for a design decision is interesting to whoever made it and
rarely to whoever is learning from it.
