---
id: T-071
title: Tutorial: chapter 8 no longer ends it; Next links through chapter 18
status: done
scope:
  - docs/tutorial/
manual:
  - none
done: 2026-09-14
created: 2026-09-14
---

## Goal

Chapter 8 still closes with "That is the end of the tutorial", written
when it was the last chapter; ten chapters now follow it. Chapters 1 to
7 end with a "Next:" link and 8 to 18 do not, so a reader who arrives
from the index stops at 8.

## Design

Chapter 8's closing paragraph says what the first eight chapters were
(the whole path from a tensor to a served model) and what the rest are
(the same tools on real problems), and links on. Chapters 8 to 17 get
the same "Next:" line the early chapters have; chapter 18 closes the
tutorial and points at the manual. Documentation only.

## Acceptance

- No chapter but the last says the tutorial ends there.
- Every chapter 1 to 17 ends with a link to the next; every link
  target exists.

## Notes

Decisions made while implementing, surprises, follow-ups.
