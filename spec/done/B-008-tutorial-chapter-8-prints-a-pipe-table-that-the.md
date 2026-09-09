---
id: B-008
title: tutorial chapter 8 prints a pipe table that the chapter quotes inside a code fence, so it renders as text
status: done
scope:
  - examples/tutorial/
  - docs/tutorial/
manual:
  - none
done: 2026-09-09
created: 2026-09-09
---

## Goal

The chapter-8 program ends with a Markdown pipe table, and the chapter
quotes the program's output inside a code fence, so GitHub renders the
pipes and dashes as text. Program output belongs in a fence and a table
belongs outside one.

## Design

The program prints an aligned plain-text table (`%-26s %8.2f %8.2f`);
the chapter keeps that output in its fence and adds the same numbers as
a real Markdown table above the discussion.

## Acceptance

`go run ./examples/tutorial/08-convolution` prints aligned columns; the
chapter renders a table on GitHub. No performance impact.

## Notes
