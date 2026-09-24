---
id: T-076
title: Gate cites the beat-Python rule as PROCESS.md rule 7, which is the manual rule
status: done
scope:
  - PROCESS.md
  - cmd/gate/
  - docs/manual/process.md
manual:
  - docs/manual/process.md
done: 2026-09-24
created: 2026-09-24
---

## Goal

The gate's two performance messages — the spec rejection and the
`gate new` reminder — and the manual's process page cite the
beat-Python rule as "PROCESS.md rule 7". PROCESS.md rule 7 is "The
manual is maintained", and PROCESS.md does not state the beat-Python
rule at all; the citation has been wrong since the rule was added in
abce67e. A reader, or an agent, following the reference lands on the
wrong rule.

## Design

Add the rule to PROCESS.md as rule 8, worded as the manual already
words it, and change the three citations to rule 8. The `gate new`
reminder also gains the "or state 'no performance impact'" escape the
rejection message already names, so the two messages describe the same
check.

## Acceptance

- PROCESS.md states the beat-Python rule as rule 8.
- No file cites "rule 7" for the beat-Python rule; `grep -rn 'rule 7'`
  over PROCESS.md, cmd/gate and docs/manual returns nothing.
- `go test ./cmd/gate/` passes.
- No performance impact: message text and documentation only.

## Notes

Decisions made while implementing, surprises, follow-ups.

The citation was found by a prompt audit of the text the gate shows to
Claude Code sessions; `git show abce67e:PROCESS.md` confirms rule 7 was
already the manual rule when the citation was written.
