---
id: T-012
title: Fixed bugs are tracked in BUGS-FIXED.md, enforced by the gate
status: done
scope:
  - cmd/gate/
manual:
  - docs/manual/process.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

Fixed bugs are listed in their own file, BUGS-FIXED.md, not mixed into
DONE.md with finished work items, so the bug history can be read on its
own. The gate enforces it.

## Design

- `gate done B-nnn` prepends the entry to BUGS-FIXED.md (newest first)
  instead of DONE.md and removes the open item from BUGS.md.
- `gate check` requires BUGS-FIXED.md to exist and every done `B-` spec
  to be listed there; done `T-` specs must be in DONE.md as before.
- PROCESS.md, CLAUDE.md and the manual describe the fourth list.
- The existing entry for B-001 moves from DONE.md to BUGS-FIXED.md.

## Acceptance

- `gate check` fails when a done bug spec is missing from BUGS-FIXED.md
  and passes after `gate done` on a bug spec. Unit test covers the round
  trip. No performance impact.

## Notes
