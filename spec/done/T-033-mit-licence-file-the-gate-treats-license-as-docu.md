---
id: T-033
title: MIT licence file; the gate treats LICENSE as documentation
status: done
scope:
  - LICENSE
  - cmd/gate/
manual: none
done: 2026-09-06
created: 2026-09-06
---

## Goal

The repository is about to be published and has no licence file. Add
the MIT licence, and stop the process gate from treating `LICENSE` as a
code file so future edits to it (or to `NOTICE`, `AUTHORS` and the like)
need no spec.

## Design

- `LICENSE` at the repository root: the MIT text, copyright 2026 Sven
  Engelhardt.
- `cmd/gate`: `exempt` also returns true for extension-less root files
  that are conventionally documentation: `LICENSE`, `LICENSE.*`,
  `NOTICE`, `AUTHORS`, `CONTRIBUTORS`, `CODEOWNERS`. Test added.
- README gains a one-line licence note.

## Acceptance

- `go test ./cmd/gate` passes with the new exempt cases.
- `LICENSE` exists with the MIT text; `go run ./cmd/gate check` is clean.

## Notes
