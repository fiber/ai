---
id: T-053
title: Depend on a released ai-data instead of a pseudo-version
status: done
scope:
  - go.mod
  - go.sum
manual:
  - none
done: 2026-09-12
created: 2026-09-12
---

## Goal

`examples/mnist` pulls its digits from github.com/fiber/ai-data, which
T-052 added at whatever pseudo-version the tool derived from the tip of
its main branch. A pseudo-version says nothing to a reader about what
they are getting, and it moves the moment that branch does. The data
module is now tagged v0.1.0, so the requirement should name it.

## Design

`go get github.com/fiber/ai-data@v0.1.0`, which rewrites the requirement
and the checksum. Nothing else changes: the module is imported only by
the example, so no consumer of the framework is affected either way.

## Acceptance

- `go.mod` requires `github.com/fiber/ai-data v0.1.0`, no pseudo-version.
- `go build ./...` and `go test ./examples/mnist/` pass against it.

## Notes

