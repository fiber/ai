---
id: B-007
title: tokenizer TestLoadTime asserts a wall-clock bound that a slower or loaded machine misses
status: done
scope:
  - tokenizer/
manual:
  - none
done: 2026-09-09
created: 2026-09-09
---

## Goal

`TestLoadTime` fails when loading tokenizer.json takes more than one
second. The bound was set from the M2 Pro (0.7 s); the Xeon Gold 6130
loads it in 1.7 s under `go test ./...` and the suite goes red for a
machine that is merely slower. A unit test must not encode a wall-clock
budget of one particular machine.

## Design

Keep the test as a regression guard against a pathological load path
(the parse once took 20 s) with a bound of 10 s, log the measured time,
and leave the performance figure to the benchmark. Nothing else changes.

## Acceptance

`go test ./tokenizer` passes on the Xeon under `go test ./...`; the load
time is still printed with `-v`.

## Notes
