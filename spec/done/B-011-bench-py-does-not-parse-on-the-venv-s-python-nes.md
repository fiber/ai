---
id: B-011
title: bench.py does not parse on the venv's Python: nested quotes in an f-string need 3.12
status: done
scope:
  - benchmarks/python/
manual:
  - none
done: 2026-09-12
created: 2026-09-12
---

## Goal

`benchmarks/python/bench.py` does not parse under the Python in its own
`.venv`. One f-string nests double quotes inside double quotes, which
became legal only in Python 3.12; the venv is 3.9.6, so the file is a
`SyntaxError` before a single benchmark runs. The command BENCHMARKS.md
prints under "Reproduce" therefore produces nothing but a traceback for
anyone following it with the checked-in environment.

## Design

Compute the embedding dimension into a local before the f-string, which
is what the nesting was there to avoid, and works on every version.

## Acceptance

- `.venv/bin/python -m py_compile bench.py` succeeds on Python 3.9.
- `.venv/bin/python bench.py` produces the full Markdown report, and the
  NumPy and PyTorch figures it prints are the ones BENCHMARKS.md quotes:
  at n=2048, NumPy 2 225 and PyTorch 2 192 GFLOPS.

## Notes

Found while re-measuring for T-054. The line predates that work, so the
last Python figures in BENCHMARKS.md were produced either by a newer
interpreter than the venv pins or before the line was written; either way
the documented command had stopped working and nothing noticed, because
nothing runs it but a person preparing a release.
