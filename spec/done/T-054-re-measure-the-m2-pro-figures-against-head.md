---
id: T-054
title: Re-measure the M2 Pro figures against HEAD
status: done
scope:
  - BENCHMARKS.md
  - benchmarks/
manual:
  - docs/manual/performance.md
done: 2026-09-12
created: 2026-09-12
---

## Goal

BENCHMARKS.md carries two M2 Pro measurements, the matrix-multiply table
from 5 September and the headline from 8 September. Three changes have
landed since the later of them and all three touch the paths being
measured: T-047 made the performance cores the default worker count on
Apple Silicon, the attention rework of 9 September changed the packed
path and made it the default again, and T-050 rebuilt the small-GEMM
path. A spot check of one shape shows 2 460 GFLOPS at n=1024 where the
file records 2 183 — we are about 13 % faster than we claim.

Understating ourselves is still publishing a number a reader cannot
reproduce, and the website is about to quote these figures. Every number
on that site has to come back when a reader runs one command.

## Design

Re-run all three sides the way the file's own Reproduce block says, into
`benchmarks/results/m2pro-2026-09-12/`, and update the tables and the
prose that quotes them:

    go run ./cmd/bench
    FIBERAI_KERNEL=generic go run ./cmd/bench -quick
    benchmarks/python/.venv/bin/python bench.py

The older result directories stay: they are what the earlier sections
were measured from and the file names their dates. Sections about other
machines — the M4 Air, the Xeon, the AVX2 KVM guest — are not re-measured
here and keep their dates, because this spec only has the M2 Pro.

Prose that draws a conclusion from a number gets re-read against the new
figures, not only the tables. `docs/manual/performance.md` quotes several
and is in the manual list for that reason.

## Acceptance

- Every M2 Pro figure in BENCHMARKS.md comes from the 2026-09-12 run, and
  the raw output of all three sides is committed under
  `benchmarks/results/m2pro-2026-09-12/`.
- NumPy and PyTorch are re-measured in the same session on the same
  machine, so both sides of every comparison share a date. Baseline to
  beat or match at n=2048: NumPy 2 241 and PyTorch 2 243 GFLOPS, which
  the AMX path met at 2 301 on 5 September; at n=1024 Accelerate was
  ahead, 2 693 against 2 183, and the new figures say plainly where we
  stand rather than quoting only the size we win.
- Where a claim in the prose no longer follows from the numbers, the
  claim changes, not the numbers.
- `docs/manual/performance.md` agrees with BENCHMARKS.md.

## Notes
The premise was half wrong. I opened this because a spot check seemed to
show n=1024 at 2 460 GFLOPS against the file's 2 183 — but that compared
the new B-packed-once figure with the old packed-per-call one. Like for
like the number is 2 118, three per cent below, and two runs of one
binary an hour apart gave 2 089 and 2 157, so the difference is the
machine, not the code.

What the re-measurement actually found:

- **n=128 went from 376 to 553 GFLOPS**, which is T-050 arriving in the
  table. Everything from 256 up is unchanged within the spread.
- **tanh went from 5.0 to 90 GB/s** and LayerNorm from 3.37 to 1.81 ms.
  Both were listed as losses against PyTorch and are now wins; the file
  had been claiming we were slower than we are.
- **The MLP table was NEON-era**: forward 1.51 ms where it now measures
  310 µs. It moved when GEMM moved to AMX and nobody re-ran it.
- **The convolution figure depends on what ran before it.** In a full
  suite it reports 184 GFLOPS, alone 334 to 352. The element-wise section
  leaves 404 MiB retained in the mapped pool and forces some 3 900
  collections, and the convolution pays for that. The file now quotes the
  isolated figure and says so, but the harness should reset the pool
  between sections; that needs its own spec.
- **`bench.py` did not parse** under the venv it ships with (B-011), so
  the documented reproduce command produced a traceback.

Two figures are not from this session and are marked where they appear:
the PyTorch EmbeddingGemma row, because the venv has no
`sentence-transformers`, and every non-M2 machine, which keeps the date
in its heading.

Also removed a reference to a private project by name in the small-
products section. It should not have been in a public repository, and
`spec/done/T-050` still contains the same word.
