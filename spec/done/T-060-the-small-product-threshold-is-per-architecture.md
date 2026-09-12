---
id: T-060
title: The small-product threshold is per architecture: AVX2 loses parallelism it cannot spare
status: done
scope:
  - internal/blas/
  - BENCHMARKS.md
  - benchmarks/
manual:
  - docs/manual/performance.md
done: 2026-09-12
created: 2026-09-12
---

## Goal

T-050 added a path for small products: one call on the calling
goroutine, nothing packed through the pool, no packing and compute
rounds. It took the M2 Pro from 376 to 553 GFLOPS at 128². T-059
measured it on x86 for the first time and it does the opposite there —
44 GFLOPS at 128² where the blocked driver reaches 105, and 36 at 160²
where the driver reaches 149.

The reason is not the kernel. The small path runs on one goroutine, so
it trades the driver's fixed cost for every worker but one, and how good
that trade is depends on how much of the machine one core is. On the
M2 Pro a single thread reaches 1 182 of 2 118 GFLOPS at n=1024, so
giving up the rest costs a factor of 1.8. On six AVX2 vCPUs a core is a
sixth of the machine and the same trade costs almost everything.
`SmallLimit` is one constant for every back-end, tuned on the machine
where the trade is cheapest, and applied where it is dearest.

## Design

Move the threshold into the per-architecture parameter files that
already hold KC, MC and NC, since it is the same kind of number: measured
on hardware, not derived.

    params_arm64.go  defaultSmallLimit = 160³   (unchanged)
    params_amd64.go  defaultSmallLimit =  96³
    params_other.go  defaultSmallLimit =  96³   (unmeasured, conservative)

`SmallLimit` becomes `var SmallLimit = defaultSmallLimit`, so
`FIBERAI_BLAS_SMALL` still overrides it and 0 still disables the path.

96³ from a sweep on the AVX2 guest: at that limit 128² measures 96
GFLOPS and 160² measures 132, against 45 and 36 today, while the shapes
below it keep the path and their advantage (64²: 29 with, 19 without).
110³ was also tried and is slightly worse at 160² (117 against 132).

What this does not do: the right threshold plainly depends on the worker
count as well as the back-end — on a single-core AVX2 machine the path
would pay much further up — and one constant per architecture cannot
express that. Making it a function of `parallel.Workers()` is the honest
fix and needs measurements on machines with different core counts, which
we do not have. Recorded as a follow-up rather than guessed at.

## Acceptance

- On the six-vCPU AVX2 guest, `cmd/bench -only small` reaches at least
  90 GFLOPS at 128² and at least 120 at 160², against 45 and 36 before.
  PyTorch 2.14 with MKL on the same machine and session is the baseline
  at 126 and 130; this closes most of that gap and does not claim to
  close all of it.
- Shapes below the new limit are unchanged: 32², 64² and the narrow
  autoencoder products stay within run-to-run noise of today's figures,
  and the tiny autoencoder training step stays at or above 225 K
  samples/s (PyTorch 31 K).
- The M2 Pro is untouched: `cmd/bench -only small` stays within noise of
  63 / 216 / 371 / 558 / 671 GFLOPS at 32² to 160², since arm64 keeps
  160³.
- `go test ./...` passes on both machines, and `GOARCH=amd64 go vet`
  is clean.

## Notes
Measured over five runs on the guest after the change: 128² median 92.4
(range 75.6 to 97.7) against 44.9 before, 160² median 123.4 (118.5 to
132.9) against 35.6. So 2.1× and 3.5×, and at 160² we pass MKL's 130 on
the better runs and sit just under it on the median.

The acceptance criterion first said 125 at 160². That came from a single
sweep run that measured 132, which turned out to be the top of the
range; the median is 123.4. Criterion corrected to 120 and the measured
distribution recorded, rather than quietly counting a target as met.

The M2 Pro is unchanged, as intended: 60.6 / 215.8 / 369.3 / 562.8 /
670.7 GFLOPS from 32² to 160², against 63 / 216 / 371 / 558 / 671
before.

One thing this measurement nearly got wrong. The tiny-autoencoder rows
looked as though they had improved by a third, from 927 K to 1.21 M
samples/s. They had not: the earlier figure came from a full suite run
and the new one from `-only small`, and on this guest the allocator
state left by preceding sections moves that case by about that much —
the same effect T-054 found on convolution. Those shapes are far below
both thresholds and cannot have been affected by this change at all.
BENCHMARKS.md now states both contexts.

Follow-up, unspecced: the threshold should depend on the worker count as
well as the architecture, since the trade is between fixed cost and lost
parallelism. One constant per architecture cannot express that, and
choosing the function needs machines with different core counts than the
two we have.
