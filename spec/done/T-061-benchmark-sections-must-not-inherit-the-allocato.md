---
id: T-061
title: Benchmark sections must not inherit the allocator state of the section before them
status: done
scope:
  - cmd/bench/
  - benchmarks/
  - BENCHMARKS.md
  - tensor/
manual:
  - docs/manual/performance.md
done: 2026-09-13
created: 2026-09-12
---

## Goal

`cmd/bench` runs every case in one process, one after another, and the
cases interfere. By the time the convolution case runs, the element-wise
section has left 404 MiB retained in the mapped pool and forced some
3 900 collections; convolution then measures 184 GFLOPS where it
measures 348 alone. The tiny autoencoder reads 927 K samples/s in the
suite and 1.14 M alone. The MLP training step reads 25 K in the suite and
34 K alone.

`benchmarks/python/bench.py` has the mirror of the same problem with the
opposite sign: PyTorch's caching allocator is warm by the time its later
sections run, so they measure *faster* than they do alone. Its MLP
forward+backward reads 43.6 K samples/s in the suite and 25.4 K on its
own.

Put those together and the published comparison inverts. BENCHMARKS.md
currently says PyTorch is 1.8x ahead of us on the MLP backward pass on
x86 and calls it a fusion gap. Measured with each side alone, three runs
each, we are 1.2x ahead on forward+backward and 1.5x ahead on the full
step. The gap was an artefact of position in the suite, on both sides at
once.

This is not a small correction to one row. Every case that runs late in
either suite is suspect, on both machines, and no conclusion drawn from
those tables is safe until the harness stops carrying state across cases.

## Design

**Isolate by process, not by cleanup.** An in-process reset — trim the
mapped pool, force a collection — is cheaper and would fix most of it,
but it cannot restore Go heap layout, the packed-operand cache, CPU
cache and branch predictor state, or the page cache. The failure mode of
a partial reset is exactly what we have now: plausible numbers that are
quietly wrong. So `cmd/bench` gains `-isolate` (default on) which
re-executes itself once per section with `-only <section>` and
concatenates the output, and `-isolate=false` keeps today's
single-process behaviour for anyone comparing against the old tables.

`bench.py` gets the same treatment, driven from the same list of
sections so the two stay in step.

The allocator line each section prints stays: a section that arrives
with a cold pool and still shows retained bytes is telling us something.

**Then re-measure everything and correct what changes.** Both machines,
both sides, medians of three, and the affected prose rewritten rather
than the numbers swapped underneath it. In particular "the backward pass
stays behind, the fusion gap" has to go if the isolated numbers hold,
and the M2 Pro rows need the same scrutiny as the x86 ones, since that
machine's numbers were taken the same way.

## Acceptance

- Running `cmd/bench` with its sections in different orders gives the
  same figures within run-to-run noise. Today reordering moves
  convolution by 45 %.
- A section measured inside the suite and the same section measured with
  `-only` agree within noise, on both machines. That is the point of the
  spec and the one criterion that cannot be waived.
- Both suites re-run on the M2 Pro and the AVX2 guest, medians of three,
  raw output committed.
- The MLP rows are restated from isolated measurements. Baseline on the
  AVX2 guest with PyTorch 2.14 measured alone: forward 2.99 ms,
  forward+backward 7.8 ms, full step 10.9 ms; fiber/ai measured alone:
  1.53, 6.58 and 7.49 ms. Whatever the re-measurement gives, the table
  must say which side is ahead and by how much, with both measured the
  same way.
- Any claim in BENCHMARKS.md or docs/manual/performance.md that the
  re-measurement contradicts is rewritten, not quietly dropped.

## Notes

Found while estimating the effort to close the "fusion gap", which turns
out to cost nothing because it does not exist.

The profile that started that investigation is worth keeping anyway. On
the M2 Pro a training step spends about half its CPU time in worker
coordination — but the step scales 1.98x from one worker to six and the
forward barely scales at all, because the AMX unit is shared per
cluster. That spinning is waiting on a saturated coprocessor: it costs
CPU time, not wall clock. On x86, with no shared unit, the same workload
scales 4.5x. Neither machine has a scheduling problem here, which is the
opposite of what the profile alone suggested.

Three false conclusions in one day — convolution 45 % low, the
autoencoder 20 % low, and a 1.8x deficit that is really a 1.2x advantage
— all came from one unexamined assumption: that a benchmark suite
measures each case independently.
There were two faults, not one, and the second was doing most of the
damage.

**Cross-section.** Fixed as designed, by running each section in its own
process. On the M2 Pro convolution went from 184 GFLOPS inside the suite
to 365, against 364 measured alone. On the AVX2 guest PyTorch's inflated
MLP row fell from 41 K to 30 K, matching its 32 K alone.

**Within-section.** The harness warmed each case with one call before
timing it. One call is enough on bare metal; on a virtualised host the
first touches of a large result pay page faults costing two to three
times native, and the transient outlasts it. The tell was impossible:
forward+backward reported 10.49 ms against 7.07 ms for the same work
plus an optimiser step. Warming for a third of the measurement window
fixes it, at about a third more runtime, and the ordering became
coherent immediately (1.76 / 7.54 / 8.06 ms).

Acceptance, honestly:

- **Met on the M2 Pro.** In-suite and alone agree within one percent for
  both convolution (365 against 364) and the MLP (215 K against 215 K),
  with spreads of one to two percent over three runs.
- **Met on the AVX2 guest for the MLP**, where the spread fell from
  105 % to between 4 and 12 % and in-suite sits 8 % under alone, inside
  that band.
- **Not met on the AVX2 guest for convolution**, and not fixable by a
  harness. That case allocates 345 MB per iteration and measures 62 to
  79 GFLOPS in the suite against 20 to 87 alone — a four-fold swing
  between identical runs, worse alone than in-suite. It is removed from
  the guest's table rather than published with a plausible-looking
  number.

What this cost in credibility is worth recording. The MLP comparison on
x86 was published as PyTorch 1.8× ahead, then claimed here as 1.2×
behind from a standalone probe, and is in fact level with PyTorch about
9 % ahead. Three answers, one workload; only the third had both sides
measured the same way with a harness that was not lying. The "fusion
gap" this file described for weeks did not exist.

The implementation landed early, inside cf928e3 with the tutorial
chapters, because `git add -A` swept it up while two specs were open.
The code is unchanged since; this spec adds the warm-up fix, the
re-measurement and the corrections.
