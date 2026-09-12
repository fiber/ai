---
id: T-059
title: Re-measure the AVX2 cloud VM against HEAD
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

Production for this work is x86: a six-vCPU AVX2 KVM guest and a Xeon,
not the Apple laptop everything is developed and benchmarked on. The
guest's figures in BENCHMARKS.md were from 9 September and predate
off-heap mapped storage, T-047, T-050 and B-009 — so the platform that
matters was the one measured least, and the allocation-cost section's
claim that pooled buffers would be "the single most valuable change for
x86 deployments" had never been checked on x86.

## Design

Build on the guest rather than cross-compiling, so the toolchain is the
one a user there would have: Go 1.27.1 installed under the test account,
the repository cloned from GitHub, `go build ./...`. NumPy and PyTorch in
a virtualenv in the same account, at the versions the September run used
(2.5.2 and 2.14.0+cpu) so both sides stay comparable with it.

Three runs of everything, medians reported, ranges quoted where they are
wide. The guest is shared and its spread is far larger than the Mac's;
single runs there are not evidence. Attention and convolution measured
with `-only`, as on the M2 Pro, for the allocator-state reason recorded
in T-054.

Raw output for all three sides under
`benchmarks/results/kvm-avx2-2026-09-12/`.

## Acceptance

- Every figure in the cloud VM section is a median of three runs from
  12 September, with the run-to-run spread stated so a reader knows what
  a difference has to exceed to mean anything.
- The section says what changed since 9 September and what got worse,
  not only what improved.
- Both sides measured in one session: NumPy 2.5.2 and PyTorch 2.14.0+cpu
  at six threads. Baseline to compare against, from that session:
  PyTorch SGEMM 2048² 188, 1024² 173, 512² 185, 128² 104 GFLOPS; MLP
  training step 32 K samples/s; tiny autoencoder training step 31 K.
- `go build ./...`, `go vet ./...` and `go test ./...` pass on
  linux/amd64 with the AVX2 back-end and Go 1.27.1.

## Notes

The result worth carrying elsewhere is not in the GEMM table. On the
production-like guest, a 24→16→3→16→24 autoencoder at batch 64 — the
shape of the anomaly-detection models this is actually for — runs at
927 K samples/s forward against PyTorch's 231 K, and 225 K on a full
training step against 31 K. Four times and seven times. At that size a
step is per-operation overhead and that is precisely what a Python
framework spends, while the arithmetic is too small for MKL's advantage
to appear.

The mirror image is that between 96² and 192² PyTorch is two to
two-and-a-half times ahead on this machine, and T-050's small-product
path — worth 376 to 553 GFLOPS at 128² on the M2 Pro — does nothing here
and may cost a little. That spec recorded "Xeon and VM: pending"; this is
the answer, and the AVX2 small path is now the clearest piece of
unfinished performance work.

Off-heap storage did what it was predicted to do, and more on this
machine than on the Mac: transpose-and-copy over a 64 MB result went
from 74 ms to 21.6 ms, because page faults cost 2–3× more under a
hypervisor and the free list removes most of them.

Method note for next time: three runs was the minimum that made this
readable. SGEMM 2048² spanned 215 to 262 GFLOPS and attention 124 to 185
across three runs of one binary. Anything reported from this guest as a
single number is noise dressed as a measurement.
