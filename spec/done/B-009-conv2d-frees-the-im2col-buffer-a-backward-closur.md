---
id: B-009
title: Conv2D frees the im2col buffer a backward closure still reads
status: done
scope:
  - tensor/
manual:
  - docs/manual/internals.md
done: 2026-09-12
created: 2026-09-12
---

## Goal

`Conv2D` builds the im2col matrix, multiplies the filters by it, and then
frees it:

    col := Im2Col(x, kh, kw, stride, pad)
    prod := w.Reshape(o, -1).MatMul(col)
    col.Release()   // "no-op while a graph holds it"

The comment is wrong, and so is the check behind it. `Release` refuses
when *the tensor it is called on* has a backward node or requires a
gradient. `col` has neither whenever the input images do not require a
gradient, which is the case for the first convolution of every network.
But `matmul2D` captured `yd := col.saved()` for its own backward, and
`saved` deliberately does not mark the storage shared, so nothing stops
the release. The im2col buffer is handed back while the product's
backward closure still needs it to compute the filter gradient.

What the free does next decides the symptom, because storage of 64 KiB
and more is mapped outside the Go heap:

| `FIBERAI_MAPPED_LIMIT` | the freed mapping | symptom |
|---|---|---|
| unset (512 MB retained) | reused by the next allocation of that size | wrong gradients, silently |
| `0` | unmapped at once | SIGSEGV in `blas.packRows4` |
| `-1` | Go heap, collector-owned | correct |

Both halves are observed. A two-stage CNN backward faults on every run
under `FIBERAI_MAPPED_LIMIT=0`. Under the default it trains to gradients
that differ from the on-heap run by 0.175 % in `sum|g|` (3563.45 against
3557.20 over 20490 elements, `max|g|` unchanged, forward bit-identical),
and `examples/mnist` loses about 0.15 points of test accuracy to it.
Silent wrong numbers are the worse half.

A single convolution does not show it: the buffer goes back to the pool
and, with nothing else allocating that size class, is still intact when
the backward reads it. It takes the allocation churn of a second stage
for the reuse to land on top of live data. That is why `Conv2D` alone,
`Conv2D`+`MaxPool2D` and a plain `Linear` are all bit-identical between
storage settings, and only the full network diverges.

## Design

**The fix.** `Release` must refuse whenever a recorded node captured the
tensor, not only when the tensor itself carries one. `record` already
counts that: every input of a node gets `consumers++`. So the guard
becomes

    if t.node != nil || t.requiresGrad || t.consumers > 0 { return }

which turns `col.Release()` into the no-op its comment claims, while
leaving the inference path — no autograd, no consumers — free to recycle
as before. That is the whole behavioural fix; `Conv2D` itself does not
change.

**The latent half.** The same class of defect is one step away at every
site that hands a raw slice to a kernel: `mat()` passes `t.data`, and a
slice into off-heap memory keeps no Go object reachable, so the collector
may free the storage while the kernel reads it. `matmul_fused.go`,
`attention.go`, `rope.go`, `inplace.go`, `reduce.go`, `autograd.go` and
`tensor.go` call `runtime.KeepAlive` for that reason; `matmul.go`,
`conv.go`, `nn.go`, `ops_binary.go`, `ops_unary.go`, `packcache.go` and
`view.go` do not. They get the guards too. KeepAlive emits no
instructions — it only extends the compiler's notion of a live range —
so it costs nothing.

**The rule, written down.** `docs/manual/internals.md` gains it where an
operation is added: a tensor whose data went to a kernel stays alive
until the kernel returns, and a tensor a backward closure saved is not
yours to release.

Alternatives rejected. Marking the storage shared in `saved()` would stop
the release, but it also stops Backward from reclaiming intermediates,
which is what `saved` exists to allow. Dropping `col.Release()` alone
would fix this instance and leave the same trap for the next caller.

## Acceptance

- A regression test in `tensor` runs a two-stage convolution and its
  backward under all three storage settings — off-heap retained (the
  default), off-heap unretained (`SetMappedLimit(0)`, every free an
  immediate unmap) and on-heap (`SetMappedLimit(-1)`) — and requires
  bit-identical gradients from all three. Today the unretained setting
  faults and the retained one produces different numbers.
- A test that a tensor captured by a backward node survives `Release`.
- `go test ./...`, `go vet ./...` and `GOARCH=amd64 go vet ./...` clean;
  the AVX2 path exercised under Rosetta.
- No throughput change: `cmd/bench` GEMM, MLP and attention rows stay
  within run-to-run noise of BENCHMARKS.md. The guard is one integer
  comparison on a path that already tests two fields, and KeepAlive
  generates no code.
- `examples/mnist` stops depending on the storage setting, and its
  accuracy over seeds 12, 13 and 14 becomes what the on-heap setting
  gives today — 98.85 / 98.64 / 98.44, mean 98.64 — instead of
  98.76 / 98.47 / 98.28, mean 98.50. PyTorch 2.8 is the baseline on the
  same machine, data, model, optimiser and thread count at
  98.72 / 98.75 / 98.72, mean 98.73, and 51.6 s against our 35.6 s; the
  0.09 points still between us and PyTorch after the fix are not part of
  this bug and are not claimed to close.

## Notes
The guard is `t.consumers > 0` in `Release`, one integer comparison added
to a check that already read two fields. `Conv2D` is untouched: its
`col.Release()` now does what its comment always claimed.

Getting to it took three wrong turns worth recording, because each looked
convincing:

- *Initialisation.* He normal against PyTorch's `kaiming_uniform_(a=√5)`
  is a real difference, 2.45× in σ, and it explained nothing: PyTorch run
  with our initialisation scores no worse than with its own.
- *Nondeterminism.* The spread across seeds looked six times PyTorch's,
  which suggested noise in the convolution. It was a measurement error —
  `-model both` shares one RNG stream, so its CNN starts from different
  weights than `-model cnn`. Three runs at one seed give the same number
  to the last digit. The example now seeds each model separately.
- *The collector.* The first diagnosis was a missing `runtime.KeepAlive`,
  which is a real hazard and is fixed here too, but it was not this bug:
  adding the guards left the fault exactly where it was. The buffer was
  never collected; it was handed back deliberately.

What finally located it was refusing to argue from the training numbers
and comparing one forward and one backward pass instead. The forward was
bit-identical between storage settings and the loss agreed to nine
digits, while `sum|g|` differed by 0.175 %. That put the fault in the
backward and nowhere else, and made the search small enough to read.

`FIBERAI_MAPPED_LIMIT=0` deserves to be part of the routine: it turns
every use-after-free of off-heap storage from a silent wrong number into
an immediate fault. Worth running the test suite under it now and then,
and worth a follow-up spec to do so in CI.

After the fix, `examples/mnist` gives 98.85 / 98.64 / 98.44 over seeds
12, 13 and 14 (mean 98.64), which is what the on-heap setting produced
before it, and the MLP is unchanged at 97.82 % — it never built an im2col
matrix. PyTorch 2.8 on the same machine and settings: 98.72 / 98.75 /
98.72, mean 98.73, in 51.6 s against our 35.6 s.
