---
id: T-043
title: AVX-512 exp, exp-sum and tanh: 16 lanes with every constant in a register
status: done
scope:
  - internal/kernel/
manual:
  - docs/manual/performance.md
done: 2026-09-09
created: 2026-09-09
---

## Goal

The AVX-512 back-end shares the AVX2 element-wise kernels: eight lanes,
and every constant a memory operand because sixteen ymm registers do not
hold them (the exp needs twelve, the tanh seventeen). Twelve loads per
eight elements on two load ports is six cycles before any arithmetic,
which is why the AVX2 exp and tanh run at about 1.2 ns per element on
the Xeon Gold 6130 against 0.4 on the M2's NEON. After T-042 the
exponential is the second item in the pinned attention profile, tanh
(the GELU of every feed-forward and the tanh rows) is 9× behind MKL's
vector library, and softmax rows pay the same exp. Give the AVX-512
back-end its own exp, exp-sum and tanh: sixteen lanes, every constant
broadcast once into a zmm register, two vectors per iteration.

## Design

- **`expAVX512`, `expSumAVX512`, `tanhAVX512`** in a new
  `kernel_avx512_math_amd64.s`: the algorithms of the AVX2 routines
  (same constant tables, broadcast with `VBROADCASTSS` into Z16–Z31 at
  entry), `VRNDSCALEPS` for the rounding, mask registers for the
  compares: the flush below the exp clamp is a zeroing masked move on
  the keep mask, the tanh tiny/±1 cases are merge-masked moves. n % 16
  == 0; the wrappers (`wrapExp`, `wrapExpSum`, `wrapUnary` with width
  16) run the generic tail.
- The `avx512` and `avx512x12` implementations take the three routines;
  `avx2` is unchanged. The start-up verification compares them with the
  generic code like every other kernel and falls back with a warning if
  they disagree, so a wrong routine costs speed, not results.
- Not in scope: log and sqrt (rare), sigmoid and GELU (composed from
  exp and tanh, they inherit the gain), an AVX-512 `DotNorms`.
- The M2 cannot run AVX-512 (Rosetta lacks it), so the routines are
  assembled here and verified on the Xeon: `go test ./internal/kernel`
  there runs every implementation against float64 references.

## Acceptance

- `go test ./internal/kernel` passes on the Xeon (the exp, exp-sum and
  tanh accuracy tests over every implementation, the flush test) and
  under `GOARCH=amd64` here (assembles; AVX2 path runs); start-up
  verification reports no warning on the Xeon; `go test ./...` on the
  Xeon passes with the default back-end and with
  `FIBERAI_KERNEL=avx512x12`.
- Baselines (Xeon Gold 6130, one socket, 2026-09-09): single-core exp
  and tanh about 1.2 ns per element (`go test ./internal/kernel -bench
  'Exp|Tanh'`); bench rows tanh 1M 484 µs (MKL 54 µs), exp 16M 10.7 ms
  (PyTorch 13.8 ms, memory-bound: expected unchanged), attention
  [8×8×512×64] 618 GFLOPS (PyTorch 1 141), EmbeddingGemma 58
  sentences/s (37). Targets: exp and tanh ≤ 0.5 ns per element on one
  core, tanh 1M ≤ 200 µs, attention ≥ 660, EmbeddingGemma ≥ 60; M2
  unchanged.
- Manual: performance.md's back-end section says the AVX-512 back-end
  has its own transcendental kernels and what they were worth.

## Notes

- Written and assembled on the M2 (`GOARCH=amd64 go vet`), which cannot
  execute AVX-512: correctness rests on the Xeon's `go test
  ./internal/kernel` (every implementation against float64 references)
  and the start-up verification. If the bench header ever says `backend
  avx2` on the Xeon, a routine failed verification and was disabled;
  `tensor.BackendWarnings()` names it.
- The flush below the exp clamp is a merge-masked move into a zeroed
  register on the keep mask (x ≥ lo); tanh's tiny and ±1 cases are
  merge-masked moves on |x| < tiny and |x| ≥ 9. |x| and the sign bit
  come from broadcast memory operands (`.BCST`), the only loads in the
  tanh loop besides x.
- Xeon measurement pending (targets in Acceptance).
