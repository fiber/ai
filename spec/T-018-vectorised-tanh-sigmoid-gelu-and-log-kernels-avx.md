---
id: T-018
title: Vectorised tanh, sigmoid, GELU and log kernels (AVX2, NEON, Go)
status: open
scope:
  - internal/kernel/
  - cmd/bench/
manual:
  - docs/manual/performance.md
  - docs/manual/internals.md
created: 2026-09-05
---

## Goal

`tanh`, `sigmoid`, `GELU` and `log` are scalar `math` calls today, one
float64 conversion per element. On the Xeon Gold 6130 `tanh` over 1M
elements takes 2.73 ms against PyTorch's 54 µs (NumPy 421 µs); on the M2
Pro 2.1 ms. That is the worst ratio in the whole benchmark and it sits
on the path of every GELU/tanh network. Folds TODO item T-002.

## Design

- `tanh`: Eigen's float rational approximation (odd degree-13 numerator
  over even degree-6 denominator after clamping to ±7.9988, where the
  approximation is exactly ±1; `x` itself below |x| < 4e-4), pure FMAs plus one division; AVX2 (8 lanes; the
  AVX-512 implementation shares it like the other vector ops), NEON (4
  lanes) and a Go implementation with the same formula so the start-up
  self-verification can compare bit-for-bit-ish (2e-6 relative).
- `log`: Cephes `logf` (mantissa/exponent split by integer ops,
  degree-9 polynomial, exponent times split ln 2); x ≤ 0 gives NaN or
  -Inf as `math.Log` does, denormals are flushed to the smallest normal.
- `sigmoid(x) = ½ tanh(x/2) + ½` and GELU's tanh form are composed from
  the vector `Tanh` plus the existing vector kernels (`Scale`, `Mul`,
  `Axpy`, `AddScalar`) over the 4K-element chunks the tensor layer
  already uses, so every pass stays in L1. `GELUGrad` likewise.
- `impl` gets `tanh` and `log` entries, verified at init like `exp`;
  the exported `Tanh`/`Log` become re-pointable vars like `Exp`.
- `cmd/bench` adds `sigmoid` and `gelu` rows.

Rejected: tanh via `1 - 2/(1+exp(2x))` (cancellation near zero), and
computing in float64 lanes (half the throughput for no visible gain).

## Acceptance

- Accuracy tests against `math.Tanh` / `math.Log` over dense sweeps of
  the whole range on every implementation: tanh within 2e-6 relative (or
  1e-7 absolute), log within 2e-6 relative; special values (0, ±Inf,
  negative) behave as documented.
- M2 Pro, `cmd/bench` `tanh(x)` 1M ≤ 150 µs (from 2.1 ms; PyTorch on
  the M2 Pro is 723 µs, NumPy 986 µs).
- Xeon Gold 6130, one socket: `tanh(x)` 1M ≤ 120 µs without release
  (from 2.73 ms; PyTorch 54 µs, NumPy 421 µs), and with the result
  released at or below PyTorch's 54 µs; `exp`, `sigmoid`, `gelu` rows
  in the same range.
- Manual: performance.md's "what is fast" table lists the vectorised
  transcendentals; internals.md names the algorithms.

## Notes

Implemented for AVX2 (shared by the AVX-512 table), NEON and Go. Two
details found on the way: the classic clamp at 7.9053 leaves tanh(∞) at
0.99999976, and the 7.9988 clamp at which the rational is exactly 1
holds only with fused multiply-add (Rosetta/amd64 without FMA gave
0.9999998), so the kernels saturate explicitly to ±1 from |x| ≥ 9, where
float32 tanh is 1 anyway; that also fixed a visible error in `GELUGrad`
at x = -8. Go's assembler lacks `VSCVTF`, `FCMEQ/FCMGT/FCMGE` vector
forms, `FABS` and `BSL` as mnemonics; they are `WORD` encodings.

M2 Pro (`cmd/bench`, before → after): `tanh` 64K 161 → 16 µs, 1M 2.1 ms
→ 101–135 µs (PyTorch 723 µs, NumPy 986 µs), 16M 27 → 1.4 ms (PyTorch
11.6 ms); `sigmoid` 1M 175 µs; `gelu` 1M 289 µs. Kernel alone
(`BenchmarkTanh`, one core, L1-resident): 0.42 ns/element NEON against
6.6 ns for `math.Tanh`. The NEON `exp` kernel does 1.4 ns/element on the
same benchmark, three times tanh's cost; unrolling two vectors per
iteration is a follow-up worth taking before it shows up on the 16-core
Xeon, where `exp` over 1M would otherwise be compute-bound. Xeon numbers
pending.
Done for NEON `exp`: two vectors per iteration took the kernel from 1.4
to 0.4 ns/element (`BenchmarkExp` 4096: 5.7 → 1.6 µs; 1M one core 792
→ 382 µs). The M2's out-of-order engine evidently does not overlap
consecutive iterations of the single-vector loop. The AVX2 kernels are
left single-vector until the Xeon `BenchmarkExp`/`BenchmarkTanh` figures
say whether they need the same.

Xeon (one socket, GOMAXPROCS 16) after the first version: `tanh` 1M
2.73 ms → 484 µs, 16M 47.9 → 10.6 ms, i.e. the same memory-bound figure
as `exp` and `relu` (PyTorch 54 µs at 1M is cache-resident, see
T-015/T-017: with `Release()` the 1M rows sit at ~30 µs). But the AVX2
kernels themselves ran at 1.7 ns/element (`BenchmarkExp` 4096: 7.0 µs,
`BenchmarkTanh` 7.7 µs), the same non-overlap symptom as NEON, and
`sigmoid`/`gelu` at 16M (16.0/23.4 ms against 10.6 for `tanh`) showed
the composed passes running over 1 MiB chunks instead of L1. Both fixed:
the AVX2 `exp`, `tanh` and `log` now process two vectors per iteration
with eight-fold replicated constants as memory operands (no register
pressure), and the composed kernels block internally at 4096 elements.
Tensor-level GEMM with the result released: n=1024 658 → 1 117 GFLOPS,
n=2048 1 165 (kernel alone 1 242/1 285).

Second Xeon round: `sigmoid`/`gelu` 16M now 10.6 ms like `tanh`, but the
AVX2 unroll only took `exp` 4096 from 7.0 to 5.2 µs (1.27 ns/element,
about 20–30 cycles per 8-wide vector where the two FMA ports allow ~7).
Hypothesis: 4K aliasing. `make` hands out 16 KiB buffers at the same
page offset and the mmap allocator page-aligns every result, so the
kernel's store to z[i] and its load of x[i+8] agree in the low 12 bits
and the load waits for the store on Intel cores. `BenchmarkExpAliasing`
compares same-offset against shifted output buffers. Caveat for the M2
figures in these notes: single-goroutine benchmarks there vary up to
4× between runs depending on whether the goroutine lands on a P- or an
E-core (GOMAXPROCS counts both), so only the Xeon numbers are reliable
for kernel-level conclusions.
