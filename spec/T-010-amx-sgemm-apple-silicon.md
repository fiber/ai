---
id: T-010
title: AMX SGEMM kernel for Apple Silicon via the undocumented AMX instructions
status: open
scope:
  - internal/kernel/
  - internal/blas/
manual:
  - docs/manual/performance.md
  - docs/manual/internals.md
created: 2026-09-05
---

## Goal

On Apple M1–M3 the only matrix unit is Apple's undocumented AMX
coprocessor, which Accelerate uses to reach 2.2–2.7 TFLOPS SGEMM (M2 Pro
baseline in benchmarks/results/python.md) against ~600 GFLOPS from ten
NEON cores. An AMX kernel would bring GEMM to parity with Python on those
machines, and the end-to-end training step ahead of it, because the rest
of the stack already is. This is an opt-in back-end for development
machines; it ranks after x86 (T-008) and SME (T-006).

## Design

Reference: github.com/corsix/amx — instruction encodings, register file
and semantics, reverse-engineered on M1 Max, M2, M3 and M4 Max (the AMX
instructions still exist on the M4 alongside SME; versions differ per
generation: M2 adds bf16, M3 extra modes, M4 changed offset handling).

- **Encoding.** AMX instructions are system-instruction encodings emitted
  as `WORD` with the operand descriptor in a general register; needed:
  `set`, `clr`, `ldx`, `ldy`, `ldz`, `stz`, `fma32` (f32 outer product
  into Z). X and Y hold 32 float32 each per row, Z is the 64×64-byte
  accumulator: one `fma32` is a 16×16 outer product, Z row selection
  lets several tiles accumulate independently.
- **Thread state.** AMX state belongs to the OS thread and is enabled by
  `set`; the kernel runs `set … loads/fma32 … stz … clr` in one `NOSPLIT`
  assembly function. The same feasibility probes as in T-006 (preemption
  signal during the kernel, context switch with live state) run first.
- **Micro-kernel.** MR = NR = 32 using four Z tile groups, k-step: load 32
  A values into X, 32 B values into Y, four `fma32`. Packed operands from
  the existing driver; `Gemm(k, a, b, c, ldc)` signature unchanged.
- **Detection and safety.** Enabled only when `machdep.cpu.brand_string`
  names an Apple M-series chip and — in the first release — only with
  `FIBERAI_AMX=1`, because an illegal instruction cannot be caught by the
  self-test. The per-generation differences documented by corsix are
  handled by using only the M1-level instruction subset.
- **Measure** against the M2 Pro Python baseline.

Risks: Apple may change or remove the instructions; throughput is shared
per cluster, so more goroutines than clusters do not help.

## Acceptance

- Feasibility probes documented in Notes (state survives preemption and
  context switches; behaviour on unsupported chips confirmed).
- `go test ./...` passes on an M2 Pro with `FIBERAI_AMX=1` and without.
- SGEMM n=2048 ≥ 1.8 TFLOPS on the M2 Pro (PyTorch: 2.24 TFLOPS); MLP
  training step ≥ PyTorch's 116 K samples/s.
- Opt-in documented in `docs/manual/performance.md`; the design in
  `internals.md`.

## Notes
