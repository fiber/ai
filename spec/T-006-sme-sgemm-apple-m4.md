---
id: T-006
title: SME SGEMM kernel for Apple M4-class chips
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

On Apple M4 and later, reach the matrix-unit throughput that Accelerate
gets (1.9 TFLOPS SGEMM on the base M4; PyTorch/NumPy baseline in
benchmarks/results/python-m4.md) from pure Go assembly, using the
documented ARM Scalable Matrix Extension instead of NEON. This closes the
3–10× GEMM gap to Python on Apple development machines; production is
x86 (T-008), so this ranks second.

Detected features on the target (M4, `sysctl hw.optional.arm`):
FEAT_SME, FEAT_SME2, SME_F32F32, SME_F64F64, SME_I8I32, SME_B16F32,
`sme_max_svl_b = 64` (streaming vector length 512 bit: 16 float32 lanes;
ZA is 64×64 bytes = four 16×16 float32 tiles).

## Design

**Phase 0 — feasibility (answered before any kernel work).**

1. Streaming mode and Go. An SME kernel runs between `SMSTART SM,ZA` and
   `SMSTOP` inside one `NOSPLIT` assembly function that neither calls Go
   nor allocates. Verify empirically that Go's asynchronous preemption
   (SIGURG) and GC signal handling do not corrupt streaming state: a
   stress test running the kernel in many goroutines under heavy
   allocation for minutes, checking results. macOS must save/restore ZA
   on context switch and signal delivery (Accelerate relies on it); a
   probe checks this directly by parking a known ZA pattern across a
   forced context switch.
2. Which cores execute SME. Measure single-goroutine throughput pinned by
   `runtime.LockOSThread`; if only the performance cluster has the unit,
   limit parallelism accordingly.
3. Encodings. The Go assembler has no SME mnemonics; every instruction is
   a `WORD`. Needed: `SMSTART`/`SMSTOP` (MSR SVCR), `ZERO {ZA}`, `LD1W`
   (streaming SVE loads into Z registers), `FMOPA ZAn.S, Pg/M, Pg/M,
   Zn.S, Zm.S` (outer-product accumulate, f32), `MOVA`/`ST1W` of ZA tile
   rows, `PTRUE`. Verify each encoding with `go tool objdump` and a
   single-instruction test.

**Phase 1 — micro-kernel.** MR = NR = 32 (two 16-lane vectors each); the
four ZA tiles hold the 32×32 C block: per k-step two loads of A, two of
B, four `FMOPA`. Packing as in `internal/blas` with the new tile size;
the kernel signature stays `Gemm(k, a, b, c, ldc)`, so the blocked driver
is unchanged. Edge tiles use the existing scratch-tile path.

**Phase 2 — integration.** New `impl` "sme" in `kernel_arm64.go`,
selected ahead of "neon" when `hw.optional.arm.FEAT_SME` (via
`syscall.Sysctl`) and `SME_F32F32` are set; the init self-verification
applies. Element-wise kernels stay NEON. Blocking parameters for the M4
via a sweep.

**Phase 3 — measure** against the M4 Python baseline; update BENCHMARKS.md.

Rejected: cgo/Accelerate (T-007 keeps that as an optional backend);
AMX on the M4 (T-010 targets the AMX encodings, which corsix/amx reports
as present up to M4 Max, but SME is the documented path there).

## Acceptance

- Phase 0 report in Notes: preemption/signal safety shown by the stress
  test, core availability measured, encodings verified.
- `go test ./...` passes on the M4 with `sme` selected, and still with
  `FIBERAI_KERNEL=neon`.
- SGEMM n=2048 ≥ 1.5 TFLOPS on the base M4 (PyTorch: 1.88 TFLOPS); n=512
  ≥ 1.2 TFLOPS. MLP training step ≥ PyTorch's 117 K samples/s on the M4.
- No change in behaviour on machines without SME.
- `docs/manual/performance.md` and `internals.md` describe the back-end.

## Notes
