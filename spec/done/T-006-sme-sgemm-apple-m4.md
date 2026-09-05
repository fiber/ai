---
id: T-006
title: SME SGEMM kernel for Apple M4-class chips
status: done
scope:
  - internal/kernel/
  - internal/blas/
manual:
  - docs/manual/performance.md
  - docs/manual/internals.md
done: 2026-09-05
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

Phase 0 prepared (8 September, night): `internal/kernel/smeprobe` is a
stand-alone program for the M4 (`go run ./internal/kernel/smeprobe`)
that checks `hw.optional.arm.FEAT_SME`, runs one `fmopa` outer product
and verifies the ZA layout, measures the 32×32 tile body (four `fmopa`
per k-step over the four f32 tiles) on 1–10 goroutines, and stresses
outer products in 32 goroutines under forced collections. Instruction
words come from the LLVM MC tests, not from guesswork: `smstart`
D503477F, `smstop` D503467F, `zero {za}` C00800FF, `ptrue p0.s`
2598E3E0, `ld1w {Zt.s}, p0/z, [Xn]` A540A000 | Rn<<5 | Zt, `st1w`
E540E000 | Rn<<5 | Zt, `fmopa ZAda.s, p0/m, p0/m, Zn.s, Zm.s` 80800000 |
Zm<<16 | Zn<<5 | ZAda, `mova Zd.s, p0/m, ZAn h.s[w12, imm]` C0820000 |
imm<<5 | ZAn<<7 | Zd. The M2 Pro cannot run it (no SME); results from
the M4 decide phase 1. Open question the probe answers first: whether
macOS leaves streaming mode across signal delivery (Go's handler uses
NEON), which the stress test would show as SIGILL or mismatches.

Phase 0 results (M4, 8 September). Encodings and ZA layout verified:
`fmopa ZAda, Zn, Zm` accumulates Zn[row]·Zm[col], rows are Zn lanes.
Throughput of the 32×32 tile body (four `fmopa` per k-step): 1 031–1 062
GFLOPS on one thread, 1 386–1 399 on six; software pipelining of the
loads changes nothing (1 062), unlike AMX, whose in-order queue gained 3×
from it. The same tile through the AMX instructions on the same M4:
1 400 per thread, 1 738 on all cores. State safety: Go's asynchronous
preemption signal landing in streaming mode costs the Z registers (the
architecture zeroes them on the streaming-mode change into and out of
the handler; the macOS signal frame carries only the NEON halves):
11–14 wrong results in 640 000 outer products under a GC every 5 ms,
none with `GODEBUG=asyncpreemptoff=1`, and none with all signals
blocked on the thread around each kernel call (`sigprocmask`, one
syscall per task). AMX under the same stress: 960 000 tiles, no error,
its state is thread state the kernel saves.

Decision: the kernel is not built now. AMX is the default on M1–M4,
about 20 % faster on the M4 for this tile, and needs no signal masking.
SME becomes worth a kernel when an Apple chip drops AMX or when SME's
documented status matters for support; `internal/kernel/smeprobe`,
the instruction words above and the masking recipe are the starting
point then, and the driver hooks (`GemmBegin`/`GemmEnd` with a locked
thread) already fit a masked SME task. Acceptance criteria for the
kernel therefore stay unmet by choice; the phase-0 report is complete.
