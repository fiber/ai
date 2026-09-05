---
id: T-008
title: x86 validation and benchmarks: AVX-512 on hardware, tuning, comparison with OpenBLAS/MKL
status: open
scope:
  - internal/kernel/
  - internal/blas/
  - cmd/bench/
  - benchmarks/
manual:
  - docs/manual/performance.md
created: 2026-09-05
---

## Goal

Production runs on Intel/AMD servers and VMs, not on Macs. The amd64
back-ends must therefore be first-class: the AVX-512 GEMM kernel has to
run on real hardware (so far it is only assembled and vetted), the
blocking parameters must fit x86 cache hierarchies, and the benchmark
comparison must be made against NumPy (OpenBLAS) and PyTorch (MKL /
oneDNN) on the same x86 box — the like-for-like comparison that Apple's
matrix units made impossible on the Macs. Ground rule: beat Python
wherever we can, and know precisely where we do not yet.

## Design

1. **Test matrix.** A script under `benchmarks/` that prints CPU model,
   flags (avx2, fma, avx512f/bw/vl, avx512_vnni, amx_*), cache sizes
   (`lscpu`), then runs `go vet ./...`, `go test -count=1 ./...`,
   `FIBERAI_KERNEL=avx2 go test ./...`, `FIBERAI_KERNEL=generic go test
   ./...` and `go run ./cmd/bench`, storing everything under
   `benchmarks/results/<host>/`. Runs on: at least one AVX2-only machine
   (Zen 2/3 or Haswell–Skylake client), one AVX-512 machine (Zen 4, Ice
   Lake or Sapphire Rapids) and one cloud VM (frequency licensing and
   noisy-neighbour effects).
2. **AVX-512 verification.** `TestVerifyAll` and `TestGemmMicroKernel`
   natively with `avx512` selected; fix whatever the self-test finds.
   Decide on AVX-512 element-wise kernels by measurement: keep the AVX2
   ones unless wider versions win on L2-resident sizes.
3. **Micro-kernel and blocking per micro-architecture.** Compare the
   current 12×32 tile with 14×32 (28 accumulators) and 8×48. Choose
   KC/MC/NC from the cache sizes reported by CPUID leaf 4 (L1d, L2 per
   core) at init instead of compile-time constants, with the existing
   constants as fallback; validate against a sweep on each test machine.
   Typical: L1d 32–48 KiB → KC 256–384; L2 1–2 MiB → MC 96–192.
4. **Frequency effects.** Record sustained clocks during the GEMM
   benchmark (`turbostat` / `perf stat` where available) so AVX-512
   downclocking is visible in the report.
5. **Python baseline on the same box** with the existing script; document
   which BLAS NumPy links (`numpy.show_config()`) and which PyTorch
   backend is active; same thread count as Go.
6. **Report.** An x86 section in BENCHMARKS.md with the side-by-side
   table per machine.

## Acceptance

- `go test ./...` passes natively on AVX2-only and AVX-512 hardware with
  the implementation selected by detection (no `FORCE`).
- SGEMM single-core ≥ 80 % of the core's FMA peak on both machines
  (peak = FMA units × lanes × 2 × sustained clock).
- On the AVX-512 machine, fiber/ai SGEMM n=2048 all-cores ≥ 90 % of
  PyTorch's GFLOPS on the same machine; n ≤ 256 within 2×. Every
  non-GEMM workload in the benchmark table at least on par with PyTorch;
  where not, a TODO item with the cause.
- BENCHMARKS.md gains the x86 tables; `docs/manual/performance.md` gains
  x86 guidance (which back-end to expect where, tuning knobs).
- Results directories committed under `benchmarks/results/`.

## Notes
