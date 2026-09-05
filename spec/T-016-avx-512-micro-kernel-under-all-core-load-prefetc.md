---
id: T-016
title: AVX-512 micro-kernel under all-core load: prefetch, tile shape, C layout
status: open
scope:
  - internal/kernel/
  - internal/blas/
manual:
  - docs/manual/performance.md
created: 2026-09-05
---

## Goal

After the driver work of T-014 the Skylake-SP socket (16 cores, 1.95 GHz
all-core AVX-512, ~2 000 GFLOPS FMA peak) runs our SGEMM at 1 147 GFLOPS
for n=1024 (57 % of peak) while a single core reaches 84 % of its own
peak and MKL reaches 72 % on all cores (1 434). perf shows 2.3 IPC and
few LLC misses with every core busy, so the remaining loss is inside
the micro-kernel when 16 copies run at once: memory-level effects that
one core hides and sixteen do not.

## Design

Experiments, each measured on the Xeon with the blas benchmark at
n=1024 and 2048, 16 workers; keep what wins:

1. **Software prefetch in the AVX-512 kernel**: `PREFETCHT0` of the next
   A panel lines and B lines a few k-steps ahead, and of the C tile rows
   before the store phase (C rows are 4 KiB apart at n=1024, twelve of
   them map to one L1 set). MKL and OpenBLAS kernels do this.
2. **Tile shape**: compare 12×32 with 14×32 (28 accumulators, higher
   arithmetic intensity) and 8×48; measure single-core and all-core.
3. **KC vs L1**: with KC=512 the B panel is 64 KiB, twice L1; measure
   KC=256/384 again with prefetch in place, as prefetch changes the
   trade-off.
4. **Packing in assembly** for A (currently ~7 % of CPU in `packA` with
   strided stores) if it still shows after 1–3.

Non-goals: AMX/SME (T-006/T-010), changes to the driver (T-014 done).

## Acceptance

- Xeon Gold 6130, one socket, 16 workers: n=1024 ≥ 1 300 GFLOPS (from
  1 147; PyTorch/MKL 1 434, NumPy/OpenBLAS 1 303), n=2048 ≥ 1 250 (from
  1 139; PyTorch 780, NumPy 881). Single-core within noise of 150.
- No regression of the AVX2 kernel under Rosetta tests or of NEON on the
  Macs; `go test ./...` on all three.
- BENCHMARKS.md and `docs/manual/performance.md` updated with the result.

## Notes

Experiment 1 (software prefetch of A/B ahead and the C tile at entry):
Xeon, 1 worker 143/151 GFLOPS (n=1024/2048), 16 workers 1 119/1 177 —
within the ±3 % noise of the runs before (1 147/1 139). Kept, no effect.
Next: tile shape (14×32) and KC with the tile change.

Experiment 2 (14×32 tile, 28 accumulators): Xeon 1 worker 156 → 164 /
155 → 168 GFLOPS (n=1024/2048), 16 workers 1 132 → 1 242 / 1 194 → 1 285.
Adopted as the AVX-512 default (`avx512`); the 12×32 kernel stays as
`avx512x12`. n=1024 now at 87 % of MKL (1 434), n=2048 at 165 % of MKL
(780) and 146 % of OpenBLAS (881).
