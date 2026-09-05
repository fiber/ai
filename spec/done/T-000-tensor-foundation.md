---
id: T-000
title: Tensor foundation: SIMD kernels, blocked SGEMM, tensor/autograd, nn, optim, benchmarks
status: done
scope:
  - internal/
  - tensor/
  - nn/
  - optim/
  - examples/
  - cmd/bench/
  - benchmarks/
  - go.mod
manual:
  - docs/manual/getting-started.md
  - docs/manual/tensors.md
  - docs/manual/autograd.md
  - docs/manual/nn-and-optim.md
  - docs/manual/performance.md
  - docs/manual/internals.md
created: 2026-09-05
done: 2026-09-05
---

## Goal

A float32 tensor compute layer for fiber/ai that is fast enough to compete
with Python frameworks on CPU: SIMD kernels in Go assembly (NEON, AVX2,
AVX-512), a cache-blocked GEMM, reverse-mode autograd and goroutine
parallelism, plus the nn/optim layer needed to train models.

(Written retrospectively: this work predates the process. It records what
was built so the lists start complete.)

## Design

- `internal/parallel`: dynamic work distribution over goroutines.
- `internal/kernel`: element-wise, reduction, exp and GEMM micro-kernels;
  NEON floating-point vector ops as verified raw encodings; CPUID
  dispatch; self-verification at init; `FIBERAI_KERNEL` override.
- `internal/blas`: Goto/BLIS packed SGEMM, 8×12 / 6×16 / 12×32 tiles,
  2-D task grid, gemv paths, per-architecture blocking.
- `tensor`: strided views, broadcasting, reductions, batched MatMul,
  autograd with per-op backward closures, fused Softmax / LogSoftmax /
  CrossEntropy / MSELoss / LayerNorm.
- `nn`, `optim`: modules and SGD/Adam ported from the `vektor` prototype.
- `cmd/bench`, `benchmarks/python/bench.py`, BENCHMARKS.md.

## Acceptance

- `go test ./...` on arm64 (NEON) and generic; AVX2 under Rosetta 2. Met.
- SGEMM ≥ 80 % of single-core FMA peak on Apple M2 Pro. Met (87 %).
- Gradient checks against finite differences for every differentiable op.
  Met.
- Benchmarks documented with a NumPy/PyTorch comparison. Met
  (BENCHMARKS.md).

## Notes

AVX-512 GEMM is assembled and vetted but has not run on hardware (T-008).
Large GEMM on Apple Silicon trails Accelerate because of the AMX/SME
matrix unit (T-006, T-007).
