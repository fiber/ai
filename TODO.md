# TODO

Open work items. Each needs a spec in `spec/` before code is written
(see PROCESS.md). Ordered by priority: production is Intel/AMD Linux,
macOS is the development platform.

- [ ] T-008 — x86 validation and benchmarks: AVX-512 on hardware, tuning, comparison with OpenBLAS/MKL (spec/T-008-x86-validation-and-benchmarks.md)
- [ ] T-002 — Vectorised tanh, GELU and log kernels (NEON, AVX2, AVX-512, Go)
- [ ] T-003 — Element-wise ops writing into an existing tensor (avoid per-call allocation and zeroing)
- [ ] T-004 — Fused LayerNorm statistics in one pass over each row
- [ ] T-005 — GEMM path reading B in place for small M (batch-1 inference)
- [ ] T-006 — SME SGEMM kernel for Apple M4-class chips (spec/T-006-sme-sgemm-apple-m4.md)
- [ ] T-010 — AMX SGEMM kernel for Apple Silicon via the undocumented AMX instructions (spec/T-010-amx-sgemm-apple-silicon.md)
- [ ] T-007 — Optional Accelerate BLAS backend behind a cgo build tag
- [ ] T-009 — Conv1D/Conv2D via im2col, multi-head attention module
