# TODO

Open work items. Each needs a spec in `spec/` before code is written
(see PROCESS.md). Ordered roughly by expected payoff.

- [ ] T-002 — Vectorised tanh, GELU and log kernels (NEON, AVX2, Go)
- [ ] T-003 — Element-wise ops writing into an existing tensor (avoid per-call allocation and zeroing)
- [ ] T-004 — Fused LayerNorm statistics in one pass over each row
- [ ] T-005 — GEMM path reading B in place for small M (batch-1 inference)
- [ ] T-006 — SME SGEMM kernel for Apple M4 (FEAT_SME2, SVL 512 bit, SME_F32F32); first clarify Go runtime interaction with streaming mode and ZA state
- [ ] T-007 — Optional Accelerate BLAS backend behind a cgo build tag
- [ ] T-008 — x86 benchmarks: AVX2/AVX-512 vs OpenBLAS/MKL, first run of the AVX-512 kernel on real hardware
- [ ] T-009 — Conv1D/Conv2D via im2col, multi-head attention module
- [ ] T-010 — AMX SGEMM kernel for Apple M1–M3 via the undocumented AMX instructions (reference: github.com/corsix/amx); check thread-state and Go runtime constraints first
