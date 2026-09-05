# TODO

Open work items. Each needs a spec in `spec/` before code is written
(see PROCESS.md). Ordered by priority: production is Intel/AMD Linux,
macOS is the development platform.

- [ ] T-008 — x86 validation and benchmarks: AVX-512 on hardware, tuning, comparison with OpenBLAS/MKL (spec/T-008-x86-validation-and-benchmarks.md) — kernel validated on Skylake-SP; remaining: blocking sweep, multi-core scaling, x86 doc
- [ ] T-013 — Topology-aware defaults: worker count from physical cores of the current affinity set / NUMA node, not GOMAXPROCS; measure goroutine pinning on hyperthreaded sockets
- [ ] T-005 — GEMM path reading B in place for small M (batch-1 inference)
- [ ] T-006 — SME SGEMM kernel for Apple M4-class chips (spec/T-006-sme-sgemm-apple-m4.md)
- [ ] T-007 — Optional Accelerate BLAS backend behind a cgo build tag
- [ ] T-009 — Conv1D/Conv2D via im2col, multi-head attention module
- [ ] T-016 — AVX-512 micro-kernel under all-core load: prefetch, tile shape, C layout (spec/T-016-avx-512-micro-kernel-under-all-core-load-prefetc.md)
- [ ] T-021 — Tutorial: AI for Go developers with fiber/ai, eight chapters with runnable programs (spec/T-021-tutorial-ai-for-go-developers-with-fiber-ai-eigh.md)
