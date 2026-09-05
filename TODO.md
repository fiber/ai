# TODO

Open work items. Each needs a spec in `spec/` before code is written
(see PROCESS.md). Ordered by priority: production is Intel/AMD Linux,
macOS is the development platform.

- [ ] T-013 — Topology-aware defaults: worker count from physical cores of the current affinity set / NUMA node, not GOMAXPROCS; measure goroutine pinning on hyperthreaded sockets
- [ ] T-005 — Small GEMM shapes: read B in place for M ≤ MR, and cut the per-call cost that leaves n=128 at 73 GFLOPS on the Xeon (PyTorch 236) and the MLP shapes at half the kernel rate
- [ ] T-006 — SME SGEMM kernel for Apple M4-class chips (spec/T-006-sme-sgemm-apple-m4.md)
- [ ] T-009 — Conv1D/Conv2D via im2col, multi-head attention module
- [ ] T-022 — Syslog building blocks: log template mining at 50M lines a day, k-means and nearest-centre search over embeddings (spec/T-022-syslog-building-blocks-log-template-mining-at-50.md)
