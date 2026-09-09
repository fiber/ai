# TODO

Open work items. Each needs a spec in `spec/` before code is written
(see PROCESS.md). Ordered by priority: production is Intel/AMD Linux,
macOS is the development platform.

Nothing open. Candidates for the next round, to be specified when picked up: small-result recycling on hypervisors (KVM guest: an unreleased 256 KB result costs 788 µs of page faults against 15 µs released; a lower forced-collection budget for small size classes, or the heap path for them), an AMX tile variant for narrow products (n ≤ 16: half the 32-wide tile idles, [256×24]·[24×16] 36 against Accelerate's 92 GFLOPS), NUMA-aware worker placement on Linux (pin worker threads to one node's cores with sched_setaffinity when the affinity mask spans nodes; the unpinned two-socket default costs a third against numactl on the Xeon: 46K vs 67K training samples/s), implicit-GEMM convolution (no column matrix; the Xeon runs conv at 73 GFLOPS against 611 for PyTorch/oneDNN), Python bench row for EmbeddingGemma (sentence-transformers, 32 × 64 tokens) and its Xeon measurement, bf16 weight storage expanded during GEMM packing (halves the 1.2 GB), Python bench rows for attention and convolution, Python venv on the Xeon for the bench.py attention/conv/embed rows. (Measured and closed: the 1M parallel threshold is right when pinned, n=128 gains 99 vs 65 GFLOPS; the unpinned default is the NUMA item above.)
