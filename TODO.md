# TODO

Open work items. Each needs a spec in `spec/` before code is written
(see PROCESS.md). Ordered by priority: production is Intel/AMD Linux,
macOS is the development platform.

Nothing open. Candidates for the next round, to be specified when picked up: implicit-GEMM convolution (no column matrix), Python bench row for EmbeddingGemma (sentence-transformers, 32 × 64 tokens) and its Xeon measurement, bf16 weight storage expanded during GEMM packing (halves the 1.2 GB), Python bench rows for attention and convolution, Xeon measurement of the 1M parallel threshold and the unpinned worker default.
