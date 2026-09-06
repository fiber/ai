# TODO

Open work items. Each needs a spec in `spec/` before code is written
(see PROCESS.md). Ordered by priority: production is Intel/AMD Linux,
macOS is the development platform.

Nothing open. Candidates for the next round, to be specified when picked up: implicit-GEMM convolution (no column matrix), safetensors loader and tokenizer for running EmbeddingGemma natively, Python bench rows for attention and convolution, Xeon measurement of the 1M parallel threshold and the unpinned worker default.
- [ ] T-031 — Native EmbeddingGemma: safetensors loader, SentencePiece tokenizer, Gemma 3 encoder, sentence embeddings (spec/T-031-native-embeddinggemma-safetensors-loader-sentenc.md)
