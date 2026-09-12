---
id: T-056
title: Benchmark EmbeddingGemma against llama.cpp at matched precision
status: open
scope:
  - benchmarks/
  - BENCHMARKS.md
manual:
  - docs/manual/performance.md
created: 2026-09-12
---

## Goal

BENCHMARKS.md compares against NumPy and PyTorch, which answers "is the
Go code competent at linear algebra". It does not answer the question a
Go developer actually asks, which is whether they should call out to a
C++ runtime or stay in Go. The most visible AI project in the Go
ecosystem is Ollama, and Ollama is Go orchestration around llama.cpp:
the maths is handed to C++ through cgo. What that choice costs, measured,
is the most useful number this project could publish.

There is exactly one workload where the comparison is honest.
`models/gemma` runs EmbeddingGemma end to end — tokenizer to unit vector,
24 layers — and llama.cpp runs the same model through
`llama-embedding`. Everything else would be a category error: llama.cpp
is a serving runtime, not a tensor library, so there is no GEMM row to
compare, and we have no quantised inference and no decoder model
shipping, so chasing it on quantised LLM decoding would compare things we
do not do.

Current figure to beat or lose to honestly: 105 sentences/s for 32 copies
of a 65-token sentence in one batch, against PyTorch with
`sentence-transformers` at 88 (8 September) on the same machine.

## Design

**Precision must match or the number measures quantisation.** llama.cpp's
reputation rests on Q4 and Q5; we are float32. The comparable run is a
GGUF at F32. F16 may be added as a separate, labelled row because it is
what people actually deploy, but it is not the headline comparison and
must never be presented as one.

**Parity first, speed second.** `models/gemma` holds cosine 1.000000
against `sentence-transformers` on 64 short and 3 long inputs (up to
1753 tokens); see docs/manual/models.md. The llama.cpp side gets the same
treatment before any timing is recorded, because its pooling and
normalisation defaults for embedding models differ from
sentence-transformers' and a mismatch would show up as a speed difference
that is really a different computation. If parity cannot be reached, the
row is not published and the spec says why instead.

**Same protocol as every other row.** Six performance cores on both sides
(`-t 6` for llama.cpp, which is our default and PyTorch's here), 32
sentences of about 65 tokens as one batch, one warm-up, then repeated for
at least 0.7 s and the mean reported. Model load time excluded on both
sides and reported separately, since it is a real cost that differs by an
order of magnitude and belongs in the text rather than hidden in or out
of the number.

**Artefacts.** The GGUF conversion command, the llama.cpp commit, and the
build flags go in `benchmarks/results/` next to the run, the same way the
Python side records its versions. A number nobody can reproduce is worth
less than no number.

**Reproduce path.** A `--only embed` equivalent on the llama.cpp side, so
the row reruns with the rest rather than being a one-off someone did by
hand and cannot repeat in three months.

## Acceptance

- Cosine similarity between our embeddings and llama.cpp's is at least
  0.9999 on the same 64 short and 3 long inputs used for the
  sentence-transformers parity check, recorded before any timing.
- One row in BENCHMARKS.md: sentences/s for fiber/ai, llama.cpp F32 and,
  if measured, llama.cpp F16, all on six performance cores of the M2 Pro
  in one session, with load time reported separately.
- The llama.cpp commit, GGUF conversion command and build flags are
  committed under `benchmarks/results/`.
- The existing baselines stay in the same table so the three can be read
  together: fiber/ai 105 sentences/s, PyTorch with
  `sentence-transformers` 88.
- The text states plainly where we land, including if we lose. A factor
  of two behind a mature C++ runtime with no cgo is a result worth
  publishing; pretending it did not happen is not.

## Notes

Scheduled for the week of 2026-09-14; specced on 2026-09-12 so the
protocol is fixed before the numbers exist, which is the only way to stop
the protocol drifting toward whichever configuration flatters us.
