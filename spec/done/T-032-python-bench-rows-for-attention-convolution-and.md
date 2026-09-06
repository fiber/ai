---
id: T-032
title: Python bench rows for attention, convolution and EmbeddingGemma; long-input parity test
status: done
scope:
  - benchmarks/python/
  - models/gemma/
manual: none
done: 2026-09-06
created: 2026-09-06
---

## Goal

`cmd/bench` has attention, convolution and EmbeddingGemma sections that
the Python counterpart lacks, so the Xeon comparison for those rows has
to be improvised by hand. Give `benchmarks/python/bench.py` the same
three sections with the same shapes, so one run per side produces the
full table.

## Design

- `bench_attention`: PyTorch `scaled_dot_product_attention` on
  [8×8×512×64] float32 with and without a causal mask, under `no_grad`,
  reported as time and GFLOPS over the two products like the Go row.
- `bench_conv`: `torch.nn.functional.conv2d` on [32×64×56×56] with 64
  3×3 filters, padding 1, time and GFLOPS.
- `bench_embed`: `sentence_transformers.SentenceTransformer` in float32
  on 32 copies of the 65-token sentence `cmd/bench` uses, `batch_size=32`,
  `normalize_embeddings=True`; skipped with a note when
  `sentence-transformers` is not installed or `FIBERAI_MODELS` is unset.
  Thread count follows `torch.get_num_threads()` and is printed, so the
  Go side can be run with the same `FIBERAI_WORKERS`.
- Sections selectable with `--only attention,conv,embed` mirroring the Go
  flag.

- `models/gemma`: a parity test for inputs longer than the 512-token
  sliding window (879, 1147 and 1753 tokens), where sliding and full
  attention layers differ and the `WindowMask` path runs; reference
  produced by the committed `make_long_reference.py`.

## Acceptance

- `bench.py --only attention,conv` runs on a machine without the model;
  `--only embed` prints a skip line without `FIBERAI_MODELS` and a result
  row with it. Shapes and units match `cmd/bench` exactly.
- Long-input cosine to the reference ≥ 0.999 for all three texts.
- The M2 Pro numbers are recorded in BENCHMARKS.md beside the Go rows;
  the Xeon rows follow with the next Xeon run.

## Notes

Results, Apple M2 Pro, 2026-09-06 (PyTorch 2.14 with Accelerate, 6
threads by default; fiber/ai numbers from `cmd/bench`):

| row | PyTorch | fiber/ai |
|---|---:|---:|
| attention [8×8×512×64], GFLOPS | 561 | ~950 |
| same, causal mask | 566 | ~900 |
| conv [32×64×56×56]·64×3×3, GFLOPS | 314 | 367 |
| EmbeddingGemma 32 × 65 tokens, sentences/s | 91 | 94–98 |

The long-input parity test found a real bug: for bidirectional models
transformers uses a window bound of `sliding_window/2 + 1` (config 512 →
257, i.e. 256 tokens each side), not the configured 512. With the
configured value the cosine to the reference was 0.982–0.987 for 879,
1147 and 1753-token inputs; with the halved bound it is 1.000000 for all
three. Inputs under 257 tokens were never affected, which is why the
first parity suite passed. `Config.window()` now encodes the rule.

