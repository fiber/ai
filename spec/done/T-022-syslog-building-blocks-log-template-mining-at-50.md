---
id: T-022
title: Syslog building blocks: log template mining at 50M lines a day, k-means and nearest-centre search over embeddings
status: done
scope:
  - logtemplate/
  - cluster/
  - tensor/
  - examples/
manual:
  - docs/manual/README.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

The syslog project of the network workbook: 50 million lines a day
cannot be embedded line by line (EmbeddingGemma on CPU manages a few
hundred lines a second), so lines are reduced to templates first and only
templates are embedded and clustered. Two packages: `logtemplate`, which
turns a stream of lines into template IDs and counts fast enough for
that volume on one core, and `cluster`, which runs k-means and
nearest-centre search over embedding rows with the tensor layer.

## Design

`logtemplate`:
- `Mask(line string) string` replaces IPv4/IPv6 addresses, MAC
  addresses, hex identifiers, numbers and quoted strings by placeholders
  (`<IP>`, `<MAC>`, `<HEX>`, `<NUM>`, `<STR>`), leaving words intact.
- `Miner` implements the Drain algorithm: lines split into tokens, a
  tree keyed by token count and the first tokens, leaves holding
  templates; a line joins the template with the highest share of equal
  tokens above a threshold (default 0.5), differing positions become
  `<*>`; otherwise a new template is created. `Add(line) int` returns the
  template ID, `Templates()` the list with counts, `Template(id)`. A
  `Reset` for daily roll-over, counts per template exported for the
  time-series side. Single-goroutine by design (one core suffices);
  callers shard by source if they want more.
- Benchmark on generated syslog-like lines; target below.

`cluster`:
- `Normalize(x)` scales rows to unit length so that the matrix product
  is cosine similarity.
- `KMeans(x, k, iters, seed)` with k-means++ seeding, assignment by one
  `MatMul` against the centres per iteration, centre update by summing
  assigned rows; returns centres and assignments.
- `Nearest(x, centres)` returns for each row the best centre and its
  similarity, the "is this a known kind of message" test.
- An example under `examples/syslog/` that mines templates from a
  generated day of logs, fakes embeddings (random projections of the
  template tokens, deterministic) and clusters them, printing the
  dashboard the workbook describes.

## Acceptance

- `logtemplate` tests: masking of the listed patterns; the Drain miner
  merges lines that differ in one token and keeps distinct messages
  apart; counts add up. Throughput ≥ 300 000 lines/s on one M2 Pro core
  (50 M lines in under 3 minutes); no Python baseline exists for this
  path in the benchmark suite, and the tensor layer is not on it: no
  performance impact on existing operations.
- `cluster` tests: k-means recovers three planted clusters; `Nearest`
  returns the planted centre with similarity above 0.9 for members and
  below the threshold for a random vector. 50 000 × 768 against 40
  centres: one `MatMul`, under 20 ms on the M2 Pro (the tensor layer's
  GEMM figures cover it; NumPy/PyTorch would do the same product at the
  same speed).
- Manual README lists both packages; the example runs in under ten
  seconds.

## Notes

Implemented. The first `Mask` was six regular expressions and cost 16 µs
a line (62 000 lines/s, a fifth of the target); the single-pass scanner
that replaced it does 0.54 µs, and `Miner.Add` 0.74 µs including it:
1.17 million lines/s on one M2 Pro core in `examples/syslog` (two million
generated lines → 9 templates in 1.7 s). Design choices recorded there:
digit runs inside words are masked (`eth0` → `eth<NUM>`) because
interface numbers vary; pure hex-letter words are not, because `cafe`
and `added` are hex too. The miner treats any token containing a
placeholder as variable when keying its tree, otherwise `ge-…` and
`xe-…` lines never met. `cluster`: k-means recovers three planted
clusters at similarity ≥ 0.85 (noise 0.05 per dimension over 64);
`Nearest` on 50 000 × 768 against 40 centres runs in 9.7 ms. Manual page
`applications.md` added.
