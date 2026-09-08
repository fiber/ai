---
id: T-036
title: cluster.Cosine and cluster.Similarities: shared, vectorised cosine similarity for embeddings
status: done
scope:
  - cluster/
  - internal/kernel/
  - examples/
  - models/gemma/
manual:
  - docs/manual/applications.md
done: 2026-09-08
created: 2026-09-08
---

## Goal

The EmbeddingGemma example and the gemma tests each carry their own
scalar cosine function (the example even its own square root). Cosine
similarity is what every user of the embeddings computes first; it
belongs in the library, vectorised, in the two forms it is needed: one
pair of vectors, and every query against every document.

## Design

- `cluster.Cosine(a, b []float32) float32` over a new fused kernel
  `kernel.DotNorms(x, y) (dot, xx, yy)` that accumulates a·b, a·a and
  b·b in one pass (NEON, AVX2/AVX-512 via the AVX2 entry, generic); three
  separate `Dot` calls read the data three times and measured no faster
  than a scalar loop (196 vs 236 ns for 768 elements). 0 when either
  vector is all zero.
- `cluster.Similarities(queries, docs *tensor.Tensor) *tensor.Tensor`:
  rows of unit length in, the [q×d] matrix of cosines out through one
  GEMM (`Normalize` first when the rows are not unit length; the
  EmbeddingGemma outputs already are). Normalising inside was tried and
  doubled the time at 1 000×10 000 (14 ms against 7.4): the two extra
  passes over 30 MB of documents cost as much as the product, and the
  documents are reused across calls. `Nearest` stays as the argmax
  convenience over the same product.
- The example uses `cluster.Cosine`; the gemma tests keep a local copy so
  the package under test does not depend on `cluster`.

## Acceptance

- `Cosine` matches a float64 reference to 1e-6 on random vectors, gives
  1 for identical, −1 for opposite and 0 for a zero vector.
- Python baseline (M2 Pro, 768-d float32 pair): NumPy
  `dot/(norm·norm)` 1 968 ns, `torch.nn.functional.cosine_similarity`
  5 576 ns per pair; NumPy 1 000×10 000 normalised similarities through
  `Qn @ Dn.T` 7.3 ms. Target: `Cosine` under 200 ns per pair (10× NumPy,
  the fused kernel against the scalar loop's 236 ns), `Similarities` at
  the GEMM rate, i.e. within 10 % of the NumPy matrix figure.
- `Similarities` equals the pairwise `Cosine` matrix to 1e-5.
- Manual: the cluster section names both functions and when to use
  which.

## Notes

Results (M2 Pro, 2026-09-08): `kernel.DotNorms` fused, 768 elements: NEON
120 ns against 186 ns for three `Dot` calls and 232 ns for the scalar
float64 loop; AVX2 under Rosetta 250 ns against 363 ns. `Cosine` 128 ns
per pair, 15× NumPy (1 968 ns) and 44× `torch.cosine_similarity`
(5 576 ns). `Similarities` 1 000×10 000×768 with unit rows 7.35 ms,
level with NumPy's `Qn @ Dn.T` (7.3 ms, Accelerate). Found on the way:
`Normalize` over 10 000×768 takes 4.8 ms (6 GB/s) because it is composed
from four passes; a fused row-normalise kernel is a small follow-up.

