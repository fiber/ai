---
id: T-058
title: MultiHeadAttention can apply rotary positions to its queries and keys
status: done
scope:
  - nn/
  - tensor/
manual:
  - docs/manual/nn-and-optim.md
done: 2026-09-12
created: 2026-09-12
---

## Goal

Attention compares pairs of tokens and nothing in it knows where a token
sits, so position has to be injected. Two schemes are in use. A learned
table with one vector per slot, added to the input, is what GPT-2 did and
what a reader can build from `nn.Embedding` today; it cannot say anything
about a position beyond the window it was trained on. Rotary positions
rotate each query and key by an angle proportional to its index just
before the dot product, which makes a score depend on the distance
between two tokens rather than their absolute slots, and it degrades
gracefully past the trained length. Every current decoder — Llama,
Gemma, Qwen — uses the second.

`tensor.RoPE` implements it and `nn.MultiHeadAttention` cannot reach it.
The rotation has to happen after the Q and K projections and before
`tensor.Attention`, and at that point `q` and `k` are locals inside
`Cross` with no field, argument or callback that touches them. That is
why `models/gemma` writes its own attention rather than using the `nn`
module: not because the module is wrong, but because it has no seam.

A tutorial chapter that builds a small language model runs into the same
wall. It either teaches the dated scheme or teaches readers to bypass
`nn`, and neither is what the chapter is for.

## Design

Two fields on `MultiHeadAttention`:

- `RoPEBase float64` — the theta. Zero, the default, means no rotation,
  so every existing use is unchanged. Gemma uses 1e6 for global layers
  and 1e4 for local ones.
- `PosOffset int` — the position of the first query. Keys always start at
  zero. This is what incremental decoding needs later: one query at
  position n against n+1 cached keys.

`Cross` applies `tensor.RoPE` to `q` and `k` after the split to
[batch, heads, T, head_dim], which is the shape RoPE documents and the
shape Gemma already passes it. RoPE makes its input contiguous itself, so
the permuted views need no special handling.

Rejected: a `PositionFn` callback, which is more surface for one
arithmetic sequence; and applying the rotation before the head split,
which would rotate along the wrong dimension.

## Acceptance

- No performance impact: `RoPEBase` defaults to zero and the rotation is
  skipped entirely, so `cmd/bench` attention rows stay within run-to-run
  noise of 1 005 GFLOPS for [8×8×512×64] on the M2 Pro. The comparison
  against PyTorch for that row (590 GFLOPS) is unchanged, since the
  feature is off unless asked for.
- A test that attention scores with `RoPEBase` set depend only on the
  distance between two positions: shifting a query and its key by the
  same amount leaves the score unchanged, which is the property rotary
  positions exist to provide.
- A test that `PosOffset` shifts the query positions, so one query at
  offset n scores against cached keys exactly as the n-th query of a full
  sequence does.
- Gradients flow through the rotation (`checkGrad` on a small case).
- `RoPEBase` zero reproduces today's numbers bit for bit.

## Notes
Found by prototyping the language-model tutorial chapter: the first thing
a decoder needs after attention and a causal mask is a position scheme,
and the module had no way to express the one everybody uses. The
prototype fell back to a learned table, which works and is what GPT-2
did, but it would have taught the reader a scheme they will not meet in
any current model.

The seam is two fields and five lines because `tensor.RoPE` already takes
exactly the shape `Cross` has after the head split, and makes its input
contiguous itself. Gemma can now drop its private copy of this, though
that is a separate change with its own parity tests to re-run.
