---
id: T-064
title: A KV cache for incremental decoding, and tutorial chapter 14
status: done
scope:
  - nn/
  - examples/tutorial/
  - docs/tutorial/
manual:
  - docs/manual/nn-and-optim.md
done: 2026-09-12
created: 2026-09-12
---

## Goal

Generating text one token at a time re-runs the whole prefix for every
token. In causal attention nothing that comes later changes an earlier
token's keys and values, so all of that work but the last row is
repeated exactly. Chapter 13's model generates 398 characters a second
while doing roughly 128 times the arithmetic it needs, and the library
has no way to avoid it: `MultiHeadAttention.Cross` projects K and V from
its context argument on every call, which is precisely the work a cache
exists to skip.

This is the last structural piece missing for inference. `PosOffset`
(T-058) already lets a single query know which position it is at, so
rotary positions work against keys rotated when they were stored; what
is absent is somewhere to keep them.

## Design

A `KVCache` holding the projected keys and values of one attention
layer, and a step method on `MultiHeadAttention` that projects only the
new tokens and appends:

    type KVCache struct { K, V *tensor.Tensor } // [batch, heads, seen, headDim]
    func (m *MultiHeadAttention) Step(x *tensor.Tensor, c *KVCache) *tensor.Tensor

`Step` takes [batch, new, dim], projects Q, K and V for the new tokens
only, rotates Q at `c.Len()` and K at the same offset, appends K and V
to the cache with `tensor.Cat` along the token axis, and attends with no
mask — every key in the cache is in the past by construction, which is
the second saving after the arithmetic.

`Step` accepts more than one token so a prompt can be ingested in one
call before decoding begins, which is what a real runtime does and costs
nothing extra to support.

The cache is explicit and caller-owned rather than a field on the
module: the same weights serve many concurrent sequences, and hiding
per-sequence state inside a shared module is the bug that design invites.

Chapter 14 then measures it, and explains why the speedup is far short
of the 128× the arithmetic suggests: with one token per step every
matrix product becomes a matrix–vector product, and the step is bound by
reading the weights rather than by arithmetic. That floor is why
inference is memory-bound, why quantisation pays, and why serving
batches.

## Acceptance

- Cached decoding produces the same tokens as the uncached path from the
  same seed and prompt: a test that generates 64 tokens both ways and
  compares them exactly, and a test that the cached attention output
  matches a full forward's last row within 1e-4.
- Chapter 14 reports measured characters per second with and without the
  cache on a named machine, and states the memory the cache costs.
- No performance impact on training or on anything that does not call
  `Step`: `cmd/bench` attention rows stay within run-to-run noise of
  1 005 GFLOPS for [8×8×512×64] on the M2 Pro, against PyTorch's 590 on
  the same machine and session.
- `go test ./...`, `go vet ./...` and `GOARCH=amd64 go vet ./...` clean.

## Notes
Measured on an M2 Pro, 400 characters from a 3.19M model: 354
characters/s re-running the prefix, 1303 with the cache, 3.7x. The cache
holds 3.3 MB after 400 characters.

Two things the implementation only got right because the example checked
them.

`Trim` needs the cache to count its absolute position separately from
the rows it holds. The first version rotated the next query at `Len`,
so after trimming a query at true position 300 was rotated as though it
were at 127 while the retained keys still carried rotations from 173 to
299 — every distance wrong, silently. `Pos` fixes it.

The example first asserted that cached and uncached generation are
identical, and past the window they are not. They agree for 156 of 400
characters and then diverge, because the sliding window rotates at
positions starting from 0 while the cache rotates at true positions:
mathematically the same differences, not the same float32. Sampling
turns a one-in-a-million difference into a different character and from
there a different text. The example now reports how far they agree,
which is more informative than a boolean, and the test asserts exactness
only where it is exact — inside the window.

The chapter also does not repeat the usual claim that this is
memory-bound. At 3.19M parameters a step reads 12.8 MB, and 1303
characters/s is 16.6 GB/s against 147 GB/s that this machine streams in
the element-wise benchmark. The limit is per-operation overhead at batch
one — roughly forty operations at about 19 microseconds each — which is
the same wall the small-product path exists for.
