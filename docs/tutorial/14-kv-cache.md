# 14. The KV cache: not recomputing the past

Chapter 13 ended with a confession. To produce one character, the model
ran the whole 128-character prefix through all four blocks and kept the
last row. The other 127 rows were computed and discarded — and they were
computed identically the step before.

This chapter fixes that, measures the result, and then explains why the
result is much smaller than the arithmetic promises. The second part is
the more useful.

```
go run ./examples/tutorial/14-kv-cache
```

## What can be reused

In causal attention, position *i* attends only to positions ≤ *i*. So
when a token is appended, nothing about the earlier tokens changes:
their **keys** and **values** — the "what do I have" and "here is my
content" vectors from chapter 11 — are exactly what they were. Only the
new token contributes anything new.

A **KV cache** (key–value cache) is therefore just: keep the keys and
values you already computed, append one row per step, and attend from a
single-row query against the whole stored set.

```go
var c nn.KVCache
out := attn.Step(oneToken, &c)   // projects only the new token
```

`Step` projects the query, key and value of the new token, appends the
key and value to the cache, and attends. No mask is needed: everything
in the cache is in the past by construction, which is a second saving
after the arithmetic — chapter 13's causal mask is a [128×128] matrix
built and added on every step.

## Bounding it

A cache that grows for ever is a memory leak with good manners. Real
decoders keep a window, and so does this one:

```go
c.Trim(ctx - 1)   // keep the most recent tokens
```

Dropping the oldest keys is sound **because** positions are rotary. A
score depends on the distance between two tokens, not on their absolute
places, so a query at position 400 attending to a key kept from position
300 sees exactly the rotation it would in a window that began at 273. It
would *not* be sound with a learned position table, where each token was
told which slot it occupied and the survivors would suddenly be sitting
in the wrong ones. The position scheme chosen in chapter 13 for one
reason turns out to decide something else entirely.

One subtlety the cache has to get right: `Trim` drops rows but must not
move the tokens that remain. `KVCache` therefore counts `Pos`, the
position the next token will occupy, separately from `Len`, how many
rows it holds. Rotations use `Pos`. Getting that wrong — rotating the
new query as though the trimmed prefix had never existed — silently
produces a model that attends to the wrong distances.

## Is it the same computation?

```
the two paths agree for the first 156 of 400 characters
```

Not "yes" and not "no", and the reason is worth understanding.

While the whole history fits the window, the two paths are bit-identical
— the example checks this, and it holds. Past that they compute the same
thing by different arithmetic: the uncached path re-runs a sliding
window whose positions always start at 0, while the cache holds keys
rotated at their true positions, 173 through 300. Those are
mathematically equal in their differences and not equal in float32.

Sampling then amplifies the disagreement. A draw from the distribution
is a threshold on a random number, so a difference of one part in a
million eventually lands on the other side of one, a different character
comes out, and from there the two texts have different histories and
nothing in common. 156 characters of agreement is not a bug; it is what
exact-until-it-isn't looks like when the output is fed back into the
input.

## What it buys

```
whole prefix each step:   1.13s     354 characters/s
with a KV cache:          0.31s    1303 characters/s  (3.7x)
```

3.7×. Now the interesting part: the cache removes about **128 times**
the arithmetic — one token through the network instead of 128 — and
returns less than four. Where did the rest go?

The usual answer is that inference is memory-bound: with one token per
step every matrix product becomes a matrix–vector product, and you must
read all the weights regardless. That is true of large models and it is
*not* what limits this one. The model is 3.19 M parameters, so a step
reads 12.8 MB, and at 1303 characters a second that is 16.6 GB/s — while
this machine streams 147 GB/s in the element-wise benchmark. Bandwidth
is not the constraint here; it is nine times away.

What limits it is **per-operation overhead**. A step at batch one is
dozens of tiny operations — four blocks of two normalisations, four
projections, an attention and two feed-forward products — and each one
allocates a result, dispatches to the worker pool and pays a barrier.
At 0.77 ms per character over roughly forty operations, that is about
19 µs each, for products that do a few hundred thousand
multiply-accumulates. The arithmetic is nearly free; the framing is not.

That is the same wall chapter 12's small products ran into and the same
reason `SmallLimit` exists in the matrix-multiply driver. It is also why
serving systems batch requests: sixteen sequences at once turn every
matrix–vector product back into a matrix product and amortise every
fixed cost sixteen ways, for very nearly the same wall time.

## What it costs

```
cache after 400 characters: 3.3 MB (2 x 4 layers x 400 tokens x 256 wide x 4 bytes)
```

Two tensors per layer, one row per token. Here it is nothing. Scale the
same formula to a 7-billion-parameter model — 32 layers, 4096 wide, a
4096-token context — and it is 4.3 GB **per sequence**, next to 14 GB of
weights that every sequence shares.

That single number explains much of how language models are served.
Weights are shared and the cache is not, so the cache decides how many
conversations fit on a card. It is why context length is priced,
why quantising the cache is a research area of its own, and why
attention variants that shrink it — grouped-query, multi-query — were
adopted across the field within about a year of being published.

## What to try

Set `-ctx 32` and `-ctx 512` and watch both paths move: the uncached one
scales with the window, the cached one barely notices, because its work
per step is one token either way. That divergence is the whole point,
and it is the difference between a demo and something you could serve.

Then set `-gen 2000` and read the tail. The model has 128 characters of
memory and no more, so it cannot maintain a scene, a speaker or an
argument beyond it. Everything the field has built since — longer
contexts, better positions, retrieval — is an answer to what you are
looking at.
