# 13. A language model, from nothing

Every chapter so far has ended with a number: an accuracy, an error, a
distance. This one ends with a machine that writes. The architecture is
the one behind every current language model, the training data is 1.1 MB
of Shakespeare, and the whole thing is about two hundred lines and seven
minutes of laptop time.

```
go run ./examples/tutorial/13-language-model
```

## The task

Predict the next character. That is all a language model does, and
everything else follows from it: given `To be or not to b`, put high
probability on `e`. Train that on enough text and the model has to learn
spelling to predict the end of a word, grammar to predict the end of a
clause, and something about who speaks in a play to predict what follows
a name and a colon.

The training signal is free, which is the trick that made the field move.
Every position in the corpus is an example, and its label is the next
character — no annotation, no dataset construction. A batch of 32 windows
of 128 characters is 4096 predictions at once.

## Characters, not words

The vocabulary is the 65 distinct bytes of the corpus:

```go
text, _ := shakespeare.Text()
symbols, index := shakespeare.Vocabulary(text) // 65 symbols
ids := shakespeare.Encode(text, index)
```

Real models use *subword* tokens — pieces like `ing` or ` the` — and we
have a tokenizer for them (chapter 9 uses it). It is the wrong choice
here, and the numbers say why. Gemma's vocabulary has 262 144 entries.
At width 256 the embedding table and the output layer would come to
134 M parameters, against a model body of 3 M: the lookup tables would be
forty times the model they serve. Worse, this corpus is only 321 000
subword tokens, using 13 608 distinct ones — 5 % of the vocabulary — and
5 035 of those appear exactly once. Ninety-five percent of that table
would never receive a gradient.

Characters cost the opposite way. 65 symbols is a 33 K embedding, one
percent of the model, and every symbol appears tens of thousands of
times. The price is that a character carries less meaning than a word
piece, so the model needs more layers and more context to reach the same
understanding — which is exactly the trade real models make in the other
direction once they have billions of tokens to learn from.

## The block

A transformer is one block repeated. Ours:

```go
func (b *block) Forward(x *tensor.Tensor) *tensor.Tensor {
    x = x.Add(b.attn.Forward(b.norm1.Forward(x)))   // mix across positions
    h := b.norm2.Forward(x).Reshape(B*T, D)
    h = b.down.Forward(b.up.Forward(h).GELU()).Reshape(B, T, D)
    return x.Add(h)                                  // think per position
}
```

Two sub-layers with two different jobs. **Attention mixes across
positions**: every token looks at the tokens before it and pulls in what
it needs (chapter 11). **The feed-forward part thinks about each position
on its own**, widening to four times the model width and back — that is
where most of the parameters live, and the current understanding is that
it is where most of what the model knows is stored.

Both are wrapped the same way: normalise, transform, add back. The `Add`
is a *residual connection*, and it is what makes depth work. The
unchanged `x` runs from the embedding to the output like a main line,
with each block adding a correction to it. A gradient reaching layer 12
does not have to survive a journey through eleven transformations; it can
take the main line. Removing those two `Add`s stops a four-layer model
from training at all.

`RMSNorm` — *root-mean-square normalisation*, which rescales a vector
by its own magnitude — before each sub-layer rather than after is
*pre-norm*. It is
the arrangement Llama and Gemma use, and it is the difference between a
model that trains at this depth without a warm-up schedule and one that
does not.

## Positions, by rotation

Attention has no notion of order. Shuffle the tokens and every score is
the same, because it only ever compares pairs. So position has to be put
in by hand, and there are two ways.

The old way is a lookup table with one learned vector per slot, added to
the input — one `nn.Embedding(context, dim)`. It works, and it can say
nothing at all about position 500 in a model trained at 128.

The way current models do it is **RoPE**, *rotary position embedding*:
rotate each query and key by an angle proportional to its position, just
before they are multiplied. Two
vectors rotated by their positions have a dot product that depends on the
**difference** between them, so the model learns "three tokens back"
rather than "slot 47", and it degrades gracefully past the trained
length. One field turns it on:

```go
a := nn.NewMultiHeadAttention(dim, heads)
a.Mask = tensor.CausalMask(ctx) // may attend to itself and earlier
a.RoPEBase = 1e4                // positions by rotation
```

`RoPEBase` is the rate: a larger base turns the rotation more slowly and
so keeps distant positions distinguishable, which is why Gemma raises it
to 1e6 on its long-context layers. The causal mask is the other half —
without it the model may look at the answer, and it would learn nothing
but to copy the next character.

## The one initialisation that matters

`nn.NewEmbedding` draws from N(0, 1), and that is far too wide here:

```go
tensor.NoGrad(func() { m.tok.W.MulScalarInPlace(0.02) })
```

The residual stream is a running sum that every block adds to. Start the
embeddings at unit scale and the first block receives inputs an order of
magnitude larger than anything it produces, so its contribution is noise
against the input for the first several hundred steps. 0.02 is what
GPT-2 and Gemma use. It is one line, and without it this model trains
visibly worse.

## Watching it learn

```
model 4 blocks, width 256, 4 heads, context 128: 3.19M parameters

  step    1  train 5.344  validation 4.016
  step  250  train 1.862  validation 1.929
  step  500  train 1.626  validation 1.736
  step 1000  train 1.415  validation 1.622
  step 1500  train 1.337  validation 1.615
  step 2000  train 1.296  validation 1.570

2000 steps in 402s — 201 ms/step, 20384 tokens/s
```

Loss here is in *nats* per character — the natural-logarithm unit of
cross-entropy, so a loss of L means the model is as uncertain as a fair
choice between e^L options — and it has a floor you can reason about. A model that has learned nothing spreads its guess over 65
symbols: ln(65) = 4.17, which is roughly where step 1 sits. Getting to
1.57 means the model has narrowed 65 possibilities to the equivalent of
about e^1.57 ≈ 4.8. Given the next character is usually a letter or a
space, that is close to what the text itself allows.

Nearly seven minutes on a laptop CPU, for a model that writes English.

## What it writes

```
Of Clifford of Gloucester several despair:
Where gone to Rome, if thou not to my city--

MENENIUS:
Foul.

KING RICHARD III:
What is this man, it is a custom Warwick,
I cannot plant him heaven thus with passire?
Lord me! Here knows no shows with her prince,
```

Nothing here was copied. The model has learned the *shape* of a play: a
name in capitals, a colon, a line break, then verse. It has learned that
`MENENIUS` and `KING RICHARD III` are the sort of thing that goes there,
having seen them often enough. It has learned English spelling well
enough that `passire` and `concens` are wrong in the way a foreign
speaker is wrong, not in the way random letters are.

What it has not learned is meaning. Read a sentence to the end and it
goes nowhere. Three million parameters and a megabyte of text buy
convincing surface and no content — which is worth seeing directly,
because it is the same architecture as the models that do have content,
differing by four orders of magnitude in both.

## Sampling

The model outputs 65 scores. Turning those into a character is a choice:

```go
p[i] = exp((logits[i] - max) / temperature)
```

Always taking the highest score is *greedy* decoding, and it produces
loops — `the the the` — because the most likely character given a bland
context is a bland character, which makes the next context blander.
Sampling from the distribution avoids that, and the **temperature**
controls how sharply. Below 1 it sharpens: safer, more repetitive. Above
1 it flattens: more surprising, more misspelled. 0.8 is a common default
and is what the output above used. Try `-temp 0.4` and `-temp 1.3` and
read both.

## What this costs to run

```
500 characters in 1.3s — 398 characters/s, re-running the whole prefix each time
```

That line describes a real inefficiency. To produce one character, the
example runs the whole 128-character prefix through all four blocks and
keeps only the last row. The other 127 rows are computed and discarded —
and they were computed identically the step before, because in causal
attention nothing that happens later changes an earlier token's keys and
values.

Storing them instead is a *KV cache* — a key-value cache — and it is
the subject of
[chapter 14](14-kv-cache.md). Skipping ahead: it removes about 128× the arithmetic and does
not make generation 128× faster, and the reason why is the most useful
thing in that chapter.

## What to try

Set `-layers 2` and `-layers 6` and compare validation loss against
training time; the returns are steeper than you would guess at this size.
Then remove the causal mask — delete the `a.Mask` line — and watch the
loss collapse to nearly zero while the samples become gibberish: the
model has found it can read the answer, and has learned to copy rather
than predict. It is the clearest demonstration of why the mask exists.

Finally, `-ctx 64` against `-ctx 256`. Attention costs the square of the
context, so the longer one is much slower per step; ask whether the loss
justifies it. That question, at a scale four orders of magnitude larger,
is most of what current model design is about.
