# Pretrained models

The `models` packages run real pretrained models on the fiber/ai stack, in
pure Go with no Python and no external service. The first is
`models/gemma`, which runs Google's **EmbeddingGemma** models — the
original EmbeddingGemma-300m (and other Gemma 3 text encoders) and the text
tower of **EmbeddingGemma 2** — and turns text into a sentence embedding,
the vector a similarity search, a cluster or an anomaly score is built on.
The same `Load` and `Embed` calls serve both; the architecture is read from
the checkpoint, so switching models means switching directories.

## Getting the model

The weights are distributed on Hugging Face and gated behind Google's
licence. Accept it once, then download the directory:

```
huggingface-cli download google/embeddinggemma-300m --local-dir embeddinggemma-300m
huggingface-cli download google/embeddinggemma-2 --local-dir embeddinggemma-2
```

EmbeddingGemma 2 is published under Apache 2.0 and is not gated. Its
directory has no Dense modules — the projection to 768 dimensions is part
of the model — and its single `model.safetensors` (1.5 GB in bf16) also
holds the vision and audio towers, which this package does not load.

The directory holds everything the loader needs: `config.json`,
`tokenizer.json`, the weight `*.safetensors`, and the sentence-transformers
head under `1_Pooling`, `2_Dense`, `3_Dense`. The files are not part of
fiber/ai and never enter the repository; the loader reads them from wherever
you put them.

## Embedding text

```go
m, err := gemma.Load("embeddinggemma-300m")
if err != nil {
    log.Fatal(err)
}
defer m.Close()

vecs, err := m.Embed([]string{
    "interface GigabitEthernet0/1 changed state to down",
    "a network link went down",
})
// vecs[0] and vecs[1] are 768-float unit vectors; their dot product is
// their cosine similarity.
```

`Embed` tokenises every input, groups inputs of similar length into batches
under a token budget, runs each batch through the encoder and head with no
gradient recorded, and returns one embedding per input in the original
order. It is safe to call with one string or ten thousand.

## Prompts

EmbeddingGemma was trained with task prompts, and using them matters for
retrieval quality. A search query and the documents it searches take
different prompts:

```go
docs, _ := m.Embed(templates, gemma.Prompt("document"))
query, _ := m.Embed([]string{"someone failed to log in"}, gemma.Prompt("query"))
```

`Prompt` names a prompt from the model's
`config_sentence_transformers.json` (`query`, `document`, `Clustering`,
`Classification`, `Retrieval-query`, and the rest). `PromptText` prepends a
literal prefix instead, for example `task: search result | query: `. With
no prompt option the text is embedded as given.

## Shorter embeddings (Matryoshka)

EmbeddingGemma is trained so that a prefix of the embedding is itself a
good embedding. `Dim` truncates to 512, 256 or 128 and renormalises, for
less storage and faster comparison at a small quality cost:

```go
small, _ := m.Embed(texts, gemma.Dim(256))
```

## Options

`Load` takes options: `Threads(n)` caps the cores one forward pass uses,
`MaxTokens(n)` truncates long inputs (default the model's context length,
capped at 2048 for EmbeddingGemma-300m; 8192, the documented context, for
EmbeddingGemma 2), `BatchTokens(n)` sets the padded-token budget per pass
(default 8192). `Config()` returns the hyper-parameters, `Tokenizer()` the
underlying tokenizer, `Dim()` the embedding size, `Close()` releases the
weights.

## Accuracy

The encoder reproduces the reference. On 64 varied sentences (English,
German, syslog with addresses and hex, other scripts, the empty string),
cosine similarity to the fp32 `sentence-transformers` output is:

| | cosine to reference |
|---|---:|
| mean | 1.000000 |
| minimum | 1.000000 |

Inputs longer than the sliding window are checked too (879, 1147 and
1753 tokens, cosine 1.000000 each). One detail matters there: for
bidirectional Gemma models the effective window is half the configured
`sliding_window` on each side (256 tokens for the configured 512), the
convention `transformers` applies; the loader does the same.

Embeddings do not depend on how inputs are batched (a sentence alone and in
a mixed-length batch match), and the `Dim(256)` vectors match the
truncated-and-renormalised full vectors.

## Speed and memory

fiber/ai computes in float32 throughout, as does the reference here. On the
Apple M2 Pro, 32 sentences of about 64 tokens embedded as one batch:

| machine | fiber/ai | PyTorch CPU (`sentence-transformers`) |
|---|---:|---:|
| Apple M2 Pro (AMX; PyTorch 8 threads) | 106 | 89 |
| Xeon Gold 6130, one socket (AVX-512; PyTorch MKL, 16 threads) | 58 | 37 |

Ahead of the Python stack on both machines in float32, by a fifth on
the laptop and by half on the server, which is the bar this project
holds itself to. The pre-norms are folded into the weights and the gated
feed-forward runs as one fused product (the fused-epilogue section of
[performance.md](performance.md)). The weights take about
1.2 GB in float32; total resident memory after loading and embedding a
batch stays under 1.6 GB. For throughput, hand `Embed` many texts at once
rather than calling it per text: batching amortises the per-call work and
fills the cores.

## EmbeddingGemma 2

`google/embeddinggemma-2` (October 2026) maps text, images, audio and
video into one 768-dimensional space. `models/gemma` runs its **text
path**: a 270 M-parameter encoder, 24 layers at width 512, whose vectors
are the same ones the multimodal model produces for text. Usage is the
same as above, with the same prompt names and Matryoshka sizes 512, 256
and 128:

```go
m, err := gemma.Load("embeddinggemma-2")
vecs, err := m.Embed(texts, gemma.Prompt("document"))
```

**Accuracy.** Against the reference (`transformers` main with
sentence-transformers 6.1, float32, CPU) on the same 64 sentences and
prompted query/document pair as above: cosine 1.000000 mean and minimum.
The long inputs (879, 1147 and 1753 tokens) match at 1.000000 as well.
Batching does not change the vectors, and `Dim(512|256|128)` matches the
truncated-and-renormalised full vector.

**What changed from the 300m model**, all of it handled by the loader:

- RMSNorm multiplies by its weight directly, not by `1 + w` as Gemma 3
  does. Loading these weights the old way raises no error; it just
  produces wrong vectors, so a test compares the loaded norms with the
  stored ones.
- The full-attention layers are wider than the sliding ones (head
  dimension 512 with one key/value head, against 256 with two), the
  rotary dimension follows each layer's head size, attention scores are
  not scaled, and the values get a weight-free RMSNorm.
- **The sliding window is an inclusive radius of 512 tokens** on each
  side, not the 300m model's half-window. With the old rule the long
  inputs come out at cosine 0.993 to 0.996 — plausible, and wrong; the
  long-input test is what catches it.
- Every layer has a third residual block that mixes in a per-layer
  embedding — a 512-vector per token and layer, computed once per pass
  from the token embeddings — and every layer's output is multiplied by
  a stored scalar.
- The projection from width 512 to 768 dimensions is inside the model;
  it is linear, so it is applied after pooling, which gives the same
  vector.

**Speed.** On the Apple M2 Pro, 32 short sentences as one batch, best
of five after warm-up, sentences per second:

| model | fiber/ai | PyTorch CPU, 8 threads |
|---|---:|---:|
| EmbeddingGemma-300m | 394 | 240 |
| EmbeddingGemma 2 | 256 | 164 |

The new model is slower than the old one in both stacks, by a factor of
about 1.5. Its text tower has fewer parameters but more arithmetic per
token — 136 million multiply-adds against 101 million: a wider
feed-forward relative to the model width, the wider full-attention
layers, and the per-layer-embedding block — and nine matrix products per
layer instead of seven, each at the narrower width. On 421-token
documents the ratio is the same (9.2 against 13.6 documents per
second). The text weights take about 1.1 GB in float32.

**Not supported:** the vision and audio towers, and therefore image,
audio and video input. The API takes text only, and text that contains
one of the model's image, audio or video placeholder tokens is refused
with an error rather than embedded as ordinary text, which would produce
a vector that looks valid and means something else.

## What is loaded, and what is not

For EmbeddingGemma-300m, `models/gemma` implements the Gemma 3 text
encoder: scaled token
embeddings, grouped-query attention with two RoPE bases (a local base for
the sliding-window layers, a global base for the full-attention layers),
per-head query and key normalisation, the gated-GELU feed-forward, and
Gemma's `(1 + w)` RMSNorm, followed by mean pooling, the Dense head and L2
normalisation; the EmbeddingGemma 2 differences are listed in its section
above. It is inference only. Training on top of a loaded encoder,
bf16 or int8 weight storage, and text-generating decoder models are not in
this package yet.

The pieces it is built from are general and documented elsewhere:
`safetensors` reads and writes the weight files (see the package
documentation and *Exporting safetensors* in the
[nn and optim page](nn-and-optim.md)),
`tokenizer` is the byte-level BPE tokenizer (*byte-pair encoding*: a
vocabulary grown by repeatedly merging the most frequent adjacent pair of
symbols, with every byte kept as a fallback so no input is unencodable;
[tutorial chapter 17](../tutorial/17-tokenization.md) trains one from
scratch and measures what it buys), and the new tensor operations
(`RoPE`, `WindowMask`, `AttentionScaled` for grouped-query attention, and
`Recycle` for reusing off-heap buffers during inference) are in the
[tensors page](tensors.md).
