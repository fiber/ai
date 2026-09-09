# Pretrained models

The `models` packages run real pretrained models on the fiber/ai stack, in
pure Go with no Python and no external service. The first is
`models/gemma`, which runs Google's **EmbeddingGemma** (and other Gemma 3
text encoders): it turns text into a sentence embedding, the vector a
similarity search, a cluster or an anomaly score is built on.

## Getting the model

The weights are distributed on Hugging Face and gated behind Google's
licence. Accept it once, then download the directory:

```
huggingface-cli download google/embeddinggemma-300m --local-dir embeddinggemma-300m
```

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
capped at 2048), `BatchTokens(n)` sets the padded-token budget per pass
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
| Xeon Gold 6130, one socket (AVX-512; PyTorch MKL, 16 threads) | 49 | 37 |

Ahead of the Python stack on both machines in float32, by a fifth on
the laptop and by a third on the server, which is the bar this project
holds itself to. The pre-norms are folded into the weights and the gated
feed-forward runs as one fused product (the fused-epilogue section of
[performance.md](performance.md)). The weights take about
1.2 GB in float32; total resident memory after loading and embedding a
batch stays under 1.6 GB. For throughput, hand `Embed` many texts at once
rather than calling it per text: batching amortises the per-call work and
fills the cores.

## What is loaded, and what is not

`models/gemma` implements the Gemma 3 text encoder: scaled token
embeddings, grouped-query attention with two RoPE bases (a local base for
the sliding-window layers, a global base for the full-attention layers),
per-head query and key normalisation, the gated-GELU feed-forward, and
Gemma's `(1 + w)` RMSNorm, followed by mean pooling, the Dense head and L2
normalisation. It is inference only. Training on top of a loaded encoder,
bf16 or int8 weight storage, and text-generating decoder models are not in
this package yet.

The pieces it is built from are general and documented elsewhere:
`safetensors` reads the weight files (see the package documentation),
`tokenizer` is the byte-level BPE tokenizer, and the new tensor operations
(`RoPE`, `WindowMask`, `AttentionScaled` for grouped-query attention, and
`Recycle` for reusing off-heap buffers during inference) are in the
[tensors page](tensors.md).
