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
holds the vision and audio towers, each loaded on its first use.

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
video into one 768-dimensional space. `models/gemma` runs its **text,
image and audio paths**: a 270 M-parameter text encoder, 24 layers at
width 512, and the vision and audio towers that feed it (see *Images*
and *Audio* below). Usage is the
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

### Images

Images go into the same space as text:

```go
vecs, err := m.EmbedImages([]image.Image{photo, diagram})

// Text and images in one input, one vector for the whole item:
vecs, err = m.EmbedInputs([]gemma.Input{{
    Text:   "Rack 4, top: <|image|> rear view: <|image|>",
    Images: []image.Image{front, rear},
}})
```

Each `<|image|>` in the text is filled by the next image; a placeholder
without an image, or an image without a placeholder, is an error.
`Prompt` and `Dim` work as for text. The vision tower (170 M parameters,
about 680 MB in float32) is read from the checkpoint on the first image,
so a program that only embeds text never loads it.

Inside, an image is resized, keeping its aspect ratio, to the largest
size that is a multiple of 48 pixels and fits 2520 patches of 16×16; a
16-layer encoder with axial 2-D rotary positions runs over the patches;
3×3 blocks are averaged into at most 280 soft tokens; and those take the
place of the placeholder in the text sequence, between begin- and
end-of-image markers. The text tower then runs over the whole sequence.

**Accuracy.** Against the reference on eight images — square, landscape,
portrait, a 1200×900 grayscale image, a 40×30 image enlarged thirty
times, a 1000×120 strip, an image with an alpha channel and a photo —
cosine 1.000000 mean and 0.999997 minimum; a text-and-image input and
an image under a prompt match at 1.000000. The resize is Pillow's
bicubic filter, ported exactly: it reproduces Pillow's output bit for
bit, which is also what PyTorch computes on x86. On Apple Silicon
PyTorch's own resize differs from Pillow by one or two levels in up to
1 % of the values when enlarging, which is the 0.999997.

Use any `image.Image`; register decoders with blank imports
(`image/png`, `image/jpeg`). Go's JPEG decoder rounds differently from
libjpeg, which the Python stack uses: the same photo decoded both ways
embeds at cosine 0.99978.

**Cost.** About 0.9 s per image on the M2 Pro, against 1.06 s for
PyTorch on 8 threads with batches of 8 and 1.27 s one image at a time.
The vision tower does roughly 490 billion multiply-adds per image at the
default 280 soft tokens; the text tower's share is small.

### Audio

Speech and other sound go into the same space:

```go
f, _ := os.Open("call.wav")
samples, rate, err := gemma.ReadWAV(f) // 16-bit PCM, channels averaged
// rate must be 16000: EmbedAudio does not resample.
vecs, err := m.EmbedAudio([][]float32{samples})

vecs, err = m.EmbedInputs([]gemma.Input{{
    Text:  "Voicemail from the NOC: <|audio|>",
    Audio: [][]float32{samples},
}})
```

Input is 16 kHz mono in [−1, 1]; clips longer than 30 s are cut at 30 s,
as the reference does. The audio tower (300 M parameters, about 1.2 GB
in float32) loads on the first clip.

Inside: 128-bin log-mel frames every 10 ms (20 ms Hann window, 512-point
FFT, HTK mel scale up to 8 kHz); two strided convolutions reduce them to
25 frames per second; twelve conformer layers follow — two half-weighted
feed-forward blocks, local self-attention over the current and the
eleven previous frames with a relative position term and a soft cap,
and a causal depthwise convolution — and the frames, projected into the
text width, replace the placeholder between begin- and end-of-audio
markers. Every projection in the tower clamps its input and output to
bounds stored in the checkpoint.

**Accuracy.** Against the reference on public-domain speech (3 s and
20 s of a LibriVox recording, and 40 s cut to 30), a tone, noise,
near-silence and a 50 ms clip: cosine 1.000000 mean and minimum; a
text-and-audio input matches at 1.000000. The log-mel features agree
with the reference's to within 4·10⁻⁴ on a log scale that reaches 7 —
the reference's FFT runs in single precision, this one in double — and
frame counts and padding masks are identical.

**Cost.** On the M2 Pro a 20 s clip takes 0.67 s (30× real time) and a
3 s clip 0.18 s, against 1.20 s and 0.85 s for PyTorch on 8 threads.

**Not supported:** video. The model reads video as frames sampled once
a second through the vision tower, so frames you decode yourself can be
passed as images, but there is no video input as such, and text with
the video placeholder is refused with an error. Audio at other sample
rates must be converted first.

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
