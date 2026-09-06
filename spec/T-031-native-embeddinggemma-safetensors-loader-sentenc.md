---
id: T-031
title: Native EmbeddingGemma: safetensors loader, SentencePiece tokenizer, Gemma 3 encoder, sentence embeddings
status: open
scope:
  - safetensors/
  - tokenizer/
  - models/
  - tensor/
  - nn/
  - examples/
  - cmd/bench/
manual:
  - docs/manual/models.md
  - docs/manual/nn-and-optim.md
  - docs/manual/tensors.md
created: 2026-09-06
---

## Goal

Run a real pretrained model in pure Go: `google/embeddinggemma-300m`
(308 M parameters, 768-dimensional sentence embeddings, Gemma 3 encoder
with bidirectional attention) loaded from the Hugging Face files and
executed on the fiber/ai stack, with results interchangeable with the
`sentence-transformers` reference. This turns "call Ollama over HTTP"
into a function call in one binary, is the embedding step of the syslog
template work, and is the first thing an outside user will try.

Everything below is inference only, float32 weights and activations.
bf16 weight storage, int8 and decoder (generating) models are follow-up
specs; the design must not block them (see Notes).

## Design

### Packages

Four new packages, each usable on its own; nothing outside `models/`
knows about Gemma.

**`safetensors/`: read weights.**

- `Open(path string) (*File, error)`: parses the 8-byte little-endian
  header length, the JSON header (`{name: {dtype, shape, data_offsets},
  "__metadata__": ...}`) and maps the data section read-only (`mmap` on
  unix, `os.ReadFile` fallback elsewhere; the file is 0.6 GB in bf16).
  `OpenDir(dir string)` opens every `*.safetensors` in a directory,
  including sharded `model-0000n-of-0000m.safetensors`, into one name
  space; an `index.json` is honoured when present, otherwise the shards
  are scanned.
- `(*File).Names() []string`, `Info(name) (Dtype, []int, bool)`,
  `Tensor(name) (*tensor.Tensor, error)`: materialises a float32 tensor
  in fiber/ai's own storage (so it can be mapped, released, packed).
  Dtypes: `F32` (copy), `BF16` (shift left 16, exact), `F16` (full
  conversion including subnormals, ±Inf, NaN), `F64` (narrowing),
  integer types only via `Raw(name) []byte`. Unknown dtype is an error
  that names the tensor.
- `(*File).Close()` unmaps. Tensors already materialised stay valid.
- No writer in this spec; a writer is trivial and comes with the first
  need to export.

**`tokenizer/`: byte-fallback BPE from `tokenizer.json`.**

- The Hugging Face `tokenizer.json` format is used, not the protobuf
  `tokenizer.model`, because it needs only `encoding/json`. The
  EmbeddingGemma file is a **BPE** model (`model.type == "BPE"`, 262 144
  pieces, 514 906 merges, `byte_fallback`, `fuse_unk`), not Unigram as
  first assumed. Supported: `vocab` as `{piece: id}`, `merges` as
  `[[a, b], ...]` (or the older `"a b"` line form), `byte_fallback`;
  `normalizer` `Replace` (the Gemma file maps " " to the metaspace
  character U+2581, no prepend), refuse `NFKC`/`NFC`/`Prepend`/`Sequence`
  since the file needs none; `pre_tokenizer` `Split` on " " (a no-op
  after normalisation, accepted and ignored) or `Metaspace`;
  `post_processor` `TemplateProcessing` for BOS/EOS (BOS=2, EOS=1, both
  added); `added_tokens` (6 415 of them, all raw/`normalized:false`),
  matched literally against the raw text longest-first before
  normalisation. Anything else is an error at load time with the
  offending key, never silently ignored.
- Encoding: added-token extraction over a byte trie (leftmost-longest),
  then per chunk normalisation (" " to U+2581) and BPE. BPE builds the
  initial symbols from the runes (a rune absent from the vocab falls back
  to its UTF-8 bytes as `<0xNN>` pieces) and merges adjacent pairs by
  rank with a doubly linked list and a candidate heap, O(n log n).
  Decoding is the inverse: `<0xNN>` pieces become their bytes, U+2581
  becomes a space, added tokens render as their literal content.
- API: `Load(path string) (*Tokenizer, error)`, `Encode(text string)
  []int` (with the file's BOS/EOS template applied), `EncodeRaw(text)
  []int` without template, `Decode([]int) string`, `Piece(id) string`,
  `ID(piece) (int, bool)`, `VocabSize() int`.
- Load time matters (the JSON is about 33 MB): target under one second;
  parse with a streaming decoder, build the trie as a flat array. No
  cache file in this spec.

**`models/gemma/`: the encoder and the sentence pipeline.**

- `Config` read from `config.json`: hidden size, intermediate size,
  layers, attention heads, key/value heads, head dim, vocab size, sliding
  window, layer types (or `sliding_window_pattern`, default 6, every
  sixth layer full attention), `rope_theta` (global, 1e6),
  `rope_local_base_freq` (local, 1e4), `query_pre_attn_scalar`,
  `rms_norm_eps`, `hidden_activation` (must be `gelu_pytorch_tanh`, our
  `GELU`), `use_bidirectional_attention`, `max_position_embeddings`.
  Expected values for embeddinggemma-300m (verified at load by the test,
  read from the file at run time, never hard-coded): hidden 768,
  intermediate 1152, 24 layers, 3 query heads, 1 key/value head, head
  dim 256, vocab 262 144, sliding window 512, context 2048.
- `Encoder` built from `nn` modules: scaled token embedding
  (× √hidden), per layer: RMSNorm → attention → RMSNorm → residual,
  RMSNorm → gated MLP `down(GELU(gate(x)) · up(x))` → RMSNorm →
  residual; final RMSNorm. Gemma's RMSNorm uses `(1 + w)`: the loader
  adds 1 to the weight once, the module stays the plain `nn.RMSNorm`.
  Attention: q and k pass through per-head RMSNorm (`q_norm`,
  `k_norm` over head dim), rotary embedding with the layer's base, scale
  `query_pre_attn_scalar^-0.5` instead of `1/√D`, grouped-query
  (3 query heads share 1 k/v head), bidirectional; sliding layers add a
  window mask `|i−j| < 512`. All projections are without bias.
- Weight names as in the Hugging Face file (`embed_tokens.weight`,
  `layers.N.self_attn.{q,k,v,o}_proj.weight`, `q_norm`, `k_norm`,
  `mlp.{gate,up,down}_proj.weight`, `input_layernorm`,
  `post_attention_layernorm`, `pre_feedforward_layernorm`,
  `post_feedforward_layernorm`, `norm.weight`), with or without a
  `model.` prefix. A missing or misshaped tensor is an error naming it;
  unused tensors are reported by `Load(..., Strict)`.
- `Pipeline` from `modules.json` in the sentence-transformers layout:
  `Transformer` (the encoder, with `max_seq_length`), `Pooling` (mean
  over non-padding tokens; the mode is read from `1_Pooling/config.json`
  and cls/last-token supported too), `Dense` layers from
  `n_Dense/config.json` + their own safetensors (in/out features,
  activation `Identity` or `Tanh`, bias flag), `Normalize` (L2). For
  embeddinggemma this is expected to be mean pooling, Dense 768→3072,
  Dense 3072→768, normalise; whatever the files say is built.
- `Load(dir string, opts ...Option) (*Model, error)` reads config,
  tokenizer, weights and pipeline from one directory (the `huggingface
  download` layout). Options: `Threads(n)`, `MaxTokens(n)` (default the
  pipeline's `max_seq_length`, 2048 ceiling), `Strict`.
- `(*Model).Embed(texts []string, opts ...EmbedOption) ([][]float32,
  error)`: tokenises, sorts by length into batches whose padded token
  count stays under a budget (default 8192 tokens per batch, so about 32
  sentences of 256 tokens), runs the batch under `NoGrad` with a padding
  mask, pools, applies Dense and Normalize, restores the input order,
  releases every intermediate. Options: `Prompt(name)` prepending the
  named prompt from `config_sentence_transformers.json` (`query`,
  `document`, `Retrieval-query`, `Clustering`, ...) or `PromptText(s)`
  for a literal prefix such as `task: search result | query: `;
  `Dim(d)` for Matryoshka truncation (512, 256, 128) followed by
  renormalisation; `Batch(tokens)`.
- `(*Model).Tokenizer() *tokenizer.Tokenizer`, `Config() Config`,
  `Close()` releasing the weights.

### Operations added to `tensor/` and `nn/`

- `tensor.RoPE(x, base float64, positions []int) *Tensor` on
  `[B,H,T,D]`, the Hugging Face rotate-half convention (pairs `(i,
  i+D/2)`), cos/sin tables built once per (base, T, D) and cached in the
  model, applied in one pass; no autograd needed now, implemented with
  a backward anyway since it is a plain rotation (inverse = negative
  angle) and training on top of the encoder is a foreseeable ask.
- `tensor.WindowMask(n, w int) *Tensor`: additive `[n×n]` mask, 0 where
  `|i−j| < w`, −1e9 elsewhere; composes with `PaddingMask` by addition
  like `CausalMask`.
- Grouped-query attention: `tensor.Attention` and the fused inference
  path accept k, v with `Hkv` heads where `H % Hkv == 0`; the query head
  `h` uses k/v head `h / (H/Hkv)`. No materialised repeat: the batched
  products index the k/v head by division. The fused path's scratch
  sizing (`attentionRows`) is checked for head dim 256.
- `tensor.Attention` gains an explicit scale (`AttentionScaled(q, k, v,
  mask, scale)`), the existing function keeps `1/√D`.
- `nn.Linear` without bias: `NewLinearNoBias(in, out)` (or a `Bias
  bool` option, whichever reads better in the manual); `Forward` skips
  the add, `Parameters` omits it, `SaveParams`/`LoadParams` unchanged
  in format.
- `nn.RMSNorm` works on the last dimension of a 4-D tensor (needed for
  q/k norm over head dim); check, extend if it does not.
- Per-head RMSNorm, gated MLP and the Gemma block live in
  `models/gemma`, not in `nn`, until a second model wants them.

### Benchmarks and example

- `cmd/bench` row `embed`: 32 sentences of about 64 tokens, batch as
  one, reported as sentences/s and ms/batch; skipped with a note when
  `FIBERAI_MODELS` is unset. The Python bench gets the same row with
  `sentence-transformers` on CPU, fp32, same thread count
  (`torch.set_num_threads`), `encode(batch_size=32, convert_to_numpy)`.
- `examples/models/embeddinggemma`: loads the model from `-model`,
  embeds a list of syslog templates and free-text queries with the
  retrieval prompts, prints the nearest templates for each query and
  the timing. The example is the manual's worked case.
- Reference fixtures under `models/gemma/testdata/`: `sentences.txt`
  (64 lines: English, German, syslog with IPs, hex and timestamps,
  Unicode, an empty string, a 600-token line) and
  `reference.bin` (64 × 768 float32 from `sentence-transformers` in
  fp32, plus the token ids per line), produced once by the committed
  script `make_reference.py` (documented; needs the gated download).
  Also `tokens.json` with token ids for 200 tokenizer test strings.

### Alternatives considered

- `tokenizer.model` (protobuf): needs a protobuf decoder; the
  `tokenizer.json` BPE data carries the same information.
- Repeating k/v heads with `Expand` before the products: correct and
  simple, but doubles the traffic in the attention products; indexing
  by division costs nothing.
- A generic HF model loader with architecture dispatch: premature.
  `models/gemma` is explicit; a second architecture will show what the
  shared part is.
- GGUF as the source format: would pick up Ollama's quantised files,
  but the quantisation formats are a separate project. safetensors is
  the canonical distribution and what int8 will be derived from later.

## Acceptance

- **safetensors:** a test writes a file with F32, BF16 and F16 tensors
  (including F16 subnormals, ±Inf, NaN, and a zero-size tensor) and
  reads back bit-exact values; sharded directory with index; error
  messages name the tensor for bad dtype, bad shape and missing name.
  Opening the 0.6 GB model file and materialising all tensors takes
  under 2 s on the M2 Pro (the conversion is memory-bound).
- **tokenizer:** identical token ids to the Hugging Face tokenizer on
  all 200 fixture strings (English, German with umlauts, syslog lines
  with IPv4/IPv6, MACs, hex dumps, ISO timestamps, JSON, URLs, tabs and
  repeated spaces, emoji, CJK, the empty string, a lone byte-fallback
  character); `Decode(Encode(s)) == s` for every fixture string after
  normalisation. Load under 1 s; encode throughput at least 1 M
  characters/s single-threaded (50 M syslog lines of 120 characters is
  6 GB a day, so this must never be the bottleneck).
- **model parity:** on the 64-sentence fixture, cosine similarity to the
  fp32 reference mean ≥ 0.9995 and minimum ≥ 0.999; the `Dim(256)`
  vectors match the truncated-and-renormalised reference to the same
  bound; a sentence embedded alone and inside a mixed-length batch
  differs by less than 1e-4 (padding invariance); prompts change the
  vector exactly as the reference does (the fixture has query and
  document versions of the same text). Tests skip with a clear message
  when `FIBERAI_MODELS/embeddinggemma-300m` is absent, and CI runs them
  when the directory is present.
- **memory:** RSS under 1.6 GB after loading and embedding one batch
  (1.2 GB weights plus activations); repeated `Embed` calls do not grow
  the mapped-storage statistics.
- **performance (rule 7 baseline):** `cmd/bench` `embed` row versus
  `sentence-transformers` on PyTorch CPU fp32 with the same thread count
  on the same machine. Target: at least 1.2× the PyTorch sentences/s on
  the M2 Pro (AMX) and on the Xeon Gold 6130 (one socket, 16 threads);
  Ollama's `embeddinggemma` (Metal on the Mac) is reported alongside for
  orientation, not as the target. Expected scale for orientation only:
  about 32 sentences × 64 tokens = 2048 tokens per batch ≈ 1.3 TFLOP,
  roughly 0.6 s on the M2 Pro and 1.1 s on the Xeon.
- **gates:** `go vet ./...`, `GOARCH=amd64 go vet ./...`, `go test
  ./...` including the AVX2-under-Rosetta run; no regression in existing
  bench rows.
- **manual:** `docs/manual/models.md` (new, linked from the manual
  README): download, directory layout, `Load`/`Embed`, prompts,
  Matryoshka, threads and batching advice, memory, the parity numbers
  and the benchmark table; `tensors.md`: `RoPE`, `WindowMask`,
  grouped-query attention and the scale argument; `nn-and-optim.md`:
  bias-free `Linear`. README package tree and TODO updated.

## Notes

Open points to settle from the files at implementation time, each one
a likely source of a "cosine 0.97" bug:

- whether the tokenizer template adds EOS as well as BOS;
- whether the pooling includes the prompt tokens (`include_prompt` in
  the Pooling config; sentence-transformers defaults to including them);
- the exact Dense pipeline in `modules.json` and the Dense activations;
- the rotary embedding of full-attention layers: `rope_scaling` is
  expected to be absent for this model, error if it is not;
- the attention mask of sliding layers is `|i−j| < 512`, bidirectional,
  combined with padding, per the transformers implementation;
- the embedding scale is applied in the weight dtype in PyTorch (√768
  rounded to bf16 = 27.75 versus 27.7128); the reference is produced in
  fp32, so the fp32 value is right for parity with it, and the bf16
  behaviour is noted in the manual.

Follow-ups, not in scope: bf16 weight storage expanded inside GEMM
packing (halves the 1.2 GB), int8 weights, decoder models with KV cache
(Gemma 3 270M/1B share this code), a safetensors writer, a tokenizer
cache file.
