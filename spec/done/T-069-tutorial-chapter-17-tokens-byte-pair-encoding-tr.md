---
id: T-069
title: Tutorial chapter 17: tokens — byte-pair encoding, trained and measured
status: done
scope:
  - docs/tutorial/
  - examples/tutorial/
  - docs/manual/models.md
  - docs/manual/getting-started.md
  - README.md
manual:
  - docs/manual/models.md
done: 2026-09-14
created: 2026-09-14

---

## Goal

Chapter 15 predicts characters and explains in a paragraph why it does
not use Gemma's 262 144-entry vocabulary. Chapter 10 loads a model whose
input is subword pieces and never says where they come from. Between
the two there is a hole where every real model's first layer sits: the
tokenizer, and the trade it makes between vocabulary size and sequence
length. The manual calls `tokenizer` "the byte-level BPE tokenizer"
without expanding BPE, which is the acronym debt noted in T-065.

The chapter shows the trade with numbers rather than describing it:
train a byte-pair encoding on the Shakespeare corpus in a few dozen
lines, watch the token count fall as merges accumulate, retrain chapter
15's model on the pieces, and compare the two models on the only fair
scale, bits per character of held-out text — which is where most
readers will be surprised, because a per-token loss cannot be compared
across vocabularies and it is the number every framework prints.

## Design

A new chapter 17, `examples/tutorial/17-tokenization`, appended after
the KV cache so that the language-model chapters stay together and
nothing is renumbered. The chapter refers back to 15 and 16 rather than
the other way round; chapter 15's "Characters, not words" section gains
a one-line forward reference.

- **Three views of one sentence**: `To be, or not to be` as 65-symbol
  character ids, as whitespace words, and — when `FIBERAI_MODELS` points
  at EmbeddingGemma — as Gemma pieces via `tokenizer.Load`. The Gemma
  part is optional and the program says so and continues without it.
- **BPE trained in the example.** Pre-tokenise the corpus into chunks
  (a run of letters with its preceding space attached, or a single
  other byte, so that merges never cross a word boundary, as in every
  production tokenizer), count distinct chunks, then repeat: count
  adjacent symbol pairs weighted by chunk frequency, merge the most
  frequent into a new symbol. Print the first merges and the corpus
  token count at 0, 64, 256, 512 and 1024 merges, with mean characters
  per token. Encoding applies the merges in rank order per chunk, cached
  by chunk; decoding concatenates the pieces. The example owns this
  code; the `tokenizer` package loads trained vocabularies and is not
  extended.
- **The same model on two vocabularies.** Chapter 15's decoder, same
  width, depth, context and batch, trained for the same number of steps
  on characters and on the 1024-merge pieces. Report per-token
  validation loss (not comparable), bits per character on the held-out
  tenth of the corpus (comparable), milliseconds per step, and
  characters per second when generating. A sample from the piece model
  shows that the decoded text is whole words.
- **What it costs where it matters**: the KV cache of chapter 16 is per
  token, so the same 400 characters of context hold 2–3× fewer rows,
  and every generation step produces a whole piece. The chapter also
  says what the piece model cannot do: spell a word it never saw as a
  unit, which is why byte fallback exists.

The default step count is lower than chapter 15's so that two training
runs finish in comparable time; the chapter says so and gives the
figure.

## Acceptance

- `go run ./examples/tutorial/17-tokenization` trains the BPE, prints
  the merge table and token counts, trains both models and prints the
  comparison, without Gemma weights present.
- A test asserts that encode followed by decode reproduces the corpus
  byte for byte, that the token count is strictly decreasing across the
  reported merge counts, and that the encoded corpus is shorter than
  the character sequence.
- Every figure quoted in the chapter comes from a run of the committed
  code on a named machine with nothing else running.
- `docs/manual/models.md` expands BPE where it names the tokenizer and
  links the chapter; index rows, README and getting-started chapter
  counts read seventeen; the stale chapter numbers in the header
  comments of examples 13–16 are corrected.
- No library code changes; no performance impact.

## Notes

Measured on the M2 laptop with nothing else running (`ch17-run`, 1000
steps per model): characters 224 ms/step, 1.634 nats/token, 2.357
bits/char, 400 chars/s generated; 1089 pieces 235 ms/step, 3.749
nats/token, 2.230 bits/char, 885 chars/s. The per-token loss says the
piece model is worse by 2.3×; bits per character says it is better by
5 %, having read 2.5× more text in the same number of steps. That
reversal is the chapter's point, and the reason the comparison is at
equal steps rather than equal text: it is the comparison a reader will
make by accident.

Gemma's 262 144-piece vocabulary, learned on the web, encodes the
corpus at 3.47 chars/token against 2.54 for 1024 merges learned on the
corpus itself; the 240× larger vocabulary buys 37 % shorter sequences.
Merge ` the` is the twelfth; ` VINCENTIO` earns a merge before the
512th, which the chapter uses to show that a vocabulary carries its
corpus with it.

The first draft's prose said the piece model "cannot invent words";
the sample showed `injurives` and `Signery`. It invents them from
syllable-sized pieces, which is a different and more interesting
statement, and the chapter now makes that one.
