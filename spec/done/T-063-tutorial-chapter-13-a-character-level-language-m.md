---
id: T-063
title: Tutorial chapter 13: a character-level language model, trained and sampled
status: done
scope:
  - examples/tutorial/
  - docs/tutorial/
  - go.mod
  - go.sum
manual:
  - docs/manual/applications.md
done: 2026-09-12
created: 2026-09-12
---

## Goal

The tutorial ends on handwritten digits, a benchmark from 1998. Chapter
11 explains attention and stops there; nothing in the repository puts
attention, a causal mask and positions together into the architecture
every current model is built from, and nothing generates text.

A reader who has followed twelve chapters can be shown the whole thing:
a decoder-only transformer built from `nn` parts, trained from scratch
on 1.1 MB of Shakespeare in a few minutes of CPU time, and then sampled
from until it produces something that looks like a play. That is the
chapter that makes the rest of the tutorial worth finishing, and it is
the strongest demonstration this project has of what Go can do here.

## Design

`examples/tutorial/13-language-model`, a character-level decoder:

- Vocabulary of 65 bytes from `github.com/fiber/ai-data/shakespeare`
  (v0.2.0). A subword vocabulary is the wrong choice at this corpus
  size and the chapter says why with numbers: Gemma's 262 144 entries
  would put 134 M parameters in the embedding and output layers against
  a 3 M model body, and 95 % of the rows would never see a gradient.
- Four blocks, width 256, four heads, context 128, batch 32. Pre-norm:
  `x + attention(RMSNorm(x))`, then `x + MLP(RMSNorm(x))`.
- Rotary positions through `RoPEBase` (T-058), not a learned table. The
  prototype used a learned table because the hook did not exist; the
  chapter should teach what current models do.
- Sampling written out in the example rather than hidden in a helper:
  temperature, and the difference between greedy and sampled output.

Generation re-runs the whole prefix for each character, which is what a
KV cache exists to avoid; the chapter says so and leaves the cache to
chapter 14, which needs library support that does not exist yet.

## Acceptance

- `go run ./examples/tutorial/13-language-model` trains and prints
  samples; the default settings reach a validation loss under 1.7 and
  produce recognisable words and speaker names, in under ten minutes on
  an M2 Pro.
- A test that runs a handful of steps on a small model and asserts the
  loss falls, so the chapter cannot rot; it must not need minutes.
- Every figure in the chapter — parameter count, ms per step, tokens per
  second, final loss, the sample itself — comes from a run of the
  committed code on a named machine.
- `go vet ./...`, `GOARCH=amd64 go vet ./...` and `go test ./...` pass.
- No performance impact on the library: the example adds no code to
  `tensor` or `nn`.

## Notes
Measured on an M2 Pro: 3.19M parameters, 2000 steps in 402s (201 ms per
step, 20 384 tokens/s), train loss 1.296 and validation 1.570. The
sample carries speaker names, verse line breaks and English spelling.

Rotary positions were worth taking: the prototype with a learned
position table reached validation 1.64 on the same budget, and the
model is slightly smaller without the table.

The training run's 250-step intervals were 51, 50, 50, 50, 50, 50, 50,
51 seconds, which is the evidence that nothing else was competing for
the machine while it ran.
