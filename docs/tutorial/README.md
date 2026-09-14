# Tutorial: AI for Go developers

Seventeen chapters for people who write Go and never learned NumPy or
PyTorch. Each chapter is a short text and a program you can run; the
text quotes what the program prints. No mathematics beyond "a slope";
the ideas arrive through the code.

| | Chapter | Program |
|---|---|---|
| 1 | [A tensor is a slice with a shape](01-tensors.md) | `go run ./examples/tutorial/01-tensors` |
| 2 | [Broadcasting](02-broadcasting.md) | `go run ./examples/tutorial/02-broadcasting` |
| 3 | [A gradient without formulas](03-gradient.md) | `go run ./examples/tutorial/03-gradient` |
| 4 | [Linear regression by hand](04-regression.md) | `go run ./examples/tutorial/04-regression` |
| 5 | [One unit, and what it cannot do](05-perceptron.md) | `go run ./examples/tutorial/05-perceptron` |
| 6 | [The first classifier](06-classifier.md) | `go run ./examples/tutorial/06-classifier` |
| 7 | [Anatomy of a training loop](07-training-loop.md) | `go run ./examples/tutorial/07-training-loop` |
| 8 | [A model in service](08-service.md) | `go run ./examples/tutorial/08-service` |
| 9 | [Convolutions over time](09-convolution.md) | `go run ./examples/tutorial/09-convolution` |
| 10 | [Embeddings: text as geometry](10-embeddings.md) | `go run ./examples/tutorial/10-embeddings` |
| 11 | [The autoencoder: unusual by reconstruction](11-autoencoder.md) | `go run ./examples/tutorial/11-autoencoder` |
| 12 | [A score is not a decision](12-threshold.md) | `go run ./examples/tutorial/12-threshold` |
| 13 | [Attention: choosing what to look at](13-attention.md) | `go run ./examples/tutorial/13-attention` |
| 14 | [Handwritten digits: the first real data](14-mnist.md) | `go run ./examples/tutorial/14-mnist` |
| 15 | [A language model, from nothing](15-language-model.md) | `go run ./examples/tutorial/15-language-model` |
| 16 | [The KV cache: not recomputing the past](16-kv-cache.md) | `go run ./examples/tutorial/16-kv-cache` |
| 17 | [Tokens: what a model actually reads](17-tokenization.md) | `go run ./examples/tutorial/17-tokenization` |

Chapters 1 to 5 use only the `tensor` package, or barely more; 6 to 17 add `nn` and
`optim`, chapters 9 to 12 also `data` and `metrics`; chapter 10 uses the
pretrained `models/gemma` and needs its weights once (see the models
manual). Chapters 14 to 17 train on data collected by somebody else and
pull it from the separate `github.com/fiber/ai-data` module: handwritten
digits, then 1.1 MB of Shakespeare.

Most programs run in a few seconds on a laptop. Chapter 14 takes about
half a minute for its convolutional model, chapter 15 about seven
minutes to train a language model from scratch, chapter 16 about a
minute and a half, and chapter 17 about eight minutes for two training
runs. Start with
[getting started](../manual/getting-started.md) if the module is not
installed yet.
