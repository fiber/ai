# Tutorial: AI for Go developers

Twelve chapters for people who write Go and never learned NumPy or
PyTorch. Each chapter is a short text and a program you can run; the
text quotes what the program prints. No mathematics beyond "a slope";
the ideas arrive through the code.

| | Chapter | Program |
|---|---|---|
| 1 | [A tensor is a slice with a shape](01-tensors.md) | `go run ./examples/tutorial/01-tensors` |
| 2 | [Broadcasting](02-broadcasting.md) | `go run ./examples/tutorial/02-broadcasting` |
| 3 | [A gradient without formulas](03-gradient.md) | `go run ./examples/tutorial/03-gradient` |
| 4 | [Linear regression by hand](04-regression.md) | `go run ./examples/tutorial/04-regression` |
| 5 | [The first classifier](05-classifier.md) | `go run ./examples/tutorial/05-classifier` |
| 6 | [Anatomy of a training loop](06-training-loop.md) | `go run ./examples/tutorial/06-training-loop` |
| 7 | [A model in service](07-service.md) | `go run ./examples/tutorial/07-service` |
| 8 | [Convolutions over time](08-convolution.md) | `go run ./examples/tutorial/08-convolution` |
| 9 | [Embeddings: text as geometry](09-embeddings.md) | `go run ./examples/tutorial/09-embeddings` |
| 10 | [The autoencoder: unusual by reconstruction](10-autoencoder.md) | `go run ./examples/tutorial/10-autoencoder` |
| 11 | [Attention: choosing what to look at](11-attention.md) | `go run ./examples/tutorial/11-attention` |
| 12 | [Handwritten digits: the first real data](12-mnist.md) | `go run ./examples/mnist` |

Chapters 1 to 4 use only the `tensor` package; 5 to 12 add `nn` and
`optim`, chapters 8 to 12 also `data` and `metrics`; chapter 9 uses the
pretrained `models/gemma` and needs its weights once (see the models
manual). Chapter 12 is the only one that trains on data collected by
somebody else, and pulls it from the separate `github.com/fiber/ai-data`
module. Every program but that one runs in a few seconds on a laptop;
chapter 12 takes about half a minute for its convolutional model. Start
with [getting started](../manual/getting-started.md) if the module is not
installed yet.
