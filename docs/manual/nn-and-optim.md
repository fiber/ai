# Neural networks and optimisers

## Modules

```go
type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Params() []*tensor.Tensor
}
```

That is the whole abstraction. `nn.Sequential` is a slice of modules;
anything with those two methods composes with it.

| Module | Notes |
|---|---|
| `NewLinear(in, out)` | `y = x·W + b`, W is `[in out]` (He initialisation); `NewLinearNoBias`; in inference the bias is applied in the product's epilogue |
| `ReLU{} GELU{} Tanh{} Sigmoid{}` | stateless activations |
| `Softmax{Dim: -1}` | softmax along a dimension |
| `Flatten{}` | `[batch, ...]` → `[batch, features]` |
| `NewDropout(p)` | active while `Training` is true; identity otherwise |
| `NewLayerNorm(n)` | normalises the last dimension, learnable scale and shift |
| `NewEmbedding(vocab, dim)` | `Lookup(ids []int)` returns `[len(ids) dim]` |

`nn.NumParams(m)` counts scalars, `nn.ZeroGrad(m)` clears gradients,
`nn.SetTraining(m, on)` flips modules with a training mode (Sequential
forwards it to its children).

Under `tensor.NoGrad` `Linear` adds its bias inside the matrix product
and `Sequential` fuses a `ReLU{}` or `GELU{}` that directly follows a
`Linear` into that product (`tensor.MatMulFused`), so the inference
forward of an MLP is its matrix products and nothing else. With
gradients recording, the modules compute the same result from the
ordinary operations; see the fused-epilogue section of
[performance.md](performance.md).

## Attention

`nn.MultiHeadAttention(dim, heads)` projects [batch, tokens, dim] inputs to
`heads` sets of queries, keys and values, runs scaled dot-product
attention per head and projects back. `Forward(x)` is self-attention;
`Cross(x, context)` takes queries from `x` and keys and values from
`context`. Set `Mask` to `tensor.CausalMask(n)` for autoregressive
models, to `tensor.PaddingMask(lengths, n)` for padded batches, or to
their sum; masks are additive and broadcast over batch and heads.

Position is not part of attention: it compares pairs of tokens and a
shuffled sequence would score identically, so a model has to be told
where its tokens are. Set `RoPEBase` to rotate queries and keys by their
positions before the scores (`tensor.RoPE`), which makes a score depend
on the distance between two tokens rather than their absolute slots and
is what Llama- and Gemma-class decoders do; 1e4 and 1e6 are the usual
values. Zero, the default, leaves the rotation out, and the alternative
is then a learned vector per slot added to the input — one
`nn.Embedding(context, dim)` indexed by position — which is simpler but
says nothing about a position longer than the training window.
`PosOffset` sets the position of the first query, for decoding a token
at a time against earlier keys.

For generation, `nn.KVCache` holds the keys and values a decoder has
already computed and `m.Step(x, cache)` projects only the new tokens,
appends them and attends over the whole cache — no mask needed, since
everything cached precedes every new query. `Step` takes any number of
tokens, so a prompt goes in with one call before decoding proceeds one
token at a time. A cache belongs to one sequence, not to the module: the
same weights serve many sequences and per-sequence state inside a shared
module is a race waiting to happen.

`Trim(keep)` bounds the memory by dropping the oldest rows, which is
sound with rotary positions because a score depends on the distance
between two tokens rather than their absolute places — and is not sound
with a learned position table. `Len` is how many rows the cache holds
and `Pos` is the position the next token takes; rotations use `Pos`, so
trimming does not move the tokens that remain.

`nn.NewRMSNorm(dim)` is the normalisation of Gemma- and Llama-class
models; `Embedding.Lookup(ids)` gathers token vectors with a scatter-add
gradient. `examples/nn/attention` puts them together as a two-layer
pre-norm transformer that learns to continue repeating patterns.

Without gradients recording, the attention itself runs from the GEMM
micro-kernel with K and V packed once per head and a three-pass softmax;
see the attention section of [performance.md](performance.md).

Under `NoGrad`, or when neither input needs a gradient, `RMSNorm` runs a fused one-pass kernel that writes a single output buffer; while a gradient is recorded it is composed from `Square`, `Mean`, `Sqrt`, `Div` and `Mul` so autograd provides the backward pass. Both give the same values.

## Convolutions

`nn.NewConv2D(in, out, k)` and `nn.NewConv1D(in, out, k)` are convolution
layers over [batch, channels, height, width] and [batch, channels,
length]; `Stride` and `Pad` are fields (defaults 1 and k/2, which keeps
the size for odd k). `nn.NewMaxPool2D(k)` halves the spatial size,
`nn.Flatten{}` leads into `Linear`. Convolutions run as one matrix
product per image over an im2col layout, so they use the GEMM path;
`examples/nn/conv` trains a small CNN to 97 % on generated shape images
in under a second.

## Losses and fused primitives

All in package `tensor`:

- `MSELoss(pred, target)` — mean squared error.
- `CrossEntropyWeighted(logits, targets, weights)` — cross-entropy with one
  weight per class, normalised by the batch's total target weight; for
  uneven classes.
- `CrossEntropy(logits, targets []int)` — mean negative log-likelihood of
  integer classes with the log-softmax fused in. Feed raw logits, not
  softmax output.
- `x.Softmax(dim)`, `x.LogSoftmax(dim)` — numerically stable, single pass.
- `LayerNorm(x, gamma, beta, eps)` — one fused op with its own backward.

## Optimisers

```go
opt := optim.NewSGD(params, 0.1)          // plain SGD
sgd := optim.NewSGD(params, 0.05)
sgd.Momentum, sgd.Nesterov, sgd.WeightDecay = 0.9, true, 1e-4

adam := optim.NewAdam(params, 1e-3)       // β1 0.9, β2 0.999, ε 1e-8
adamw := optim.NewAdamW(params, 1e-3, 1e-2)
```

`Step()` applies one update from the parameters' current `Grad()` (skipping
parameters without one) and `ZeroGrad()` clears them. Updates run under
`NoGrad` internally.

## A complete training loop

```go
model := nn.Sequential{
	nn.NewLinear(784, 512), nn.ReLU{}, nn.NewDropout(0.1),
	nn.NewLinear(512, 10),
}
opt := optim.NewAdamW(model.Params(), 1e-3, 1e-2)

for epoch := 0; epoch < epochs; epoch++ {
	model.SetTraining(true)
	for _, batch := range batches {            // batch.X [b 784], batch.Y []int
		loss := tensor.CrossEntropy(model.Forward(batch.X), batch.Y)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	model.SetTraining(false)
	correct := 0
	tensor.NoGrad(func() {
		pred := model.Forward(testX).Argmax(1)
		for i, p := range pred {
			if p == testY[i] {
				correct++
			}
		}
	})
	fmt.Printf("epoch %d accuracy %.2f%%\n", epoch, 100*float64(correct)/float64(len(testY)))
}
```

Building a batch from rows of a larger tensor:
`tensor.Stack(0, x.Row(i), x.Row(j), ...)` copies the rows; a contiguous
range is a free view: `x.Narrow(0, start, n)`.

## Writing your own module

```go
type Residual struct{ Inner nn.Module }

func (r Residual) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Add(r.Inner.Forward(x)) }
func (r Residual) Params() []*tensor.Tensor                 { return r.Inner.Params() }
```

Parameters are ordinary tensors created with `SetRequiresGrad(true)`;
there is no registration step.
