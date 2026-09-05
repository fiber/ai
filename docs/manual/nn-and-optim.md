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
| `NewLinear(in, out)` | `y = x·W + b`, W is `[in out]` (He initialisation); `NewLinearNoBias` |
| `ReLU{} GELU{} Tanh{} Sigmoid{}` | stateless activations |
| `Softmax{Dim: -1}` | softmax along a dimension |
| `Flatten{}` | `[batch, ...]` → `[batch, features]` |
| `NewDropout(p)` | active while `Training` is true; identity otherwise |
| `NewLayerNorm(n)` | normalises the last dimension, learnable scale and shift |
| `NewEmbedding(vocab, dim)` | `Lookup(ids []int)` returns `[len(ids) dim]` |

`nn.NumParams(m)` counts scalars, `nn.ZeroGrad(m)` clears gradients,
`nn.SetTraining(m, on)` flips modules with a training mode (Sequential
forwards it to its children).

## Losses and fused primitives

All in package `tensor`:

- `MSELoss(pred, target)` — mean squared error.
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
