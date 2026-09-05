# Autograd

fiber/ai implements reverse-mode automatic differentiation. Every
differentiable operation records, on its result, a node holding its inputs
and a closure that pushes the result's gradient back to those inputs.

## Marking parameters

```go
w := tensor.Randn(4, 8).SetRequiresGrad(true)
```

Only leaf tensors (tensors you created, not results of operations) can be
marked. A result requires grad when any input does; it is then a non-leaf
with `IsLeaf() == false`.

## Computing gradients

```go
loss := x.MatMul(w).Tanh().Square().Mean()   // 0-D
loss.Backward()
g := w.Grad()                                 // *Tensor of w's shape
```

`Backward` requires a single-element tensor; for other shapes pass the
upstream gradient explicitly with `BackwardWith(grad)`.

Gradients **accumulate** into `Grad()` across calls, which is what a
training loop wants between optimiser steps and what you have to reset
otherwise: call `ZeroGrad()` on each parameter (optimisers do it for you
via `opt.ZeroGrad()`, and `nn.ZeroGrad(module)` covers a module).

Only leaves keep their gradient. Intermediate results are released as soon
as their gradient has been propagated. To inspect an intermediate, call
`RetainGrad()` on it before `Backward`.

## What is differentiable

All element-wise operations, scalar variants, reductions, `MatMul` in all
its forms, all views, `Cat`/`Stack`, `Softmax`, `LogSoftmax`,
`CrossEntropy`, `MSELoss`, `LayerNorm`. Broadcasting is handled: the
gradient of a broadcast operand is summed back to its shape. Views route
gradients through the inverse view, so a gradient computed through
`w.T()` lands in `w`.

Non-differentiable points follow the usual conventions: `ReLU` and `Abs`
have gradient 0 at 0, `Maximum` gives the gradient to the first operand on
ties, `Max` shares it equally between tied elements, `Clamp` passes it
inside the range only.

## NoGrad, Detach, in-place

`tensor.NoGrad(func() { ... })` runs code without recording a graph. Use it
for evaluation and for parameter updates. It is a process-wide switch, so
wrap whole phases, not single calls, when other goroutines are computing.

`t.Detach()` returns a tensor sharing `t`'s storage with no graph
connection — the way to feed a computed value back in as a constant.

In-place operations (`AddInPlace`, `Fill`, ...) panic on tensors that
require grad while grad mode is on, because they would corrupt values that
backward closures still need. Inside `NoGrad` they are allowed:

```go
tensor.NoGrad(func() {
	w.AddScaledInPlace(w.Grad(), -lr)   // w -= lr * dw
})
```

## Checking a gradient

Central finite differences catch most mistakes in a custom loss:

```go
func numericGrad(loss func() *tensor.Tensor, p *tensor.Tensor, i int) float32 {
	d := p.Data()
	orig := d[i]
	var lp, lm float32
	tensor.NoGrad(func() {
		d[i] = orig + 1e-2
		lp = loss().Item()
		d[i] = orig - 1e-2
		lm = loss().Item()
	})
	d[i] = orig
	return (lp - lm) / 2e-2
}
```

float32 limits the agreement to a few percent relative; keep inputs O(1).
The test suite applies this to every operation (`tensor/autograd_test.go`).

## Costs

Recording a node is one small allocation per operation and happens only
when an input requires grad and grad mode is on. Backward closures capture
detached views of their inputs, so a forward pass under `NoGrad` — the
inference path — has no autograd overhead at all.
