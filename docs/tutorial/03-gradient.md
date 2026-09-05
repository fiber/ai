# 3. A gradient without formulas

"Training" means: make a guess, measure how wrong it is, and adjust the
guess in the direction that makes it less wrong. The gradient is the
direction. This chapter shows what that number is and where it comes
from, with nothing more than `y = 3x + 1`.

```
go run ./examples/tutorial/03-gradient
```

## Slope

Take `y = 3x + 1`. At `x = 2`, `y` is 7. If `x` grows by a little, `y`
grows three times as much. That factor is the gradient of `y` with
respect to `x`. The library computes it when you ask:

```go
x := tensor.Scalar(2).SetRequiresGrad(true)
y := x.MulScalar(3).AddScalar(1)
y.Backward()
x.Grad().Item() // 3
```

```
y = 3x + 1 at x = 2: 7  dy/dx = 3
```

`SetRequiresGrad(true)` marks `x` as something we want the gradient
for. From then on every operation on `x` remembers what it did.
`Backward()` walks back through those operations and leaves the result
in `x.Grad()`.

For a curve the slope depends on where you stand:

```
y = x² at x = 2: slope 4
y = x² at x = -1: slope -2
y = x² at x = 0: slope 0
```

At the bottom of the bowl the slope is zero. That is what a trained
model looks like: a place where no small change makes the loss smaller.

## The loss and its gradient

Now the version that matters. We have a guess `w = 1` and the truth is
5. The *loss* is how wrong we are, squared: `(w − 5)² = 16`. Squaring
makes the loss positive and punishes big misses more than small ones.

```go
w := tensor.Scalar(1).SetRequiresGrad(true)
loss := w.Sub(tensor.Scalar(5)).Square()
loss.Backward()
```

```
guess w = 1, target 5: loss 16, dloss/dw -8
the gradient is negative, so increasing w lowers the loss
```

The gradient is −8: increasing `w` decreases the loss. So we move `w`
*against* the gradient, by a fraction of it. That fraction is the
learning rate.

```go
w.AddScaledInPlace(w.Grad(), -0.25)   // w -= 0.25 · gradient
```

```
step 1: w = 3.0000  loss = 16.0000
step 2: w = 4.0000  loss = 4.0000
step 3: w = 4.5000  loss = 1.0000
step 4: w = 4.7500  loss = 0.2500
step 5: w = 4.8750  loss = 0.0625
```

Each step halves the distance to 5. The loss printed in a step is the
loss *before* that step's update, which is why step 1 shows 16.

Three details of the code are worth a look, because every training loop
has them:

- `w.ZeroGrad()` before each `Backward()`. Gradients accumulate by
  design (useful when a batch is too big for one pass); between steps
  you clear them.
- The update runs inside `tensor.NoGrad(func() { … })`. Changing `w` is
  not part of the model; without `NoGrad` the library would refuse to
  modify a tensor that a recorded graph still depends on.
- `AddScaledInPlace` writes into `w` instead of making a new tensor,
  so the optimizer's state stays where it is.

## Many inputs at once

A model has millions of numbers, not one. The gradient is then one
number per input, all computed in the same `Backward()`:

```go
v := tensor.New([]float32{1, 2, 3}, 3).SetRequiresGrad(true)
v.Square().Sum().Backward()   // loss = 1 + 4 + 9
v.Grad()                      // [2 4 6]
```

```
v = [1 2 3], loss = sum(v²): gradient [2 4 6]
```

Each entry says how much the loss would change if that one input moved.
Moving all of them a little against their gradients is one training
step, for three numbers or for a billion.

## What happens under the hood, in one paragraph

Every operation you call on a tensor that requires a gradient records a
small node: which operation, which inputs, and how to push a gradient
back through it. `Backward()` visits those nodes from the loss backwards
and applies each rule in turn; the chain of rules is called
backpropagation. You never write the rules; each operation of the
library brings its own. That is also why in-place changes to recorded
tensors are refused: they would falsify what a node remembers.

## What to remember

- The gradient says which way to move each number to make the loss
  smaller; the learning rate says how far.
- `SetRequiresGrad`, a loss, `Backward`, `Grad`: that is the whole
  interface.
- `ZeroGrad` between steps, `NoGrad` around updates.

Next: [4. Linear regression by hand](04-regression.md), the same loop
on real inputs.
