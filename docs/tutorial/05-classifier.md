# 5. The first classifier

Chapter 4 predicted a number. Most of what people call AI predicts a
*choice*: spam or not, which of ten digits, which of three kinds of
device. This chapter trains the classic toy classifier, three
interleaved spiral arms, and introduces the two packages that take the
boilerplate out of the loop: `nn` for layers and `optim` for the update.

```
go run ./examples/tutorial/05-classifier
```

## The data

900 points in the plane, 300 on each of three spiral arms that wind
around each other. Each point comes with its arm number, 0, 1 or 2: the
label. No straight line separates the arms, which is exactly why the
problem needs more than the model of chapter 4.

## Layers

```go
model := nn.Sequential{
    nn.NewLinear(2, 64), nn.ReLU{},
    nn.NewLinear(64, 64), nn.ReLU{},
    nn.NewLinear(64, 3),
}
```

`nn.Linear` is chapter 4's `x.MatMul(w).Add(b)` with its own `w` and
`b`. Three of them in a row would still be one straight line, because a
line of a line is a line. `ReLU` between them replaces every negative
number by zero; that small kink is what lets the stack bend, and enough
kinks approximate any shape. The last layer has three outputs, one score
per arm.

`Sequential` calls the layers in order and collects their parameters
with `Params()`, which is what the optimizer needs:

```go
opt := optim.NewAdam(model.Params(), 3e-3)
```

Adam is chapter 3's "step against the gradient" with two refinements:
it remembers the direction of recent steps (momentum) and it scales each
parameter's step by how noisy that parameter's gradient has been. In
practice it means one learning rate works for most problems; `3e-3` and
`1e-3` are the values to try first.

## The loss for choices

```go
logits := model.Forward(x)                  // [900×3], a score per class
loss := tensor.CrossEntropy(logits, labels) // labels: []int, one per row
```

The three output scores are called logits. `CrossEntropy` turns them
into probabilities (with softmax, so they sum to one) and measures how
much probability the model gave the *right* arm: none is a loss of
about 1.1 for three classes (that is `ln 3`), all of it is a loss of
zero. You never need the formula; the shape is what matters: scores
`[n classes]`, labels `[]int` of length `n`.

## The loop

```go
for epoch := 1; epoch <= 200; epoch++ {
    loss := tensor.CrossEntropy(model.Forward(x), labels)
    opt.ZeroGrad()
    loss.Backward()
    opt.Step()
}
```

Compare with chapter 4: `ZeroGrad` and the update moved into the
optimizer; nothing else changed.

```
before training: accuracy 33.3% (guessing would be 33.3%)
epoch   1  loss 1.210  accuracy 33.7%
epoch  40  loss 0.486  accuracy 79.7%
epoch  80  loss 0.145  accuracy 97.8%
epoch 120  loss 0.065  accuracy 99.4%
epoch 160  loss 0.043  accuracy 99.8%
epoch 200  loss 0.033  accuracy 99.8%
```

Accuracy is the number a person understands: how many of the 900 points
land on the right arm. It is computed under `NoGrad` from `Argmax(1)`,
the index of the highest score per row.

## Using it

```go
tensor.NoGrad(func() {
    probs := model.Forward(points).Softmax(1)
    probs.Argmax(1)
})
```

```
two new points, probability per arm:
[[   0.6254    0.3746 2.231e-06]
 [0.0007976   0.05902    0.9402]]
predicted arms: [0 2]
```

The first point is a coin toss between arms 0 and 1; the second is
clearly arm 2. This is the difference between `Argmax` and `Softmax`:
the first gives you the answer, the second tells you whether to trust
it. A system that acts on a 62 % answer the same way as on a 94 % one is
throwing away the most useful thing the model produces.

## What to remember

- Classification: scores per class, `CrossEntropy` with `[]int`
  labels, `Argmax` for the answer, `Softmax` for the confidence.
- `nn.Sequential` of `Linear` and `ReLU`; `optim.NewAdam` with `1e-3`
  to `3e-3`.
- Training is still the same four lines.

Next: [6. Anatomy of a training loop](06-training-loop.md): batches,
validation, and the one failure mode every model has.
