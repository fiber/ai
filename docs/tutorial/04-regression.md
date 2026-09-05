# 4. Linear regression by hand

The smallest model that deserves the name: a weight per input and one
bias, trained by the loop from chapter 3. No `nn` package yet; plain
tensors, so that every line is visible.

```
go run ./examples/tutorial/04-regression
```

## The data

We make it ourselves, so that we know the answer. Two inputs, a truth
of `y = 2·x₁ − 3·x₂ + 0.5`, and a little noise so it looks like a
measurement:

```go
n := 200
x := tensor.Randn(n, 2)                    // 200 examples, 2 inputs each
trueW := tensor.New([]float32{2, -3}, 2, 1)
y := x.MatMul(trueW).AddScalar(0.5).Add(tensor.Randn(n, 1).MulScalar(0.1))
```

`x` is `[200 2]`, `trueW` is `[2 1]`, their product is `[200 1]`: one
number per example. That is the shape of every regression: examples down,
inputs across, one prediction per row.

## The model

The same shape as the truth, starting from zero:

```go
w := tensor.Zeros(2, 1).SetRequiresGrad(true)
b := tensor.Zeros(1).SetRequiresGrad(true)
```

A prediction for all 200 examples at once is one matrix product and a
broadcast add:

```go
pred := x.MatMul(w).Add(b)
```

## The loop

```go
lr := float32(0.1)
for step := 0; step <= 60; step++ {
    pred := x.MatMul(w).Add(b)
    loss := tensor.MSELoss(pred, y)     // mean of (pred − y)²
    w.ZeroGrad(); b.ZeroGrad()
    loss.Backward()
    tensor.NoGrad(func() {
        w.AddScaledInPlace(w.Grad(), -lr)
        b.AddScaledInPlace(b.Grad(), -lr)
    })
}
```

`MSELoss` is chapter 3's "difference squared", averaged over the 200
examples so that the loss does not grow with the data size. Everything
else is the loop you already know.

```
step  0  loss 12.3617  w = [0.400 -0.539]  b = 0.114
step 10  loss 0.2083  w = [1.825 -2.656]  b = 0.476
step 20  loss 0.0128  w = [1.978 -2.947]  b = 0.498
step 30  loss 0.0094  w = [1.994 -2.987]  b = 0.498
step 40  loss 0.0094  w = [1.996 -2.993]  b = 0.498
step 50  loss 0.0094  w = [1.996 -2.993]  b = 0.498
step 60  loss 0.0094  w = [1.996 -2.993]  b = 0.498
truth:           w = [2.000 -3.000]  b = 0.500
```

Two things to notice. The loss stops at 0.0094 and does not go to zero:
that is the noise we added (spread 0.1, squared is 0.01), and a model
that went below it would be memorising noise. And the weights land at
1.996 and −2.993, not exactly 2 and −3, for the same reason; with 200
noisy examples this is as close as anyone can get.

## Using it

```go
tensor.NoGrad(func() {
    newX := tensor.New([]float32{1, 1}, 1, 2)
    newX.MatMul(w).Add(b).Item()
})
```

```
prediction for x = [1 1]: -0.499 (truth 2 - 3 + 0.5 = -0.5)
```

`NoGrad` matters here for a practical reason: without it the library
would record the operations for a `Backward()` that never comes.

## Try it

- Set `lr` to 1.0. The loss explodes: the steps overshoot the bottom of
  the bowl and land higher on the other side each time. Set it to 0.001
  and it crawls. Choosing the learning rate is the first thing you tune
  in every project, and there is no formula; a factor of 3 up and down
  from a working value is the usual search.
- Take the noise out (`MulScalar(0)`). The loss now goes to zero and the
  weights to exactly 2 and −3.
- Add a third input that the truth ignores. Its weight should go to
  zero; watch how long that takes compared with the others.

## What to remember

- Examples down, inputs across: `[n inputs]` times `[inputs 1]`.
- `MSELoss` for numbers, one loop for everything.
- The loss floor is the noise. Do not chase it.

Next: [5. The first classifier](05-classifier.md), where `nn` and
`optim` take the boilerplate.
