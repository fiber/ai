# 7. Anatomy of a training loop

Chapters 4 to 6 trained on all the data at once and reported how well
the model did on the very examples it had seen. Real training does
neither. This chapter adds the three things every serious loop has:
mini-batches, a validation set, and a way to notice when the model has
stopped learning and started memorising.

```
go run ./examples/tutorial/07-training-loop
```

## Mini-batches

Rather than the whole data set per step, the loop takes a random slice
of 32 examples, updates, and takes the next slice. Two reasons: memory
(a million examples do not fit at once) and speed of learning (thirty
small noisy steps per pass beat one exact step). `Rows` assembles a
batch from an index list, and the index list is reshuffled every epoch
so the model cannot learn anything from the order:

```go
perm := r.Perm(n)
for epoch := 1; epoch <= epochs; epoch++ {
    r.Shuffle(n, func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
    for lo := 0; lo < n; lo += 32 {
        idx := perm[lo:min(lo+32, n)]
        xb, yb := xTrain.Rows(idx), yTrain.Rows(idx)
        loss := tensor.MSELoss(model.Forward(xb), yb)
        opt.ZeroGrad(); loss.Backward(); opt.Step()
    }
}
```

`MSELoss` is the *mean squared error*: the average of the squared
differences between prediction and target, which is the loss chapter 4
minimised by hand.

An *epoch* is one pass over all the training data. With 200 examples
and batches of 32 that is 7 steps per epoch. The `data` package has the
same loop as one line, `for idx := range data.Batches(n, 32, r)`, once
you have seen it written out.

## The validation set

Before training starts, part of the data is set aside and never used
for an update. Here the last 40 of 240 examples:

```go
xTrain, yTrain := x.Narrow(0, 0, 200), y.Narrow(0, 0, 200)
xVal, yVal := x.Narrow(0, 200, 40), y.Narrow(0, 200, 40)
```

The loss on those 40 is the only number that tells you how the model
will do on data it has not seen. The training loss tells you how well
it remembers.

With time-ordered data (anything measured: traffic, sensors, logs) the
split must be by time, later data for validation, never a random
sample; otherwise the model is validated on the neighbours of its
training examples and looks better than it is.

## What `Step` does

`opt.Step()` has been in every loop since chapter 6 without a word about
what it does with the gradient. It is the first thing this chapter's
program prints, the same model and the same data five times over, with
nothing different but the update rule:

```
the same model and data, only the update rule differs (200 epochs)
                                 ep 10     ep 50    ep 200  val<0.30
SGD lr 0.003                     1.397     0.898     0.451     never
SGD lr 0.05                      0.821     0.283     0.291     ep 36
SGD lr 0.5                         NaN       NaN       NaN     never
SGD lr 0.05, momentum 0.9        0.307     0.278     0.219      ep 7
Adam lr 0.001                    1.702     0.907     0.243    ep 145
```

**Plain SGD** is chapter 4's hand-written step, `optim.NewSGD(params, lr)`:

```
w ← w − lr·g
```

One knob. At 0.003 the model is still at 0.451 after two hundred epochs
— it is learning, just not in this lifetime. At 0.5 the first few steps
overshoot, the overshoot feeds the next gradient, and within a few
batches the weights are infinite: the loss reads `NaN` and every later
step keeps it there. A `NaN` loss is almost always the learning rate,
and it is the one failure that announces itself. Between crawling and
exploding lies 0.05, and the useful range is narrow: a factor of about
170 separates the two failures, so search it by multiplying and dividing
by three, not by ten percent.

**Momentum** keeps a running sum of the gradients and steps along that
instead:

```
v ← μ·v + g          w ← w − lr·v
```

A direction the gradient keeps pointing in accumulates — with μ = 0.9 a
steady gradient builds up to ten times its own size — while directions
that flip sign from batch to batch cancel out. Mini-batch gradients are
noisy by construction, so this is mostly free speed: same learning rate,
epoch 7 instead of epoch 36. `optim.NewSGD` takes it as a field, and 0.9
is the value nearly everyone uses.

**Adam** adds a second running average, of the *squared* gradient, and
divides by its root:

```
m ← β₁·m + (1−β₁)·g       v ← β₂·v + (1−β₂)·g²
w ← w − lr·m̂ / (√v̂ + ε)
```

The division is the point. A parameter whose gradients are consistently
tiny gets a proportionally larger step, one with large gradients a
smaller one, so a single learning rate means roughly the same thing in
every layer of a model whose layers are on very different scales. `m̂`
and `v̂` are `m` and `v` corrected for having started at zero, which
otherwise makes the first steps far too small. `optim.NewAdamW` adds
weight decay applied to the weights directly rather than through the
gradient, which is what "decoupled" means and what every current model
is trained with.

And here Adam is the *slowest* of the three that work: epoch 145 against
epoch 7. That is not a mistake in the table. This model has 321
parameters, one hidden layer and well-scaled inputs — exactly the case
where there is nothing for the per-parameter scaling to fix, and Adam's
conservative default of 0.001 simply takes smaller steps. What Adam buys
is that it works without knowing any of that in advance. On the models
from chapter 13 on, where an embedding table, an attention projection
and a layer norm want completely different step sizes, tuning SGD per
layer is not a reasonable use of anybody's afternoon.

So: reach for `AdamW` at 1e-3 first, because it is the choice that needs
no knowledge of the problem. If the loss goes `NaN`, the rate is too
high; if it barely moves, too low; change it by factors of three. And
keep in the back of your mind that a tuned SGD with momentum still beats
it on plenty of problems — this table being one of them.

## Overfitting, shown

The data set is small (200 examples of 8 inputs) and noisy by design;
a perfect model would still have a loss of 0.09, the noise. Three models
train for 600 epochs each:

```
big model, no regularisation: memorises the 200 examples
epoch  100  train 0.067  validation 0.265
epoch  200  train 0.003  validation 0.201
epoch  300  train 0.003  validation 0.197
epoch  400  train 0.002  validation 0.172
epoch  500  train 0.000  validation 0.161
epoch  600  train 0.003  validation 0.168
```

A training loss of 0.000 on data whose noise alone is worth 0.09 means
one thing: the model has memorised the noise. The validation loss, 0.17,
is what it is actually worth.

```
small model: cannot memorise, generalises better
epoch  100  train 0.297  validation 0.423
epoch  200  train 0.127  validation 0.256
epoch  300  train 0.086  validation 0.251
epoch  400  train 0.064  validation 0.256
epoch  500  train 0.049  validation 0.254
epoch  600  train 0.039  validation 0.249
```

The small model cannot get close to the training data and also does
not generalise as well here; too little capacity is a failure mode too.

```
big model with dropout: noise during training keeps it honest
epoch  100  train 0.095  validation 0.310
epoch  200  train 0.054  validation 0.226
epoch  300  train 0.036  validation 0.208
epoch  400  train 0.022  validation 0.209
epoch  500  train 0.038  validation 0.263
epoch  600  train 0.031  validation 0.262
```

`nn.NewDropout(0.3)` zeroes a random 30 % of a layer's outputs at every
training step, so no single path through the model can be relied on and
memorising gets harder. Its validation loss is best around epoch 300
and then drifts up: the model is still overfitting, just more slowly.
Note `model.SetTraining(false)` before evaluating; dropout must be off
when the model is used.

The lesson is in the shape of the curves, not the final numbers: the
training loss always falls; the validation loss falls, bottoms out, and
rises. The epoch where it bottoms out is when to stop, and the model to
keep is the one from that epoch (chapter 8 shows how to save it). This
is called early stopping and it is the most important regularisation
there is.

## Three habits

- **Print both losses every epoch**, from the first run on. A training
  loss alone has never told anyone anything.
- **Change one thing at a time**: batch size, learning rate, model
  size, dropout. Each of them moves the curves, and the effects mix.
- **Read a `NaN` as a learning rate**, not as a bug in the model.

## What to remember

- Shuffle, batch with `Rows`, one pass is an epoch.
- Hold out validation data; split by time when the data is a time
  series.
- Training loss down and validation loss up means stop.
- `Step` is a choice: AdamW when you know nothing, SGD with momentum
  when the problem is small and you are willing to tune.

Next: [8. A model in service](08-service.md).
