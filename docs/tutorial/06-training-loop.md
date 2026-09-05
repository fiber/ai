# 6. Anatomy of a training loop

Chapters 4 and 5 trained on all the data at once and reported how well
the model did on the very examples it had seen. Real training does
neither. This chapter adds the three things every serious loop has:
mini-batches, a validation set, and a way to notice when the model has
stopped learning and started memorising.

```
go run ./examples/tutorial/06-training-loop
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

An *epoch* is one pass over all the training data. With 200 examples
and batches of 32 that is 7 steps per epoch.

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
keep is the one from that epoch (chapter 7 shows how to save it). This
is called early stopping and it is the most important regularisation
there is.

## Two habits

- **Print both losses every epoch**, from the first run on. A training
  loss alone has never told anyone anything.
- **Change one thing at a time**: batch size, learning rate, model
  size, dropout. Each of them moves the curves, and the effects mix.

## What to remember

- Shuffle, batch with `Rows`, one pass is an epoch.
- Hold out validation data; split by time when the data is a time
  series.
- Training loss down and validation loss up means stop.

Next: [7. A model in service](07-service.md).
