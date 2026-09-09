# 11. Attention: choosing what to look at

Chapter 8's convolution treats every minute of the window the same way:
the same filter, everywhere. Chapter 11 builds a layer that decides,
for each input, which minutes matter, and says so in numbers you can
print. This is attention, the layer inside every transformer, in its
smallest form: about twenty lines, written here rather than taken from
`nn`, because the point is to see the weights.

```
go run ./examples/tutorial/11-attention
```

The task is chapter 8's: the number of aircraft in Dublin's zone half an
hour ahead, from the last two hours of eight airports. Each minute of
the window becomes a *token* of ten numbers: the eight airports' counts
and two numbers that say where in the window the minute sits, because
attention by itself has no idea of order.

```
EIDW, 30 minutes ahead from 120 tokens of 10 numbers; 5611 examples train, 1440 validate (2025-01-23)
attention pooling + head, 913 parameters
```

## The layer

Three learned pieces: a matrix that turns a token into a *key*, a
matrix that turns it into a *value*, and one *query* vector. The score
of a minute is the dot product of its key with the query; a softmax
over the 120 scores turns them into weights that sum to 1; the output
is the weighted sum of the values.

```go
type attentionPool struct {
    Wk, Wv *tensor.Tensor // [10 × 16]
    Q      *tensor.Tensor // [16 × 1]
}

func (a *attentionPool) weights(x *tensor.Tensor) *tensor.Tensor {
    b := x.Dim(0)
    k := x.Reshape(b*120, 10).MatMul(a.Wk)        // a key per minute
    scores := k.MatMul(a.Q).Reshape(b, 120)        // one number per minute
    return scores.MulScalar(1 / 4.0).Softmax(1)    // weights, sum 1 per window
}

func (a *attentionPool) Forward(x *tensor.Tensor) *tensor.Tensor {
    b := x.Dim(0)
    w := a.weights(x)                              // [b × 120]
    v := x.Reshape(b*120, 10).MatMul(a.Wv).Reshape(b, 120, 16)
    return w.Unsqueeze(2).Mul(v).Sum(1)            // [b × 16]
}
```

That is the whole mechanism. The scaling by 1/√16 keeps the scores
from saturating the softmax early; the rest is two matrix products, a
softmax and a sum, and autograd handles the backward pass through all
of it. A transformer uses several of these side by side (heads), lets
every token be a query rather than one learned vector, and stacks the
result; `nn.MultiHeadAttention` is that version. The reading is the
same.

## What it learns

```
epoch  5  train 2.12  validation 2.18 aircraft
epoch 20  train 1.89  validation 2.19 aircraft

chapter 8 on the same task: persistence 3.02, flat MLP 2.52, 1-D CNN 2.63 aircraft
```

With 913 parameters, fewer than the convolution and seventy times
fewer than the flat MLP, the attention model beats both, and its
validation error stops improving by epoch 5 while the training error
keeps falling only slowly: it is not memorising, it is selecting.

## Where it looks

The weights for one window, the two hours before 08:00 on the
validation day, summed per ten minutes:

```
  120 to 110 minutes ago   12.6%  #########################
  110 to 100 minutes ago   12.0%  ########################
  100 to  90 minutes ago   13.8%  ###########################
   90 to  80 minutes ago   13.9%  ###########################
   80 to  70 minutes ago    6.5%  #############
   70 to  60 minutes ago    4.6%  #########
   60 to  50 minutes ago    6.7%  #############
   50 to  40 minutes ago    6.1%  ############
   40 to  30 minutes ago    3.6%  #######
   30 to  20 minutes ago    6.0%  ###########
   20 to  10 minutes ago    6.7%  #############
   10 to   0 minutes ago    7.2%  ##############
```

More than half of the weight sits 80 to 120 minutes back, not on the
most recent minutes. Nobody told the model this; the data did. An
aircraft that will be in Dublin's zone in thirty minutes was over
Manchester, Amsterdam or Paris about ninety minutes before, which is
the flight time, and the tokens from that part of the window carry the
neighbours' counts. The convolution of chapter 8 could not express
"look two hours back, not one"; a filter of width seven sees seven
minutes. Attention expresses it with one query vector.

That is the reason the layer took over the field: it lets the model
pick, per input, which parts of a long input to combine, and the
picking is visible. Whenever a model built on attention does something
surprising, its weights are the first place to look, exactly as here.

## What to try

Print the weights for a window at 14:00 or 22:00 and see whether the
model looks somewhere else when the traffic pattern differs. Replace
the single learned query by the last token's key (so the most recent
minute asks the question) and compare. Then read `nn/attention.go`,
which is this chapter with heads and a mask.
