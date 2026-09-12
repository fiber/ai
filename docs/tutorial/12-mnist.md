# 12. Handwritten digits: the first real data

Every chapter so far generated its own data. Spirals, sine waves,
counters from a simulated day: convenient for teaching, because the
answer is known and the file is never missing. Real work does not begin
that way. It begins with somebody else's bytes in somebody else's
format, and the first thing you have to do is read them.

This chapter trains on MNIST — 60000 handwritten digits for training and
10000 for testing, 28 by 28 pixels of grey, collected from American
census workers and high-school students in the early nineties. It is the
oldest benchmark still in daily use, and the reason to run it here is
that everybody knows roughly what the numbers should be. If a framework
reports 89% on MNIST, something is broken, and you know it without
reading the code.

```
go run ./examples/mnist
```

## Where the data comes from

The four files live in a second module, `github.com/fiber/ai-data`,
embedded in the package and verified by checksum. The example fetches
nothing at run time.

That is a deliberate choice with two reasons behind it. The first is
that an example which downloads from a public server on every run is a
small denial-of-service attack with a friendly face: one person testing
it is nothing, a thousand readers of a tutorial doing the same is a
problem for whoever pays for that bandwidth. The airspace example in
this repository learned that lesson the hard way and now identifies
itself and backs off. The second is that 11 MB of digits would otherwise
sit in the framework's own module and be downloaded by everyone who ever
imports `tensor`, whether they look at MNIST or not. A separate module
keeps the framework small: Go fetches only the modules it needs to build
what you import.

The format is `idx`, which predates most things: a big-endian magic
number, the count, the dimensions, then the pixels, one byte each, with
no separators and no header per image. Thirty lines of `encoding/binary`
read it, and `ai-data/mnist` is those thirty lines plus the checks that
turn a truncated file into an error instead of a panic.

```go
train, _ := mnist.Train()
fmt.Println(train.N, train.Rows, train.Cols) // 60000 28 28
x := train.Float32(0, buf)                   // 784 values, 0 to 1
y := train.Labels[0]                         // the digit
```

## Scaling, and one trap

Networks train better when the inputs are centred near zero with a
deviation near one. The obvious move is `data.Fit`, which computes a mean
and a deviation *per feature* — and here it would produce `NaN`.

The reason is worth understanding, because it is the kind of thing real
data does and generated data never does. A pixel in the corner of an
MNIST image is zero in all 60000 training images: nobody writes a digit
into the corner. Its standard deviation is exactly zero, and dividing by
it destroys the image. So the example uses one mean and one deviation for
the whole set:

```
scaled by mean 0.1307 and standard deviation 0.3081 of the training pixels
```

Those two numbers are the ones every MNIST tutorial in every framework
uses, which is a pleasant way to confirm the loader read the file
correctly. Note also which split they come from: the training split
measures itself, the test split borrows the training numbers. At
prediction time you do not yet know the statistics of the data arriving,
so a model that needs them has already cheated.

## The baseline

The model is chapter 5's classifier, widened. Two coordinates became 784
pixels, three spiral arms became ten digits, and nothing else changed:

```go
nn.Sequential{
    nn.NewLinear(784, 256), nn.ReLU{},
    nn.NewLinear(256, 10),
}
```

```
MLP  784-256-10: 203530 parameters, 60000 images, 5 epochs of batch 128
  epoch  1  loss 0.2473  test accuracy 96.39%  0.2s
  epoch  2  loss 0.0992  test accuracy 96.97%  0.2s
  epoch  3  loss 0.0672  test accuracy 97.61%  0.2s
  epoch  4  loss 0.0481  test accuracy 97.52%  0.2s
  epoch  5  loss 0.0358  test accuracy 97.82%  0.2s
  MLP  784-256-10: 97.82% after 1.0s of training
```

97.8% of ten thousand test digits, from two hundred lines of Go, in one
second of training on a laptop. Every value in that run went through the
same `tensor` package chapter 1 introduced.

Notice epoch 4: accuracy went *down* while the loss kept falling. That is
not a bug, it is the ordinary noise of a validation number computed on
ten thousand images; a difference of 0.1 points is nine digits. Treat
small differences in this table as nothing at all.

## What convolution buys

Chapter 8 slid filters along a time series. The same idea in two
dimensions slides them across an image, and the argument for it is the
same: a stroke is a stroke wherever it appears, so the thing that
recognises it should not have to be learned again for every position.

```go
nn.Sequential{
    toImages{},                                   // [batch, 1, 28, 28]
    nn.NewConv2D(1, 16, 3), nn.ReLU{}, nn.NewMaxPool2D(2),   // 28 -> 14
    nn.NewConv2D(16, 32, 3), nn.ReLU{}, nn.NewMaxPool2D(2),  // 14 -> 7
    nn.Flatten{},
    nn.NewLinear(32*7*7, 10),
}
```

`toImages` is a four-line module with no parameters that reshapes the
flat rows into pictures; the rest is `nn`. Each convolution stage looks
at 3 by 3 neighbourhoods and the pooling then halves the picture, so the
second stage sees features of features over a wider area, and the linear
head at the end reads 32 channels of 7 by 7 instead of raw pixels.

```
CNN  16-32 filters: 20490 parameters, 60000 images, 5 epochs of batch 128
  epoch  1  loss 0.2418  test accuracy 97.58%  7.2s
  epoch  5  loss 0.0325  test accuracy 98.85%  7.1s
  CNN  16-32 filters: 98.85% after 35.6s of training
```

| | parameters | accuracy | training |
|---|---:|---:|---:|
| MLP 784-256-10 | 203530 | 97.82% | 1.0s |
| CNN 16-32 filters | 20490 | 98.85% | 35.6s |

The convolutional model is **ten times smaller** and better. That is the
result to take away: the improvement did not come from capacity, it came
from a structure that matches the data. The filters are shared across
every position in the image, so 16 filters of 3 by 3 are 160 numbers
doing the work that a dense layer would need tens of thousands for.

It is also thirty-five times slower to train, which is the other half of
the truth and the half tutorials usually skip. Convolution does far more
arithmetic per parameter than a matrix multiply does, and the first
stage runs 16 filters over 784 positions of every image in the batch.
Whether one point is worth 35 seconds depends entirely on what you are
building. On this laptop — an Apple M2 Pro — both are fast enough that
the question never comes up; at a hundred times the data it is the only
question.

## The same thing in PyTorch

Numbers mean little without something to compare them against, so
`benchmarks/python/mnist.py` is this example line for line in PyTorch:
the same idx files, the same scaling, the same two architectures, Adam at
1e-3, batch 128, five epochs, and a hand-written batch loop rather than a
`DataLoader`, so the shape of the work matches. Both run on the six
performance cores of the same M2 Pro — PyTorch chooses six threads by
default here, and `defaultWorkers` on macOS is the performance-core
count, so no thread flag is needed on either side.

| | fiber/ai | PyTorch 2.8 |
|---|---:|---:|
| MLP accuracy | 97.82% | 97.88% |
| MLP training | **1.0 s** | 2.0 s |
| CNN accuracy (mean of seeds 12, 13, 14) | 98.64% | 98.73% |
| CNN training | **35.6 s** | 51.6 s |

Twice as fast on the perceptron, about 1.45× on the convolutional model,
with accuracy the same to within the spread across seeds — ours is
98.85 / 98.64 / 98.44 over those three, PyTorch 98.72 / 98.75 / 98.72.
PyTorch is the steadier of the two and holds a tenth of a point on the
mean; we are the faster.

Two things are worth saying about that comparison rather than leaving
them for the reader to find. It is CPU against CPU: on a machine with a
CUDA card PyTorch is not in the same race, and this library does not try
to be. And the numbers were wrong until recently — the convolution used
to score 98.50 because its backward pass read a buffer that had already
been freed. What exposed it was exactly this comparison, which is the
argument for having one at all.

## Where the mistakes are

The example ends by printing the pairs the model confuses most:

```
  most frequent confusions (true digit read as):
    6 read as 0: 7 times
    7 read as 2: 7 times
    9 read as 4: 7 times
    9 read as 7: 7 times
    5 read as 3: 5 times
```

115 mistakes in ten thousand, and no pair accounts for more than seven of
them. That flatness is the interesting part.

An average accuracy tells you how often the model is right; the confusion
matrix tells you *how* it is wrong, and the two shapes call for different
work. Error concentrated in one pair can be attacked directly — more
examples of that pair, a feature that separates them, a model that
abstains when the two top scores are close. Error spread evenly across
ninety pairs usually means the model has learned what is there to learn,
and the remaining images are the ones where the writer's intent is not
recoverable from the pixels. Print a few of the failures as ASCII art and
you will agree with the model more often than not.

Note which pairs do surface: nine read as four, nine read as seven, six
read as zero. Closed loops and open ones, in a script where a hurried
writer closes the wrong stroke.

## What to try

Run `-model mlp -epochs 20`. The training loss keeps falling — 0.0132 at
epoch 10, 0.0034 at epoch 20 — while test accuracy sits at 97.9 % and
does not move. That is overfitting, visible in four lines of output: the
model is still learning, but only things that are true of the training
set. Add `nn.NewDropout(0.2)` between the layers and see whether it
helps.

Then `-limit 1000 -epochs 5`, one image in sixty: the MLP gets 87.7 % and
the CNN 86.5 %, which looks like the convolution being no help at all.
Raise it to `-epochs 40` and the order reverses — 89.2 % against 93.4 %.
A thousand images and five epochs is forty optimiser steps, and the
convolutional model has not finished starting. The lesson is worth more
than the digits: comparing two architectures at a fixed epoch count
compares how fast they converge as much as what they can reach, and the
smaller model is usually the slower starter.

Finally, change the filter counts from 16 and 32 to 8 and 16 and compare
accuracy against training time. The curve is flatter than you would
guess, which is the same point the parameter counts already made.
