# 5. One unit, and what it cannot do

Chapter 4 fitted a line to numbers. This chapter uses the same idea to
make a *decision* — and then runs into the wall that shaped the next
thirty years of the field.

```
go run ./examples/tutorial/05-perceptron
```

## The unit

A perceptron is one weighted sum and a threshold. Two inputs, two
weights, a bias; if the sum is positive it answers 1, otherwise 0.

```go
func (p *perceptron) predict(x []float32) float32 {
    if p.w0*x[0]+p.w1*x[1]+p.b > 0 {
        return 1
    }
    return 0
}
```

That is the whole model. Frank Rosenblatt built it in 1957, and the
reason to start here is not history: it is that the unit is small enough
to hold in your head, so when it fails you can see exactly why.

## Learning without gradients

Chapter 3 introduced the gradient, and it is easy to come away thinking
that learning *is* gradient descent. It is not. The perceptron has its
own rule, a decade older:

```go
err := y[i] - p.predict(x)     // 0 if right, +1 or -1 if wrong
p.w0 += rate * err * x[0]
p.w1 += rate * err * x[1]
p.b  += rate * err
```

Read it in words: when the answer is wrong, move the weights toward the
input that was misjudged. There is no loss function, nothing is
differentiated, and no chain rule appears. It also comes with a
guarantee gradient descent cannot offer — if the data *can* be separated
by a line, this rule will find one, in finite time.

```
AND: separated after 6 epochs
OR:  separated after 4 epochs
```

Six passes over four points. The guarantee is real.

## Then XOR

Exclusive-or: answer 1 when exactly one input is 1.

```
XOR: still wrong after 1000 epochs
  XOR  (0,0)->1 want 0  (0,1)->1 want 1  (1,0)->0 want 1  (1,1)->0 want 0
```

Not slow. Not badly tuned. **Impossible.** Change the learning rate,
the initialisation, run it for a million epochs: it will never separate
XOR, and the guarantee above is why we can say so without trying.

The reason is in the one line of code that makes the decision. The unit
asks whether `w0*x0 + w1*x1 + b` is positive — and the set of points
where that flips from negative to positive is a straight line. One side
answers 1, the other 0. So plot the four points and ask where you would
draw it:

```
  1 |  (0,1)=1      (1,1)=0
    |
  0 |  (0,0)=0      (1,0)=1
    +---------------------
         0             1
```

The two 1s are diagonally opposite, and so are the two 0s. No straight
line separates one diagonal from the other. The model is not
underpowered or undertrained — the shape it can express does not contain
the answer.

Minsky and Papert published this in 1969 and interest in the field
collapsed for a decade. The useful part is not the history: it is that
"my model is not learning" and "my model *cannot* learn this" look
exactly the same from the outside, and only one is fixed by tuning.

## One layer more

Give the network a hidden layer — two units that each draw their own
line, and an output unit that combines them — and train it the modern
way, with a loss and gradients:

```go
model := nn.Sequential{
    nn.NewLinear(2, hiddenUnits), nn.Tanh{},
    nn.NewLinear(hiddenUnits, 1),
}
```

```
1 hidden unit(s): loss 0.1674  outputs  0.34  0.33  1.00  0.33   want  0 1 1 0
2 hidden unit(s): loss 0.0000  outputs -0.00  1.00  1.00 -0.00   want  0 1 1 0
4 hidden unit(s): loss 0.0000  outputs  0.00  1.00  1.00  0.00   want  0 1 1 0
```

One hidden unit is still one line, and still fails. **Two solve it
exactly.** Each hidden unit draws a line; two lines carve the plane into
regions; the output unit says which regions mean 1.

Notice what had to be there for this to work. Not more units — the jump
from one to two matters and the jump from two to four does not. What
matters is that something *between* the layers is not a straight line.
`nn.Tanh` bends; without it, two linear layers collapse into one and you
are back to a single line however many units you stack. Chapter 6 states that as a fact about `ReLU`; here you can watch it
happen, and change one line to break it again.

## Why this matters when you are not doing XOR

You will not meet exclusive-or in production. You will meet its shape:
a model that trains, converges, and is wrong — where the loss stops
falling at some value well above zero and stays there no matter what you
change.

Three things look identical from the outside and need different fixes:

| symptom | cause | what to do |
|---|---|---|
| loss falls, then plateaus above zero, tuning changes nothing | the model cannot express the answer | more capacity: a layer, a nonlinearity, better features |
| loss falls on training data, rises on validation | too much capacity for the data | less capacity, more data, regularisation |
| loss jumps around or goes to NaN | learning rate or scaling | lower the rate, check the input scale |

This chapter is the first row, isolated so you can recognise it. The
tell is that **tuning does nothing**: the perceptron on XOR gives the
same answer at any learning rate, any initialisation, any number of
epochs. When you see a floor that ignores every knob, stop tuning and
change the model — the shape you have does not contain the answer you
want.

That is worth four data points to learn, because on a real problem the
floor is at some unremarkable number and nothing tells you whether you
are looking at row one or row two.

## What to try

Print the two hidden units' weights after training and work out which
line each one drew — they are usually the two diagonal boundaries you
would draw by hand. Then replace `nn.Tanh{}` with nothing at all, so the
network is two linear layers in a row, and watch the loss settle at
exactly where the single unit settled. That is a line of a line being a
line, measured.

Next: [6. The first classifier](06-classifier.md), where the same idea
meets data that is not four points.
