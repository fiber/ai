# 7. A model in service

A trained model is a set of numbers and the code that uses them. This
chapter is about the part after training: keeping the numbers, loading
them in a program that only predicts, and what a prediction costs.

```
go run ./examples/tutorial/07-service
```

## Saving and loading

The architecture is Go code; only the parameters go to disk:

```go
f, _ := os.Create("model.fiberai")
nn.SaveParams(f, model)
f.Close()
```

```
saved 6 parameters, 76888 bytes
```

Six parameters, because three `Linear` layers have a weight matrix and
a bias each; 76 kB, because 16·128 + 128 + 128·128 + 128 + 128·4 + 4
numbers of four bytes each. Loading reads them into a freshly built
model of the same shape:

```go
served := newModel()          // the same layers, random numbers
f, _ := os.Open("model.fiberai")
nn.LoadParams(f, served)      // now the trained numbers
```

`LoadParams` checks that count and shapes match and returns an error
that names the parameter if they do not, which is what you get when
the code has moved on and the file has not. Keep the function that
builds the model in one place and call it from both the trainer and
the service.

## Predicting

Three things belong in every prediction path:

```go
served.SetTraining(false)          // dropout off, once, at start-up
tensor.NoGrad(func() {             // record no graph
    out := served.Forward(in)
    class = out.Argmax(1)[0]
    out.Release()                  // hand the result's memory back at once
})
```

`NoGrad` is the important one: without it every prediction builds an
autograd graph that nobody will ever call `Backward` on. `Release`
returns the output's storage immediately instead of leaving it to the
garbage collector, so the next request computes into the same, still
cache-warm memory; it is safe to call on anything you are done with
(see the [performance chapter](../manual/performance.md#off-heap-results)).

```
first request: class 1 (label 1)
```

## What a request costs

```
threads 10:    6.0 µs per request
threads  4:    6.2 µs per request
threads  1:    6.1 µs per request
batch of 256:   90.9 µs per batch, 0.35 µs per example
```

Six microseconds for one example through a model with 20 000
parameters. The thread count makes no difference here because an input
this small never reaches the parallel paths; the library runs it inline.
`tensor.SetThreads(n)` matters for large batches and large models, where
it caps how many cores a single call may take; in a server that handles
requests concurrently you usually want it low so that requests do not
fight over cores, and `SetThreads(0)` restores the default.

The last line is the one to design around: 256 examples at once cost
15 times one example, not 256 times. Whenever requests can be collected
for a millisecond and answered together, do it.

## What to remember

- `SaveParams` and `LoadParams`; the architecture stays in code, built
  by one function used by both sides.
- `SetTraining(false)` once, `NoGrad` around every prediction,
  `Release` on results you have read.
- Batch when you can; measure with the real input size.

That is the end of the tutorial. The [application packages](../manual/applications.md)
show these pieces on log and counter data, and the
[manual](../manual/README.md) covers the API in full.
