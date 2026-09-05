# 1. A tensor is a slice with a shape

If you can use a `[]float32`, you can use a tensor. A tensor is a slice
of numbers plus a *shape* that says how to read them: six numbers with
shape `[2 3]` are two rows of three. That is all the word means here.
Everything a model does, from a temperature table to a language model's
weights, is arithmetic on such slices, and the shape is what keeps the
arithmetic honest.

Run the chapter's program:

```
go run ./examples/tutorial/01-tensors
```

## Making one

```go
temps := []float32{18.5, 21.0, 19.5, 22.0, 24.5, 23.0}
t := tensor.New(temps, 2, 3)
```

`New` copies the slice and attaches the shape. Printing shows the two
rows of three:

```
t =
[[18.5   21 19.5]
 [  22 24.5   23]]
shape: [2 3]  elements: 6  dims: 2
```

`Shape()` returns the dimensions, `Size()` the number of elements,
`Dims()` how many dimensions there are (a vector has one, a matrix two,
a batch of images four). `At(1, 2)` reads one element, `Row(0)` one row,
`Select(1, 1)` one column:

```
t[1,2]  = 23
row 0   = [18.5   21 19.5]
column 1 = [  21 24.5]
```

## Views: the same numbers, read differently

This is the one idea that is different from ordinary slices of slices.
`Reshape`, `T` (transpose) and `Narrow` do not copy anything. They return
a *view*: a new tensor object that reads the same storage with a
different shape or a different stride between elements.

```go
v := t.Reshape(3, 2)   // the six numbers as three rows of two
t.T()                  // rows and columns swapped
```

```
t.Reshape(3, 2) =
[[18.5   21]
 [19.5   22]
 [24.5   23]]
t.T() (rows and columns swapped, still no copy) =
[[18.5   22]
 [  21 24.5]
 [19.5   23]]
```

Because views share storage, a write through one is visible in all of
them, exactly like two slices over the same array:

```
after v.Set(100, 0, 0): t[0,0] = 100
```

Views are why `x.MatMul(w.T())` costs nothing extra for the transpose,
and why a model can hand out a 4 GB weight matrix in a hundred ways
without copying it once. When you do want your own copy, `Float32s()`
gives you a fresh slice; `Data()` gives you the backing slice itself,
which is faster but ties the storage to your code for good (the
[performance chapter of the manual](../manual/performance.md#off-heap-results)
explains why).

```
Float32s() copy changed, t[0,0] still 100
```

## Constructors

```go
tensor.Zeros(2, 2)          // all zero
tensor.Ones(3)              // all one
tensor.Full(0.5, 2, 3)      // all 0.5
tensor.Arange(0, 5, 1)      // 0 1 2 3 4
tensor.Randn(2, 3)          // normal noise: mean 0, spread 1
tensor.Rand(2, 3)           // uniform in [0, 1)
```

`Randn` is the one you will use most, because every model starts from
random weights. `tensor.Seed(n)` makes the randomness repeatable, which
you want while you are learning: the same program should print the same
numbers twice.

```
Randn(2, 3) (normal noise, mean 0, spread 1) =
[[0.8107 -1.302 -1.181]
 [ -1.22  1.679  1.753]]
```

## Mistakes are panics, with a message

Shapes that do not fit are programming errors, so the library panics
with a message that says what did not fit. When you would rather have an
`error`, wrap the call in `tensor.Try`:

```go
err := tensor.Try(func() { tensor.New(temps, 4, 2) })
// tensor: New: 6 elements do not fit shape [4 2]
```

## What to remember

- A tensor is numbers plus a shape. `Shape`, `Size`, `At`, `Row`.
- `Reshape`, `T`, `Narrow`, `Select` are views: free, and shared.
- `Float32s` copies, `Data` does not.
- Start every experiment with `tensor.Seed`.

Next: [2. Broadcasting](02-broadcasting.md), where one line of code
replaces the loop you were about to write.
