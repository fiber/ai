# 2. Broadcasting

Most of the arithmetic in a model is not "matrix times matrix". It is
"subtract this row from every row", "scale every column by its own
factor", "add the bias to each example". Broadcasting is the rule that
lets a small tensor act on a big one without you writing the loop, and
without the small one being copied to the big one's size.

```
go run ./examples/tutorial/02-broadcasting
```

## The situation

Five days of readings from three sensors. Rows are days, columns are
sensors, and the sensors have nothing in common in scale: temperature
around 20, pressure around 1000, humidity around 40.

```
x (days × sensors) =
[[20.1 1013   41]
 [21.4 1009   45]
 [19.8 1021   38]
 [  23 1004   52]
 [22.2 1011   47]]
```

## A row acts on every row

```go
offset := tensor.New([]float32{20, 1000, 40}, 3)
x.Sub(offset)
```

The shapes are `[5 3]` and `[3]`. The row is applied to each of the five
days:

```
x - offset =
[[ 0.1   13    1]
 [ 1.4    9    5]
 [-0.2   21   -2]
 [   3    4   12]
 [ 2.2   11    7]]
```

Nothing was copied. The kernel reads the row five times from the same
three numbers, in one parallel pass.

## A column acts on every column

A `[5 1]` tensor, five rows of one, is applied across the three sensors
instead:

```go
weight := tensor.New([]float32{1, 1, 1, 0.5, 0.5}, 5, 1)
x.Mul(weight)
```

```
x * weight (a column) =
[[ 20.1  1013    41]
 [ 21.4  1009    45]
 [ 19.8  1021    38]
 [ 11.5   502    26]
 [ 11.1 505.5  23.5]]
```

## The rule

Compare the two shapes from the right. In each position the sizes must
be equal, or one of them must be 1 (that one is stretched); a dimension
that is missing counts as 1.

| left | right | result |
|---|---|---|
| `[5 3]` | `[3]` | `[5 3]`, the row repeats down |
| `[5 3]` | `[5 1]` | `[5 3]`, the column repeats across |
| `[5 3]` | `[5]` | error: 3 and 5 do not match |
| `[4 1 3]` | `[2 3]` | `[4 2 3]` |

```
x + Ones(5): tensor: Add: shapes [5 3] and [5] are not broadcastable
x + Ones(5, 1) works, shape [5 3]
```

The error case is the one to remember: a vector of five is a *row* of
five, not a column. When you mean a column, say so with `Reshape(5, 1)`
or `Unsqueeze(1)`.

## The one use you will need in every project: standardising

A model given raw sensor values learns mostly about pressure, because
its numbers are fifty times bigger than the others. Before training,
every column is shifted to mean 0 and scaled to spread 1. With
broadcasting this is one line, and it is the same line for three columns
or three thousand:

```go
mean := x.Mean(0)   // one value per column, shape [3]
std := x.Std(0)
z := x.Sub(mean).Div(std)
```

```
mean per sensor = [21.3 1012 44.6]
std  per sensor = [1.217 5.571 4.841]
standardised =
[[-0.9864  0.2513 -0.7436]
 [ 0.0822 -0.4667 0.08262]
 [ -1.233   1.687  -1.363]
 [  1.397  -1.364   1.528]
 [ 0.7398 -0.1077  0.4957]]
check: column means = [-9.298e-07  -6.57e-06 -4.888e-07]  column stds = [1 1 1]
```

The column means come out at a millionth rather than exactly zero. That
is float32 arithmetic, and it is fine: models never depend on the last
digits.

Keep `mean` and `std`. Whatever you feed the model later has to be
shifted and scaled with the *same* numbers, or the model sees a world
it was never trained on.

## Scalars

`AddScalar`, `MulScalar`, `SubScalar`, `DivScalar` take a plain
`float32` and avoid building a tensor for it:

```go
celsius.MulScalar(1.8).AddScalar(32)
```

## What to remember

- Shapes are compared from the right; a 1 stretches.
- A `[3]` is a row. A column is `[3 1]`.
- Standardise inputs with `Mean(0)` and `Std(0)`, and keep both numbers.

Next: [3. A gradient without formulas](03-gradient.md).
