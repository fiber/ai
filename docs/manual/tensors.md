# Tensors

A `*tensor.Tensor` is an n-dimensional float32 array. There is one element
type; the design keeps kernels small and fast. Tensors are values with
reference semantics: methods return new tensors, views share storage.

## Creating tensors

| Constructor | Result |
|---|---|
| `New(data, shape...)` | copy of `data` in the given shape |
| `FromSlice(data, shape...)` | wraps `data` without copying (writes to `data` are visible) |
| `Zeros(shape...)`, `Ones(shape...)`, `Full(v, shape...)` | filled tensors |
| `Scalar(v)` | 0-D tensor |
| `Eye(n)` | identity matrix |
| `Arange(start, stop, step)`, `Linspace(start, stop, n)` | 1-D ranges |
| `Rand`, `Randn`, `Uniform(lo, hi, shape...)` | random; `Seed(s)` makes them reproducible, `*From(r, ...)` takes your own `*rand.Rand` |
| `OneHot(indices, classes)` | `[len(indices), classes]` |
| `ZerosLike(t)`, `OnesLike(t)` | same shape as `t` |

Shapes are `tensor.Shape` (`[]int`). A shape of `[]` is a scalar with one
element; a dimension of size 0 gives an empty tensor.

## Inspecting

```go
t.Shape()          // Shape, a copy
t.Dims()           // number of dimensions
t.Size()           // number of elements
t.Dim(-1)          // size of a dimension, negative counts from the end
t.At(i, j)         // one element; negative indices allowed
t.Item()           // the value of a single-element tensor
t.Data()           // []float32 in row-major order: the storage itself if contiguous (pins it, see below), otherwise a copy
t.Float32s()       // always a fresh copy
t.IsContiguous()   // dense row-major storage?
t.Strides()        // element strides per dimension
```

`Data()` on a contiguous tensor returns the backing slice, so writes
through it are visible in the tensor. Because the caller may keep that
slice, the tensor's storage is excluded from buffer reuse from then on
and, for results of 128 KiB and more, stays mapped for the life of the
process (see [performance.md](performance.md#off-heap-results)); read
with `Float32s()` or `At` when you do not need the live slice.

`fmt.Println(t)` prints a NumPy-style nested layout; dimensions longer
than `tensor.PrintOptions.Threshold` (8) are elided to the first and last
`EdgeItems` (3). `%#v` shows shape, strides and autograd flags.

## Element-wise arithmetic and broadcasting

`Add Sub Mul Div Maximum Minimum` take another tensor and broadcast with
NumPy rules: dimensions are aligned from the right, a size-1 dimension
stretches, missing leading dimensions are added.

```go
a := tensor.Randn(4, 3)
a.Add(tensor.Randn(3))       // row vector added to every row
a.Mul(tensor.Randn(4, 1))    // column vector scales every row
a.Sub(tensor.Scalar(1))      // scalar tensor
a.Add(tensor.Randn(2, 1, 3)) // result [2 4 3]
```

Scalar variants avoid allocating a tensor: `AddScalar SubScalar MulScalar
DivScalar`. Unary functions: `Neg Exp Log Sqrt Square Abs Pow(p) Tanh
Sigmoid ReLU GELU Clamp(lo, hi)`. `Dropout(p)` zeroes each element with
probability `p` and scales the rest by `1/(1-p)`; the mask is drawn from
the package random source (`tensor.Seed`) and used again in the backward
pass.

## Reductions

`Sum Mean Max Min Var Std` take zero or more dimensions. With none they
reduce everything to a 0-D tensor; with dimensions those dimensions are
removed (use `Unsqueeze` to keep them for broadcasting).

```go
x := tensor.Arange(0, 24, 1).Reshape(2, 3, 4)
x.Sum()          // 276 (0-D)
x.Sum(0)         // [3 4]
x.Mean(-1)       // [2 3]
x.Max(0, 2)      // [3]
x.Argmax(1)      // []int of length 2*4, index of the max along dim 1
```

## Matrix products

`MatMul` follows NumPy: 2-D × 2-D is the matrix product; a 1-D operand is
treated as a row (left) or column (right) vector and the extra dimension is
dropped; with more dimensions the leading batch dimensions broadcast.
`Dot` and `Outer` are the 1-D conveniences.

```go
w := tensor.Randn(8, 4)
x.Reshape(6, 4).MatMul(w.T())          // [6 8]; the transpose is a view, nothing is copied
tensor.Randn(3, 2, 5, 4).MatMul(w.T()) // [3 2 5 8] (batched)
```

## Views

These return tensors sharing the same storage, in O(1):

| Method | Effect |
|---|---|
| `T()` | 2-D transpose |
| `Transpose(d0, d1)`, `Permute(dims...)` | reorder dimensions |
| `Reshape(shape...)` | new shape, one `-1` inferred; copies only if the tensor is not contiguous |
| `Flatten()` | 1-D |
| `Squeeze(dims...)`, `Unsqueeze(dim)` | remove / insert size-1 dimensions |
| `Expand(shape...)` | broadcast size-1 dimensions (stride 0), `-1` keeps a size |
| `Narrow(dim, start, len)`, `Slice(dim, start, end)`, `Select(dim, i)`, `Row(i)` | sub-ranges |

`Contiguous()` returns the tensor itself when it is dense and a dense copy
otherwise; `Clone()` always copies. `Cat(dim, ts...)` and `Stack(dim,
ts...)` join tensors (these copy).

Operations accept views directly: `MatMul` reads strided operands,
element-wise operations iterate with strides, reductions call
`Contiguous()` first. Writing through a view (`t.Narrow(0, 2, 1).Zero()`)
modifies the underlying tensor.

## In-place operations

`t.Release()` hands a tensor's storage back for immediate reuse when you
know the result is dead; see
[performance.md](performance.md#off-heap-results). It refuses silently
when a view, a `Data()` slice or autograd may still need the storage.

`Fill Zero CopyFrom AddInPlace SubInPlace MulInPlace DivInPlace
MulScalarInPlace AddScaledInPlace(u, alpha)` modify the tensor. They are
not recorded in the autograd graph and refuse to run on a tensor that
requires grad unless inside `NoGrad` (see [autograd.md](autograd.md)).

## Errors

Shape and argument errors are programming errors and panic with a
`*tensor.Error` that names the operation:

```
tensor: MatMul: shape mismatch [2 3] · [2 3]
```

Where an `error` value is preferable, wrap the code:

```go
err := tensor.Try(func() { out = a.MatMul(b) })
```

Panics that are not `*tensor.Error` pass through `Try` unchanged.
