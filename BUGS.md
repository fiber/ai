# BUGS

Open bugs. Record a bug here first; fixing it needs a `B-` spec in `spec/`
(see PROCESS.md). Fixed bugs move to BUGS-FIXED.md.

- 2026-09-12 — **MatMul and the other unguarded kernel call sites use
  off-heap storage after it has been freed.** Tensors of `mapMin` floats
  (64 KiB) and more are allocated with anonymous `mmap` outside the Go
  heap; their lifetime is governed by a weak pointer plus a cleanup that
  either returns the mapping to `mapPool` or `munmap`s it. `mat()` hands
  the raw `t.data` slice to BLAS and nothing then keeps the `*Tensor`
  reachable — a slice into off-heap memory keeps no Go object alive — so
  the cleanup can run while a kernel is reading the buffer.
  `matmul_fused.go`, `attention.go`, `rope.go`, `inplace.go`, `reduce.go`,
  `autograd.go` and `tensor.go` call `runtime.KeepAlive` for exactly this
  reason. `matmul.go` (7 kernel calls), `conv.go`, `nn.go` (37),
  `ops_unary.go` (18), `ops_binary.go` (11), `packcache.go` (3) and
  `view.go` (1) do not.

  What the pool does with the freed mapping decides the symptom:

  | `FIBERAI_MAPPED_LIMIT` | freed mapping | symptom |
  |---|---|---|
  | unset (512 MB retained) | retained and reused by another tensor | silent corruption |
  | `0` | unmapped at once | SIGSEGV |
  | `-1` | mapping off, all storage on the Go heap | correct |

  Reproducer, deterministic: a two-stage CNN backward under
  `FIBERAI_MAPPED_LIMIT=0` faults on every run in `blas.packRows4` via
  `packBPanel` from `tensor.matmul2D` (matmul.go:52 and the backward
  closure at matmul.go:58), `fatal error: fault`, `SIGSEGV code=0x2` at
  the base address of the source mapping. With the default retention the
  same model instead trains to slightly wrong gradients: forward is
  bit-identical between mapped and unmapped runs, the loss is identical,
  but the gradients of one training step differ by 0.175% in `sum|g|`
  (3563.45 against 3557.20 over 20490 elements, `max|g|` unchanged) —
  three orders of magnitude more than float32 rounding order accounts
  for. Single operations are unaffected: a lone Linear, a lone Conv2D and
  Conv2D+MaxPool2D are all bit-identical between the two configurations.
  It takes the allocation churn of a deeper model to make the pool recycle
  a buffer that is still in use.

  Consequence for the numbers: `examples/mnist` reaches 98.50% mean test
  accuracy over three seeds as it stands and 98.64% with
  `FIBERAI_MAPPED_LIMIT=-1`, against PyTorch's 98.73%. The deficit is a
  symptom of this bug, not a separate shortcoming of the convolution.

  Not at fault, checked: Adam (identical to PyTorch down to epsilon's
  place), the loss, padding/stride/pooling geometry, max-pool tie breaking
  (both take the first maximum), and initialisation (PyTorch run with He
  normal and zero bias via `benchmarks/python/mnist.py --init he` scores
  no worse than with its own default). Training is deterministic: three
  runs at one seed give 98.76% every time.
(none open)
