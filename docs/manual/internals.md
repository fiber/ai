# Internals

For contributors. Read [performance.md](performance.md) first if you only
want to use the library well.

## Layers

```
tensor  ──►  internal/blas  ──►  internal/kernel  ──►  CPU
   │              │                    ▲
   └──────────────┴────────────────────┘
                internal/parallel
```

- `internal/parallel` — `For(n, fn)` hands work items to up to
  `Workers()` goroutines from an atomic counter. `Range(n, minChunk, fn)`
  splits the range into up to four chunks per worker and claims them by
  per-chunk CAS, owner first: worker w (the caller is 0, helper k is k+1)
  takes chunks w, w+workers, … before stealing what is left, so repeated
  operations on the same tensors put the same chunk on the same
  goroutine, and the data a core wrote last time is in its own cache.
  A persistent pool of helpers picks up jobs published through an atomic
  generation counter, spins ~200 µs between jobs before parking on a
  condition variable, and never blocks the caller waiting for a helper to
  start (nested calls are safe). Panics stop the job and are re-raised
  in the caller.
- `internal/kernel` — float32 kernels on contiguous slices, one
  implementation table per ISA (`impl` struct), selected in `init`.
- `internal/blas` — `Gemm(c, a, b Mat)` on strided `Mat` views.
- `tensor` — everything user-facing.

## Kernels

Each kernel exists in `generic.go` (portable Go, also the reference) and in
assembly: `kernel_neon_arm64.s`, `kernel_avx2_amd64.s`,
`kernel_avx512_amd64.s`. The Go side declares the assembly functions with
pointer+length signatures and wraps them into slice-based functions
(`wrapBinary`, `wrapScalar`, ...) that validate lengths.

`verify(impl)` runs at init and compares every kernel of a candidate
implementation with the generic one on lengths that exercise all vector
and scalar tail paths; a mismatch disables the implementation. The same
comparison runs in `go test` for every implementation the machine can
execute (`implementations()` in `kernel_test.go`).

Conventions in the assembly:

- Element-wise kernels process 4 vectors per iteration, then one vector,
  then a scalar tail (NEON: 16/4/1 floats, AVX2: 32/8/1).
- Reductions keep four accumulators and reduce them at the end; the scalar
  tail runs after the horizontal reduction because VEX scalar instructions
  zero the upper lanes.
- The Go assembler has no mnemonics for NEON floating-point vector
  arithmetic (`FADD`, `FMUL`, ...). They are emitted as `WORD` encodings
  through macros at the top of `kernel_neon_arm64.s`; verify any new
  encoding with `go tool objdump`.
- Multi-line macros are defined before the first `TEXT` because
  `go vet`'s asmdecl attributes macro lines to the preceding function.

## GEMM

`internal/blas/gemm.go` is a Goto/BLIS blocked SGEMM:

```
for jc in N step NC:
  for pc in K step KC:
    round 1 (all workers): pack B[pc.., jc..] into NR-wide panels and
                           every A[.., pc..] row panel into one shared buffer
    round 2 (all workers): grid of (row block × panel range) compute tasks,
                           ~16 per worker, handed out from an atomic counter;
                           each task streams its panels against its L2-hot A block
```

Packing happens once per K block and never inside a compute task; a
fine task grid keeps the tail of each round short. The rounds run on
`internal/parallel`'s persistent helpers, which spin on a generation
counter between rounds instead of parking, so the two barriers per K
block cost no thread wake-ups. `FIBERAI_BLAS_KC/MC/NC`,
`FIBERAI_BLAS_TASKS` and `FIBERAI_BLAS_STRATEGY=rows` (one task per row
block with private A packing) exist for experiments.

The micro-kernel (`kernel.Gemm`) receives packed panels in k-major order
and accumulates the tile in registers: 8×12 on NEON (24 accumulators),
6×16 on AVX2, 14×32 on AVX-512 (12×32 selectable as `FIBERAI_KERNEL=avx512x12`). Edge tiles are computed into a scratch
tile and added into `C`. Blocking parameters live in `params_<arch>.go`.
Matrix–vector shapes take `dot`/`axpy` paths before packing.

Tests compare against a float64 reference for many shapes, all three
operand layouts (row-major, transposed, oddly strided), block-boundary
sizes and worker counts.

## Adding an operation to `tensor`

1. Compute the forward into a new tensor with `unaryOp`, `binaryOp` or
   your own loop over `Data()`; parallelise with `parallel.Range`.
2. Wrap it in `record(out, "Name", inputs, backward)`. The backward
   closure receives the output gradient and must call `in.accumGrad(g)`
   for each input; capture inputs as `x.Detach()` so no graph is recorded
   during backward. Use `sumTo(g, shape)` for broadcast inputs.
3. Add a gradient check in `tensor/autograd_test.go` (`checkGrad`) and a
   value test against a naive reference.
4. Document it in the manual (the process gate insists).

## Adding an architecture

1. Create `kernel_<arch>.go` with the assembly declarations, an `impl`
   value and `candidates()` / `allImpls()` (see `kernel_arm64.go`).
2. Write the assembly; start with the element-wise kernels, run
   `go test ./internal/kernel` until `TestVerifyAll` passes, then the GEMM
   micro-kernel (`TestGemmMicroKernel`).
3. Add `internal/blas/params_<arch>.go` with blocking parameters and tune
   them with a sweep such as the one described in `internal/blas`.
4. `go vet` checks frame sizes and argument offsets of the assembly against
   the Go declarations — keep both in sync.
