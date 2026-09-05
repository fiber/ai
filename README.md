# fiber/ai

A fast, modern AI framework for Go. No cgo, no dependencies outside the
standard library, hand-written SIMD kernels.

This is the foundation layer: a float32 tensor library with a cache-blocked
SIMD GEMM, NumPy-style broadcasting, zero-copy views, reverse-mode autograd,
neural-network modules and optimisers.

```
github.com/fiber/ai
├── tensor/            Tensor, broadcasting, views, reductions, MatMul, autograd, fused NN ops
├── nn/                Linear, activations, Dropout, LayerNorm, Embedding, Sequential
├── optim/             SGD (momentum, Nesterov, weight decay), Adam / AdamW
├── internal/kernel/   float32 kernels: NEON, AVX2, AVX-512 assembly + Go fallback, CPU dispatch
├── internal/blas/     cache-blocked, packed, multi-threaded SGEMM
├── internal/parallel/ goroutine work distribution
├── cmd/bench/         throughput benchmarks (Markdown output)
├── cmd/gate/          development-process gate (specs, lists, manual)
├── docs/manual/       the user manual
├── benchmarks/        results and the NumPy/PyTorch comparison script
└── examples/          tensor basics, autograd, MLP regression, spiral classification
```

The [manual](docs/manual/README.md) covers the API in depth;
[PROCESS.md](PROCESS.md) describes how changes are made.

## Quick start

```go
import "github.com/fiber/ai/tensor"

x := tensor.Randn(64, 32)                             // [64 32]
w := tensor.Randn(32, 8).SetRequiresGrad(true)
b := tensor.Zeros(8).SetRequiresGrad(true)

y := x.MatMul(w).Add(b).Tanh()                        // broadcasting bias add
loss := y.Square().Mean()
loss.Backward()                                       // w.Grad(), b.Grad() populated

tensor.NoGrad(func() {                                // plain SGD step
    w.AddScaledInPlace(w.Grad(), -0.1)
    b.AddScaledInPlace(b.Grad(), -0.1)
})
```

With modules and an optimiser:

```go
model := nn.Sequential{
    nn.NewLinear(784, 512), nn.ReLU{},
    nn.NewLinear(512, 10),
}
opt := optim.NewAdam(model.Params(), 1e-3)

loss := tensor.CrossEntropy(model.Forward(x), labels) // fused log-softmax + NLL
opt.ZeroGrad()
loss.Backward()
opt.Step()
```

Run the examples:

```sh
go run ./examples/tensor/basics     # API tour
go run ./examples/tensor/autograd   # XOR with hand-written SGD
go run ./examples/nn/sine           # MLP time-series regression, prints samples/s
go run ./examples/nn/spiral         # 3-class spiral classification with CrossEntropy
go run ./cmd/bench                  # GEMM / element-wise / reduction / MLP throughput
```

## The tensor API

| Area | Operations |
|---|---|
| Construction | `New`, `FromSlice` (zero-copy), `Zeros`, `Ones`, `Full`, `Scalar`, `Eye`, `Arange`, `Linspace`, `Rand`, `Randn`, `Uniform`, `OneHot`, `*Like` |
| Element-wise | `Add Sub Mul Div Maximum Minimum` (broadcasting), `AddScalar MulScalar …`, `Neg Exp Log Sqrt Square Abs Pow Tanh Sigmoid ReLU GELU Clamp` |
| Reductions | `Sum Mean Max Min Var Std` over any dimensions, `Argmax` |
| Linear algebra | `MatMul` (2-D, batched with broadcasting, vector forms), `Dot`, `Outer` |
| Views (no copy) | `T Transpose Permute Reshape Squeeze Unsqueeze Expand Narrow Slice Select Row`, `Cat`, `Stack` |
| NN primitives | `Softmax LogSoftmax` (any dim), `CrossEntropy`, `MSELoss`, `LayerNorm` — fused forward and backward |
| In-place | `Fill Zero CopyFrom AddInPlace … AddScaledInPlace` (guarded against corrupting the graph) |
| Autograd | `SetRequiresGrad Backward BackwardWith Grad ZeroGrad Detach RetainGrad NoGrad` |

Shape or argument errors panic with a `*tensor.Error`; `tensor.Try` turns
that into an `error` where you prefer one. `tensor.SetThreads` limits the
goroutines used, `tensor.Backend()` names the active SIMD implementation.

## How it is fast

**Kernels.** Every hot loop (`add/sub/mul/div/max`, scalar broadcasts,
`axpy`, `dot`, `sum`, `max`, a vectorised `exp` and the GEMM micro-kernel)
exists as Go assembly for NEON (arm64), AVX2+FMA and AVX-512F (amd64), plus
a portable Go version. The Go assembler has no mnemonics for the NEON
floating-point vector arithmetic, so those instructions are emitted as
verified raw encodings. `exp` uses Cody–Waite range reduction and a
degree-6 polynomial (≈1 ulp) and runs 16× faster than `math.Exp`; it drives
`Softmax`, `CrossEntropy` and `Sigmoid`.
The best implementation is chosen at start-up via CPUID and — before it is
activated — verified against the Go version on random data. A kernel that
disagrees is dropped and reported in `tensor.BackendWarnings()`.

**GEMM.** `internal/blas` is a Goto/BLIS-style blocked SGEMM: B is packed
into NR-wide panels that stay in L2, A into MR-wide panels that stream
through L1, and the SIMD micro-kernel accumulates an MR×NR tile of C in
registers. Tiles are 8×12 (NEON, 24 accumulators), 6×16 (AVX2) and 14×32
(AVX-512, 28 accumulators). Packing removes all strides from the inner loop, so transposed or
otherwise strided operands cost nothing extra — `x.MatMul(w.T())` never
copies. Work is distributed over goroutines as a 2-D grid of (row block ×
column panel range) tasks with dynamic scheduling, which keeps all cores
busy on both tall and wide products and on heterogeneous (P/E) cores.
Matrix-vector shapes take dedicated `dot`/`axpy` paths.

On an Apple M2 Pro the NEON path reaches ~98 GFLOPS on one core (≈ 87 % of
the core's FMA peak) and ~600 GFLOPS on all ten cores. See
[BENCHMARKS.md](BENCHMARKS.md) for the full tables and the comparison with
NumPy and PyTorch.

**Parallelism.** Element-wise operations, reductions, packing and the GEMM
tile grid all run through `internal/parallel`, which hands out work items
from an atomic counter to up to `GOMAXPROCS` goroutines (the caller
participates). Small problems stay on the calling goroutine.

**Autograd.** A tensor is a slice, a shape, strides and — if it was produced
from something requiring grad — one node with a backward closure. Nothing is
recorded under `NoGrad` or when no input requires grad, so inference has no
autograd overhead. Backward closures capture detached inputs, run in
topological order and release intermediate gradients as soon as they have
been propagated. Broadcast gradients are reduced with `sumTo`; view
gradients are routed through the inverse view.

## Verify

```sh
go test ./...                     # kernels vs Go reference on every available ISA, GEMM vs float64,
                                  # broadcasting vs a naive reference, finite-difference gradient checks
go test -race ./...
go vet ./...                      # also checks assembly frame sizes and argument offsets
GOARCH=amd64 go vet ./...         # assembles the AVX2 / AVX-512 kernels
FIBERAI_KERNEL=generic go test ./...
go test ./internal/blas -bench Gemm -run x
```

On an Apple Silicon Mac the amd64 AVX2 kernels can be executed under
Rosetta 2, which hides AVX from CPUID; bypass detection with
`FIBERAI_KERNEL=avx2 FIBERAI_KERNEL_FORCE=1 GOARCH=amd64 go test ./...`.

### Status of the SIMD back-ends

| ISA | Kernels | Verified |
|---|---|---|
| NEON (arm64) | all + exp + 8×12 GEMM | natively on Apple M2 Pro |
| AVX2 + FMA (amd64) | all + exp + 6×16 GEMM | under Rosetta 2 (all tests pass) |
| AVX-512F (amd64) | 14×32 GEMM (vector ops use AVX2, they are memory-bound) | on a Xeon Gold 6130 (Skylake-SP): 1 242 GFLOPS at n=1024 on 16 cores, see BENCHMARKS.md |
| Apple AMX (macOS arm64, opt-in `FIBERAI_AMX=1`) | 32×32 GEMM on the matrix coprocessor | on an Apple M2 Pro: 2 250 GFLOPS at n=2048, level with PyTorch/Accelerate; training step 146 K samples/s vs 116 K |

## Roadmap

1. Fused GEMM epilogues (`C = A·B + bias`, optionally through an activation).
2. Skip B packing for small M (batch-1 inference) — read B in place.
3. Vectorised `tanh`/`log` (exp is done) — GELU, Tanh and Log are still
   `math.*` per element.
4. Conv1D/Conv2D via im2col + GEMM, multi-head attention, more modules.
5. Quantised (int8 / bf16) dot products with SDOT / VNNI for inference.
6. float64 / bf16 tensors.

## Development notes

- `orig/` contains the earlier `vektor` prototype this work builds on. It is
  a separate module and not compiled.
- Blocking parameters live in `internal/blas/params_*.go`;
  `go test ./internal/blas -run TestTune` style sweeps are easy to add.
