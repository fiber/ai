# Getting started

## Requirements

- Go 1.26 or newer.
- No C compiler, no cgo, no external libraries. The SIMD kernels are Go
  assembly and are selected at start-up; on any other architecture (or with
  `FIBERAI_KERNEL=generic`) the portable Go code runs.

Supported CPU back-ends: NEON (arm64, e.g. Apple Silicon, Graviton), AVX2
with FMA and AVX-512F (amd64).

## Install

```sh
go get github.com/fiber/ai
```

Packages:

| Import path | Purpose |
|---|---|
| `github.com/fiber/ai/tensor` | tensors, broadcasting, views, autograd, fused NN primitives |
| `github.com/fiber/ai/nn` | layers and activations as `Module`s |
| `github.com/fiber/ai/optim` | SGD, Adam, AdamW |

## First program

```go
package main

import (
	"fmt"

	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

func main() {
	tensor.Seed(1)
	x := tensor.Randn(256, 4)                    // 256 samples, 4 features
	y := x.MatMul(tensor.New([]float32{1, -2, 0.5, 3}, 4, 1)).AddScalar(0.25)

	model := nn.Sequential{nn.NewLinear(4, 16), nn.Tanh{}, nn.NewLinear(16, 1)}
	opt := optim.NewAdam(model.Params(), 1e-2)

	for step := 1; step <= 500; step++ {
		loss := tensor.MSELoss(model.Forward(x), y)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step%100 == 0 {
			fmt.Printf("step %d loss %.5f\n", step, loss.Item())
		}
	}
}
```

Three things to notice: operations are methods that chain
(`x.MatMul(w).AddScalar(0.25)`), a loss is an ordinary 0-D tensor whose
`Backward` fills the `Grad()` of every parameter, and the optimiser is the
only thing that mutates parameters.

## Running the examples

```sh
go run ./examples/tensor/basics     # tour of the tensor API
go run ./examples/tensor/autograd   # XOR with hand-written SGD, gradient inspection
go run ./examples/nn/sine           # MLP regression on a time series, prints samples/s
go run ./examples/nn/spiral         # 3-class classification with CrossEntropy
go run ./cmd/bench                  # throughput tables (see performance.md)
```

## Checking your machine

```go
fmt.Println(tensor.Backend())          // "neon", "avx512", "avx2" or "generic"
fmt.Println(tensor.Threads())          // goroutines used for parallel work
fmt.Println(tensor.BackendWarnings())  // non-empty only if a SIMD kernel failed self-verification
```
