// Tutorial chapter 7: a model in service — saving and loading parameters,
// predicting without a graph, releasing results, thread settings and the
// latency of a single request.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

func newModel() nn.Sequential {
	return nn.Sequential{nn.NewLinear(16, 128), nn.ReLU{}, nn.NewLinear(128, 128), nn.ReLU{}, nn.NewLinear(128, 4)}
}

func main() {
	tensor.Seed(7)
	// Train something small so there is a model worth keeping: 16 inputs,
	// 4 classes, the label is the quadrant of the first two inputs.
	x := tensor.Randn(4096, 16)
	labels := make([]int, 4096)
	for i := range labels {
		if x.At(i, 0) > 0 {
			labels[i] = 1
		}
		if x.At(i, 1) > 0 {
			labels[i] += 2
		}
	}
	model := newModel()
	opt := optim.NewAdam(model.Params(), 2e-3)
	for epoch := 0; epoch < 30; epoch++ {
		loss := tensor.CrossEntropy(model.Forward(x), labels)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}

	// Save: the parameters, nothing else. The architecture lives in code.
	path := filepath.Join(os.TempDir(), "tutorial-07.fiberai")
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	if err := nn.SaveParams(f, model); err != nil {
		panic(err)
	}
	f.Close()
	info, _ := os.Stat(path)
	fmt.Printf("saved %d parameters, %d bytes\n", len(model.Params()), info.Size())

	// Load into a fresh model of the same architecture, as a service would
	// at start-up.
	served := newModel()
	f, err = os.Open(path)
	if err != nil {
		panic(err)
	}
	if err := nn.LoadParams(f, served); err != nil {
		panic(err)
	}
	f.Close()

	// Serving: NoGrad (no graph is recorded, nothing kept for a Backward
	// that never comes), one request at a time, and Release on the result
	// once it has been read so the next request reuses the same memory.
	served.SetTraining(false)
	request := x.Narrow(0, 0, 1)
	predict := func(in *tensor.Tensor) int {
		var class int
		tensor.NoGrad(func() {
			out := served.Forward(in)
			class = out.Argmax(1)[0]
			out.Release()
		})
		return class
	}
	fmt.Printf("first request: class %d (label %d)\n", predict(request), labels[0])

	// Latency per request: a single example through a small model is a
	// question of microseconds, and it depends on how many threads the
	// library is allowed to wake up. For tiny inputs, fewer is faster.
	for _, threads := range []int{tensor.Threads(), 4, 1} {
		tensor.SetThreads(threads)
		predict(request) // warm up
		const n = 2000
		start := time.Now()
		for i := 0; i < n; i++ {
			predict(request)
		}
		fmt.Printf("threads %2d: %6.1f µs per request\n", threads, float64(time.Since(start).Microseconds())/n)
	}

	// Batching amortises everything: 256 requests at once cost little more
	// than one.
	tensor.SetThreads(0)
	batch := x.Narrow(0, 0, 256)
	start := time.Now()
	for i := 0; i < 200; i++ {
		predict(batch)
	}
	per := float64(time.Since(start).Microseconds()) / 200
	fmt.Printf("batch of 256: %6.1f µs per batch, %.2f µs per example\n", per, per/256)
	os.Remove(path)
}
