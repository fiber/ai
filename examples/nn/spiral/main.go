// Example: classify a three-arm spiral (the classic CS231n toy problem)
// with an MLP trained by cross-entropy and Adam. Demonstrates CrossEntropy,
// Argmax-based accuracy and evaluation under NoGrad.
package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"time"

	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

const (
	classes   = 3
	perClass  = 300
	epochs    = 300
	batchSize = 64
)

// spiral generates the dataset: points on three interleaved spiral arms.
func spiral(r *rand.Rand) (*tensor.Tensor, []int) {
	n := classes * perClass
	x := tensor.Zeros(n, 2)
	labels := make([]int, n)
	xd := x.Data()
	for c := 0; c < classes; c++ {
		for i := 0; i < perClass; i++ {
			k := c*perClass + i
			radius := float64(i) / perClass
			theta := float64(c)*4 + radius*4 + r.NormFloat64()*0.2
			xd[k*2] = float32(radius * math.Sin(theta))
			xd[k*2+1] = float32(radius * math.Cos(theta))
			labels[k] = c
		}
	}
	return x, labels
}

func accuracy(model nn.Module, x *tensor.Tensor, labels []int) float64 {
	var correct int
	tensor.NoGrad(func() {
		pred := model.Forward(x).Argmax(1)
		for i, p := range pred {
			if p == labels[i] {
				correct++
			}
		}
	})
	return float64(correct) / float64(len(labels))
}

func main() {
	fmt.Printf("backend %s, %d threads\n", tensor.Backend(), tensor.Threads())
	tensor.Seed(7)
	r := rand.New(rand.NewPCG(7, 0))
	x, labels := spiral(r)
	n := x.Dim(0)

	model := nn.Sequential{
		nn.NewLinear(2, 64), nn.ReLU{},
		nn.NewLinear(64, 64), nn.ReLU{},
		nn.NewLinear(64, classes),
	}
	opt := optim.NewAdam(model.Params(), 3e-3)

	perm := r.Perm(n)
	start := time.Now()
	for epoch := 1; epoch <= epochs; epoch++ {
		r.Shuffle(n, func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
		var total float64
		batches := 0
		for lo := 0; lo < n; lo += batchSize {
			hi := min(lo+batchSize, n)
			rows := make([]*tensor.Tensor, 0, hi-lo)
			targets := make([]int, 0, hi-lo)
			for _, i := range perm[lo:hi] {
				rows = append(rows, x.Row(i))
				targets = append(targets, labels[i])
			}
			xb := tensor.Stack(0, rows...)
			loss := tensor.CrossEntropy(model.Forward(xb), targets)
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
			total += float64(loss.Item())
			batches++
		}
		if epoch%50 == 0 || epoch == 1 {
			fmt.Printf("epoch %3d  loss %.4f  accuracy %.1f%%\n", epoch, total/float64(batches), 100*accuracy(model, x, labels))
		}
	}
	fmt.Printf("trained in %.1fs, final accuracy %.1f%%\n", time.Since(start).Seconds(), 100*accuracy(model, x, labels))
}
