// Tutorial chapter 6: anatomy of a training loop — shuffling, batches,
// a validation split, and what overfitting looks like.
package main

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

// A small, noisy data set: 8 inputs, a target that depends on a curved
// combination of the first three, plus noise. Few examples on purpose,
// so that a big model can memorise them.
func makeData(r *rand.Rand, n int) (*tensor.Tensor, *tensor.Tensor) {
	x := tensor.RandnFrom(r, n, 8)
	xd := x.Data()
	y := make([]float32, n)
	for i := 0; i < n; i++ {
		a, b, c := float64(xd[i*8]), float64(xd[i*8+1]), float64(xd[i*8+2])
		y[i] = float32(math.Sin(a)*b + 0.5*c*c + r.NormFloat64()*0.3)
	}
	return x, tensor.New(y, n, 1)
}

func mse(m nn.Module, x, y *tensor.Tensor) float32 {
	var v float32
	tensor.NoGrad(func() { v = tensor.MSELoss(m.Forward(x), y).Item() })
	return v
}

func train(name string, model nn.Sequential, r *rand.Rand, xTrain, yTrain, xVal, yVal *tensor.Tensor, epochs int) {
	opt := optim.NewAdam(model.Params(), 1e-3)
	n := xTrain.Dim(0)
	perm := r.Perm(n)
	const batch = 32
	fmt.Printf("\n%s\n", name)
	for epoch := 1; epoch <= epochs; epoch++ {
		// A fresh order every epoch: the model must not learn the order.
		r.Shuffle(n, func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
		model.SetTraining(true)
		for lo := 0; lo < n; lo += batch {
			idx := perm[lo:min(lo+batch, n)]
			xb, yb := xTrain.Rows(idx), yTrain.Rows(idx) // one mini-batch
			loss := tensor.MSELoss(model.Forward(xb), yb)
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
		}
		model.SetTraining(false)
		if epoch%100 == 0 || epoch == epochs {
			fmt.Printf("epoch %4d  train %.3f  validation %.3f\n", epoch, mse(model, xTrain, yTrain), mse(model, xVal, yVal))
		}
	}
}

func main() {
	tensor.Seed(6)
	r := rand.New(rand.NewPCG(6, 0))
	// 240 examples, split by position: the first 200 train, the last 40
	// are never trained on and tell us how the model does on new data.
	x, y := makeData(r, 240)
	xTrain, yTrain := x.Narrow(0, 0, 200), y.Narrow(0, 0, 200)
	xVal, yVal := x.Narrow(0, 200, 40), y.Narrow(0, 200, 40)
	fmt.Println("noise floor (the loss a perfect model would still have): 0.090")

	big := nn.Sequential{nn.NewLinear(8, 256), nn.ReLU{}, nn.NewLinear(256, 256), nn.ReLU{}, nn.NewLinear(256, 1)}
	train("big model, no regularisation: memorises the 200 examples", big, r, xTrain, yTrain, xVal, yVal, 600)

	small := nn.Sequential{nn.NewLinear(8, 32), nn.ReLU{}, nn.NewLinear(32, 1)}
	train("small model: cannot memorise, generalises better", small, r, xTrain, yTrain, xVal, yVal, 600)

	dropout := nn.Sequential{nn.NewLinear(8, 256), nn.ReLU{}, nn.NewDropout(0.3), nn.NewLinear(256, 256), nn.ReLU{}, nn.NewDropout(0.3), nn.NewLinear(256, 1)}
	train("big model with dropout: noise during training keeps it honest", dropout, r, xTrain, yTrain, xVal, yVal, 600)
}
