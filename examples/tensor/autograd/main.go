// Example: reverse-mode autograd on a tiny two-layer network that learns
// XOR, trained with hand-written SGD. Shows gradient inspection, NoGrad
// parameter updates and in-place ops.
package main

import (
	"fmt"

	"github.com/fiber/ai/tensor"
)

func main() {
	tensor.Seed(42)

	x := tensor.New([]float32{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	y := tensor.New([]float32{0, 1, 1, 0}, 4, 1)

	w1 := tensor.Randn(2, 8).SetRequiresGrad(true)
	b1 := tensor.Zeros(8).SetRequiresGrad(true)
	w2 := tensor.Randn(8, 1).SetRequiresGrad(true)
	b2 := tensor.Zeros(1).SetRequiresGrad(true)
	params := []*tensor.Tensor{w1, b1, w2, b2}

	forward := func(x *tensor.Tensor) *tensor.Tensor {
		h := x.MatMul(w1).Add(b1).Tanh()
		return h.MatMul(w2).Add(b2).Sigmoid()
	}

	// the gradient of a scalar loss w.r.t. every parameter
	loss := tensor.MSELoss(forward(x), y)
	loss.Backward()
	fmt.Printf("initial loss %.4f\n", loss.Item())
	fmt.Println("dLoss/dw2 =")
	fmt.Println(w2.Grad())

	const lr = 0.5
	for step := 1; step <= 2000; step++ {
		for _, p := range params {
			p.ZeroGrad()
		}
		loss = tensor.MSELoss(forward(x), y)
		loss.Backward()
		// parameter updates must not be recorded in the graph
		tensor.NoGrad(func() {
			for _, p := range params {
				p.AddScaledInPlace(p.Grad(), -lr)
			}
		})
		if step%400 == 0 {
			fmt.Printf("step %4d  loss %.5f\n", step, loss.Item())
		}
	}

	var pred *tensor.Tensor
	tensor.NoGrad(func() { pred = forward(x) })
	fmt.Println("\ninput      target  prediction")
	for i := 0; i < 4; i++ {
		fmt.Printf("%v   %v     %.3f\n", x.Row(i).Data(), y.At(i, 0), pred.At(i, 0))
	}
}
