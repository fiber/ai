// Tutorial chapter 4: linear regression by hand, with plain tensors.
package main

import (
	"fmt"

	"github.com/fiber/ai/tensor"
)

func main() {
	tensor.Seed(4)
	// The truth we pretend not to know: y = 2·x₁ − 3·x₂ + 0.5, plus noise.
	// 200 examples with two inputs each.
	n := 200
	x := tensor.Randn(n, 2)
	trueW := tensor.New([]float32{2, -3}, 2, 1)
	y := x.MatMul(trueW).AddScalar(0.5).Add(tensor.Randn(n, 1).MulScalar(0.1))

	// The model: the same shape as the truth, but starting from zero.
	w := tensor.Zeros(2, 1).SetRequiresGrad(true)
	b := tensor.Zeros(1).SetRequiresGrad(true)
	lr := float32(0.1)

	for step := 0; step <= 60; step++ {
		pred := x.MatMul(w).Add(b)          // [200×1]: the model's guess for every example
		loss := tensor.MSELoss(pred, y)     // mean of (guess − truth)²
		w.ZeroGrad()
		b.ZeroGrad()
		loss.Backward()                     // gradients for w and b
		tensor.NoGrad(func() {              // the update is not part of the model
			w.AddScaledInPlace(w.Grad(), -lr)
			b.AddScaledInPlace(b.Grad(), -lr)
		})
		if step%10 == 0 {
			fmt.Printf("step %2d  loss %.4f  w = [%.3f %.3f]  b = %.3f\n",
				step, loss.Item(), w.At(0, 0), w.At(1, 0), b.At(0))
		}
	}
	fmt.Println("truth:           w = [2.000 -3.000]  b = 0.500")

	// A prediction for a new example, without recording anything.
	tensor.NoGrad(func() {
		newX := tensor.New([]float32{1, 1}, 1, 2)
		fmt.Printf("\nprediction for x = [1 1]: %.3f (truth 2 - 3 + 0.5 = -0.5)\n", newX.MatMul(w).Add(b).Item())
	})
}
