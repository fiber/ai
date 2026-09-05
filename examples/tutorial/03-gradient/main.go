// Tutorial chapter 3: a gradient without formulas.
package main

import (
	"fmt"

	"github.com/fiber/ai/tensor"
)

func main() {
	// y = 3x + 1. At x = 2, y is 7. If x grows a little, how fast does y
	// grow? Three times as fast: that number, 3, is the gradient of y with
	// respect to x. Backward computes it for us.
	x := tensor.Scalar(2).SetRequiresGrad(true)
	y := x.MulScalar(3).AddScalar(1)
	y.Backward()
	fmt.Println("y = 3x + 1 at x = 2:", y.Item(), " dy/dx =", x.Grad().Item())

	// The same for something curved: y = x². At x = 2 the slope is 4, at
	// x = -1 it is -2: the gradient depends on where you stand.
	for _, at := range []float32{2, -1, 0} {
		x := tensor.Scalar(at).SetRequiresGrad(true)
		x.Square().Backward()
		fmt.Printf("y = x² at x = %v: slope %v\n", at, x.Grad().Item())
	}

	// Now the case that matters for learning. A guess w, a target 5, and
	// the "loss": how far off the guess is, squared. The gradient of the
	// loss tells us which way to move w to make the loss smaller.
	w := tensor.Scalar(1).SetRequiresGrad(true)
	target := tensor.Scalar(5)
	loss := w.Sub(target).Square()
	loss.Backward()
	fmt.Printf("\nguess w = 1, target 5: loss %v, dloss/dw %v\n", loss.Item(), w.Grad().Item())
	fmt.Println("the gradient is negative, so increasing w lowers the loss")

	// Take a step against the gradient and look again. This loop, run a
	// few thousand times over millions of numbers, is all that training is.
	for step := 1; step <= 5; step++ {
		w.ZeroGrad()
		loss := w.Sub(target).Square()
		loss.Backward()
		tensor.NoGrad(func() {
			w.AddScaledInPlace(w.Grad(), -0.25) // w -= 0.25 * gradient
		})
		fmt.Printf("step %d: w = %.4f  loss = %.4f\n", step, w.Item(), loss.Item())
	}

	// With several inputs the gradient is one number per input: the
	// direction in which each of them should move.
	v := tensor.New([]float32{1, 2, 3}, 3).SetRequiresGrad(true)
	v.Square().Sum().Backward() // loss = 1 + 4 + 9
	fmt.Println("\nv = [1 2 3], loss = sum(v²): gradient", v.Grad())
}
