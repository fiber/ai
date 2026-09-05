// Tutorial chapter 5: the first classifier, with nn and optim.
package main

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

// spiral makes three interleaved arms of points; the label is the arm.
func spiral(r *rand.Rand, perClass int) (*tensor.Tensor, []int) {
	n := 3 * perClass
	pts := make([]float32, 0, 2*n)
	labels := make([]int, 0, n)
	for c := 0; c < 3; c++ {
		for i := 0; i < perClass; i++ {
			radius := float64(i) / float64(perClass)
			angle := float64(c)*4 + radius*4 + r.NormFloat64()*0.2
			pts = append(pts, float32(radius*math.Sin(angle)), float32(radius*math.Cos(angle)))
			labels = append(labels, c)
		}
	}
	return tensor.New(pts, n, 2), labels
}

func accuracy(m nn.Module, x *tensor.Tensor, labels []int) float64 {
	correct := 0
	tensor.NoGrad(func() {
		for i, p := range m.Forward(x).Argmax(1) {
			if p == labels[i] {
				correct++
			}
		}
	})
	return 100 * float64(correct) / float64(len(labels))
}

func main() {
	tensor.Seed(5)
	r := rand.New(rand.NewPCG(5, 0))
	x, labels := spiral(r, 300)

	// Two inputs (the point's coordinates), two hidden layers of 64, three
	// outputs (one score per arm). ReLU between the layers is what lets the
	// model bend; without it three Linear layers would be one straight line.
	model := nn.Sequential{
		nn.NewLinear(2, 64), nn.ReLU{},
		nn.NewLinear(64, 64), nn.ReLU{},
		nn.NewLinear(64, 3),
	}
	opt := optim.NewAdam(model.Params(), 3e-3)

	fmt.Printf("before training: accuracy %.1f%% (guessing would be 33.3%%)\n", accuracy(model, x, labels))
	for epoch := 1; epoch <= 200; epoch++ {
		logits := model.Forward(x)                  // [900×3]: a score per class
		loss := tensor.CrossEntropy(logits, labels) // how far the scores are from the right answer
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if epoch == 1 || epoch%40 == 0 {
			fmt.Printf("epoch %3d  loss %.3f  accuracy %.1f%%\n", epoch, loss.Item(), accuracy(model, x, labels))
		}
	}

	// A prediction is the class with the highest score; Softmax turns the
	// scores into probabilities that sum to one when you want to know how
	// sure the model is.
	tensor.NoGrad(func() {
		probe := tensor.New([]float32{0.5, 0.2, -0.6, 0.3}, 2, 2)
		probs := model.Forward(probe).Softmax(1)
		fmt.Println("\ntwo new points, probability per arm:")
		fmt.Println(probs)
		fmt.Println("predicted arms:", probs.Argmax(1))
	})
}
