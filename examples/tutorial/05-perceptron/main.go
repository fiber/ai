// Tutorial chapter 5: one unit, the rule that trains it, and the
// problem it cannot solve.
package main

import (
	"flag"
	"fmt"

	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

// The four points of a two-input logic function. Everything in this
// chapter is these four rows.
var inputs = [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}

func targets(f func(a, b int) int) []float32 {
	out := make([]float32, 4)
	for i, in := range inputs {
		out[i] = float32(f(int(in[0]), int(in[1])))
	}
	return out
}

func and(a, b int) int { return a & b }
func or(a, b int) int  { return a | b }
func xor(a, b int) int { return a ^ b }

// perceptron is Rosenblatt's 1957 unit: weights, a bias, and a step.
// It predicts 1 when w·x + b is positive and 0 otherwise.
type perceptron struct{ w0, w1, b float32 }

func (p *perceptron) predict(x []float32) float32 {
	if p.w0*x[0]+p.w1*x[1]+p.b > 0 {
		return 1
	}
	return 0
}

// train applies the perceptron rule, which predates gradients by a
// decade: for every example that is wrong, move the weights toward the
// answer by the size of the mistake. There is no loss function here and
// nothing is differentiated.
func (p *perceptron) train(y []float32, rate float32, epochs int) (converged bool, at int) {
	for epoch := 1; epoch <= epochs; epoch++ {
		wrong := 0
		for i, x := range inputs {
			err := y[i] - p.predict(x)
			if err != 0 {
				wrong++
				p.w0 += rate * err * x[0]
				p.w1 += rate * err * x[1]
				p.b += rate * err
			}
		}
		if wrong == 0 {
			return true, epoch
		}
	}
	return false, epochs
}

func (p *perceptron) report(name string, y []float32) {
	fmt.Printf("  %-4s", name)
	for i, x := range inputs {
		fmt.Printf("  (%g,%g)->%g want %g", x[0], x[1], p.predict(x), y[i])
	}
	fmt.Println()
}

// A network with one hidden layer, trained the modern way. Two units in
// the middle are enough: each can draw one line, and the output unit
// combines them.
func hidden(y []float32, hiddenUnits, steps int) (float32, []float32) {
	tensor.Seed(4)
	x := tensor.New([]float32{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	want := tensor.New(y, 4, 1)
	model := nn.Sequential{
		nn.NewLinear(2, hiddenUnits), nn.Tanh{},
		nn.NewLinear(hiddenUnits, 1),
	}
	opt := optim.NewAdam(model.Params(), 0.1)
	var loss *tensor.Tensor
	for range steps {
		loss = tensor.MSELoss(model.Forward(x), want)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	var out []float32
	tensor.NoGrad(func() { out = model.Forward(x).Float32s() })
	return loss.Item(), out
}

func main() {
	epochs := flag.Int("epochs", 1000, "passes over the four points for the perceptron rule")
	rate := flag.Float64("rate", 0.1, "perceptron learning rate")
	steps := flag.Int("steps", 400, "gradient steps for the two-layer network")
	flag.Parse()

	fmt.Println("One unit, trained by the perceptron rule:")
	for _, c := range []struct {
		name string
		f    func(a, b int) int
	}{{"AND", and}, {"OR", or}, {"XOR", xor}} {
		y := targets(c.f)
		p := &perceptron{}
		ok, at := p.train(y, float32(*rate), *epochs)
		if ok {
			fmt.Printf("\n%s: separated after %d epochs\n", c.name, at)
		} else {
			fmt.Printf("\n%s: still wrong after %d epochs\n", c.name, *epochs)
		}
		p.report(c.name, y)
	}

	fmt.Printf("\nXOR is not the rule's fault. No single unit can do it: the unit\n" +
		"draws one straight line, and no straight line puts (0,1) and (1,0)\n" +
		"on one side with (0,0) and (1,1) on the other.\n")

	fmt.Println("\nThe same XOR with one hidden layer, trained by gradient descent:")
	y := targets(xor)
	for _, h := range []int{1, 2, 4} {
		loss, out := hidden(y, h, *steps)
		fmt.Printf("  %d hidden unit(s): loss %.4f  outputs", h, loss)
		for i := range out {
			fmt.Printf("  %.2f", out[i])
		}
		fmt.Printf("   want  %g %g %g %g\n", y[0], y[1], y[2], y[3])
	}
	fmt.Println("\nOne hidden unit is still one line and still fails. Two are enough.")
}
