// Example: predict the next value of a noisy, drifting oscillation from
// the previous 32 samples with an MLP. Prints loss, throughput and the
// backend in use. Run with FIBERAI_KERNEL=generic to compare against the
// SIMD path.
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
	window  = 32
	hidden  = 256
	batch   = 256
	steps   = 1500
	nSeries = 20000
)

// series returns a noisy two-frequency oscillation with a slow drift,
// standardised to zero mean and unit variance so the network sees inputs
// in a sane range.
func series(n int, r *rand.Rand) []float32 {
	s := make([]float32, n)
	var sum, sq float64
	for i := range s {
		t := float64(i)
		v := math.Sin(t*0.07) + 0.5*math.Sin(t*0.23+1.0) + 0.0002*t + 0.1*r.NormFloat64()
		s[i] = float32(v)
		sum += v
		sq += v * v
	}
	mean := sum / float64(n)
	std := math.Sqrt(sq/float64(n) - mean*mean)
	for i := range s {
		s[i] = float32((float64(s[i]) - mean) / std)
	}
	return s
}

func makeBatch(s []float32, r *rand.Rand, xs, ys []float32) {
	for b := 0; b < batch; b++ {
		i := r.IntN(len(s) - window - 1)
		copy(xs[b*window:(b+1)*window], s[i:i+window])
		ys[b] = s[i+window]
	}
}

func main() {
	fmt.Printf("backend %s, %d threads\n", tensor.Backend(), tensor.Threads())
	tensor.Seed(42)
	r := rand.New(rand.NewPCG(42, 0))
	data := series(nSeries, r)

	model := nn.Sequential{
		nn.NewLinear(window, hidden), nn.GELU{},
		nn.NewLinear(hidden, hidden), nn.GELU{},
		nn.NewLinear(hidden, 1),
	}
	fmt.Printf("%d parameters\n", nn.NumParams(model))
	opt := optim.NewAdam(model.Params(), 1e-3)

	x := tensor.Zeros(batch, window)
	y := tensor.Zeros(batch, 1)
	start := time.Now()
	for step := 1; step <= steps; step++ {
		makeBatch(data, r, x.Data(), y.Data())
		loss := tensor.MSELoss(model.Forward(x), y)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step%100 == 0 {
			el := time.Since(start).Seconds()
			fmt.Printf("step %4d  loss %.5f  %.0f samples/s\n", step, loss.Item(), float64(step*batch)/el)
		}
	}

	// evaluation without graph recording
	var mse float64
	tensor.NoGrad(func() {
		makeBatch(data, r, x.Data(), y.Data())
		mse = float64(tensor.MSELoss(model.Forward(x), y).Item())
	})
	fmt.Printf("held-out MSE %.5f (noise floor ≈ %.5f after standardisation)\n", mse, 0.01/0.83)
}
