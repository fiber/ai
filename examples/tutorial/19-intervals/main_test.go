package main

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

// coverage and width are what the chapter reports; check them against a
// case whose answer can be read off by hand.
func TestEvaluateCountsCoverage(t *testing.T) {
	// Three rows, quantile outputs already in "aircraft" (mu 0, sd 1):
	// the truth is inside the first band, below the second, above the
	// third, and the third is also out of order.
	m := fixed{tensor.New([]float32{
		8, 10, 12,
		8, 10, 12,
		12, 10, 8,
	}, 3, 3)}
	y := tensor.New([]float32{10, 2, 20}, 3, 1)
	r := evaluate(m, tensor.Zeros(3, 1), y, 0, 1)
	if math.Abs(r.coverage-100.0/3) > 1e-6 {
		t.Errorf("coverage %.2f%%, want one row in three", r.coverage)
	}
	if math.Abs(r.width-(4+4-4)/3.0) > 1e-6 {
		t.Errorf("width %.3f, want the mean of 4, 4 and -4", r.width)
	}
	if want := (0 + 8 + 10) / 3.0; math.Abs(r.mae-want) > 1e-6 {
		t.Errorf("median MAE %.3f, want %.3f", r.mae, want)
	}
	if r.crossed != 1 {
		t.Errorf("crossed = %d, want 1", r.crossed)
	}
}

type fixed struct{ out *tensor.Tensor }

func (f fixed) Forward(*tensor.Tensor) *tensor.Tensor { return f.out }
func (fixed) Params() []*tensor.Tensor                { return nil }

// The model must accept a history length it was not built for: that is
// what GlobalAvgPool1D is in the architecture for.
func TestForecasterAcceptsAnyLength(t *testing.T) {
	tensor.Seed(19)
	m := forecaster()
	for _, hist := range []int{60, 120, 240} {
		out := m.Forward(tensor.Randn(4, len(airports)*hist))
		if !out.Shape().Equal(tensor.Shape{4, len(quantiles)}) {
			t.Errorf("history %d: shape %v, want [4 3]", hist, out.Shape())
		}
	}
}

// Trained on data whose spread is known, the three outputs must come out
// ordered and near the true quantiles: the chapter's whole premise.
func TestQuantilesTrainToOrder(t *testing.T) {
	tensor.Seed(5)
	rng := rand.New(rand.NewPCG(5, 0))
	// The target is noise around a constant, so the answer is the
	// normal's quantiles: about -1.28, 0, 1.28.
	const n = 256
	xs := make([]float32, n*len(airports)*window)
	ys := make([]float32, n)
	for i := range ys {
		ys[i] = float32(rng.NormFloat64())
	}
	x, y := tensor.New(xs, n, len(airports)*window), tensor.New(ys, n)
	m := forecaster()
	opt := optim.NewAdamW(m.Params(), 3e-2, 0)
	for range 400 {
		loss := tensor.PinballLoss(m.Forward(x), y, quantiles...)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	var got []float32
	tensor.NoGrad(func() { got = m.Forward(tensor.Zeros(1, len(airports)*window)).Float32s() })
	if !(got[0] < got[1] && got[1] < got[2]) {
		t.Fatalf("quantile outputs out of order: %v", got)
	}
	for i, want := range []float32{-1.2816, 0, 1.2816} {
		if math.Abs(float64(got[i]-want)) > 0.4 {
			t.Errorf("quantile %v: %v, want about %v", quantiles[i], got[i], want)
		}
	}
}

var _ = nn.Module(forecaster())
