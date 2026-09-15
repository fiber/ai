package tensor

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

// pinball is the definition, written out: the loss of one element is
// max(q·d, (q−1)·d) with d = target − prediction.
func pinballRef(pred, target []float32, quantiles []float32) float32 {
	nq := len(quantiles)
	var total float64
	for r := range target {
		for j, q := range quantiles {
			d := target[r] - pred[r*nq+j]
			total += math.Max(float64(q*d), float64((q-1)*d))
		}
	}
	return float32(total / float64(len(target)*nq))
}

func TestPinballValue(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 7))
	qs := []float32{0.1, 0.5, 0.9}
	pred := RandnFrom(rng, 8, 3)
	target := RandnFrom(rng, 8)
	got := PinballLoss(pred, target, qs...).Item()
	want := pinballRef(pred.Float32s(), target.Float32s(), qs)
	if !approx(got, want, 1e-6) {
		t.Errorf("PinballLoss = %v, want %v", got, want)
	}
}

// With the single quantile 0.5 the loss is half the mean absolute error,
// which is the cheapest way to be sure the two branches are symmetric.
func TestPinballMedianIsHalfMAE(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	pred := RandnFrom(rng, 16, 1)
	target := RandnFrom(rng, 16)
	got := PinballLoss(pred, target, 0.5).Item()
	var mae float64
	p, y := pred.Float32s(), target.Float32s()
	for i := range y {
		mae += math.Abs(float64(y[i] - p[i]))
	}
	want := float32(mae / float64(2*len(y)))
	if !approx(got, want, 1e-6) {
		t.Errorf("median pinball = %v, want half the MAE %v", got, want)
	}
}

func TestPinballGrad(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 11))
	// Targets well away from the predictions, so no element sits on the
	// kink at d = 0 where the loss has no derivative and a finite
	// difference would straddle both branches.
	pred := RandnFrom(rng, 6, 2).MulScalar(0.1).Detach().SetRequiresGrad(true)
	target := RandnFrom(rng, 6).MulScalar(0.1).AddScalar(5).Detach().SetRequiresGrad(true)
	checkGrad(t, "pinball", func() *Tensor { return PinballLoss(pred, target, 0.25, 0.75) }, pred, target)
}

// A three-dimensional forecast: [batch, horizon, quantiles] against
// [batch, horizon], which is the shape a multi-horizon forecaster has.
func TestPinballMultiHorizon(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 5))
	qs := []float32{0.2, 0.8}
	pred := RandnFrom(rng, 4, 3, 2)
	target := RandnFrom(rng, 4, 3)
	got := PinballLoss(pred, target, qs...).Item()
	want := pinballRef(pred.Float32s(), target.Float32s(), qs)
	if !approx(got, want, 1e-6) {
		t.Errorf("PinballLoss = %v, want %v", got, want)
	}
}

// What quantile regression promises: minimising the loss puts the q-th
// prediction at the q-th quantile of the targets. Here the targets are
// pure noise around a constant, so the answer is known.
func TestPinballFindsQuantiles(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 13))
	const n = 4000
	target := make([]float32, n)
	for i := range target {
		target[i] = float32(rng.NormFloat64())
	}
	y := New(target, n)
	qs := []float32{0.1, 0.5, 0.9}
	pred := Zeros(1, len(qs)).SetRequiresGrad(true)
	// Plain subgradient descent with a decaying step: the gradient of a
	// pinball loss does not shrink near the optimum, it only changes
	// sign, so a constant step would oscillate around the answer.
	for step := range 3000 {
		loss := PinballLoss(pred.Expand(n, len(qs)).Contiguous(), y, qs...)
		pred.ZeroGrad()
		loss.Backward()
		lr := float32(10 / (1 + float64(step)/300))
		NoGrad(func() { pred.SubInPlace(pred.Grad().MulScalar(lr)) })
	}
	// The quantiles of a standard normal.
	want := []float32{-1.2816, 0, 1.2816}
	got := pred.Float32s()
	for j := range qs {
		if math.Abs(float64(got[j]-want[j])) > 0.15 {
			t.Errorf("quantile %v: learned %v, want about %v", qs[j], got[j], want[j])
		}
	}
	if !(got[0] < got[1] && got[1] < got[2]) {
		t.Errorf("quantile predictions out of order: %v", got)
	}
}

func TestPinballErrors(t *testing.T) {
	for _, c := range []struct {
		name string
		fn   func()
		want string
	}{
		{"no quantiles", func() { PinballLoss(Zeros(2, 1), Zeros(2)) }, "no quantiles"},
		{"quantile 0", func() { PinballLoss(Zeros(2, 1), Zeros(2), 0) }, "not in (0, 1)"},
		{"quantile 1", func() { PinballLoss(Zeros(2, 1), Zeros(2), 1) }, "not in (0, 1)"},
		{"1-D predictions", func() { PinballLoss(Zeros(2), Zeros(2), 0.5) }, "[..., quantiles]"},
		{"count mismatch", func() { PinballLoss(Zeros(2, 3), Zeros(2), 0.5) }, "quantiles"},
		{"target shape", func() { PinballLoss(Zeros(2, 2), Zeros(2, 2), 0.5, 0.9) }, "want [2]"},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("no panic")
				}
				if msg, ok := r.(error); !ok || !strings.Contains(msg.Error(), c.want) {
					t.Fatalf("panic %v, want one mentioning %q", r, c.want)
				}
			}()
			c.fn()
		})
	}
}
