package optim

import (
	"testing"

	"github.com/fiber/ai/tensor"
)

// fit trains w to solve a small least-squares problem and returns the
// final loss.
func fit(t *testing.T, mk func(params []*tensor.Tensor) Optimizer, steps int) float32 {
	t.Helper()
	tensor.Seed(3)
	x := tensor.Randn(32, 4)
	trueW := tensor.New([]float32{1, -2, 0.5, 3}, 4, 1)
	y := x.MatMul(trueW).AddScalar(0.25)
	w := tensor.Zeros(4, 1).SetRequiresGrad(true)
	b := tensor.Zeros(1).SetRequiresGrad(true)
	opt := mk([]*tensor.Tensor{w, b})
	var loss *tensor.Tensor
	for i := 0; i < steps; i++ {
		loss = tensor.MSELoss(x.MatMul(w).Add(b), y)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	return loss.Item()
}

func TestSGD(t *testing.T) {
	if l := fit(t, func(p []*tensor.Tensor) Optimizer { return NewSGD(p, 0.1) }, 300); l > 1e-3 {
		t.Fatalf("SGD loss %v", l)
	}
	if l := fit(t, func(p []*tensor.Tensor) Optimizer {
		s := NewSGD(p, 0.05)
		s.Momentum = 0.9
		return s
	}, 200); l > 1e-3 {
		t.Fatalf("SGD+momentum loss %v", l)
	}
	if l := fit(t, func(p []*tensor.Tensor) Optimizer {
		s := NewSGD(p, 0.05)
		s.Momentum, s.Nesterov = 0.9, true
		return s
	}, 200); l > 1e-3 {
		t.Fatalf("SGD+nesterov loss %v", l)
	}
	if l := fit(t, func(p []*tensor.Tensor) Optimizer {
		s := NewSGD(p, 0.1)
		s.WeightDecay = 1e-3
		return s
	}, 300); l > 1e-2 {
		t.Fatalf("SGD+decay loss %v", l)
	}
}

func TestAdam(t *testing.T) {
	if l := fit(t, func(p []*tensor.Tensor) Optimizer { return NewAdam(p, 0.05) }, 400); l > 1e-3 {
		t.Fatalf("Adam loss %v", l)
	}
	if l := fit(t, func(p []*tensor.Tensor) Optimizer { return NewAdamW(p, 0.05, 1e-3) }, 400); l > 1e-2 {
		t.Fatalf("AdamW loss %v", l)
	}
}

func TestSkipsParamsWithoutGrad(t *testing.T) {
	w := tensor.Ones(2).SetRequiresGrad(true)
	unused := tensor.Ones(2).SetRequiresGrad(true)
	opt := NewAdam([]*tensor.Tensor{w, unused}, 0.1)
	w.Sum().Backward()
	opt.Step()
	if unused.At(0) != 1 || w.At(0) == 1 {
		t.Fatal("Step must update only parameters with gradients")
	}
}
