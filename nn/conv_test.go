package nn

import (
	"testing"

	"github.com/fiber/ai/tensor"
)

func TestConvModulesShapesAndGradients(t *testing.T) {
	tensor.Seed(12)
	m := Sequential{NewConv2D(1, 4, 3), ReLU{}, NewMaxPool2D(2), NewConv2D(4, 8, 3), ReLU{}, NewMaxPool2D(2), Flatten{}, NewLinear(8*4*4, 3)}
	x := tensor.Randn(2, 1, 16, 16).SetRequiresGrad(true)
	out := m.Forward(x)
	if !out.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("shape %v", out.Shape())
	}
	if n := len(m.Params()); n != 6 {
		t.Fatalf("%d params", n)
	}
	// gradient check on a smooth variant: ReLU and max pooling have kinks
	// that finite differences with h = 1e-2 step across
	c1, c2 := NewConv2D(1, 4, 3), NewConv2D(4, 8, 3)
	c1.Stride, c2.Stride = 2, 2
	sm := Sequential{c1, GELU{}, c2, GELU{}, Flatten{}, NewLinear(8*4*4, 3)}
	w := tensor.Randn(2, 3)
	loss := func() *tensor.Tensor { return sm.Forward(x).Mul(w).Sum() }
	loss().Backward()
	numericCheck(t, "x", x, loss, 97, 3e-2)
	numericCheck(t, "conv1.W", c1.W, loss, 5, 3e-2)
	numericCheck(t, "conv2.B", c2.B, loss, 3, 3e-2)

	d1 := NewConv1D(3, 5, 3)
	y := d1.Forward(tensor.Randn(4, 3, 24))
	if !y.Shape().Equal(tensor.Shape{4, 5, 24}) {
		t.Fatalf("Conv1D shape %v", y.Shape())
	}
}
