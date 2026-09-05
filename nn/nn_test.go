package nn

import (
	"testing"

	"github.com/fiber/ai/tensor"
)

func TestLinearSequential(t *testing.T) {
	tensor.Seed(1)
	m := Sequential{NewLinear(4, 8), ReLU{}, NewLayerNorm(8), NewLinear(8, 3), Softmax{Dim: -1}}
	if got := NumParams(m); got != 4*8+8+8+8+8*3+3 {
		t.Fatalf("NumParams = %d", got)
	}
	x := tensor.Randn(5, 4)
	y := m.Forward(x)
	if !y.Shape().Equal(tensor.Shape{5, 3}) {
		t.Fatalf("shape %v", y.Shape())
	}
	if s := y.Sum(1); !s.AllClose(tensor.Ones(5), 1e-5, 1e-5) {
		t.Fatalf("softmax rows: %v", s)
	}
	loss := y.Log().Neg().Mean()
	loss.Backward()
	for i, p := range m.Params() {
		if p.Grad() == nil || !p.Grad().Shape().Equal(p.Shape()) {
			t.Fatalf("param %d has no gradient", i)
		}
	}
	ZeroGrad(m)
	if m.Params()[0].Grad().Sum().Item() != 0 {
		t.Fatal("ZeroGrad")
	}
	nb := NewLinearNoBias(4, 2)
	if len(nb.Params()) != 1 || !nb.Forward(x).Shape().Equal(tensor.Shape{5, 2}) {
		t.Fatal("no-bias linear")
	}
}

func TestDropout(t *testing.T) {
	d := NewDropout(0.5)
	x := tensor.Ones(1000)
	y := d.Forward(x)
	zeros := 0
	for _, v := range y.Data() {
		switch v {
		case 0:
			zeros++
		case 2:
		default:
			t.Fatalf("unexpected value %v", v)
		}
	}
	if zeros < 400 || zeros > 600 {
		t.Fatalf("dropped %d of 1000", zeros)
	}
	SetTraining(d, false)
	if d.Forward(x) != x {
		t.Fatal("eval-mode dropout must be identity")
	}
	s := Sequential{d}
	s.SetTraining(true)
	if !d.Training {
		t.Fatal("Sequential.SetTraining")
	}
}

func TestEmbeddingAndFlatten(t *testing.T) {
	e := NewEmbedding(10, 4)
	v := e.Lookup([]int{3, 3, 7})
	if !v.Shape().Equal(tensor.Shape{3, 4}) || !v.Row(0).Equal(e.W.Row(3)) || !v.Row(2).Equal(e.W.Row(7)) {
		t.Fatalf("Lookup: %v", v)
	}
	v.Sum().Backward()
	g := e.W.Grad()
	if g.At(3, 0) != 2 || g.At(7, 0) != 1 || g.At(0, 0) != 0 {
		t.Fatalf("embedding grad: %v", g)
	}
	if !(Flatten{}).Forward(tensor.Zeros(2, 3, 4)).Shape().Equal(tensor.Shape{2, 12}) {
		t.Fatal("Flatten")
	}
}
