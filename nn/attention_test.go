package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/fiber/ai/tensor"
)

func numericCheck(t *testing.T, name string, p *tensor.Tensor, loss func() *tensor.Tensor, stride int, tol float64) {
	t.Helper()
	grad := p.Grad().Float32s()
	d := p.Data()
	for i := 0; i < len(d); i += stride {
		const h = 1e-2
		orig := d[i]
		var lp, lm float32
		d[i] = orig + h
		tensor.NoGrad(func() { lp = loss().Item() })
		d[i] = orig - h
		tensor.NoGrad(func() { lm = loss().Item() })
		d[i] = orig
		num := (lp - lm) / (2 * h)
		if diff := math.Abs(float64(grad[i] - num)); diff > tol*(1+math.Abs(float64(num))) {
			t.Fatalf("%s[%d]: analytic %v, numeric %v", name, i, grad[i], num)
		}
	}
}

func TestMultiHeadAttentionGradient(t *testing.T) {
	tensor.Seed(9)
	m := NewMultiHeadAttention(8, 2)
	m.Mask = tensor.CausalMask(5)
	x := tensor.Randn(2, 5, 8).SetRequiresGrad(true)
	w := tensor.Randn(2, 5, 8)
	loss := func() *tensor.Tensor { return m.Forward(x).Mul(w).Sum() }
	out := m.Forward(x)
	if !out.Shape().Equal(tensor.Shape{2, 5, 8}) {
		t.Fatalf("shape %v", out.Shape())
	}
	loss().Backward()
	numericCheck(t, "x", x, loss, 7, 2e-2)
	numericCheck(t, "Q.W", m.Q.W, loss, 9, 2e-2)
	numericCheck(t, "V.W", m.V.W, loss, 11, 2e-2)
	numericCheck(t, "O.b", m.O.B, loss, 3, 2e-2)
	if len(m.Params()) != 5 {
		t.Fatalf("%d params", len(m.Params()))
	}
}

func TestRMSNormModuleAndLookup(t *testing.T) {
	r := rand.New(rand.NewPCG(2, 2))
	n := NewRMSNorm(6)
	x := tensor.RandnFrom(r, 3, 6).SetRequiresGrad(true)
	y := n.Forward(x)
	rms := y.Square().Mean(-1).Sqrt()
	for i := 0; i < 3; i++ {
		if v := rms.At(i); v < 0.999 || v > 1.001 {
			t.Fatalf("row %d rms after RMSNorm %v", i, v)
		}
	}
	e := NewEmbedding(10, 4)
	emb := e.Lookup([]int{3, 3, 7})
	if !emb.Shape().Equal(tensor.Shape{3, 4}) || emb.At(0, 1) != e.W.At(3, 1) {
		t.Fatalf("lookup %v", emb)
	}
	emb.Sum().Backward()
	if e.W.Grad().At(3, 0) != 2 || e.W.Grad().At(7, 0) != 1 || e.W.Grad().At(0, 0) != 0 {
		t.Fatalf("lookup gradient %v", e.W.Grad())
	}
}
