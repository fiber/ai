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

// Rotary positions exist so that a score depends on the distance between
// two tokens, not on where the pair sits in the window. Shifting a query
// and its key by the same amount must leave the score unchanged.
func TestRoPEScoresDependOnDistance(t *testing.T) {
	tensor.Seed(4)
	const dim, heads, T = 32, 4, 8
	m := NewMultiHeadAttention(dim, heads)
	m.RoPEBase = 1e4

	x := tensor.Randn(1, T, dim)
	// One token repeated, so any difference between positions comes from
	// the rotation rather than from the content.
	row := x.Reshape(T, dim).Rows([]int{0}).Float32s()
	rep := make([]float32, 0, T*dim)
	for range T {
		rep = append(rep, row...)
	}
	x = tensor.New(rep, 1, T, dim)

	score := func(qi, ki int) float32 {
		var out float32
		tensor.NoGrad(func() {
			q := m.Q.Forward(x).Reshape(1, T, heads, dim/heads).Permute(0, 2, 1, 3)
			k := m.K.Forward(x).Reshape(1, T, heads, dim/heads).Permute(0, 2, 1, 3)
			q = tensor.RoPE(q, m.RoPEBase, seq(0, T))
			k = tensor.RoPE(k, m.RoPEBase, seq(0, T))
			qc, kc := q.Contiguous().Float32s(), k.Contiguous().Float32s()
			hd := dim / heads
			for d := range hd { // head 0 only
				out += qc[qi*hd+d] * kc[ki*hd+d]
			}
		})
		return out
	}

	// Pairs at the same distance must score alike; different distances
	// must not, or the rotation is doing nothing.
	a, b := score(3, 1), score(6, 4) // distance 2 in both cases
	if diff := math.Abs(float64(a - b)); diff > 1e-3 {
		t.Errorf("same distance scored %v and %v, difference %v", a, b, diff)
	}
	if c := score(6, 1); math.Abs(float64(a-c)) < 1e-3 {
		t.Errorf("distance 2 and distance 5 both scored %v; the rotation is not applied", a)
	}
}

// PosOffset is what decoding one token at a time needs: a single query at
// position n must behave exactly as the n-th query of a whole sequence.
func TestRoPEPosOffsetMatchesFullSequence(t *testing.T) {
	tensor.Seed(5)
	const dim, heads, T = 32, 4, 6
	full := NewMultiHeadAttention(dim, heads)
	full.RoPEBase = 1e4
	x := tensor.Randn(1, T, dim)

	var whole, step []float32
	tensor.NoGrad(func() {
		whole = full.Forward(x).Rows([]int{0}).Float32s()

		// The same weights, one query at the last position, attending over
		// the whole sequence as a cache would supply it.
		one := *full
		one.PosOffset = T - 1
		last := x.Reshape(T, dim).Rows([]int{T - 1}).Reshape(1, 1, dim)
		step = one.Cross(last, x).Float32s()
	})
	if len(whole) != T*dim {
		t.Fatalf("full output has %d values", len(whole))
	}
	for i := range dim {
		if got, want := step[i], whole[(T-1)*dim+i]; math.Abs(float64(got-want)) > 1e-4 {
			t.Fatalf("value %d: single-query %v, full sequence %v", i, got, want)
		}
	}
}

// The rotation must not break the gradient: it is orthogonal, so the
// backward pass rotates by the negative angle.
func TestRoPEGradients(t *testing.T) {
	tensor.Seed(6)
	const dim, heads, T = 16, 2, 4
	m := NewMultiHeadAttention(dim, heads)
	m.RoPEBase = 1e4
	x := tensor.Randn(1, T, dim).SetRequiresGrad(true)
	m.Forward(x).Sum().Backward()
	if x.Grad() == nil {
		t.Fatal("no gradient reached the input")
	}
	var nonzero int
	for _, v := range x.Grad().Float32s() {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Error("every input gradient is zero")
	}
	for _, p := range m.Params() {
		if p.Grad() == nil {
			t.Error("a projection has no gradient")
		}
	}
}

// With RoPEBase left at zero nothing changes, so existing models keep
// their numbers exactly.
func TestRoPEOffByDefault(t *testing.T) {
	tensor.Seed(7)
	const dim, heads, T = 16, 2, 4
	m := NewMultiHeadAttention(dim, heads)
	x := tensor.Randn(1, T, dim)
	var a, b []float32
	tensor.NoGrad(func() {
		a = m.Forward(x).Float32s()
		m.PosOffset = 3 // ignored while RoPEBase is zero
		b = m.Forward(x).Float32s()
	})
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("value %d changed from %v to %v with the rotation off", i, a[i], b[i])
		}
	}
}
