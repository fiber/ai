package tensor

import (
	"math"
	"math/rand/v2"
	"testing"
)

// checkGrad compares analytic gradients of loss() with central finite
// differences for every parameter. Inputs are kept O(1) so the float32
// differences stay meaningful.
func checkGrad(t *testing.T, name string, loss func() *Tensor, params ...*Tensor) {
	t.Helper()
	for _, p := range params {
		p.grad = nil
		p.SetRequiresGrad(true)
	}
	l := loss()
	if l.Dims() != 0 {
		t.Fatalf("%s: loss must be 0-D, got %v", name, l.Shape())
	}
	l.Backward()
	analytic := make([][]float32, len(params))
	for i, p := range params {
		if p.Grad() == nil {
			t.Fatalf("%s: param %d got no gradient", name, i)
		}
		if !p.Grad().Shape().Equal(p.Shape()) {
			t.Fatalf("%s: param %d grad shape %v != %v", name, i, p.Grad().Shape(), p.Shape())
		}
		analytic[i] = p.Grad().Float32s()
	}
	const eps = 1e-2
	for pi, p := range params {
		data := p.Data()
		for i := range data {
			orig := data[i]
			var lp, lm float32
			NoGrad(func() {
				data[i] = orig + eps
				lp = loss().Item()
				data[i] = orig - eps
				lm = loss().Item()
			})
			data[i] = orig
			num := (lp - lm) / (2 * eps)
			an := analytic[pi][i]
			if math.Abs(float64(num-an)) > 2e-2*(1+math.Abs(float64(an))) {
				t.Errorf("%s: param %d[%d] analytic %g numeric %g", name, pi, i, an, num)
			}
		}
		p.grad = nil
	}
}

func TestGradMLP(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	x := RandnFrom(rng, 6, 5)
	w1 := RandnFrom(rng, 5, 7).MulScalar(0.5).Detach()
	b1 := RandnFrom(rng, 7)
	w2 := RandnFrom(rng, 7, 3).MulScalar(0.5).Detach()
	b2 := RandnFrom(rng, 3)
	y := RandnFrom(rng, 6, 3)
	checkGrad(t, "mlp", func() *Tensor {
		h := x.MatMul(w1).Add(b1).Tanh()
		return MSELoss(h.MatMul(w2).Add(b2), y)
	}, w1, b1, w2, b2, x)
}

func TestGradElementwise(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2))
	x := RandnFrom(rng, 4, 6)
	w := RandnFrom(rng, 6, 6)
	weights := RandnFrom(rng, 4, 6)
	for name, act := range map[string]func(*Tensor) *Tensor{
		"sigmoid": (*Tensor).Sigmoid, "tanh": (*Tensor).Tanh, "gelu": (*Tensor).GELU, "exp": (*Tensor).Exp,
		"square": (*Tensor).Square, "neg": (*Tensor).Neg,
		"mulscalar": func(a *Tensor) *Tensor { return a.MulScalar(1.7) },
		"addscalar": func(a *Tensor) *Tensor { return a.AddScalar(-0.3) },
		"divscalar": func(a *Tensor) *Tensor { return a.DivScalar(2) },
		"pow3":      func(a *Tensor) *Tensor { return a.Pow(3) },
	} {
		checkGrad(t, name, func() *Tensor { return act(x.MatMul(w)).Mul(weights).Sum() }, x, w)
	}
	// kinks: keep inputs away from the non-differentiable points
	xp := RandnFrom(rng, 4, 6)
	for i, v := range xp.data {
		if math.Abs(float64(v)) < 0.15 {
			xp.data[i] = v + 0.3
		}
	}
	checkGrad(t, "relu", func() *Tensor { return xp.ReLU().Mul(xp).Sum() }, xp)
	checkGrad(t, "abs", func() *Tensor { return xp.Abs().Mul(weights).Sum() }, xp)
	xc := Linspace(-2, 2, 24).Reshape(4, 6) // no value within eps of the clamp bounds
	checkGrad(t, "clamp", func() *Tensor { return xc.Clamp(-0.5, 0.85).Mul(weights).Sum() }, xc)
	checkGrad(t, "maximum", func() *Tensor { return xp.Maximum(xp.Neg().MulScalar(0.5)).Mul(weights).Sum() }, xp)
	checkGrad(t, "minimum-scalar", func() *Tensor { return xp.Minimum(Scalar(0.2)).Mul(weights).Sum() }, xp)
	pos := RandnFrom(rng, 3, 4).Exp().AddScalar(0.5).Detach()
	checkGrad(t, "log", func() *Tensor { return pos.Log().Sum() }, pos)
	checkGrad(t, "sqrt", func() *Tensor { return pos.Sqrt().Sum() }, pos)
	checkGrad(t, "pow-0.5", func() *Tensor { return pos.Pow(-0.5).Sum() }, pos)
}

func TestGradBroadcasting(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	a := RandnFrom(rng, 5, 4)
	row := RandnFrom(rng, 4)
	col := RandnFrom(rng, 5, 1)
	s := Scalar(1.5)
	den := RandnFrom(rng, 5, 4).Exp().AddScalar(0.5).Detach()
	b3 := RandnFrom(rng, 2, 1, 4)
	checkGrad(t, "add-row", func() *Tensor { return a.Add(row).Mul(a).Sum() }, a, row)
	checkGrad(t, "add-col", func() *Tensor { return a.Add(col).Mul(a).Sum() }, a, col)
	checkGrad(t, "row-add", func() *Tensor { return row.Add(a).Mul(a).Sum() }, a, row)
	checkGrad(t, "sub-scalar", func() *Tensor { return a.Sub(s).Mul(a).Sum() }, a, s)
	checkGrad(t, "scalar-sub", func() *Tensor { return s.Sub(a).Mul(a).Sum() }, a, s)
	checkGrad(t, "mul-row", func() *Tensor { return a.Mul(row).Mul(a).Sum() }, a, row)
	checkGrad(t, "mul-col", func() *Tensor { return a.Mul(col).Mul(a).Sum() }, a, col)
	checkGrad(t, "mul-scalar", func() *Tensor { return a.Mul(s).Mul(a).Sum() }, a, s)
	checkGrad(t, "div", func() *Tensor { return a.Div(den).Sum() }, a, den)
	checkGrad(t, "div-row", func() *Tensor { return a.Div(row.Exp().AddScalar(0.5)).Sum() }, a, row)
	checkGrad(t, "scalar-div", func() *Tensor { return s.Div(den).Sum() }, s, den)
	checkGrad(t, "3d-broadcast", func() *Tensor { return a.Add(b3).Mul(b3).Sum() }, a, b3)
	checkGrad(t, "expand", func() *Tensor { return row.Expand(5, 4).Mul(a).Sum() }, row, a)
}

func TestGradViewsAndReductions(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 4))
	a := RandnFrom(rng, 3, 4)
	w := RandnFrom(rng, 4, 3)
	c := RandnFrom(rng, 2, 3, 4)
	weights := RandnFrom(rng, 2, 4, 3)
	checkGrad(t, "transpose", func() *Tensor { return a.T().Mul(a.T()).Sum() }, a)
	checkGrad(t, "transpose-matmul", func() *Tensor { return a.T().MatMul(w.T()).Sum() }, a, w)
	checkGrad(t, "permute", func() *Tensor { return c.Permute(0, 2, 1).Mul(weights).Sum() }, c)
	checkGrad(t, "reshape", func() *Tensor { return a.Reshape(2, 6).Mul(a.Reshape(2, 6)).Sum() }, a)
	checkGrad(t, "reshape-view", func() *Tensor { return a.T().Reshape(12).Mul(Arange(0, 12, 1)).Sum() }, a)
	checkGrad(t, "narrow", func() *Tensor { return a.Narrow(1, 1, 2).Square().Sum() }, a)
	checkGrad(t, "select", func() *Tensor { return c.Select(1, 2).Mul(a.Narrow(0, 0, 2)).Sum() }, c, a)
	checkGrad(t, "squeeze-unsqueeze", func() *Tensor { return a.Unsqueeze(1).Squeeze(1).Mul(a).Sum() }, a)
	checkGrad(t, "cat", func() *Tensor { return Cat(1, a, a.Square()).Mul(RandnFrom(rand.New(rand.NewPCG(0, 0)), 3, 8)).Sum() }, a)
	checkGrad(t, "stack", func() *Tensor { return Stack(0, a, w.T()).Square().Sum() }, a, w)
	checkGrad(t, "clone-contiguous", func() *Tensor { return a.T().Contiguous().Clone().Square().Sum() }, a)
	checkGrad(t, "sum-dim", func() *Tensor { return c.Sum(1).Mul(a.Narrow(0, 0, 2)).Sum() }, c)
	checkGrad(t, "sum-dims", func() *Tensor { return c.Sum(0, 2).Mul(RandnFrom(rand.New(rand.NewPCG(1, 0)), 3)).Sum() }, c)
	checkGrad(t, "mean-dim", func() *Tensor { return c.Mean(2).Square().Sum() }, c)
	checkGrad(t, "mean-all", func() *Tensor { return c.Square().Mean() }, c)
	checkGrad(t, "max-dim", func() *Tensor { return c.Max(1).Mul(a.Narrow(0, 0, 2)).Sum() }, c)
	checkGrad(t, "max-all", func() *Tensor { return c.Max().Mul(Scalar(2)) }, c)
	checkGrad(t, "min-dim", func() *Tensor { return c.Min(2).Square().Sum() }, c)
	checkGrad(t, "var", func() *Tensor { return a.Var(1).Sum() }, a)
	checkGrad(t, "std", func() *Tensor { return a.Std().Mul(Scalar(3)) }, a)
	checkGrad(t, "dot", func() *Tensor { return a.Row(0).Dot(w.Select(1, 0)) }, a, w)
	checkGrad(t, "outer", func() *Tensor { return a.Row(0).Outer(w.Select(1, 0)).Square().Sum() }, a, w)
}

func TestGradBatchedMatMul(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 5))
	x := RandnFrom(rng, 2, 3, 4, 5)
	y := RandnFrom(rng, 3, 5, 2)
	z := RandnFrom(rng, 5, 3)
	w := RandnFrom(rng, 2, 3, 4, 2)
	checkGrad(t, "batched-broadcast", func() *Tensor { return x.MatMul(y).Mul(w).Sum() }, x, y)
	checkGrad(t, "batched-2d", func() *Tensor { return x.MatMul(z).Square().Sum() }, x, z)
	checkGrad(t, "batched-transposed", func() *Tensor { return x.MatMul(y.Transpose(1, 2).Contiguous().Transpose(1, 2)).Mul(w).Sum() }, x, y)
	v := RandnFrom(rng, 5)
	checkGrad(t, "matvec", func() *Tensor { return x.MatMul(v).Square().Sum() }, x, v)
	v5 := RandnFrom(rng, 5)
	checkGrad(t, "vecmat", func() *Tensor { return v5.MatMul(z).Square().Sum() }, v5, z)
}

func TestGradSoftmaxLossesLayerNorm(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 6))
	x := RandnFrom(rng, 5, 6)
	w := RandnFrom(rng, 6, 4)
	targets := []int{0, 3, 1, 2, 3}
	weights := RandnFrom(rng, 5, 4)
	checkGrad(t, "softmax", func() *Tensor { return x.MatMul(w).Softmax(1).Mul(weights).Sum() }, x, w)
	checkGrad(t, "softmax-dim0", func() *Tensor { return x.MatMul(w).Softmax(0).Mul(weights).Sum() }, x, w)
	checkGrad(t, "logsoftmax", func() *Tensor { return x.MatMul(w).LogSoftmax(1).Mul(weights).Sum() }, x, w)
	checkGrad(t, "crossentropy", func() *Tensor { return CrossEntropy(x.MatMul(w), targets) }, x, w)
	tgt := RandnFrom(rng, 5, 4)
	checkGrad(t, "mse", func() *Tensor { return MSELoss(x.MatMul(w), tgt) }, x, w, tgt)
	x8 := RandnFrom(rng, 4, 8)
	gamma := Ones(8).Add(RandnFrom(rng, 8).MulScalar(0.1)).Detach()
	beta := RandnFrom(rng, 8)
	w8 := RandnFrom(rng, 4, 8)
	checkGrad(t, "layernorm", func() *Tensor { return LayerNorm(x8, gamma, beta, 1e-5).Mul(w8).Sum() }, x8, gamma, beta)
	x3 := RandnFrom(rng, 2, 3, 8)
	checkGrad(t, "layernorm-3d", func() *Tensor { return LayerNorm(x3, gamma, beta, 1e-5).Square().Sum() }, x3, gamma)
}

func TestGraphMechanics(t *testing.T) {
	x := New([]float32{1, 2, 3}, 3).SetRequiresGrad(true)
	// diamond: y used twice
	y := x.MulScalar(2)
	z := y.Mul(y).Add(y)
	z.Sum().Backward()
	// d/dx (4x² + 2x) = 8x + 2
	assertClose(t, "diamond", x.Grad(), New([]float32{10, 18, 26}, 3), 1e-6)

	// accumulation across Backward calls and ZeroGrad
	x.ZeroGrad()
	x.Sum().Backward()
	x.Sum().Backward()
	assertClose(t, "accumulate", x.Grad(), Full(2, 3), 0)
	x.ZeroGrad()
	assertClose(t, "zerograd", x.Grad(), Zeros(3), 0)

	// intermediate gradients are dropped unless retained
	h := x.Exp().RetainGrad()
	out := h.Mul(x)
	out.Sum().Backward()
	if out.Grad() != nil {
		t.Fatal("non-leaf gradient should be released")
	}
	assertClose(t, "retain", h.Grad(), x.Detach(), 1e-6)

	// NoGrad records nothing; Detach cuts the graph
	var ng *Tensor
	NoGrad(func() { ng = x.Mul(x) })
	if ng.RequiresGrad() || !ng.IsLeaf() {
		t.Fatal("NoGrad recorded a graph")
	}
	d := x.Mul(x).Detach()
	if d.RequiresGrad() || !d.IsLeaf() {
		t.Fatal("Detach must produce a leaf without grad")
	}
	if !x.Mul(x).RequiresGrad() || x.Mul(x).IsLeaf() {
		t.Fatal("op on grad tensor must record")
	}
	if !GradEnabled() {
		t.Fatal("grad should be enabled by default")
	}
	NoGrad(func() {
		if GradEnabled() {
			t.Fatal("GradEnabled inside NoGrad")
		}
	})

	// tensors without grad requirement carry no graph
	if Ones(2).Add(Ones(2)).RequiresGrad() {
		t.Fatal("grad requirement leaked")
	}

	// BackwardWith on a non-scalar
	x.ZeroGrad()
	x.MulScalar(3).BackwardWith(New([]float32{1, 0, -1}, 3))
	assertClose(t, "BackwardWith", x.Grad(), New([]float32{3, 0, -3}, 3), 0)

	// gradient of a strided leaf (view of a parameter) reaches the parameter
	p := Zeros(2, 3).SetRequiresGrad(true)
	p.T().Sum().Backward()
	assertClose(t, "view of leaf", p.Grad(), Ones(2, 3), 0)

	expectError(t, "Backward non-scalar", func() { x.Mul(x).Backward() })
	expectError(t, "Backward no grad", func() { Ones(1).Backward() })
	expectError(t, "BackwardWith shape", func() { x.Mul(x).BackwardWith(Ones(2)) })
	expectError(t, "SetRequiresGrad non-leaf", func() { x.Mul(x).SetRequiresGrad(true) })
}

func TestTrainingLoopConverges(t *testing.T) {
	// fit y = 3x - 1 with plain SGD
	rng := rand.New(rand.NewPCG(7, 7))
	xs := UniformFrom(rng, -1, 1, 64, 1)
	ys := xs.MulScalar(3).AddScalar(-1)
	w := Zeros(1, 1).SetRequiresGrad(true)
	b := Zeros(1).SetRequiresGrad(true)
	var loss *Tensor
	for step := 0; step < 300; step++ {
		loss = MSELoss(xs.MatMul(w).Add(b), ys)
		w.ZeroGrad()
		b.ZeroGrad()
		loss.Backward()
		NoGrad(func() {
			w.AddScaledInPlace(w.Grad(), -0.1)
			b.AddScaledInPlace(b.Grad(), -0.1)
		})
	}
	if loss.Item() > 1e-4 || !approx(w.Item(), 3, 1e-2) || !approx(b.Item(), -1, 1e-2) {
		t.Fatalf("did not converge: loss=%v w=%v b=%v", loss.Item(), w.Item(), b.Item())
	}
}

func TestBackwardReleasesIntermediates(t *testing.T) {
	defer SetReleaseGraph(true)
	SetReleaseGraph(true)
	x := Randn(64, 1<<12).SetRequiresGrad(true) // 1 MiB intermediates: off-heap
	h := x.MulScalar(2)
	y := h.Tanh()
	loss := y.Sum()
	loss.Backward()
	if !h.released || !y.released {
		t.Fatal("intermediates not released after Backward")
	}
	if x.Grad() == nil || x.released || loss.released {
		t.Fatal("leaf gradient or root affected by release")
	}
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("reading a released intermediate did not panic")
		}
	}()
	_ = h.Float32s()
}

func TestBackwardKeepsSharedIntermediates(t *testing.T) {
	defer SetReleaseGraph(true)
	x := New([]float32{1, 2, 3}, 3).SetRequiresGrad(true)
	h := x.MulScalar(3)       // consumed by two graphs
	loss1 := h.Square().Sum() // d/dx = 2·9x = 18x
	loss2 := h.Sum()          // d/dx = 3
	loss1.Backward()
	if h.released {
		t.Fatal("intermediate released while another graph still consumes it")
	}
	loss2.Backward()
	if !h.released {
		t.Fatal("intermediate not released after its last consumer")
	}
	for i, v := range x.Grad().Float32s() {
		want := 18*float32(i+1) + 3
		if !approx(v, want, 1e-4) {
			t.Fatalf("grad[%d] = %v, want %v", i, v, want)
		}
	}
	// RetainGrad and the switch keep intermediates
	x2 := New([]float32{1, 2, 3}, 3).SetRequiresGrad(true)
	h2 := x2.MulScalar(3).RetainGrad()
	h2.Sum().Backward()
	if h2.released || h2.Grad() == nil {
		t.Fatal("RetainGrad intermediate released")
	}
	SetReleaseGraph(false)
	h3 := x2.MulScalar(3)
	h3.Sum().Backward()
	if h3.released || h3.Float32s()[0] != 3 {
		t.Fatal("intermediate released although the switch is off")
	}
}

func TestLayerNormNumericGradient(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	x := RandFrom(rng, 7, 33).MulScalar(3).AddScalar(1).SetRequiresGrad(true)
	gamma := RandFrom(rng, 33).AddScalar(0.5).SetRequiresGrad(true)
	beta := RandFrom(rng, 33).SetRequiresGrad(true)
	w := RandFrom(rng, 7, 33) // random weighting so every output element matters
	loss := func() *Tensor { return LayerNorm(x, gamma, beta, 1e-5).Mul(w).Sum() }
	loss().Backward()
	check := func(name string, p *Tensor) {
		grad := p.Grad().Float32s()
		d := p.Data()
		for i := 0; i < len(d); i += max(1, len(d)/9) {
			const h = 1e-2
			orig := d[i]
			d[i] = orig + h
			var lp, lm float32
			NoGrad(func() { lp = loss().Item() })
			d[i] = orig - h
			NoGrad(func() { lm = loss().Item() })
			d[i] = orig
			num := (lp - lm) / (2 * h)
			if !approx(grad[i], num, 2e-2) {
				t.Fatalf("%s[%d]: analytic %v, numeric %v", name, i, grad[i], num)
			}
		}
	}
	check("x", x)
	check("gamma", gamma)
	check("beta", beta)
}

func TestCrossEntropyWeighted(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	logits := RandFrom(rng, 6, 3).MulScalar(3).SetRequiresGrad(true)
	targets := []int{0, 1, 2, 2, 2, 1}
	plain := CrossEntropy(logits, targets).Item()
	unit := CrossEntropyWeighted(logits, targets, []float32{1, 1, 1}).Item()
	if !approx(plain, unit, 1e-6) {
		t.Fatalf("unit weights differ: %v vs %v", plain, unit)
	}
	// hand computation: per-row loss weighted, divided by the total weight
	w := []float32{3, 1, 0.5}
	var want, norm float64
	lp := logits.LogSoftmax(1)
	for r, tg := range targets {
		want -= float64(lp.At(r, tg)) * float64(w[tg])
		norm += float64(w[tg])
	}
	got := CrossEntropyWeighted(logits, targets, w)
	if !approx(got.Item(), float32(want/norm), 1e-5) {
		t.Fatalf("weighted loss %v, want %v", got.Item(), want/norm)
	}
	// numeric gradient
	got.Backward()
	grad := logits.Grad().Float32s()
	d := logits.Data()
	for _, i := range []int{0, 4, 7, 11, 17} {
		const h = 1e-2
		orig := d[i]
		var lp, lm float32
		d[i] = orig + h
		NoGrad(func() { lp = CrossEntropyWeighted(logits, targets, w).Item() })
		d[i] = orig - h
		NoGrad(func() { lm = CrossEntropyWeighted(logits, targets, w).Item() })
		d[i] = orig
		if num := (lp - lm) / (2 * h); !approx(grad[i], num, 1e-2) {
			t.Fatalf("grad[%d] = %v, numeric %v", i, grad[i], num)
		}
	}
}
