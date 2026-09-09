package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// naiveAttention computes attention with plain loops for one (batch,
// head): q [T×D], k, v [S×D], mask [T×S] or nil.
func naiveAttention(q, k, v [][]float64, mask [][]float64) [][]float64 {
	T, S, D := len(q), len(k), len(q[0])
	out := make([][]float64, T)
	for i := 0; i < T; i++ {
		w := make([]float64, S)
		mx := math.Inf(-1)
		for j := 0; j < S; j++ {
			var s float64
			for c := 0; c < D; c++ {
				s += q[i][c] * k[j][c]
			}
			s /= math.Sqrt(float64(D))
			if mask != nil {
				s += mask[i][j]
			}
			w[j] = s
			mx = math.Max(mx, s)
		}
		var sum float64
		for j := range w {
			w[j] = math.Exp(w[j] - mx)
			sum += w[j]
		}
		out[i] = make([]float64, D)
		for j := 0; j < S; j++ {
			for c := 0; c < D; c++ {
				out[i][c] += w[j] / sum * v[j][c]
			}
		}
	}
	return out
}

func rows(t *Tensor, b, h int) [][]float64 {
	T, D := t.Dim(-2), t.Dim(-1)
	out := make([][]float64, T)
	for i := range out {
		out[i] = make([]float64, D)
		for c := range out[i] {
			out[i][c] = float64(t.At(b, h, i, c))
		}
	}
	return out
}

func TestAttentionMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 42))
	B, H, T, S, D := 2, 3, 5, 7, 4
	q := RandnFrom(rng, B, H, T, D)
	k := RandnFrom(rng, B, H, S, D)
	v := RandnFrom(rng, B, H, S, D)
	for _, withMask := range []bool{false, true} {
		var mask *Tensor
		var nm [][]float64
		if withMask {
			mask = PaddingMask([]int{S, 3}, S) // second batch element: only 3 keys visible
		}
		out := Attention(q, k, v, mask)
		for b := 0; b < B; b++ {
			for h := 0; h < H; h++ {
				if withMask {
					nm = make([][]float64, T)
					for i := range nm {
						nm[i] = make([]float64, S)
						for j := range nm[i] {
							nm[i][j] = float64(mask.At(b, 0, 0, j))
						}
					}
				}
				want := naiveAttention(rows(q, b, h), rows(k, b, h), rows(v, b, h), nm)
				for i := 0; i < T; i++ {
					for c := 0; c < D; c++ {
						if got := float64(out.At(b, h, i, c)); math.Abs(got-want[i][c]) > 1e-4 {
							t.Fatalf("mask=%v b=%d h=%d out[%d,%d] = %v, want %v", withMask, b, h, i, c, got, want[i][c])
						}
					}
				}
			}
		}
	}
}

func TestCausalMaskHidesTheFuture(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	T, D := 6, 8
	q := RandnFrom(rng, 1, 1, T, D)
	k := RandnFrom(rng, 1, 1, T, D)
	v := RandnFrom(rng, 1, 1, T, D)
	out := Attention(q, k, v, CausalMask(T))
	// changing a future key/value must not change earlier outputs
	k2, v2 := k.Float32s(), v.Float32s()
	for c := 0; c < D; c++ {
		k2[(T-1)*D+c] += 5
		v2[(T-1)*D+c] -= 3
	}
	out2 := Attention(q, New(k2, 1, 1, T, D), New(v2, 1, 1, T, D), CausalMask(T))
	for i := 0; i < T-1; i++ {
		for c := 0; c < D; c++ {
			if out.At(0, 0, i, c) != out2.At(0, 0, i, c) {
				t.Fatalf("position %d changed when the last key changed", i)
			}
		}
	}
	if out.At(0, 0, T-1, 0) == out2.At(0, 0, T-1, 0) {
		t.Fatal("the last position ignored its own key")
	}
}

func TestRMSNormAndGradient(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	x := RandnFrom(rng, 4, 6).MulScalar(2).SetRequiresGrad(true)
	g := RandFrom(rng, 6).AddScalar(0.5).SetRequiresGrad(true)
	y := RMSNorm(x, g, 1e-6)
	for i := 0; i < 4; i++ {
		var ms float64
		for j := 0; j < 6; j++ {
			ms += float64(x.At(i, j)) * float64(x.At(i, j))
		}
		rms := math.Sqrt(ms/6 + 1e-6)
		for j := 0; j < 6; j++ {
			want := float64(x.At(i, j)) / rms * float64(g.At(j))
			if math.Abs(float64(y.At(i, j))-want) > 1e-5 {
				t.Fatalf("y[%d,%d] = %v, want %v", i, j, y.At(i, j), want)
			}
		}
	}
	w := RandFrom(rng, 4, 6)
	loss := func() *Tensor { return RMSNorm(x, g, 1e-6).Mul(w).Sum() }
	loss().Backward()
	for name, p := range map[string]*Tensor{"x": x, "g": g} {
		grad := p.Grad().Float32s()
		d := p.Data()
		for i := 0; i < len(d); i += 5 {
			const h = 1e-2
			orig := d[i]
			var lp, lm float32
			d[i] = orig + h
			NoGrad(func() { lp = loss().Item() })
			d[i] = orig - h
			NoGrad(func() { lm = loss().Item() })
			d[i] = orig
			if num := (lp - lm) / (2 * h); !approx(grad[i], num, 2e-2) {
				t.Fatalf("%s[%d]: analytic %v, numeric %v", name, i, grad[i], num)
			}
		}
	}
}

func TestFusedAttentionMatchesComposed(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	B, H, T, S, D := 2, 3, 70, 45, 16 // T not a multiple of the block
	q := RandnFrom(rng, B, H, T, D)
	k := RandnFrom(rng, B, H, S, D)
	v := RandnFrom(rng, B, H, S, D)
	causal := CausalMask(T) // only valid when S == T; use a padding mask and a [T×S] random mask here
	_ = causal
	tsMask := RandnFrom(rng, T, S)
	pad := PaddingMask([]int{S, 20}, S)
	both := tsMask.Add(pad) // [B×1×T×S]
	cases := map[string]*Tensor{"none": nil, "TxS": tsMask, "padding": pad, "sum": both}
	for name, mask := range cases {
		sc := float32(1 / math.Sqrt(float64(D)))
		fused := attentionFused(q, k, v, mask, sc)
		if fused == nil {
			t.Fatalf("%s: fused path declined", name)
		}
		composed := attentionComposed(q, k, v, mask, sc)
		if !fused.AllClose(composed, 1e-4, 1e-5) {
			t.Fatalf("%s: fused and composed differ", name)
		}
	}
	// strided inputs: head split through Permute, and 3-D inputs
	x := RandnFrom(rng, B, T, H*D)
	split := x.Reshape(B, T, H, D).Permute(0, 2, 1, 3)
	if !attentionFused(split, split, split, nil, float32(1/math.Sqrt(float64(D)))).AllClose(attentionComposed(split, split, split, nil, float32(1/math.Sqrt(float64(D)))), 1e-4, 1e-5) {
		t.Fatal("permuted inputs differ")
	}
	q3, k3, v3 := RandnFrom(rng, 4, 9, 8), RandnFrom(rng, 4, 9, 8), RandnFrom(rng, 4, 9, 8)
	if !attentionFused(q3, k3, v3, CausalMask(9), float32(1/math.Sqrt(8.0))).AllClose(attentionComposed(q3, k3, v3, CausalMask(9), float32(1/math.Sqrt(8.0))), 1e-4, 1e-5) {
		t.Fatal("3-D causal differs")
	}
	// under grad recording the composed path is taken and gradients flow
	qg := RandnFrom(rng, 1, 1, 5, 4).SetRequiresGrad(true)
	Attention(qg, k3.Narrow(0, 0, 1).Narrow(1, 0, 5).Unsqueeze(0).Narrow(3, 0, 4), v3.Narrow(0, 0, 1).Narrow(1, 0, 5).Unsqueeze(0).Narrow(3, 0, 4), nil).Sum().Backward()
	if qg.Grad() == nil {
		t.Fatal("no gradient through Attention")
	}
}

// TestFusedAttentionShapes runs the fused path against the composed one
// over the shapes the micro-kernel driver has to pad: rows that are not a
// multiple of MR, keys that are not a multiple of NR, small and odd head
// dimensions, single rows, long sequences; with no mask, a random [T×S]
// mask, a padding mask, and (square) causal and window masks; on 3-D,
// 4-D and 5-D inputs; and grouped-query views where every head of a batch
// shares one packed K and V.
func TestFusedAttentionShapes(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 22))
	shapes := []struct{ T, S, D int }{
		{512, 512, 64}, {65, 65, 40}, {33, 7, 3}, {1, 512, 64}, {65, 1753, 256},
		{512, 65, 64}, {33, 1753, 64}, {1, 7, 3}, {65, 512, 40}, {7, 7, 1},
	}
	check := func(name string, q, k, v, mask *Tensor, D int) {
		t.Helper()
		sc := float32(1 / math.Sqrt(float64(D)))
		fused := attentionFused(q, k, v, mask, sc)
		if fused == nil {
			t.Fatalf("%s: fused path declined", name)
		}
		composed := attentionComposed(q, k, v, mask, sc)
		if !fused.AllClose(composed, 1e-4, 1e-5) {
			t.Fatalf("%s: fused and composed differ", name)
		}
	}
	for _, sh := range shapes {
		T, S, D := sh.T, sh.S, sh.D
		B, H := 2, 3
		if T*S*D > 1<<24 {
			B, H = 1, 2
		}
		q := RandnFrom(rng, B, H, T, D)
		k := RandnFrom(rng, B, H, S, D)
		v := RandnFrom(rng, B, H, S, D)
		lens := make([]int, B) // full length for batch 0, half for the rest
		for i := range lens {
			lens[i] = S
			if i > 0 {
				lens[i] = max(1, S/2)
			}
		}
		masks := map[string]*Tensor{"none": nil, "TxS": RandnFrom(rng, T, S), "padding": PaddingMask(lens, S)}
		if T == S {
			masks["causal"] = CausalMask(T)
			masks["window"] = WindowMask(T, max(1, T/4))
		}
		for name, mask := range masks {
			check(fmt.Sprintf("[%d×%d×%d×%d] %s", B, H, T, D, name), q, k, v, mask, D)
		}
		// grouped-query: one key/value head per batch, expanded over H
		k1 := RandnFrom(rng, B, 1, S, D)
		v1 := RandnFrom(rng, B, 1, S, D)
		check(fmt.Sprintf("[%d×%d×%d×%d] gqa", B, H, T, D), q, k1.Expand(B, H, S, D), v1.Expand(B, H, S, D), masks["padding"], D)
		// 3-D and 5-D leading shapes
		check(fmt.Sprintf("[%d×%d×%d] 3-D", H, T, D), q.Narrow(0, 0, 1).Squeeze(0), k.Narrow(0, 0, 1).Squeeze(0), v.Narrow(0, 0, 1).Squeeze(0), masks["TxS"], D)
		if T*S*D <= 1<<22 {
			check(fmt.Sprintf("5-D [%d×%d×%d]", T, S, D), RandnFrom(rng, 2, 2, 2, T, D), RandnFrom(rng, 2, 2, 2, S, D), RandnFrom(rng, 2, 2, 2, S, D), nil, D)
		}
	}
}

// TestMaskedAttentionWeightsAreExactZeros: with the exp kernels flushing
// sub-threshold inputs to 0 (B-004), masked positions carry weight exactly
// 0, so a causal attention's output at position 0 equals v[0] and the
// softmax of a masked row has no denormals.
func TestMaskedAttentionWeightsAreExactZeros(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	n, d := 64, 16
	q, k, v := RandnFrom(rng, 1, 1, n, d), RandnFrom(rng, 1, 1, n, d), RandnFrom(rng, 1, 1, n, d)
	var out *Tensor
	NoGrad(func() { out = Attention(q, k, v, CausalMask(n)) })
	if !out.Narrow(2, 0, 1).AllClose(v.Narrow(2, 0, 1), 1e-6, 1e-6) {
		t.Fatal("position 0 with a causal mask must reproduce v[0] exactly")
	}
	row := RandnFrom(rng, 1, n).Add(CausalMask(n).Narrow(0, 0, 1)) // first row: only column 0 unmasked
	w := row.Softmax(-1).Float32s()
	for j := 1; j < n; j++ {
		if w[j] != 0 {
			t.Fatalf("masked softmax weight %d = %g, want exactly 0", j, w[j])
		}
	}
	if w[0] != 1 {
		t.Fatalf("unmasked weight = %g, want 1", w[0])
	}
}
