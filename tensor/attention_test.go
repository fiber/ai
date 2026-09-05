package tensor

import (
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
		fused := attentionFused(q, k, v, mask)
		if fused == nil {
			t.Fatalf("%s: fused path declined", name)
		}
		composed := attentionComposed(q, k, v, mask)
		if !fused.AllClose(composed, 1e-4, 1e-5) {
			t.Fatalf("%s: fused and composed differ", name)
		}
	}
	// strided inputs: head split through Permute, and 3-D inputs
	x := RandnFrom(rng, B, T, H*D)
	split := x.Reshape(B, T, H, D).Permute(0, 2, 1, 3)
	if !attentionFused(split, split, split, nil).AllClose(attentionComposed(split, split, split, nil), 1e-4, 1e-5) {
		t.Fatal("permuted inputs differ")
	}
	q3, k3, v3 := RandnFrom(rng, 4, 9, 8), RandnFrom(rng, 4, 9, 8), RandnFrom(rng, 4, 9, 8)
	if !attentionFused(q3, k3, v3, CausalMask(9)).AllClose(attentionComposed(q3, k3, v3, CausalMask(9)), 1e-4, 1e-5) {
		t.Fatal("3-D causal differs")
	}
	// under grad recording the composed path is taken and gradients flow
	qg := RandnFrom(rng, 1, 1, 5, 4).SetRequiresGrad(true)
	Attention(qg, k3.Narrow(0, 0, 1).Narrow(1, 0, 5).Unsqueeze(0).Narrow(3, 0, 4), v3.Narrow(0, 0, 1).Narrow(1, 0, 5).Unsqueeze(0).Narrow(3, 0, 4), nil).Sum().Backward()
	if qg.Grad() == nil {
		t.Fatal("no gradient through Attention")
	}
}
