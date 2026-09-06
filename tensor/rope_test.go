package tensor

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestRoPEKnown(t *testing.T) {
	// A single 2-D vector at position 1, base 10000: angle = 1 * 1 = 1 rad
	// for the first (and only) frequency pair.
	x := New([]float32{1, 0}, 1, 2)
	got := RoPE(x, 10000, []int{1}).Float32s()
	c, s := float32(math.Cos(1)), float32(math.Sin(1))
	if math.Abs(float64(got[0]-c)) > 1e-6 || math.Abs(float64(got[1]-s)) > 1e-6 {
		t.Fatalf("got %v, want [%v %v]", got, c, s)
	}
	// Position 0 is the identity.
	id := RoPE(New([]float32{3, -4, 5, 6}, 1, 4), 10000, []int{0}).Float32s()
	for i, v := range []float32{3, -4, 5, 6} {
		if math.Abs(float64(id[i]-v)) > 1e-6 {
			t.Fatalf("position 0 not identity: %v", id)
		}
	}
}

func TestRoPEPreservesNorm(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	x := RandnFrom(rng, 2, 3, 8, 16)
	y := RoPE(x, 1e6, []int{0, 5, 9, 100, 500, 1000, 3, 7})
	// Rotation is orthogonal: each row keeps its L2 norm.
	xn, yn := x.Square().Sum(-1).Float32s(), y.Square().Sum(-1).Float32s()
	for i := range xn {
		if math.Abs(float64(xn[i]-yn[i])) > 1e-3*float64(xn[i]+1) {
			t.Fatalf("norm changed at %d: %v vs %v", i, xn[i], yn[i])
		}
	}
}

func TestRoPEGradient(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	x := RandnFrom(rng, 1, 1, 4, 6).SetRequiresGrad(true)
	pos := []int{0, 1, 2, 3}
	loss := func() float32 { return RoPE(x, 10000, pos).Square().Sum().Item() }
	RoPE(x, 10000, pos).Square().Sum().Backward()
	g := x.Grad().Float32s()
	base := x.Float32s()
	const eps = 1e-2
	for i := range base {
		orig := base[i]
		xd := append([]float32(nil), base...)
		xd[i] = orig + eps
		xp := New(xd, 1, 1, 4, 6)
		lp := RoPE(xp, 10000, pos).Square().Sum().Item()
		xd[i] = orig - eps
		xm := New(xd, 1, 1, 4, 6)
		lm := RoPE(xm, 10000, pos).Square().Sum().Item()
		num := (lp - lm) / (2 * eps)
		if math.Abs(float64(num-g[i])) > 1e-2*(1+math.Abs(float64(g[i]))) {
			t.Fatalf("grad[%d] = %v, numeric %v", i, g[i], num)
		}
	}
	_ = loss
}

func TestWindowMask(t *testing.T) {
	m := WindowMask(5, 2).Float32s() // |i-j| < 2
	want := func(i, j int) float32 {
		if j-i >= 2 || i-j >= 2 {
			return maskNeg
		}
		return 0
	}
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			if m[i*5+j] != want(i, j) {
				t.Fatalf("mask[%d,%d] = %v", i, j, m[i*5+j])
			}
		}
	}
}

func TestRMSNormFusedMatchesComposed(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	x := RandnFrom(rng, 4, 5, 64)
	g := RandnFrom(rng, 64)
	fused := RMSNorm(x, g, 1e-6) // no grad: fused path
	// composed reference
	rms := x.Square().Mean(-1).AddScalar(1e-6).Sqrt().Unsqueeze(-1)
	comp := x.Div(rms).Mul(g)
	if !fused.AllClose(comp, 1e-5, 1e-6) {
		t.Fatal("fused RMSNorm differs from composed")
	}
	// 4-D input (per-head norm over the last dimension)
	x4 := RandnFrom(rng, 2, 3, 8, 16)
	g4 := RandnFrom(rng, 16)
	f4 := RMSNorm(x4, g4, 1e-6)
	r4 := x4.Square().Mean(-1).AddScalar(1e-6).Sqrt().Unsqueeze(-1)
	if !f4.AllClose(x4.Div(r4).Mul(g4), 1e-5, 1e-6) {
		t.Fatal("fused RMSNorm 4-D differs")
	}
}

func TestRecycleReusesAndForcesSharedRelease(t *testing.T) {
	// A large mapped tensor with a live view: Release refuses, Recycle frees.
	big := Zeros(200, 768) // >128 KiB, mapped
	_ = big.Reshape(200 * 768)
	big.Release() // refused: shared
	if big.released {
		t.Fatal("Release should refuse a shared tensor")
	}
	big.Recycle()
	if !big.released {
		t.Fatal("Recycle should have freed the shared tensor")
	}
	// The freed buffer is reused by the next same-size allocation.
	reuse := Zeros(200, 768)
	reuse.Recycle()
}
