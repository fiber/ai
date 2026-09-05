package tensor

import (
	"math"
	"math/rand/v2"
	"testing"
)

// naiveConv2D computes the convolution with loops; x [N,C,H,W], w [O,C,kh,kw].
func naiveConv2D(x, w, b *Tensor, stride, pad int) *Tensor {
	n, c, h, wd := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3)
	o, kh, kw := w.Dim(0), w.Dim(2), w.Dim(3)
	ho, wo := (h+2*pad-kh)/stride+1, (wd+2*pad-kw)/stride+1
	out := Zeros(n, o, ho, wo)
	for bi := 0; bi < n; bi++ {
		for oc := 0; oc < o; oc++ {
			for oy := 0; oy < ho; oy++ {
				for ox := 0; ox < wo; ox++ {
					var s float64
					if b != nil {
						s = float64(b.At(oc))
					}
					for ci := 0; ci < c; ci++ {
						for ky := 0; ky < kh; ky++ {
							for kx := 0; kx < kw; kx++ {
								iy, ix := oy*stride-pad+ky, ox*stride-pad+kx
								if iy < 0 || iy >= h || ix < 0 || ix >= wd {
									continue
								}
								s += float64(x.At(bi, ci, iy, ix)) * float64(w.At(oc, ci, ky, kx))
							}
						}
					}
					out.Set(float32(s), bi, oc, oy, ox)
				}
			}
		}
	}
	return out
}

func TestConv2DMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, tc := range []struct{ n, c, h, w, o, k, stride, pad int }{
		{2, 3, 7, 9, 4, 3, 1, 0}, {1, 2, 8, 8, 3, 3, 1, 1}, {2, 3, 9, 9, 2, 3, 2, 1}, {1, 1, 5, 5, 1, 5, 1, 2}, {3, 4, 6, 7, 5, 2, 2, 0},
	} {
		x := RandnFrom(rng, tc.n, tc.c, tc.h, tc.w)
		w := RandnFrom(rng, tc.o, tc.c, tc.k, tc.k)
		b := RandnFrom(rng, tc.o)
		got := Conv2D(x, w, b, tc.stride, tc.pad)
		want := naiveConv2D(x, w, b, tc.stride, tc.pad)
		if !got.Shape().Equal(want.Shape()) || !got.AllClose(want, 1e-4, 1e-4) {
			t.Fatalf("%+v: conv differs\n%v\n%v", tc, got.Shape(), want.Shape())
		}
	}
	// Conv1D against Conv2D with height one
	x := RandnFrom(rng, 2, 3, 20)
	w := RandnFrom(rng, 4, 3, 5)
	got := Conv1D(x, w, nil, 2, 2)
	want := naiveConv2D(x.Reshape(2, 3, 1, 20), w.Reshape(4, 3, 1, 5), nil, 2, 0) // no vertical pad possible: compare unpadded variant
	_ = want
	if !got.Shape().Equal(Shape{2, 4, 10}) {
		t.Fatalf("Conv1D shape %v", got.Shape())
	}
	// and against a padded naive computation
	xp := Cat(2, Zeros(2, 3, 2), x, Zeros(2, 3, 2))
	want1 := naiveConv2D(xp.Reshape(2, 3, 1, 24), w.Reshape(4, 3, 1, 5), nil, 2, 0).Reshape(2, 4, 10)
	if !got.AllClose(want1, 1e-4, 1e-4) {
		t.Fatal("Conv1D differs from the padded naive computation")
	}
}

func TestConvGradients(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	x := RandnFrom(rng, 2, 2, 6, 6).SetRequiresGrad(true)
	w := RandnFrom(rng, 3, 2, 3, 3).SetRequiresGrad(true)
	b := RandnFrom(rng, 3).SetRequiresGrad(true)
	wt := RandnFrom(rng, 2, 3, 3, 3) // stride 2, pad 1 → 3×3 output
	loss := func() *Tensor { return Conv2D(x, w, b, 2, 1).Mul(wt).Sum() }
	loss().Backward()
	for name, p := range map[string]*Tensor{"x": x, "w": w, "b": b} {
		grad := p.Grad().Float32s()
		d := p.Data()
		for i := 0; i < len(d); i += max(1, len(d)/7) {
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

func TestMaxPool2D(t *testing.T) {
	x := New([]float32{
		1, 5, 2, 0,
		3, 4, 8, 1,
		0, 2, 9, 7,
		6, 1, 3, 4,
	}, 1, 1, 4, 4).SetRequiresGrad(true)
	y := MaxPool2D(x, 2, 2)
	if !Equalf(y.Float32s(), []float32{5, 8, 6, 9}) {
		t.Fatalf("pool %v", y)
	}
	y.Mul(New([]float32{1, 2, 3, 4}, 1, 1, 2, 2)).Sum().Backward()
	g := x.Grad().Float32s()
	want := make([]float32, 16)
	want[1], want[6], want[12], want[10] = 1, 2, 3, 4
	if !Equalf(g, want) {
		t.Fatalf("pool gradient %v", g)
	}
	// overlapping windows: stride 1
	z := MaxPool2D(x, 3, 1)
	if z.Dim(2) != 2 || z.At(0, 0, 0, 0) != 9 || z.At(0, 0, 1, 1) != 9 {
		t.Fatalf("stride-1 pool %v", z)
	}
	_ = math.Pi
}
