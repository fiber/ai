package data

import (
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/fiber/ai/tensor"
)

func TestRatesHandleResetsAndGaps(t *testing.T) {
	// 60 s interval, 1 Gbit/s link = 125e6 bytes/s
	c := []float64{1000, 7000, 13000, 500, 6500, math.NaN(), 2000, 1e12}
	r := Rates(c, time.Minute, 125e6)
	want := []float64{100, 100, 500.0 / 60, 100, math.NaN(), math.NaN(), math.NaN()}
	for i := range want {
		if math.IsNaN(want[i]) != (r[i] != r[i]) || (!math.IsNaN(want[i]) && math.Abs(float64(r[i])-want[i]) > 1e-3) {
			t.Fatalf("rate[%d] = %v, want %v (all: %v)", i, r[i], want[i], r)
		}
	}
}

func TestWindowsLineUp(t *testing.T) {
	s := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	x, y, pos := Windows(s, 3, 2, func(t int) []float32 { return []float32{float32(t)} })
	// t = 3: input [0 1 2 | 3], target s[4]; last t = 6: input [3 4 5 | 6], target s[7]
	if x.Dim(0) != 4 || x.Dim(1) != 4 || y.Dim(0) != 4 {
		t.Fatalf("shapes %v %v", x.Shape(), y.Shape())
	}
	if x.At(0, 0) != 0 || x.At(0, 2) != 2 || x.At(0, 3) != 3 || y.At(0, 0) != 4 || pos[0] != 3 {
		t.Fatalf("first window %v -> %v at %d", x.Row(0), y.At(0, 0), pos[0])
	}
	if x.At(3, 0) != 3 || y.At(3, 0) != 7 {
		t.Fatalf("last window %v -> %v", x.Row(3), y.At(3, 0))
	}
	nan := float32(math.NaN())
	x2, _, pos2 := Windows([]float32{0, 1, nan, 3, 4, 5, 6}, 2, 1, nil)
	if x2.Dim(0) != 2 || pos2[0] != 5 { // windows touching the NaN are dropped
		t.Fatalf("NaN handling: %v %v", x2, pos2)
	}
}

func TestSplitAndStandardizer(t *testing.T) {
	tensor.Seed(2)
	x := tensor.Randn(100, 3).Mul(tensor.New([]float32{1, 50, 0}, 3)).Add(tensor.New([]float32{0, 1000, 7}, 3))
	y := tensor.Randn(100, 1)
	xt, yt, xv, yv := SplitByTime(x, y, 0.8)
	if xt.Dim(0) != 80 || xv.Dim(0) != 20 || yt.Dim(0) != 80 || yv.Dim(0) != 20 {
		t.Fatal("split sizes")
	}
	if xv.At(0, 1) != x.At(80, 1) {
		t.Fatal("validation does not start where training ends")
	}
	s := Fit(xt)
	z := s.Transform(xt)
	for c, m := range z.Mean(0).Float32s() {
		if math.Abs(float64(m)) > 1e-4 {
			t.Fatalf("column %d mean %v after standardising", c, m)
		}
	}
	sd := z.Std(0).Float32s()
	if math.Abs(float64(sd[1])-1) > 1e-4 || sd[2] != 0 { // constant column stays constant, at zero
		t.Fatalf("stds %v", sd)
	}
	back := s.Inverse(z)
	if !back.AllClose(xt, 1e-4, 1e-3) {
		t.Fatal("inverse does not round-trip")
	}
}

func TestBatchesCoverEveryIndexOnce(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 1))
	seen := make([]int, 103)
	batches := 0
	for idx := range Batches(103, 10, r) {
		batches++
		for _, i := range idx {
			seen[i]++
		}
	}
	if batches != 11 {
		t.Fatalf("%d batches", batches)
	}
	for i, n := range seen {
		if n != 1 {
			t.Fatalf("index %d seen %d times", i, n)
		}
	}
	if len(TimeFeatures(time.Date(2026, 9, 8, 0, 0, 0, 0, time.UTC))) != 4 {
		t.Fatal("time features")
	}
}
