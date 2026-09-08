package tensor

import (
	"math/rand/v2"
	"testing"
)

// The cache must never change a result: products with a cached operand
// equal products with packing per call, an in-place change of the operand
// invalidates the entry, and an operand whose buffer escaped through
// Data() is never cached.
func TestPackedCacheCorrectness(t *testing.T) {
	SetPackedCacheLimit(512 << 20)
	defer SetPackedCacheLimit(512 << 20)
	rng := rand.New(rand.NewPCG(21, 22))
	x := RandnFrom(rng, 96, 300)
	w := RandnFrom(rng, 300, 400) // 120K elements: eligible
	ref := x.MatMul(w).Float32s()

	// second and third products: the second sighting packs, the third hits
	for i := 0; i < 3; i++ {
		if got := x.MatMul(w).Float32s(); !equalSlices(got, ref, 1e-5) {
			t.Fatalf("product %d differs from the uncached one", i)
		}
	}
	h0, _, bytes := PackedCacheStats()
	if h0 == 0 || bytes == 0 {
		t.Fatalf("expected a cache hit and bytes held, got hits=%d bytes=%d", h0, bytes)
	}

	// in-place change: the product must follow the new values
	w.MulScalarInPlace(2)
	ref2 := make([]float32, len(ref))
	for i := range ref {
		ref2[i] = 2 * ref[i]
	}
	for i := 0; i < 3; i++ {
		if got := x.MatMul(w).Float32s(); !equalSlices(got, ref2, 1e-4) {
			t.Fatalf("product %d after in-place change is stale", i)
		}
	}
	// Set: one element
	w.Set(123, 0, 0)
	ref3 := x.MatMul(w.Clone()).Float32s() // clone: fresh storage, packed per call
	if got := x.MatMul(w).Float32s(); !equalSlices(got, ref3, 1e-4) {
		t.Fatal("product after Set is stale")
	}

	// Data() escape: writes through the slice must be seen, so never cached
	v := RandnFrom(rng, 300, 400)
	x.MatMul(v)
	x.MatMul(v)
	x.MatMul(v)
	d := v.Data()
	hBefore, _, _ := PackedCacheStats()
	d[7] = 99
	x.MatMul(v)
	hAfter, _, _ := PackedCacheStats()
	if hAfter != hBefore {
		t.Fatal("an operand that escaped through Data() must not be served from the cache")
	}
	want := x.MatMul(v.Clone()).Float32s()
	if got := x.MatMul(v).Float32s(); !equalSlices(got, want, 1e-4) {
		t.Fatal("product after a write through Data() is stale")
	}

	// transposed view of a cached root is a separate key and also correct
	wt := w.T()
	y := RandnFrom(rng, 50, 400)
	wantT := y.MatMul(wt.Contiguous()).Float32s()
	for i := 0; i < 3; i++ {
		if got := y.MatMul(wt).Float32s(); !equalSlices(got, wantT, 1e-4) {
			t.Fatalf("transposed product %d differs", i)
		}
	}
	// limit 0 disables and drops everything
	SetPackedCacheLimit(0)
	if _, _, b := PackedCacheStats(); b != 0 {
		t.Fatalf("bytes held after disabling: %d", b)
	}
	if got := x.MatMul(w).Float32s(); !equalSlices(got, ref3, 1e-4) {
		t.Fatal("product with the cache disabled differs")
	}
}

func equalSlices(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > tol*(1+abs32(b[i])) {
			return false
		}
	}
	return true
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
