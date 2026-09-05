package tensor

import (
	"runtime"
	"testing"
	"time"
)

// settle gives the garbage collector and the cleanup goroutine time to
// return unreachable storage to the pool.
func settle() {
	for i := 0; i < 3; i++ {
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
	}
}

func TestSizeClass(t *testing.T) {
	for _, n := range []int{1024, 1025, 1500, 2048, 3000, 4095, 4096, 4097, 1 << 20, 1<<20 + 1, 3 << 19} {
		class, capacity := sizeClass(n)
		if capacity < n {
			t.Errorf("n=%d: capacity %d too small", n, capacity)
		}
		if float64(capacity) > 1.26*float64(n) && n > minPooled {
			t.Errorf("n=%d: capacity %d wastes more than a quarter", n, capacity)
		}
		if c2, cap2 := sizeClass(capacity); c2 != class || cap2 != capacity {
			t.Errorf("n=%d: class not stable under its own capacity (%d/%d vs %d/%d)", n, class, capacity, c2, cap2)
		}
	}
}

func TestPoolReusesBuffers(t *testing.T) {
	SetPoolLimit(64 << 20)
	defer SetPoolLimit(0)
	hits0, _, _ := PoolStats()
	for i := 0; i < 40; i++ {
		x := Randn(1 << 18) // 1 MiB, pooled
		_ = x.Add(x)
		if i%5 == 4 {
			settle()
		}
	}
	settle()
	hits1, _, _ := PoolStats()
	if hits1 <= hits0 {
		t.Fatalf("no pool hits after churning allocations (hits %d -> %d)", hits0, hits1)
	}
}

func TestViewKeepsStorageAlive(t *testing.T) {
	SetPoolLimit(64 << 20)
	defer SetPoolLimit(0)
	var v *Tensor
	var want []float32
	func() {
		base := Arange(0, 1<<18, 1)
		v = base.Narrow(0, 100, 16)
		want = v.Float32s()
	}() // base is unreachable now, v still refers to its storage
	settle()
	for i := 0; i < 20; i++ {
		_ = Full(-1, 1<<18) // would land in the recycled buffer if it were free
	}
	settle()
	if got := v.Float32s(); !Equalf(got, want) {
		t.Fatalf("view content changed after base became unreachable: %v", got[:4])
	}
}

func TestDataEscapesStorage(t *testing.T) {
	SetPoolLimit(64 << 20)
	defer SetPoolLimit(0)
	var d []float32
	func() {
		z := Zeros(1 << 18)
		d = z.Data()
	}()
	settle()
	for i := 0; i < 20; i++ {
		_ = Full(7, 1<<18)
	}
	settle()
	for i, x := range d {
		if x != 0 {
			t.Fatalf("escaped buffer was recycled: d[%d] = %v", i, x)
		}
	}
}

func TestUninitPathsProduceCorrectValues(t *testing.T) {
	SetPoolLimit(64 << 20)
	defer SetPoolLimit(0)
	// churn the pool with garbage, then check ops that use uninitialised
	// results still compute correct values
	for i := 0; i < 10; i++ {
		_ = Full(123, 1<<16)
	}
	settle()
	x := Arange(0, 1<<16, 1)
	if s := x.Sum(0).Item(); s != float32((1<<16)*((1<<16)-1)/2) {
		t.Fatalf("Sum after churn: %v", s)
	}
	if !x.Add(x).Equal(x.MulScalar(2)) {
		t.Fatal("Add after churn")
	}
	c := Cat(0, Ones(1<<15), Zeros(1<<15))
	if c.Sum().Item() != 1<<15 {
		t.Fatal("Cat after churn")
	}
	if !x.Reshape(256, 256).T().Contiguous().T().Equal(x.Reshape(256, 256)) {
		t.Fatal("Contiguous after churn")
	}
	// the softmax rows must sum to one regardless of the buffer's history
	sm := Randn(64, 1024).Softmax(1).Sum(1)
	if !sm.AllClose(Ones(64), 1e-5, 1e-5) {
		t.Fatal("Softmax after churn")
	}
}

func TestSetPoolLimit(t *testing.T) {
	defer SetPoolLimit(0)
	SetPoolLimit(0)
	if _, _, retained := PoolStats(); retained != 0 {
		t.Fatalf("retained %d after disabling", retained)
	}
	x := Randn(1 << 18)
	if x.store.pooled {
		t.Fatal("allocation pooled while pooling is disabled")
	}
	SetPoolLimit(1 << 20) // 1 MiB: at most one 1 MiB-class buffer fits
	for i := 0; i < 8; i++ {
		_ = Randn(1 << 18)
	}
	settle()
	if _, _, retained := PoolStats(); retained > 1<<20 {
		t.Fatalf("retained %d exceeds the limit", retained)
	}
}

// Equalf compares two float32 slices exactly.
func Equalf(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
