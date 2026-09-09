package tensor

import (
	"fmt"
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
	SetMappedLimit(-1)
	defer SetMappedLimit(512 << 20)
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
	SetMappedLimit(-1)
	defer SetMappedLimit(512 << 20)
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
	SetMappedLimit(-1)
	defer SetMappedLimit(512 << 20)
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
	SetMappedLimit(-1)
	defer SetMappedLimit(512 << 20)
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

func TestHeapBallast(t *testing.T) {
	if HeapBallast() != DefaultHeapBallast {
		t.Fatalf("default ballast %d", HeapBallast())
	}
	SetHeapBallast(0)
	if HeapBallast() != 0 {
		t.Fatal("ballast not removed")
	}
	SetHeapBallast(1 << 20)
	if HeapBallast() != 1<<20 {
		t.Fatal("ballast not set")
	}
	SetHeapBallast(DefaultHeapBallast)
}

// BenchmarkAllocate measures what a fresh result costs before any
// arithmetic happens: Go zero-fills every make on the allocating thread.
func TestMappedReusesBuffers(t *testing.T) {
	if !mmapSupported {
		t.Skip("no mmap on this platform")
	}
	hits0, _, _, _ := MappedStats()
	for i := 0; i < 40; i++ {
		x := Randn(1 << 18) // 1 MiB: off-heap
		if !x.store.mapped {
			t.Fatal("1 MiB result is not mapped")
		}
		_ = x.Add(x)
		if i%5 == 4 {
			settle()
		}
	}
	settle()
	hits1, _, retained, _ := MappedStats()
	if hits1 <= hits0 {
		t.Fatalf("no mapped hits after churning allocations (hits %d -> %d)", hits0, hits1)
	}
	if retained == 0 {
		t.Fatal("no mappings retained for reuse")
	}
}

func TestMappedViewKeepsStorageAlive(t *testing.T) {
	if !mmapSupported {
		t.Skip("no mmap on this platform")
	}
	var v *Tensor
	var want []float32
	func() {
		base := Arange(0, 1<<18, 1)
		v = base.Narrow(0, 100, 16)
		want = v.Float32s()
	}()
	settle()
	for i := 0; i < 20; i++ {
		_ = Full(-1, 1<<18)
	}
	settle()
	if got := v.Float32s(); !Equalf(got, want) {
		t.Fatalf("view content changed after base became unreachable: %v", got[:4])
	}
}

func TestMappedDataEscapes(t *testing.T) {
	if !mmapSupported {
		t.Skip("no mmap on this platform")
	}
	_, _, _, pinned0 := MappedStats()
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
			t.Fatalf("escaped mapping was recycled: d[%d] = %v", i, x)
		}
	}
	if _, _, _, pinned := MappedStats(); pinned <= pinned0 {
		t.Fatalf("escaped mapping not accounted as pinned (%d -> %d)", pinned0, pinned)
	}
}

func TestMappedZeroedAfterReuse(t *testing.T) {
	if !mmapSupported {
		t.Skip("no mmap on this platform")
	}
	for i := 0; i < 10; i++ {
		_ = Full(123, 1<<18)
	}
	settle()
	z := Zeros(1 << 18)
	if s := z.Sum().Item(); s != 0 {
		t.Fatalf("Zeros from a reused mapping sums to %v", s)
	}
	x := Arange(0, 1<<18, 1)
	if !x.Add(x).Equal(x.MulScalar(2)) {
		t.Fatal("Add on mapped buffers")
	}
	sm := Randn(256, 1024).Softmax(1).Sum(1)
	if !sm.AllClose(Ones(256), 1e-5, 1e-5) {
		t.Fatal("Softmax on mapped buffers")
	}
}

func TestSetMappedLimit(t *testing.T) {
	if !mmapSupported {
		t.Skip("no mmap on this platform")
	}
	defer SetMappedLimit(512 << 20)
	for i := 0; i < 8; i++ {
		_ = Full(1, 1<<20)
	}
	settle()
	SetMappedLimit(1 << 20)
	if _, _, retained, _ := MappedStats(); retained > 1<<20 {
		t.Fatalf("retained %d bytes above the limit", retained)
	}
	SetMappedLimit(-1)
	if x := Randn(1 << 18); x.store.mapped {
		t.Fatal("mapping still used after disabling it")
	}
}

// BenchmarkAllocate measures a fresh result buffer as handed to a kernel
// (untouched); BenchmarkAllocateTouch includes the first write, done in
// parallel as the kernels do it.
func BenchmarkAllocate(b *testing.B) {
	for _, n := range []int{1 << 16, 1 << 20, 1 << 24} {
		b.Run(fmtN(n), func(b *testing.B) {
			b.SetBytes(int64(4 * n))
			for i := 0; i < b.N; i++ {
				_ = newTensorUninit(Shape{n})
			}
		})
	}
}

func BenchmarkAllocateTouch(b *testing.B) {
	for _, n := range []int{1 << 16, 1 << 20, 1 << 24} {
		b.Run(fmtN(n), func(b *testing.B) {
			b.SetBytes(int64(4 * n))
			for i := 0; i < b.N; i++ {
				t := newTensorUninit(Shape{n})
				parallelClear(t.data)
			}
		})
	}
}

// BenchmarkAllocateTouchWarm is BenchmarkAllocateTouch in steady state:
// results are dropped and collected every 32 iterations, so most buffers
// come back from the free list and the forced GC is included in the
// figure.
func BenchmarkAllocateTouchWarm(b *testing.B) {
	if !mmapSupported {
		b.Skip("no mmap on this platform")
	}
	for _, n := range []int{1 << 16, 1 << 20, 1 << 24} {
		b.Run(fmtN(n), func(b *testing.B) {
			b.SetBytes(int64(4 * n))
			for i := 0; i < b.N; i++ {
				t := newTensorUninit(Shape{n})
				parallelClear(t.data)
				if i%32 == 31 {
					collectMapped()
				}
			}
		})
	}
}

func fmtN(n int) string {
	if n >= 1<<20 {
		return fmt.Sprintf("%dM", n>>20)
	}
	return fmt.Sprintf("%dK", n>>10)
}

func TestReleaseReusesImmediately(t *testing.T) {
	if !mmapSupported {
		t.Skip("no mmap on this platform")
	}
	x := Randn(1 << 18)
	p := &x.data[0]
	x.Release()
	if x.data != nil {
		t.Fatal("released tensor still holds data")
	}
	y := Zeros(1 << 18)
	if &y.data[0] != p {
		t.Fatal("next allocation of the same size did not get the released buffer")
	}
	if s := y.Sum().Item(); s != 0 {
		t.Fatalf("Zeros over a released buffer sums to %v", s)
	}
	y.Release()
	y.Release() // second call is a no-op
}

func TestReleaseRefusedWhenShared(t *testing.T) {
	if !mmapSupported {
		t.Skip("no mmap on this platform")
	}
	// a view keeps the storage
	x := Arange(0, 1<<18, 1)
	v := x.Narrow(0, 0, 4)
	x.Release()
	if x.data == nil || v.At(3) != 3 {
		t.Fatal("storage released although a view exists")
	}
	// Data() pins it
	d := Ones(1 << 18)
	_ = d.Data()
	d.Release()
	if d.data == nil {
		t.Fatal("storage released although Data() escaped")
	}
	// autograd holds it
	w := Randn(1 << 18).SetRequiresGrad(true)
	h := w.MulScalar(2)
	h.Release()
	if h.data == nil {
		t.Fatal("storage released although autograd recorded it")
	}
	w.Release()
	if w.data == nil {
		t.Fatal("storage released although the tensor requires grad")
	}
}

// TestForcedCollectionReclaimsSynchronously: results that were dropped
// without Release are back on the free list when the forced collection
// returns, not whenever the cleanup goroutine runs (T-044).
func TestForcedCollectionReclaimsSynchronously(t *testing.T) {
	if !mmapSupported {
		t.Skip("no off-heap storage on this platform")
	}
	const n = 1 << 20 // 4 MB each
	class, _ := sizeClass(n)
	for i := 0; i < 8; i++ {
		newTensorUninit(Shape{n}) // dropped, unreleased
	}
	mapPool.mu.Lock()
	mapPool.free[class] = nil // start from an empty free list for this class
	mapPool.retained = 0
	for _, l := range mapPool.free {
		for _, b := range l {
			mapPool.retained += cap(b) * 4
		}
	}
	mapPool.mu.Unlock()
	collectMapped()
	mapPool.mu.Lock()
	got := len(mapPool.free[class])
	mapPool.mu.Unlock()
	if got < 8 {
		t.Fatalf("forced collection returned %d of 8 dropped buffers to the free list", got)
	}
}
