package parallel

import (
	"sync/atomic"
	"testing"
)

func TestForCoversAll(t *testing.T) {
	for _, n := range []int{0, 1, 2, 7, 100, 1000} {
		for _, workers := range []int{1, 2, 4, 16} {
			seen := make([]int32, n)
			ForWorkers(n, workers, func(i int) { atomic.AddInt32(&seen[i], 1) })
			for i, s := range seen {
				if s != 1 {
					t.Fatalf("n=%d workers=%d: item %d executed %d times", n, workers, i, s)
				}
			}
		}
	}
}

func TestRangeCoversAll(t *testing.T) {
	for _, n := range []int{0, 1, 5, 64, 1000, 12345} {
		for _, minChunk := range []int{1, 8, 500, 100000} {
			for _, workers := range []int{1, 3, 8} {
				seen := make([]int32, n)
				RangeWorkers(n, minChunk, workers, func(lo, hi int) {
					if hi-lo <= 0 {
						t.Errorf("empty chunk %d..%d", lo, hi)
					}
					for i := lo; i < hi; i++ {
						atomic.AddInt32(&seen[i], 1)
					}
				})
				for i, s := range seen {
					if s != 1 {
						t.Fatalf("n=%d minChunk=%d workers=%d: item %d seen %d times", n, minChunk, workers, i, s)
					}
				}
			}
		}
	}
}

func TestForPropagatesPanic(t *testing.T) {
	defer func() {
		r := recover()
		if r != "boom" {
			t.Fatalf("expected panic 'boom', got %v", r)
		}
	}()
	ForWorkers(100, 4, func(i int) {
		if i == 42 {
			panic("boom")
		}
	})
}

func TestSetWorkers(t *testing.T) {
	old := Workers()
	defer SetWorkers(old)
	SetWorkers(3)
	if Workers() != 3 {
		t.Fatalf("Workers() = %d, want 3", Workers())
	}
	SetWorkers(0)
	if Workers() < 1 {
		t.Fatalf("Workers() = %d after reset", Workers())
	}
}

func TestNestedForDoesNotDeadlock(t *testing.T) {
	var total atomic.Int64
	ForWorkers(8, 4, func(i int) {
		ForWorkers(8, 4, func(j int) {
			ForWorkers(4, 4, func(k int) { total.Add(1) })
		})
	})
	if total.Load() != 8*8*4 {
		t.Fatalf("nested total %d", total.Load())
	}
}

func TestManyRoundsReuseHelpers(t *testing.T) {
	// thousands of small back-to-back rounds must neither leak goroutines
	// nor lose items
	var total atomic.Int64
	for r := 0; r < 5000; r++ {
		ForWorkers(16, 8, func(i int) { total.Add(1) })
	}
	if total.Load() != 5000*16 {
		t.Fatalf("total %d", total.Load())
	}
	if h := helpers.Load(); h > int64(max(Workers(), 16)) {
		t.Fatalf("%d helpers for %d workers", h, Workers())
	}
}

func TestPanicStopsRemainingItems(t *testing.T) {
	var ran atomic.Int64
	defer func() {
		if r := recover(); r != "boom" {
			t.Fatalf("expected panic 'boom', got %v", r)
		}
		if ran.Load() > 9000 {
			t.Fatalf("%d items ran after a panic", ran.Load())
		}
	}()
	ForWorkers(10000, 4, func(i int) {
		if i == 10 {
			panic("boom")
		}
		ran.Add(1)
	})
}

func TestRangePropagatesPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("panic not propagated from Range")
		}
	}()
	RangeWorkers(1000, 1, 4, func(lo, hi int) {
		if lo == 0 {
			panic("boom")
		}
	})
}

func TestNestedRangeDoesNotDeadlock(t *testing.T) {
	var outer, count atomic.Int64
	for round := 0; round < 20; round++ {
		RangeWorkers(64, 1, 8, func(lo, hi int) {
			outer.Add(1)
			RangeWorkers(64, 1, 8, func(lo2, hi2 int) { count.Add(int64(hi2 - lo2)) })
		})
	}
	if count.Load() != outer.Load()*64 {
		t.Fatalf("nested ranges covered %d items over %d outer chunks", count.Load(), outer.Load())
	}
}

func TestRangeWithMoreHelpersThanWorkers(t *testing.T) {
	// a wide job spawns helpers; a narrower owned job afterwards must still
	// complete although most helpers are not among its workers
	ForWorkers(64, 16, func(int) {})
	var count atomic.Int64
	RangeWorkers(1<<12, 1, 2, func(lo, hi int) { count.Add(int64(hi - lo)) })
	if count.Load() != 1<<12 {
		t.Fatalf("covered %d items", count.Load())
	}
}
