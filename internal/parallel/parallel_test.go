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
