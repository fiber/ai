// Package parallel distributes independent work items across goroutines.
//
// It is the single place in fiber/ai that decides how many goroutines a
// compute-bound operation uses. Work is scheduled dynamically (an atomic
// counter hands out items), which keeps heterogeneous cores (e.g. Apple
// performance/efficiency cores) busy without static partitioning.
package parallel

import (
	"runtime"
	"sync"
	"sync/atomic"
)

var maxWorkers atomic.Int64

func init() { maxWorkers.Store(int64(runtime.GOMAXPROCS(0))) }

// SetWorkers sets the maximum number of goroutines used by For and Range.
// A value <= 0 resets it to runtime.GOMAXPROCS(0).
func SetWorkers(n int) {
	if n <= 0 {
		n = runtime.GOMAXPROCS(0)
	}
	maxWorkers.Store(int64(n))
}

// Workers returns the maximum number of goroutines used for parallel work.
func Workers() int { return int(maxWorkers.Load()) }

// For calls fn(i) for every i in [0, n), spreading calls over up to
// Workers() goroutines. It returns once all calls have completed. A panic
// inside fn is re-raised in the calling goroutine.
func For(n int, fn func(i int)) { ForWorkers(n, Workers(), fn) }

// ForWorkers is like For but with an explicit upper bound on goroutines.
// workers <= 1 runs everything on the calling goroutine.
func ForWorkers(n, workers int, fn func(i int)) {
	if n <= 0 {
		return
	}
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}

	var (
		next   atomic.Int64
		wg     sync.WaitGroup
		once   sync.Once
		panic_ any
	)
	work := func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				once.Do(func() { panic_ = r })
				next.Store(int64(n)) // stop other workers early
			}
		}()
		for {
			i := int(next.Add(1)) - 1
			if i >= n {
				return
			}
			fn(i)
		}
	}
	wg.Add(workers)
	for w := 1; w < workers; w++ {
		go work()
	}
	work() // the caller participates
	wg.Wait()
	if panic_ != nil {
		panic(panic_)
	}
}

// Range splits [0, n) into contiguous chunks of at least minChunk elements
// and calls fn(lo, hi) for every chunk in parallel. Small ranges run inline.
func Range(n, minChunk int, fn func(lo, hi int)) {
	RangeWorkers(n, minChunk, Workers(), fn)
}

// RangeWorkers is like Range with an explicit upper bound on goroutines.
func RangeWorkers(n, minChunk, workers int, fn func(lo, hi int)) {
	if n <= 0 {
		return
	}
	if minChunk < 1 {
		minChunk = 1
	}
	chunks := workers * 4 // oversubscribe for dynamic load balancing
	if maxChunks := (n + minChunk - 1) / minChunk; chunks > maxChunks {
		chunks = maxChunks
	}
	if chunks <= 1 || workers <= 1 {
		fn(0, n)
		return
	}
	size := (n + chunks - 1) / chunks
	chunks = (n + size - 1) / size
	ForWorkers(chunks, workers, func(c int) {
		lo := c * size
		hi := min(lo+size, n)
		fn(lo, hi)
	})
}
