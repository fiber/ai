// Package parallel distributes independent work items across goroutines.
//
// It is the single place in fiber/ai that decides how many goroutines a
// compute-bound operation uses. Work is scheduled dynamically (an atomic
// counter hands out items), which keeps heterogeneous cores (e.g. Apple
// performance/efficiency cores) busy without static partitioning.
//
// Jobs run on a persistent set of helper goroutines. After finishing a
// job a helper spins briefly on an atomic generation counter before
// parking, so a sequence of rounds — the K blocks of one matrix product —
// does not pay a thread wake-up per round. The calling goroutine always takes part
// and never waits for a helper to start, only for items a helper has
// already claimed; nested calls therefore cannot deadlock.
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

// job is one For call: items [0, n) handed out from an atomic counter.
type job struct {
	n       int
	fn      func(int)
	next    atomic.Int64 // next item to hand out
	pending atomic.Int64 // items not yet finished (or skipped)
	failed  atomic.Bool
	done    chan struct{}

	panicMu  sync.Mutex
	panicVal any
}

// run claims and executes items until none are left.
func (j *job) run() {
	for {
		i := int(j.next.Add(1)) - 1
		if i >= j.n {
			return
		}
		if j.failed.Load() {
			j.finish() // an earlier item panicked: skip the rest, keep the count right
			continue
		}
		j.runItem(i)
	}
}

func (j *job) finish() {
	if j.pending.Add(-1) == 0 {
		close(j.done)
	}
}

func (j *job) runItem(i int) {
	defer func() {
		if r := recover(); r != nil {
			j.panicMu.Lock()
			if !j.failed.Load() {
				j.panicVal = r
				j.failed.Store(true)
			}
			j.panicMu.Unlock()
		}
		j.finish()
	}()
	j.fn(i)
}

// Helper pool. A job is published by storing it in cur and bumping gen;
// helpers spin on gen (one shared cache line, no lock) for a short while
// after finishing work, then park on cond. The caller wakes parked
// helpers with one Broadcast per job. Polling a channel instead showed up
// as runtime lock contention worth ~20 % of CPU on a 16-core Xeon.
// spinRounds is how long a helper polls for the next job before parking.
// Waking a parked helper costs a futex round trip, and a Broadcast wakes
// them one after another; with a dozen rounds per matrix product that
// idle time was ~30 % of the run on a 16-core Xeon. A few hundred µs of
// spinning (with a Gosched every few thousand loads so the P is not held
// hostage) bridges the gap between rounds; an idle program pays it once.
const spinRounds = 300000

var (
	cur     atomic.Pointer[job]
	gen     atomic.Uint64
	parkMu  sync.Mutex
	parkCnd = sync.NewCond(&parkMu)
	helpers atomic.Int64
	spawnMu sync.Mutex
)

func ensureHelpers(n int) {
	if int(helpers.Load()) >= n {
		return
	}
	spawnMu.Lock()
	defer spawnMu.Unlock()
	for int(helpers.Load()) < n {
		go helper()
		helpers.Add(1)
	}
}

func helper() {
	var seen uint64
	for {
		g := gen.Load()
		if g != seen {
			seen = g
			if j := cur.Load(); j != nil {
				j.run()
			}
			continue
		}
		// spin briefly: the next round of a blocked GEMM follows immediately
		spun := false
		for i := 0; i < spinRounds; i++ {
			if gen.Load() != seen {
				spun = true
				break
			}
			if i&4095 == 4095 {
				runtime.Gosched()
			}
		}
		if spun {
			continue
		}
		parkMu.Lock()
		for gen.Load() == seen {
			parkCnd.Wait()
		}
		parkMu.Unlock()
	}
}

// publish makes j the current job and wakes parked helpers.
func publish(j *job) {
	cur.Store(j)
	parkMu.Lock()
	gen.Add(1)
	parkMu.Unlock()
	parkCnd.Broadcast()
}

// For calls fn(i) for every i in [0, n), spreading calls over up to
// Workers() goroutines. It returns once all calls have completed. A panic
// inside fn stops the remaining items and is re-raised in the caller.
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
	j := &job{n: n, fn: fn, done: make(chan struct{})}
	j.pending.Store(int64(n))
	ensureHelpers(workers - 1)
	publish(j)
	j.run()
	// The last items are usually finishing on helpers right now: poll
	// briefly before blocking, so the caller does not pay a futex wake-up
	// at the end of every round.
	for i := 0; i < 20000 && j.pending.Load() > 0; i++ {
		if i&1023 == 1023 {
			runtime.Gosched()
		}
	}
	if j.pending.Load() > 0 {
		<-j.done
	}
	if j.failed.Load() {
		panic(j.panicVal)
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
