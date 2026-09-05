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
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

var maxWorkers atomic.Int64

func init() {
	n := defaultWorkers(runtime.GOMAXPROCS(0))
	if v, err := strconv.Atoi(os.Getenv("FIBERAI_WORKERS")); err == nil && v > 0 {
		n = v
	}
	maxWorkers.Store(int64(max(1, n)))
}

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

// job is one For or Range call. In counter mode items [0, n) are handed
// out from an atomic counter to whoever asks first. In owned mode
// (Range) each item is claimed by CAS; worker w first takes items
// w, w+workers, … so that repeated calls put the same chunk on the same
// goroutine, which is what keeps repeated element-wise work on the same
// tensors in the cores' private caches, and then steals what is left.
type job struct {
	n       int
	fn      func(int)
	next    atomic.Int64 // next item to hand out (counter mode)
	pending atomic.Int64 // items not yet finished (or skipped)
	failed  atomic.Bool
	done    chan struct{}

	owned   bool
	workers int
	claimed []atomic.Bool

	panicMu  sync.Mutex
	panicVal any
}

// run claims and executes items until none are left. w is the worker's
// id: 0 for the caller, k+1 for helper k.
func (j *job) run(w int) {
	if j.owned {
		if w >= j.workers {
			return // not one of this job's workers; stealing here would only cost locality
		}
		for i := w; i < j.n; i += j.workers {
			j.claim(i)
		}
		for i := 0; i < j.n; i++ {
			j.claim(i)
		}
		return
	}
	for {
		i := int(j.next.Add(1)) - 1
		if i >= j.n {
			return
		}
		j.exec(i)
	}
}

func (j *job) claim(i int) {
	if j.claimed[i].Load() || !j.claimed[i].CompareAndSwap(false, true) {
		return
	}
	j.exec(i)
}

func (j *job) exec(i int) {
	if j.failed.Load() {
		j.finish() // an earlier item panicked: skip the rest, keep the count right
		return
	}
	j.runItem(i)
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
// spinTime is how long a helper polls for the next job before parking.
// Waking a parked helper costs a futex round trip, and a Broadcast wakes
// them one after another; with a dozen rounds per matrix product that
// idle time was ~30 % of the run on a 16-core Xeon. A few hundred µs of
// spinning bridges the gap between rounds; an idle program pays it once.
// The spin does not yield the P: a profile of a training step showed
// ten helpers calling Gosched every 4 096 loads spending 70 % of all CPU
// samples in the scheduler lock behind it, and every 65 536 loads was
// still 40 %. A spinning helper holds its P for at most spinTime and then
// parks; the runtime's asynchronous preemption covers the rest.
var (
	spinTime     = defaultSpin
	pollTime     = 100 * time.Microsecond // caller's wait for the last items before blocking
	goschedEvery = 0                      // loads between Gosched calls while spinning; 0 = never
)

func init() {
	// Tuning knobs for experiments without a rebuild.
	if v, err := strconv.Atoi(os.Getenv("FIBERAI_SPIN_US")); err == nil && v >= 0 {
		spinTime = time.Duration(v) * time.Microsecond
	}
	if v, err := strconv.Atoi(os.Getenv("FIBERAI_POLL_US")); err == nil && v >= 0 {
		pollTime = time.Duration(v) * time.Microsecond
	}
	if v, err := strconv.Atoi(os.Getenv("FIBERAI_SPIN_GOSCHED")); err == nil && v > 0 {
		goschedEvery = v
	}
}

var (
	cur     atomic.Pointer[job]
	gen     atomic.Uint64
	parkMu  sync.Mutex
	parkCnd = sync.NewCond(&parkMu)
	parked  atomic.Int64 // helpers waiting on parkCnd; publish broadcasts only then
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
		go helper(int(helpers.Load()) + 1)
		helpers.Add(1)
	}
}

func helper(id int) {
	var seen uint64
	for {
		g := gen.Load()
		if g != seen {
			seen = g
			if j := cur.Load(); j != nil {
				j.run(id)
			}
			continue
		}
		// spin briefly: the next round of a blocked GEMM follows immediately
		spun := false
		start := time.Now()
		for i := 1; ; i++ {
			if gen.Load() != seen {
				spun = true
				break
			}
			if i&8191 == 0 {
				if time.Since(start) > spinTime {
					break
				}
				if goschedEvery > 0 && i%goschedEvery == 0 {
					runtime.Gosched()
				}
			}
		}
		if spun {
			continue
		}
		// Park. The count is raised under parkMu before the generation is
		// re-checked, and publish bumps the generation before it reads the
		// count, so a wake-up cannot be missed.
		parkMu.Lock()
		parked.Add(1)
		for gen.Load() == seen {
			parkCnd.Wait()
		}
		parked.Add(-1)
		parkMu.Unlock()
	}
}

// publish makes j the current job and wakes parked helpers, if any.
func publish(j *job) {
	cur.Store(j)
	gen.Add(1)
	if parked.Load() > 0 {
		parkMu.Lock()
		parkMu.Unlock()
		parkCnd.Broadcast()
	}
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
	runJob(&job{n: n, fn: fn, done: make(chan struct{})}, workers)
}

func runJob(j *job, workers int) {
	j.pending.Store(int64(j.n))
	ensureHelpers(workers - 1)
	publish(j)
	j.run(0)
	// The last items are usually finishing on helpers right now: poll
	// briefly before blocking, so the caller does not pay a futex wake-up
	// at the end of every round.
	if j.pending.Load() > 0 {
		start := time.Now()
		for i := 1; j.pending.Load() > 0; i++ {
			if i&1023 == 0 && time.Since(start) > pollTime {
				break
			}
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
	if workers > chunks {
		workers = chunks
	}
	j := &job{n: chunks, done: make(chan struct{}), owned: true, workers: workers, claimed: make([]atomic.Bool, chunks)}
	j.fn = func(c int) {
		lo := c * size
		hi := min(lo+size, n)
		fn(lo, hi)
	}
	runJob(j, workers)
}
