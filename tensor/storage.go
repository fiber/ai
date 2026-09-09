package tensor

import (
	"github.com/fiber/ai/internal/parallel"
	"math/bits"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// storage is the backing array of a tensor and of every view derived from
// it. Buffers of pooled storage return to the pool when the storage
// object becomes unreachable, i.e. when no tensor or view refers to it.
//
// Reusing buffers avoids two costs that dominate small and medium
// element-wise operations: the zero-fill of a fresh allocation and the
// page faults that follow once the GC has returned earlier results to
// the operating system. Recycled memory stays mapped and cache-warm.
type storage struct {
	buf    []float32
	pooled bool // heap buffer from the (opt-in) heap pool
	mapped bool // off-heap buffer from mapFloats
	shared atomic.Bool
	// version counts in-place modifications; the packed-operand cache
	// keys on it (see packcache.go). id is unique for the life of the
	// process, so a recycled buffer address never impersonates an earlier
	// storage in that cache.
	version atomic.Uint32
	id      uint64
	// state lives in its own allocation because the cleanup must not
	// reference the storage (see cleanupArg).
	state *atomic.Int32
}

// Storage states. Escaped: Data() handed the slice to a caller, never
// recycle. Released: Release() already returned the buffer, the cleanup
// must not touch it again.
const (
	stateLive int32 = iota
	stateEscaped
	stateReleased
)

func newState() *atomic.Int32 { return new(atomic.Int32) }

var storageIDs atomic.Uint64

// nextStorageID hands out process-unique storage identities.
func nextStorageID() uint64 { return storageIDs.Add(1) }

// Release hands the tensor's storage back for immediate reuse, without
// waiting for the garbage collector to notice that the tensor is dead.
// The tensor must not be used afterwards. It does nothing when the
// storage may still be in use elsewhere: a view of the tensor exists,
// Data() was taken, or autograd recorded the tensor; so it is safe to
// call on any intermediate result.
//
// The point is cache residency. A hot loop that discards a 4 MB result
// per step otherwise rotates through tens of buffers until a collection
// brings them back, and every step streams from memory; released
// storage is handed out again by the next allocation of that size, still
// in cache.
func (t *Tensor) Release() {
	if t.node != nil || t.requiresGrad {
		return
	}
	t.releaseStorage()
}

// Recycle returns a tensor's off-heap storage to the free list for
// immediate reuse even when views of it exist, which Release refuses. It is
// for an inference caller that owns the whole dataflow and knows every view
// of the tensor is also dead, such as a model runner freeing a layer's
// intermediates before the next layer. Using the tensor, or any view of it,
// after Recycle reads freed memory. It does nothing for pooled or
// autograd-recorded tensors.
func (t *Tensor) Recycle() {
	if t == nil || t.node != nil || t.requiresGrad {
		return
	}
	st := t.store
	if st == nil || !st.mapped {
		return
	}
	if !st.state.CompareAndSwap(stateLive, stateReleased) && !st.state.CompareAndSwap(stateEscaped, stateReleased) {
		return
	}
	buf := st.buf[:cap(st.buf)]
	st.buf, t.data = nil, nil
	t.released = true
	mapPool.mu.Lock()
	putMappedLocked(buf)
	mapPool.mu.Unlock()
}

// releaseStorage returns the storage unless a view, Data() or an earlier
// release claims it. Backward uses it on graph intermediates whose
// consumers have all run.
func (t *Tensor) releaseStorage() {
	st := t.store
	if st == nil || st.shared.Load() {
		return
	}
	if !st.state.CompareAndSwap(stateLive, stateReleased) {
		return
	}
	buf := st.buf[:cap(st.buf)]
	st.buf, t.data = nil, nil
	t.released = true
	switch {
	case st.mapped:
		mapPool.mu.Lock()
		putMappedLocked(buf)
		mapPool.mu.Unlock()
	case st.pooled:
		release(buf)
	}
}

// Pool size classes: powers of two with four quarter steps in between
// (2^k, 1.25·2^k, 1.5·2^k, 1.75·2^k), so at most 25 % of a buffer is
// slack. Buffers below minPooled floats are not pooled; a plain make is
// cheap at that size.
const (
	minPooled = 1 << 10 // 4 KiB: below this a plain make is cheaper than the bookkeeping
	maxPooled = 1 << 20 // 4 MiB: above this Go's own allocator reuses freed spans well enough,
	// and holding such buffers until a cleanup runs inflates the heap and delays reuse
	numClass = 64 * 4
)

var pool = struct {
	mu       sync.Mutex
	free     [numClass][][]float32
	retained int // bytes currently held
	limit    int // bytes we are willing to hold
	hits     uint64
	misses   uint64
}{limit: 0} // off by default, see SetPoolLimit

func init() {
	// FIBERAI_POOL_LIMIT (bytes; 0 disables) and FIBERAI_HEAP_BALLAST
	// (bytes) override the defaults for experiments without a rebuild.
	if v := os.Getenv("FIBERAI_POOL_LIMIT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			SetPoolLimit(n)
		}
	}
	if v := os.Getenv("FIBERAI_HEAP_BALLAST"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			SetHeapBallast(n)
		}
	}
	if v := os.Getenv("FIBERAI_MAPPED_LIMIT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			SetMappedLimit(n)
		}
	}
	if v := os.Getenv("FIBERAI_MAP_BUDGET"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			mapBudget = n
			mapPool.gcAt = n
		}
	}
}

// DefaultHeapBallast is the ballast installed at start-up (see
// SetHeapBallast). Measured on an Apple M2 Pro, 128 MiB halves the time
// of element-wise operations on 64K–1M elements and speeds up an MLP
// training step by 11 %; 512 MiB adds little more.
const DefaultHeapBallast = 128 << 20

var ballast = make([]byte, DefaultHeapBallast)

// SetHeapBallast keeps an allocation of the given size alive. Its pages
// are never touched, so it costs address space, not resident memory. Go's
// collector sizes the heap relative to the live data, so the ballast
// raises the point at which it runs: with a small model and results of a
// few MB the GC otherwise runs every few operations, returns freed
// results to the operating system, and the next allocation faults the
// pages back in and zero-fills them. With the ballast, freed buffers are
// reused by the allocator instead. The price is that up to about the
// ballast's size of garbage may accumulate between collections. 0
// removes the ballast; FIBERAI_HEAP_BALLAST overrides the default.
func SetHeapBallast(bytes int) {
	if bytes <= 0 {
		ballast = nil
		return
	}
	ballast = make([]byte, bytes)
}

// HeapBallast returns the current ballast size in bytes.
func HeapBallast() int { return len(ballast) }

// SetPoolLimit enables recycling of freed tensor storage (buffers of 4 KiB
// to 4 MiB) and caps the bytes kept; 0 (the default) disables it.
//
// Pooling is opt-in because its effect depends on the workload: it makes
// operations on small and medium tensors about twice as fast (the
// zero-fill and page faults of fresh allocations disappear) but the
// retained memory raises the GC's heap goal, and buffers only return
// after a GC cycle, which slowed a training step of medium-sized layers
// by ~10 % in measurements. Try 64 << 20 for inference over many small
// tensors; leave it off for training loops.
func SetPoolLimit(bytes int) {
	pool.mu.Lock()
	defer pool.mu.Unlock()
	pool.limit = bytes
	if bytes <= 0 {
		for i := range pool.free {
			pool.free[i] = nil
		}
		pool.retained = 0
	}
}

// PoolStats reports how many pooled allocations were served from the pool
// (hits) or freshly allocated (misses), and the bytes currently retained.
func PoolStats() (hits, misses uint64, retainedBytes int) {
	pool.mu.Lock()
	defer pool.mu.Unlock()
	return pool.hits, pool.misses, pool.retained
}

// sizeClass returns the class index and capacity (in floats) for n > 0.
func sizeClass(n int) (class int, capacity int) {
	if n <= minPooled {
		return 0, minPooled
	}
	k := bits.Len(uint(n)) - 1 // 2^k <= n
	base := 1 << k
	if base == n {
		return k * 4, base
	}
	quarter := base >> 2
	frac := (n - base + quarter - 1) / quarter // 1..4
	if frac == 4 {
		return (k + 1) * 4, base << 1
	}
	return k*4 + frac, base + frac*quarter
}

// Off-heap buffers. Go zero-fills every make on the allocating goroutine
// and fresh pages fault in serially; on a 16-core Xeon a 1M-float result
// cost 647 µs that way, three quarters of an element-wise operation.
// Results of mapMin floats and more are therefore mmapped: the kernel
// provides zero pages on first touch, which happens in the parallel
// kernels, and Linux gets huge pages. Freed mappings are kept in size
// classes for reuse (mapping is not free either) up to a retained limit.
// Off-heap memory does not drive Go's collector, so when the free list is
// empty and the outstanding mapped bytes have doubled since the last
// collection (at least DefaultMapBudget), a GC is run and its cleanups
// awaited before mapping more: garbage results then come back as hits
// instead of fresh, page-faulting mappings.
//
// A slice into a mapping keeps nothing alive for the GC. Operations must
// therefore keep the tensor (not just its slice) reachable while they read
// it; ending with record(out, op, inputs, ...) does that, the few
// functions that do not use runtime.KeepAlive.
const (
	mapMin           = 32 << 10  // 128 KiB
	DefaultMapBudget = 256 << 20 // outstanding mapped bytes before the first forced GC (FIBERAI_MAP_BUDGET)
)

var mapPool = struct {
	mu       sync.Mutex
	free     [numClass][][]float32
	lastUse  [numClass]uint64 // pop counter at the class's last hit, for eviction
	clock    uint64
	retained int // bytes held in free
	live     int // bytes mapped and not yet released
	gcAt     int // forced GC once live reaches this and the free list is empty
	limit    int // retained bytes we are willing to hold
	enabled  bool
	hits     uint64
	misses   uint64
	pinned   int // bytes kept mapped for good because Data() escaped them
}{limit: 512 << 20, gcAt: DefaultMapBudget, enabled: mmapSupported}

var mapBudget = DefaultMapBudget

// SetMappedLimit caps the off-heap buffers kept for reuse (default 512
// MiB). A negative limit disables off-heap allocation; results then come
// from the Go heap. FIBERAI_MAPPED_LIMIT sets it from the environment.
func SetMappedLimit(bytes int) {
	mapPool.mu.Lock()
	defer mapPool.mu.Unlock()
	mapPool.limit = max(bytes, 0)
	mapPool.enabled = mmapSupported && bytes >= 0
	trimMappedLocked(0, -1)
}

// trimMappedLocked unmaps retained buffers until retained+need fits the
// limit, taking from the least recently used size classes first and
// sparing class keep, so that a class in steady use is not starved by
// buffers left over from an earlier phase of the program.
func trimMappedLocked(need, keep int) {
	for mapPool.retained+need > mapPool.limit {
		victim := -1
		for class := range mapPool.free {
			if class == keep || len(mapPool.free[class]) == 0 {
				continue
			}
			if victim < 0 || mapPool.lastUse[class] < mapPool.lastUse[victim] {
				victim = class
			}
		}
		if victim < 0 {
			return
		}
		l := mapPool.free[victim]
		for len(l) > 0 && mapPool.retained+need > mapPool.limit {
			b := l[len(l)-1]
			l = l[:len(l)-1]
			mapPool.retained -= cap(b) * 4
			unmapFloats(b)
		}
		mapPool.free[victim] = l
	}
}

// MappedStats reports off-heap allocations served from the free list
// (hits) or freshly mapped (misses), the bytes retained for reuse, and
// the bytes pinned by Data().
func MappedStats() (hits, misses uint64, retainedBytes, pinnedBytes int) {
	mapPool.mu.Lock()
	defer mapPool.mu.Unlock()
	return mapPool.hits, mapPool.misses, mapPool.retained, mapPool.pinned
}

func getMapped(n int, zero bool) *storage {
	class, capacity := sizeClass(n)
	mapPool.mu.Lock()
	buf := popMappedLocked(class)
	if buf == nil && mapPool.live >= mapPool.gcAt {
		mapPool.mu.Unlock()
		collectMapped()
		mapPool.mu.Lock()
		buf = popMappedLocked(class)
		mapPool.gcAt = max(2*mapPool.live, mapBudget)
	}
	if buf != nil {
		mapPool.hits++
	} else {
		mapPool.misses++
	}
	mapPool.live += capacity * 4
	mapPool.mu.Unlock()
	if buf == nil {
		m, err := mapFloats(capacity)
		if err != nil {
			mapPool.mu.Lock()
			mapPool.live -= capacity * 4
			mapPool.mu.Unlock()
			return &storage{id: nextStorageID(), buf: make([]float32, n), state: newState()}
		}
		buf = m // fresh pages are zero
		touchPages(buf[:n])
	} else if zero {
		parallelClear(buf[:n])
	}
	st := &storage{id: nextStorageID(), buf: buf[:n], mapped: true, state: newState()}
	runtime.AddCleanup(st, releaseMapped, cleanupArg{buf: buf[:cap(buf)], state: st.state})
	return st
}

func popMappedLocked(class int) []float32 {
	l := mapPool.free[class]
	if len(l) == 0 {
		return nil
	}
	buf := l[len(l)-1]
	l[len(l)-1] = nil
	mapPool.free[class] = l[:len(l)-1]
	mapPool.retained -= cap(buf) * 4
	mapPool.clock++
	mapPool.lastUse[class] = mapPool.clock
	return buf
}

// collectMapped runs a GC and waits until its cleanups have been
// dispatched, so that mappings of unreachable results are back on the
// free list when it returns. A sentinel object's cleanup marks the point.
func collectMapped() {
	done := make(chan struct{})
	armSentinel(done)
	runtime.GC()
	select {
	case <-done:
		runtime.Gosched() // let concurrently running cleanups finish
	case <-time.After(20 * time.Millisecond):
	}
}

// sentinel must not be tiny-allocated (pointer-free objects of 16 bytes
// or less share blocks and are never individually unreachable), hence the
// pointer field.
type sentinel struct {
	ch chan struct{}
	_  [48]byte
}

func armSentinel(done chan struct{}) {
	s := &sentinel{ch: done}
	runtime.AddCleanup(s, func(ch chan struct{}) { close(ch) }, done)
}

// touchPages faults a fresh mapping in with all workers, one write per
// page over disjoint ranges. Left to the consumer, sixteen GEMM workers
// writing interleaved tiles into the same fresh pages each trapped and
// serialised on the page-table lock: 3.6× the faults and a third off an
// MLP training step on a Xeon Gold 6130. Reused mappings skip this.
func touchPages(buf []float32) {
	const page = 4096 / 4
	pages := (len(buf) + page - 1) / page
	parallel.Range(pages, 64, func(lo, hi int) {
		for p := lo; p < hi; p++ {
			buf[p*page] = 0
		}
	})
}

func releaseMapped(a cleanupArg) {
	switch a.state.Load() {
	case stateReleased:
		return // Release already returned it
	case stateEscaped:
		mapPool.mu.Lock()
		mapPool.live -= cap(a.buf) * 4
		mapPool.pinned += cap(a.buf) * 4 // the caller may still hold the slice
		mapPool.mu.Unlock()
		return
	}
	mapPool.mu.Lock()
	putMappedLocked(a.buf)
	mapPool.mu.Unlock()
}

// putMappedLocked returns a mapping to the free list, or unmaps it when
// the retention limit leaves no room even after evicting other classes.
func putMappedLocked(buf []float32) {
	size := cap(buf) * 4
	mapPool.live -= size
	class, _ := sizeClass(cap(buf))
	if mapPool.enabled {
		trimMappedLocked(size, class)
		if mapPool.retained+size <= mapPool.limit {
			mapPool.free[class] = append(mapPool.free[class], buf)
			mapPool.retained += size
			return
		}
	}
	unmapFloats(buf)
}

// getStorage returns storage for n floats; zero requests cleared memory.
func getStorage(n int, zero bool) *storage {
	if n >= mapMin && mapPool.enabled {
		return getMapped(n, zero)
	}
	if n < minPooled || n > maxPooled {
		return &storage{id: nextStorageID(), buf: make([]float32, n), state: newState()}
	}
	class, capacity := sizeClass(n)
	var buf []float32
	pool.mu.Lock()
	pooled := pool.limit > 0
	if pooled {
		if l := pool.free[class]; len(l) > 0 {
			buf = l[len(l)-1]
			l[len(l)-1] = nil
			pool.free[class] = l[:len(l)-1]
			pool.retained -= cap(buf) * 4
			pool.hits++
		} else {
			pool.misses++
		}
	}
	pool.mu.Unlock()
	if buf == nil {
		buf = make([]float32, capacity) // already zero
	} else if zero {
		clear(buf[:n])
	}
	st := &storage{id: nextStorageID(), buf: buf[:n], pooled: pooled, state: newState()}
	if pooled {
		// The cleanup argument must not reference st, or st would never
		// become unreachable; the buffer and the flag are separate objects.
		runtime.AddCleanup(st, releaseUnlessEscaped, cleanupArg{buf: buf[:cap(buf)], state: st.state})
	}
	return st
}

// cleanupArg carries what the cleanup needs without keeping the storage
// alive: the buffer and the escape flag (a pointer into the storage would
// keep it reachable, so the flag lives in its own allocation).
type cleanupArg struct {
	buf   []float32
	state *atomic.Int32
}

func releaseUnlessEscaped(a cleanupArg) {
	if a.state.Load() != stateLive {
		return
	}
	release(a.buf)
}

// release returns a buffer to the pool unless the limit is reached.
func release(buf []float32) {
	class, capacity := sizeClass(cap(buf))
	if cap(buf) != capacity {
		return // not one of ours
	}
	pool.mu.Lock()
	defer pool.mu.Unlock()
	if pool.limit <= 0 || pool.retained+cap(buf)*4 > pool.limit {
		return
	}
	pool.free[class] = append(pool.free[class], buf)
	pool.retained += cap(buf) * 4
}
