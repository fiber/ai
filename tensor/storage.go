package tensor

import (
	"math/bits"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
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
	buf     []float32
	pooled  bool         // buffer came from the pool and may go back
	escaped *atomic.Bool // Data() handed the slice to a caller: never recycle
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
	// FIBERAI_POOL_LIMIT (bytes; 0 disables) overrides the default for
	// experiments without a rebuild.
	if v := os.Getenv("FIBERAI_POOL_LIMIT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			SetPoolLimit(n)
		}
	}
}

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

// getStorage returns storage for n floats; zero requests cleared memory.
func getStorage(n int, zero bool) *storage {
	if n < minPooled || n > maxPooled {
		return &storage{buf: make([]float32, n), escaped: new(atomic.Bool)}
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
	st := &storage{buf: buf[:n], pooled: pooled, escaped: new(atomic.Bool)}
	if pooled {
		// The cleanup argument must not reference st, or st would never
		// become unreachable; the buffer and the flag are separate objects.
		runtime.AddCleanup(st, releaseUnlessEscaped, cleanupArg{buf: buf[:cap(buf)], escaped: st.escaped})
	}
	return st
}

// cleanupArg carries what the cleanup needs without keeping the storage
// alive: the buffer and the escape flag (a pointer into the storage would
// keep it reachable, so the flag lives in its own allocation).
type cleanupArg struct {
	buf     []float32
	escaped *atomic.Bool
}

func releaseUnlessEscaped(a cleanupArg) {
	if a.escaped.Load() {
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
