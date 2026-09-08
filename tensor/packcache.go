package tensor

import (
	"os"
	"sync"
	"unsafe"

	"github.com/fiber/ai/internal/blas"
	"github.com/fiber/ai/internal/parallel"
)

// The packed-operand cache. The GEMM driver packs its right operand into
// panel layout on every call; in inference that operand is a weight that
// does not change between calls, so the packed copy is kept and reused.
//
// An operand is cached under its storage and view geometry once it has
// been seen twice (activations used once never enter), while the storage's
// version is unchanged (in-place operations bump it) and its buffer has not
// escaped through Data() (a caller could write through the slice at any
// time). Entries are evicted least-recently-used under a byte limit.

type packKey struct {
	ptr        uintptr
	rs, cs     int
	rows, cols int
}

type packEntry struct {
	store      *storage
	version    uint32
	packed     *blas.PackedB
	bytes      int
	prev, next *packEntry
	key        packKey
}

type packCacheT struct {
	mu      sync.Mutex
	enabled bool
	limit   int
	bytes   int
	entries map[packKey]*packEntry
	seen    map[packKey]uint32 // first sighting: version at the time
	head    *packEntry         // most recently used
	tail    *packEntry
	hits    uint64
	misses  uint64
}

var packCache = packCacheT{enabled: true, limit: 512 << 20, entries: map[packKey]*packEntry{}, seen: map[packKey]uint32{}}

const (
	packMinElements = 64 << 10 // smaller B packs cheaply; not worth an entry
	packSeenMax     = 4096     // first-sighting ring
)

func init() {
	if os.Getenv("FIBERAI_PACK_CACHE") == "0" {
		packCache.enabled = false
	}
}

// SetPackedCacheLimit caps the bytes held by the packed-operand cache
// (default 512 MiB); 0 disables it and drops every entry.
func SetPackedCacheLimit(bytes int) {
	c := &packCache
	c.mu.Lock()
	defer c.mu.Unlock()
	c.limit = max(0, bytes)
	c.enabled = bytes > 0
	for c.bytes > c.limit && c.tail != nil {
		c.evictLocked(c.tail)
	}
}

// PackedCacheStats reports the cache's hits, misses (products whose right
// operand was packed on the call) and bytes held.
func PackedCacheStats() (hits, misses uint64, bytes int) {
	c := &packCache
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses, c.bytes
}

func (c *packCacheT) evictLocked(e *packEntry) {
	c.unlinkLocked(e)
	delete(c.entries, e.key)
	c.bytes -= e.bytes
}

func (c *packCacheT) unlinkLocked(e *packEntry) {
	if e.prev != nil {
		e.prev.next = e.next
	} else {
		c.head = e.next
	}
	if e.next != nil {
		e.next.prev = e.prev
	} else {
		c.tail = e.prev
	}
	e.prev, e.next = nil, nil
}

func (c *packCacheT) frontLocked(e *packEntry) {
	if c.head == e {
		return
	}
	if e.prev != nil || e.next != nil || c.tail == e {
		c.unlinkLocked(e)
	}
	e.next = c.head
	if c.head != nil {
		c.head.prev = e
	}
	c.head = e
	if c.tail == nil {
		c.tail = e
	}
}

// packedOperand returns the packed copy of a 2-D right operand when it is
// cached (packing it on the second sighting), or nil.
func packedOperand(y *Tensor) *blas.PackedB {
	c := &packCache
	if !c.enabled || len(y.shape) != 2 || y.size < packMinElements || y.store == nil {
		return nil
	}
	if y.store.state.Load() != stateLive { // escaped or released: never cache
		return nil
	}
	key := packKey{ptr: uintptr(unsafe.Pointer(&y.data[0])), rs: y.strides[0], cs: y.strides[1], rows: y.shape[0], cols: y.shape[1]}
	version := y.store.version.Load()

	c.mu.Lock()
	if e, ok := c.entries[key]; ok {
		if e.store == y.store && e.version == version {
			c.frontLocked(e)
			c.hits++
			c.mu.Unlock()
			return e.packed
		}
		c.evictLocked(e) // modified since: repack below if seen again
	}
	c.misses++
	if v, ok := c.seen[key]; !ok || v != version {
		// first sighting at this version: remember, do not pack yet
		if len(c.seen) >= packSeenMax {
			clear(c.seen)
		}
		c.seen[key] = version
		c.mu.Unlock()
		return nil
	}
	c.mu.Unlock()

	// Second sighting: pack outside the lock (packing is parallel and may
	// take a millisecond), then insert.
	packed := blas.PackB(mat(y, 0, 1), parallel.Workers())
	if packed.Bytes() > c.limit {
		return nil
	}
	e := &packEntry{store: y.store, version: version, packed: packed, bytes: packed.Bytes(), key: key}
	c.mu.Lock()
	if old, ok := c.entries[key]; ok {
		c.evictLocked(old)
	}
	for c.bytes+e.bytes > c.limit && c.tail != nil {
		c.evictLocked(c.tail)
	}
	c.entries[key] = e
	c.bytes += e.bytes
	c.frontLocked(e)
	delete(c.seen, key)
	c.mu.Unlock()
	if y.store.version.Load() != version { // modified while packing
		return nil
	}
	return packed
}
