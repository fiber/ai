//go:build darwin && arm64

package kernel

import (
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestAMXStateUnderGCPressure runs the AMX tile in many goroutines while
// the collector's preemption signals fly, comparing against the Go
// kernel: a lost or clobbered coprocessor state shows as a mismatch.
func TestAMXStateUnderGCPressure(t *testing.T) {
	if amxImpl == nil {
		t.Skip("AMX not enabled")
	}
	if testing.Short() {
		t.Skip()
	}
	const k = 64
	a := make([]float32, k*32)
	b := make([]float32, k*32)
	for i := range a {
		a[i] = float32(i%13) - 6
		b[i] = float32(i%7) - 3
	}
	want := make([]float32, 32*32)
	for p := 0; p < k; p++ {
		for i := 0; i < 32; i++ {
			for j := 0; j < 32; j++ {
				want[i*32+j] += a[p*32+i] * b[p*32+j]
			}
		}
	}
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
				time.Sleep(2 * time.Millisecond)
			}
		}
	}()
	var wg sync.WaitGroup
	var mu sync.Mutex
	mismatches := 0
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c := make([]float32, 32*32)
			local := 0
			for i := 0; i < 60000; i++ {
				clear(c)
				amxBegin()
				gemmAMX(k, &a[0], &b[0], &c[0], 32)
				amxEnd()
				for j := 0; j < len(c); j += 37 {
					if c[j] != want[j] {
						local++
						break
					}
				}
				if i%50 == 0 {
					_ = make([]byte, 1<<20)
				}
			}
			mu.Lock()
			mismatches += local
			mu.Unlock()
		}()
	}
	wg.Wait()
	close(stop)
	if mismatches != 0 {
		t.Fatalf("%d of 960 000 AMX tiles wrong under GC pressure", mismatches)
	}
}
