// Command smeprobe answers the phase-0 questions of spec T-006 on an
// Apple M4: do the SME instruction words execute, is the outer-product
// layout as expected, what does one core's fmopa throughput look like,
// how does it scale over goroutines, and does the state survive GC and
// preemption. Run it on the M4:
//
//	go run ./internal/kernel/smeprobe
package main

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"syscall"
	"time"
)

func outer(x, y, z *float32)
func tile(k int, a, b *float32)
func tile2(k int, a, b *float32)

func main() {
	if v, err := syscall.SysctlUint32("hw.optional.arm.FEAT_SME"); err != nil || v == 0 {
		fmt.Println("no SME on this machine (hw.optional.arm.FEAT_SME unset)")
		os.Exit(1)
	}
	svl, _ := syscall.SysctlUint32("hw.optional.arm.sme_max_svl_b")
	fmt.Printf("SME present, streaming vector length %d bytes\n", svl)

	// 1. one outer product, checked
	x := make([]float32, 16)
	y := make([]float32, 16)
	for i := range x {
		x[i] = float32(i + 1)
		y[i] = float32(100 * (i + 1))
	}
	z := make([]float32, 16*16)
	outer(&x[0], &y[0], &z[0])
	bad := 0
	for j := 0; j < 16; j++ {
		for i := 0; i < 16; i++ {
			if z[j*16+i] != y[j]*x[i] {
				bad++
			}
		}
	}
	fmt.Printf("outer product: %d mismatches; row 0 = %v\n", bad, z[:4])
	fmt.Printf("               row 1 = %v (expect 200 400 600 800 if rows are Zn lanes)\n", z[16:20])

	// 2. throughput of the 32×32 tile body
	k := 512
	a := make([]float32, k*32)
	b := make([]float32, k*32)
	iters := 4000
	bench := func(threads int, body func(k int, a, b *float32)) float64 {
		var wg sync.WaitGroup
		start := time.Now()
		for t := 0; t < threads; t++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := 0; i < iters; i++ {
					body(k, &a[0], &b[0])
				}
			}()
		}
		wg.Wait()
		el := time.Since(start).Seconds()
		return 2.0 * 32 * 32 * float64(k) * float64(iters) * float64(threads) / el / 1e9
	}
	for _, v := range []struct {
		name string
		body func(k int, a, b *float32)
	}{{"plain", tile}, {"pipelined", tile2}} {
		for _, t := range []int{1, 2, 4, 6} {
			best := 0.0
			for r := 0; r < 3; r++ {
				best = max(best, bench(t, v.body))
			}
			fmt.Printf("%-9s threads %d: %6.0f GFLOPS (%.0f per thread)\n", v.name, t, best, best/float64(t))
		}
	}

	// 3. state under GC and preemption: outer products in many goroutines
	// while the heap churns; every result must still be exact
	var wg sync.WaitGroup
	errs := make(chan int, 64)
	for g := 0; g < 32; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			zz := make([]float32, 256)
			mism := 0
			for i := 0; i < 20000; i++ {
				outer(&x[0], &y[0], &zz[0])
				for j := 0; j < 256; j += 17 {
					if zz[j] != y[j/16]*x[j%16] {
						mism++
					}
				}
				if i%100 == 0 {
					_ = make([]byte, 1<<20) // garbage for the collector
				}
			}
			errs <- mism
		}()
	}
	go func() {
		for i := 0; i < 50; i++ {
			runtime.GC()
			time.Sleep(10 * time.Millisecond)
		}
	}()
	wg.Wait()
	close(errs)
	total := 0
	for m := range errs {
		total += m
	}
	fmt.Printf("stress: %d mismatches over 640 000 outer products under GC\n", total)
}
