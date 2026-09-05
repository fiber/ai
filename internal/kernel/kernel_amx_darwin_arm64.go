//go:build darwin && arm64

package kernel

import (
	"os"
	"regexp"
	"runtime"
	"strings"
	"syscall"
)

// Apple AMX: the undocumented matrix coprocessor of the M1–M4 (see
// github.com/corsix/amx). The instructions are system-instruction
// encodings emitted as WORDs; executing one before AMX_SET on a thread
// is an illegal instruction, so a run of tiles is bracketed by amxBegin
// (lock the goroutine to its thread, enable the state) and amxEnd.
//
// Enabled by default on the M1–M4 (verified on an M2 Pro and an M4; the
// start-up self-test cannot catch an illegal instruction, so unknown
// chips stay on NEON unless FIBERAI_AMX=1 or FIBERAI_KERNEL=amx asks for
// it). FIBERAI_AMX=0 switches it off.

// Implemented in kernel_amx_darwin_arm64.s.
func amxSet()
func amxClr()
func gemmAMX(k int, a, b, c *float32, ldc int)
func gemmZeroAMX(k int, a, b, c *float32, ldc int)
func gemmAMXBody(k int, a, b, c *float32, ldc int)

func amxBegin() {
	runtime.LockOSThread()
	amxSet()
}

func amxEnd() {
	amxClr()
	runtime.UnlockOSThread()
}

var amx = func() impl {
	i := neon
	i.name = "amx"
	i.gemm, i.gemmZero = gemmAMX, gemmZeroAMX
	i.mr, i.nr = 32, 32
	i.gemmBegin, i.gemmEnd = amxBegin, amxEnd
	// KC=1024 amortises the 64 Z loads and stores per tile (n=2048: 1.93 →
	// 2.13 TFLOPS on the M2 Pro); the AMX units sit with the performance
	// cores, so no more workers than those (the efficiency cores' unit is
	// slow and drags the tail of every round).
	i.hints = Hints{KC: 1024, TasksPerWorker: 128, Workers: performanceCores()}
	return i
}()

func performanceCores() int {
	if n, err := syscall.SysctlUint32("hw.perflevel0.physicalcpu"); err == nil && n > 0 {
		return int(n)
	}
	return 0
}

var amxImpl = amxDetect()

var amxKnown = regexp.MustCompile(`Apple M[1-4]( |$)`)

func amxDetect() *impl {
	if os.Getenv("FIBERAI_AMX") == "0" {
		return nil
	}
	brand, err := syscall.Sysctl("machdep.cpu.brand_string")
	if err != nil || !strings.Contains(brand, "Apple M") {
		return nil
	}
	forced := os.Getenv("FIBERAI_AMX") == "1" || os.Getenv("FIBERAI_KERNEL") == "amx"
	if !forced && !amxKnown.MatchString(brand) {
		return nil // an M-series chip corsix/amx has not documented: opt-in only
	}
	return &amx
}
