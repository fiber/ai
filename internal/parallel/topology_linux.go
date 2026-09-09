//go:build linux

package parallel

import (
	"os"
	"sync"
	"syscall"
	"unsafe"
)

// defaultWorkers on Linux is the number of physical cores the process is
// allowed to run on: the CPUs of the affinity mask, hyperthread siblings
// counted once. Under `numactl --cpunodebind=0` on a two-socket
// hyperthreaded Xeon that is 16 rather than GOMAXPROCS's 32; unpinned 32
// rather than 64. Hyperthreads add nothing to FMA-bound kernels and two
// threads per core double the synchronisation cost of every round.
func defaultWorkers(gomaxprocs int) int {
	cpus := affinityCPUs()
	if cpus == nil {
		return gomaxprocs
	}
	n := physicalCores(sysfsRoot, cpus)
	if n <= 0 {
		return gomaxprocs
	}
	return min(n, gomaxprocs)
}

const sysfsRoot = "/sys/devices/system/cpu"

// affinityCPUs lists the CPUs in the calling process's affinity mask via
// sched_getaffinity(2); nil if the call fails.
func affinityCPUs() []int {
	var mask [1024 / 8]uint64 // room for 8192 CPUs
	n, _, errno := syscall.RawSyscall(syscall.SYS_SCHED_GETAFFINITY, 0, uintptr(len(mask)*8), uintptr(unsafe.Pointer(&mask[0])))
	if errno != 0 || n <= 0 {
		return nil
	}
	var cpus []int
	for i := 0; i < int(n)*8; i++ {
		if mask[i/64]&(1<<(uint(i)%64)) != 0 {
			cpus = append(cpus, i)
		}
	}
	return cpus
}

var (
	pinOnce sync.Once
	pinList []int
)

// pinCPUs returns the CPUs pool helpers pin to: one per physical core of
// the affinity mask, in package order (see coreCPUs). Empty when
// FIBERAI_PIN=0, when the topology is unreadable, or when the mask has a
// single core. Computed once.
func pinCPUs() []int {
	pinOnce.Do(func() {
		if os.Getenv("FIBERAI_PIN") == "0" {
			return
		}
		cpus := affinityCPUs()
		if cpus == nil {
			return
		}
		if l := coreCPUs(sysfsRoot, cpus); len(l) > 1 {
			pinList = l
		}
	})
	return pinList
}

// pinThread restricts the calling OS thread to one CPU with
// sched_setaffinity(2); false when the call fails (the thread then runs
// wherever the scheduler puts it).
func pinThread(cpu int) bool {
	var mask [1024 / 8]uint64
	if cpu < 0 || cpu >= len(mask)*64 {
		return false
	}
	mask[cpu/64] |= 1 << (uint(cpu) % 64)
	_, _, errno := syscall.RawSyscall(syscall.SYS_SCHED_SETAFFINITY, 0, uintptr(len(mask)*8), uintptr(unsafe.Pointer(&mask[0])))
	return errno == 0
}
