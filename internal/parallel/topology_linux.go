//go:build linux

package parallel

import (
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
