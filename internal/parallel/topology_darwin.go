//go:build darwin

package parallel

import "syscall"

// defaultWorkers on macOS is the number of performance cores on Apple
// Silicon (hw.perflevel0.physicalcpu) and GOMAXPROCS elsewhere. Every
// parallel round waits for its slowest member, and an efficiency core
// is that member: on an M4 MacBook Air (4P + 6E) four workers instead of
// ten took the MLP forward from 666 K to 1.09 M samples/s and the
// training step from 163 K to 223 K; on an M2 Pro (6P + 4E) six instead
// of ten took the training step from 147 K to 184 K. FIBERAI_WORKERS
// overrides.
func defaultWorkers(gomaxprocs int) int {
	if n, err := syscall.SysctlUint32("hw.perflevel0.physicalcpu"); err == nil && n > 0 {
		return min(int(n), gomaxprocs)
	}
	return gomaxprocs
}

// pinCPUs is empty on macOS (no affinity API); the AMX back-end locks its
// threads per task itself.
func pinCPUs() []int { return nil }

func pinThread(cpu int) bool { return false }
