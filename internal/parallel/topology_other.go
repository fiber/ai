//go:build !linux && !darwin

package parallel

// defaultWorkers keeps GOMAXPROCS where no topology information is used;
// on macOS the AMX back-end caps GEMM workers at the performance cores
// through kernel.GemmHints instead.
func defaultWorkers(gomaxprocs int) int { return gomaxprocs }

// pinCPUs is empty where threads cannot be placed (macOS has no affinity
// API; the AMX back-end locks its threads per task itself).
func pinCPUs() []int { return nil }

func pinThread(cpu int) bool { return false }
