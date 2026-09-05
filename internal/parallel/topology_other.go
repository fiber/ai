//go:build !linux

package parallel

// defaultWorkers keeps GOMAXPROCS where no topology information is used;
// on macOS the AMX back-end caps GEMM workers at the performance cores
// through kernel.GemmHints instead.
func defaultWorkers(gomaxprocs int) int { return gomaxprocs }
