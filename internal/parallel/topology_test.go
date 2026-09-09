package parallel

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestPhysicalCoresFromSysfs(t *testing.T) {
	root := t.TempDir()
	// two packages × 4 cores × 2 threads: cpu0-7 package 0, cpu8-15 package 1;
	// thread siblings are cpu i and cpu i+4 within a package
	for cpu := 0; cpu < 16; cpu++ {
		dir := filepath.Join(root, fmt.Sprintf("cpu%d", cpu), "topology")
		os.MkdirAll(dir, 0o755)
		os.WriteFile(filepath.Join(dir, "physical_package_id"), []byte(fmt.Sprintf("%d\n", cpu/8)), 0o644)
		os.WriteFile(filepath.Join(dir, "core_id"), []byte(fmt.Sprintf("%d\n", cpu%4)), 0o644)
	}
	all := make([]int, 16)
	for i := range all {
		all[i] = i
	}
	if n := physicalCores(root, all); n != 8 {
		t.Fatalf("all cpus: %d physical cores, want 8", n)
	}
	if n := physicalCores(root, []int{0, 1, 2, 3, 4, 5, 6, 7}); n != 4 {
		t.Fatalf("one package: %d, want 4", n)
	}
	if n := physicalCores(root, []int{0, 4}); n != 1 {
		t.Fatalf("two siblings: %d, want 1", n)
	}
	if n := physicalCores(root, []int{0, 99}); n != 0 {
		t.Fatalf("missing cpu should fail, got %d", n)
	}
}

// TestCoreCPUsFromSysfs: one CPU per core, lowest number first, package
// order; subsets and siblings.
func TestCoreCPUsFromSysfs(t *testing.T) {
	root := t.TempDir()
	// same fake topology: cpu0-7 package 0, cpu8-15 package 1; siblings i and i+4
	for cpu := 0; cpu < 16; cpu++ {
		dir := filepath.Join(root, fmt.Sprintf("cpu%d", cpu), "topology")
		os.MkdirAll(dir, 0o755)
		os.WriteFile(filepath.Join(dir, "physical_package_id"), []byte(fmt.Sprintf("%d\n", cpu/8)), 0o644)
		os.WriteFile(filepath.Join(dir, "core_id"), []byte(fmt.Sprintf("%d\n", cpu%4)), 0o644)
	}
	all := make([]int, 16)
	for i := range all {
		all[i] = i
	}
	want := []int{0, 1, 2, 3, 8, 9, 10, 11}
	if got := coreCPUs(root, all); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("all cpus: %v, want %v", got, want)
	}
	// mask listing the siblings first: still the lowest CPU of each core
	if got := coreCPUs(root, []int{7, 6, 5, 4, 3, 2, 1, 0}); fmt.Sprint(got) != fmt.Sprint([]int{0, 1, 2, 3}) {
		t.Fatalf("reversed package 0: %v", got)
	}
	// only second threads allowed (numactl --physcpubind=4-7): those CPUs
	if got := coreCPUs(root, []int{4, 5, 6, 7}); fmt.Sprint(got) != fmt.Sprint([]int{4, 5, 6, 7}) {
		t.Fatalf("second threads: %v", got)
	}
	if got := coreCPUs(root, []int{0, 99}); got != nil {
		t.Fatalf("missing cpu should fail, got %v", got)
	}
}
