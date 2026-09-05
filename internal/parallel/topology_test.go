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
