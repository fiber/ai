package parallel

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// physicalCores counts distinct (package, core) pairs among cpus using
// the Linux sysfs topology files under root; 0 when they cannot be read.
// Hyperthread siblings share a core_id within a package.
func physicalCores(root string, cpus []int) int {
	seen := map[string]bool{}
	for _, c := range cpus {
		dir := fmt.Sprintf("%s/cpu%d/topology", root, c)
		pkg, err1 := os.ReadFile(dir + "/physical_package_id")
		core, err2 := os.ReadFile(dir + "/core_id")
		if err1 != nil || err2 != nil {
			return 0
		}
		seen[strings.TrimSpace(string(pkg))+"/"+strings.TrimSpace(string(core))] = true
	}
	return len(seen)
}

// packages counts the distinct physical packages among cpus; 0 when the
// sysfs topology cannot be read.
func packages(root string, cpus []int) int {
	seen := map[string]bool{}
	for _, c := range cpus {
		pkg, err := os.ReadFile(fmt.Sprintf("%s/cpu%d/topology/physical_package_id", root, c))
		if err != nil {
			return 0
		}
		seen[strings.TrimSpace(string(pkg))] = true
	}
	return len(seen)
}

// coreCPUs picks one CPU per physical core among cpus (the lowest CPU
// number of each (package, core)), ordered by package then core id, so
// that fewer workers than cores fill one package first. nil when the
// sysfs topology cannot be read.
func coreCPUs(root string, cpus []int) []int {
	type core struct{ pkg, id, cpu int }
	first := map[[2]int]int{}
	var cores []core
	sorted := append([]int(nil), cpus...)
	sort.Ints(sorted) // the lowest CPU number of each core, whatever the mask's order
	for _, c := range sorted {
		dir := fmt.Sprintf("%s/cpu%d/topology", root, c)
		pkg, err1 := os.ReadFile(dir + "/physical_package_id")
		id, err2 := os.ReadFile(dir + "/core_id")
		if err1 != nil || err2 != nil {
			return nil
		}
		p, err1 := strconv.Atoi(strings.TrimSpace(string(pkg)))
		i, err2 := strconv.Atoi(strings.TrimSpace(string(id)))
		if err1 != nil || err2 != nil {
			return nil
		}
		key := [2]int{p, i}
		if _, seen := first[key]; !seen {
			first[key] = c
			cores = append(cores, core{p, i, c})
		}
	}
	sort.Slice(cores, func(a, b int) bool {
		if cores[a].pkg != cores[b].pkg {
			return cores[a].pkg < cores[b].pkg
		}
		return cores[a].id < cores[b].id
	})
	out := make([]int, len(cores))
	for i, c := range cores {
		out[i] = c.cpu
	}
	return out
}
