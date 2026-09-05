package parallel

import (
	"fmt"
	"os"
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
