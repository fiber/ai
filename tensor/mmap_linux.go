//go:build linux

package tensor

import "syscall"

// adviseHuge asks for transparent huge pages so a 4 MB result takes two
// page faults instead of a thousand, whatever the system-wide THP mode.
func adviseHuge(b []byte) {
	if len(b) >= 2<<20 {
		_ = syscall.Madvise(b, 14) // MADV_HUGEPAGE
	}
}
