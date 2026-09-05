//go:build unix

package tensor

import (
	"syscall"
	"unsafe"
)

const mmapSupported = true

// mapFloats maps n floats of anonymous memory outside the Go heap. Nothing
// zero-fills it eagerly: the kernel provides zero pages on first touch,
// which happens in the parallel kernels rather than on the allocating
// goroutine.
func mapFloats(n int) ([]float32, error) {
	b, err := syscall.Mmap(-1, 0, n*4, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return nil, err
	}
	adviseHuge(b)
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n), nil
}

func unmapFloats(f []float32) {
	b := unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), cap(f)*4)
	_ = syscall.Munmap(b)
}
