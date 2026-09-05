//go:build !unix

package tensor

import "errors"

const mmapSupported = false

func mapFloats(int) ([]float32, error) {
	return nil, errors.New("mmap is not supported on this platform")
}
func unmapFloats([]float32) {}
