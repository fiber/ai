//go:build unix

package safetensors

import (
	"fmt"
	"os"
	"syscall"
)

// mapping is a read-only view of a whole file.
type mapping struct {
	bytes  []byte
	mapped bool
}

func mapFile(path string) (*mapping, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() == 0 {
		return &mapping{}, nil
	}
	if int64(int(st.Size())) != st.Size() {
		return nil, fmt.Errorf("%s: too large to map", path)
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		// Fall back to reading; some file systems refuse mmap.
		data, rerr := os.ReadFile(path)
		if rerr != nil {
			return nil, rerr
		}
		return &mapping{bytes: data}, nil
	}
	return &mapping{bytes: b, mapped: true}, nil
}

func (m *mapping) close() error {
	if m.mapped && m.bytes != nil {
		err := syscall.Munmap(m.bytes)
		m.bytes, m.mapped = nil, false
		return err
	}
	m.bytes = nil
	return nil
}
