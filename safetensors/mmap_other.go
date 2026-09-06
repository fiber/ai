//go:build !unix

package safetensors

import "os"

type mapping struct {
	bytes []byte
}

func mapFile(path string) (*mapping, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return &mapping{bytes: data}, nil
}

func (m *mapping) close() error { m.bytes = nil; return nil }
