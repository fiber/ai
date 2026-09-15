package safetensors

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/fiber/ai/tensor"
)

// metadataKey is the header entry holding free-form metadata rather than
// a tensor.
const metadataKey = "__metadata__"

// alignment is what the data section is padded to. The format does not
// require it, but the reference implementation writes it and readers
// that map the file and cast expect it.
const alignment = 8

// Save writes the tensors to path as one safetensors file, with meta (if
// any) as the file's __metadata__. Every tensor is stored as F32, which
// is what a tensor.Tensor holds.
//
// The output is deterministic: tensors are written in sorted name order,
// so saving the same model twice produces identical bytes.
func Save(path string, tensors map[string]*tensor.Tensor, meta map[string]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	if err := Write(w, tensors, meta); err != nil {
		f.Close()
		os.Remove(path)
		return err
	}
	if err := w.Flush(); err != nil {
		f.Close()
		os.Remove(path)
		return err
	}
	return f.Close()
}

// Write writes the same bytes Save writes to w.
func Write(w io.Writer, tensors map[string]*tensor.Tensor, meta map[string]string) error {
	names := make([]string, 0, len(tensors))
	for name, t := range tensors {
		if name == "" {
			return fmt.Errorf("safetensors: a tensor has an empty name")
		}
		if name == metadataKey {
			return fmt.Errorf("safetensors: %q is reserved for metadata", metadataKey)
		}
		if t == nil {
			return fmt.Errorf("safetensors: tensor %q is nil", name)
		}
		names = append(names, name)
	}
	sort.Strings(names)

	// The header names every tensor's byte range within the data section,
	// so the layout is decided before anything is written. json.Marshal
	// sorts map keys, which together with the sorted write order makes
	// the whole file a function of its contents.
	entries := make(map[string]any, len(names)+1)
	if len(meta) > 0 {
		entries[metadataKey] = meta
	}
	var offset int64
	for _, name := range names {
		t := tensors[name]
		size := int64(t.Size()) * int64(F32.Size())
		entries[name] = headerEntry{Dtype: F32, Shape: t.Shape(), DataOffsets: [2]int64{offset, offset + size}}
		offset += size
	}
	header, err := json.Marshal(entries)
	if err != nil {
		return fmt.Errorf("safetensors: header: %w", err)
	}
	// Pad with spaces (valid JSON whitespace) so the data section starts
	// at a multiple of alignment counting from the start of the file.
	for (8+len(header))%alignment != 0 {
		header = append(header, ' ')
	}

	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(header)))
	if _, err := w.Write(length[:]); err != nil {
		return err
	}
	if _, err := w.Write(header); err != nil {
		return err
	}

	// One reusable buffer: the tensors go out in the order the header
	// promised, largest one at a time rather than the whole model.
	var values []float32
	var buf []byte
	for _, name := range names {
		t := tensors[name]
		n := t.Size()
		if cap(values) < n {
			values, buf = make([]float32, n), make([]byte, 4*n)
		}
		values, buf = values[:n], buf[:4*n]
		t.CopyTo(values)
		for i, v := range values {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
		}
		if _, err := w.Write(buf); err != nil {
			return fmt.Errorf("safetensors: writing %q: %w", name, err)
		}
	}
	return nil
}
