// Package safetensors reads the safetensors weight format used by the
// Hugging Face ecosystem: an 8-byte little-endian header length, a JSON
// header describing every tensor (dtype, shape, byte offsets), and the
// raw data. Files are mapped read-only where the platform allows it and
// tensors are materialised as float32 into fiber/ai storage on request.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/fiber/ai/tensor"
)

// Dtype is the element type of a stored tensor, spelled as in the file.
type Dtype string

// The dtypes the format defines.
const (
	F64  Dtype = "F64"
	F32  Dtype = "F32"
	F16  Dtype = "F16"
	BF16 Dtype = "BF16"
	I64  Dtype = "I64"
	I32  Dtype = "I32"
	I16  Dtype = "I16"
	I8   Dtype = "I8"
	U8   Dtype = "U8"
	Bool Dtype = "BOOL"
)

// Size returns the byte width of one element, or 0 for an unknown dtype.
func (d Dtype) Size() int {
	switch d {
	case F64, I64:
		return 8
	case F32, I32:
		return 4
	case F16, BF16, I16:
		return 2
	case I8, U8, Bool:
		return 1
	}
	return 0
}

// Info describes one stored tensor.
type Info struct {
	Dtype Dtype
	Shape []int
	// File is the path of the file holding the tensor (one of several
	// when a directory was opened).
	File string

	start, end int64
	src        *source
}

// Elements returns the number of elements the shape holds.
func (i Info) Elements() int {
	n := 1
	for _, d := range i.Shape {
		n *= d
	}
	return n
}

// source is one mapped file: header parsed, data section available as a
// byte slice.
type source struct {
	path string
	data []byte // the data section only
	m    *mapping
}

// File is a set of tensors from one safetensors file or a directory of
// shards sharing one name space.
type File struct {
	sources []*source
	infos   map[string]Info
	meta    map[string]string
}

type headerEntry struct {
	Dtype       Dtype    `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Open opens one safetensors file.
func Open(path string) (*File, error) {
	f := &File{infos: map[string]Info{}, meta: map[string]string{}}
	if err := f.add(path); err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}

// OpenDir opens every *.safetensors file in dir as one name space. A
// model.safetensors.index.json is used to check completeness when
// present; the shards are opened either way. Duplicate names across
// files are an error.
func OpenDir(dir string) (*File, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("safetensors: no *.safetensors in %s", dir)
	}
	sort.Strings(paths)
	f := &File{infos: map[string]Info{}, meta: map[string]string{}}
	for _, p := range paths {
		if err := f.add(p); err != nil {
			f.Close()
			return nil, err
		}
	}
	if idx, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json")); err == nil {
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(idx, &index); err != nil {
			f.Close()
			return nil, fmt.Errorf("safetensors: index: %w", err)
		}
		for name, shard := range index.WeightMap {
			info, ok := f.infos[name]
			if !ok {
				f.Close()
				return nil, fmt.Errorf("safetensors: index lists %q in %s but no shard holds it", name, shard)
			}
			if filepath.Base(info.File) != shard {
				f.Close()
				return nil, fmt.Errorf("safetensors: index puts %q in %s, found in %s", name, shard, filepath.Base(info.File))
			}
		}
	}
	return f, nil
}

func (f *File) add(path string) error {
	m, err := mapFile(path)
	if err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	src := &source{path: path, m: m}
	b := m.bytes
	if len(b) < 8 {
		m.close()
		return fmt.Errorf("safetensors: %s: file too short for a header", path)
	}
	n := binary.LittleEndian.Uint64(b)
	if n > uint64(len(b)-8) {
		m.close()
		return fmt.Errorf("safetensors: %s: header length %d exceeds file size %d", path, n, len(b))
	}
	header := b[8 : 8+n]
	src.data = b[8+n:]

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(header, &raw); err != nil {
		m.close()
		return fmt.Errorf("safetensors: %s: header: %w", path, err)
	}
	for name, msg := range raw {
		if name == "__metadata__" {
			var meta map[string]string
			if err := json.Unmarshal(msg, &meta); err != nil {
				m.close()
				return fmt.Errorf("safetensors: %s: __metadata__: %w", path, err)
			}
			for k, v := range meta {
				f.meta[k] = v
			}
			continue
		}
		var e headerEntry
		if err := json.Unmarshal(msg, &e); err != nil {
			m.close()
			return fmt.Errorf("safetensors: %s: tensor %q: %w", path, name, err)
		}
		if _, dup := f.infos[name]; dup {
			m.close()
			return fmt.Errorf("safetensors: tensor %q appears in more than one file", name)
		}
		info := Info{Dtype: e.Dtype, Shape: e.Shape, File: path, start: e.DataOffsets[0], end: e.DataOffsets[1], src: src}
		if info.Dtype.Size() == 0 {
			m.close()
			return fmt.Errorf("safetensors: tensor %q: unknown dtype %q", name, e.Dtype)
		}
		for _, d := range e.Shape {
			if d < 0 {
				m.close()
				return fmt.Errorf("safetensors: tensor %q: negative dimension in shape %v", name, e.Shape)
			}
		}
		if info.start < 0 || info.end < info.start || info.end > int64(len(src.data)) {
			m.close()
			return fmt.Errorf("safetensors: tensor %q: data offsets [%d, %d) outside the data section of %d bytes", name, info.start, info.end, len(src.data))
		}
		if want := int64(info.Elements()) * int64(info.Dtype.Size()); info.end-info.start != want {
			m.close()
			return fmt.Errorf("safetensors: tensor %q: %d bytes for shape %v of %s, want %d", name, info.end-info.start, e.Shape, e.Dtype, want)
		}
		f.infos[name] = info
	}
	f.sources = append(f.sources, src)
	return nil
}

// Names returns the tensor names in sorted order.
func (f *File) Names() []string {
	names := make([]string, 0, len(f.infos))
	for n := range f.infos {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Info returns the description of a tensor.
func (f *File) Info(name string) (Info, bool) {
	i, ok := f.infos[name]
	return i, ok
}

// Metadata returns the merged __metadata__ entries of the opened files.
func (f *File) Metadata() map[string]string { return f.meta }

// Raw returns the stored bytes of a tensor without conversion. The slice
// aliases the mapping and is invalid after Close.
func (f *File) Raw(name string) ([]byte, error) {
	info, ok := f.infos[name]
	if !ok {
		return nil, f.missing(name)
	}
	return info.src.data[info.start:info.end], nil
}

// Tensor materialises a floating-point tensor as float32 in fiber/ai
// storage. F32 is copied, BF16 and F16 converted exactly, F64 narrowed.
// Integer and boolean tensors are not converted; use Raw.
func (f *File) Tensor(name string) (*tensor.Tensor, error) {
	info, ok := f.infos[name]
	if !ok {
		return nil, f.missing(name)
	}
	conv, ok := converters[info.Dtype]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q has dtype %s; read it with Raw", name, info.Dtype)
	}
	src := info.src.data[info.start:info.end]
	shape := info.Shape
	if len(shape) == 0 {
		shape = []int{}
	}
	if info.Elements() == 0 {
		return tensor.Zeros(shape...), nil
	}
	return tensor.Generate(func(dst []float32) { conv(dst, src) }, shape...), nil
}

// Close unmaps every file. Tensors returned by Tensor stay valid; slices
// from Raw do not.
func (f *File) Close() error {
	var err error
	for _, s := range f.sources {
		if s.m != nil {
			err = errors.Join(err, s.m.close())
			s.m = nil
		}
	}
	f.sources = nil
	return err
}

func (f *File) missing(name string) error {
	names := f.Names()
	hint := ""
	if len(names) > 0 {
		// Point at a likely candidate: same suffix after the last dot.
		suffix := name[strings.LastIndex(name, ".")+1:]
		for _, n := range names {
			if strings.HasSuffix(n, "."+suffix) || n == suffix {
				hint = fmt.Sprintf(" (did you mean %q?)", n)
				break
			}
		}
	}
	return fmt.Errorf("safetensors: no tensor %q%s", name, hint)
}
