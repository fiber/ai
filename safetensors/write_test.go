package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/fiber/ai/tensor"
)

func testTensors() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"encoder.weight": tensor.New([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
		"encoder.bias":   tensor.New([]float32{-0.5, 0.25}, 2),
		"scale":          tensor.New([]float32{3.5}, 1),
		"block.0.w":      tensor.New([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2),
	}
}

func TestSaveRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.safetensors")
	want := testTensors()
	meta := map[string]string{"format": "pt", "trained_by": "fiber/ai"}
	if err := Save(path, want, meta); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if got, want := len(f.Names()), len(want); got != want {
		t.Fatalf("%d tensors in the file, wrote %d", got, want)
	}
	if got := f.Metadata()["trained_by"]; got != "fiber/ai" {
		t.Errorf("metadata trained_by = %q", got)
	}
	for name, w := range want {
		info, ok := f.Info(name)
		if !ok {
			t.Fatalf("%q missing from the file", name)
		}
		if info.Dtype != F32 {
			t.Errorf("%q stored as %s, want F32", name, info.Dtype)
		}
		got, err := f.Tensor(name)
		if err != nil {
			t.Fatalf("%q: %v", name, err)
		}
		if !got.Shape().Equal(w.Shape()) {
			t.Errorf("%q shape %v, want %v", name, got.Shape(), w.Shape())
		}
		g, e := got.Float32s(), w.Float32s()
		for i := range e {
			if g[i] != e[i] {
				t.Errorf("%q[%d] = %v, want %v", name, i, g[i], e[i])
			}
		}
	}
}

// The file must be a function of its contents, so a checksum of a
// checkpoint means something.
func TestSaveIsDeterministic(t *testing.T) {
	var a, b bytes.Buffer
	meta := map[string]string{"a": "1", "b": "2"}
	if err := Write(&a, testTensors(), meta); err != nil {
		t.Fatal(err)
	}
	if err := Write(&b, testTensors(), meta); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Error("two writes of the same tensors differ")
	}
}

// The layout every other implementation expects: 8-byte length, JSON
// header, data section starting at a multiple of 8, offsets relative to
// that section.
func TestSaveLayout(t *testing.T) {
	var buf bytes.Buffer
	if err := Write(&buf, testTensors(), nil); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	n := binary.LittleEndian.Uint64(b[:8])
	if (8+n)%alignment != 0 {
		t.Errorf("data section starts at %d, not a multiple of %d", 8+n, alignment)
	}
	var raw map[string]headerEntry
	if err := json.Unmarshal(b[8:8+n], &raw); err != nil {
		t.Fatalf("header is not JSON: %v", err)
	}
	data := b[8+n:]
	total := int64(0)
	for name, e := range raw {
		size := int64(4)
		for _, d := range e.Shape {
			size *= int64(d)
		}
		if e.DataOffsets[1]-e.DataOffsets[0] != size {
			t.Errorf("%q spans %d bytes, shape %v needs %d", name, e.DataOffsets[1]-e.DataOffsets[0], e.Shape, size)
		}
		if e.DataOffsets[1] > int64(len(data)) {
			t.Errorf("%q ends at %d, past the %d-byte data section", name, e.DataOffsets[1], len(data))
		}
		total += size
	}
	if total != int64(len(data)) {
		t.Errorf("tensors cover %d bytes, data section is %d", total, len(data))
	}
}

// A view saves as what it looks like, not as its storage.
func TestSaveTransposedView(t *testing.T) {
	path := filepath.Join(t.TempDir(), "v.safetensors")
	x := tensor.New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err := Save(path, map[string]*tensor.Tensor{"xt": x.T()}, nil); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	got, err := f.Tensor("xt")
	if err != nil {
		t.Fatal(err)
	}
	if !got.Shape().Equal(tensor.Shape{3, 2}) {
		t.Fatalf("shape %v, want [3 2]", got.Shape())
	}
	want := []float32{1, 4, 2, 5, 3, 6}
	for i, v := range got.Float32s() {
		if v != want[i] {
			t.Fatalf("element %d = %v, want %v (got %v)", i, v, want[i], got.Float32s())
		}
	}
}

func TestSaveRejects(t *testing.T) {
	cases := map[string]map[string]*tensor.Tensor{
		"empty name":    {"": tensor.Zeros(1)},
		"reserved name": {metadataKey: tensor.Zeros(1)},
		"nil tensor":    {"x": nil},
	}
	for name, ts := range cases {
		if err := Write(&bytes.Buffer{}, ts, nil); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

// A failed write must not leave a half-written file behind.
func TestSaveRemovesFileOnError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.safetensors")
	if err := Save(path, map[string]*tensor.Tensor{"": tensor.Zeros(1)}, nil); err == nil {
		t.Fatal("no error")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("file still exists after a failed save (%v)", err)
	}
}
