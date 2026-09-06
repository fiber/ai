package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// writeFile builds a safetensors file from raw tensors, in the order given.
type rawTensor struct {
	name  string
	dtype Dtype
	shape []int
	data  []byte
}

func writeFile(t *testing.T, path string, meta map[string]string, tensors ...rawTensor) {
	t.Helper()
	header := map[string]any{}
	if meta != nil {
		header["__metadata__"] = meta
	}
	var data []byte
	for _, r := range tensors {
		start := len(data)
		data = append(data, r.data...)
		header[r.name] = map[string]any{"dtype": r.dtype, "shape": r.shape, "data_offsets": []int{start, len(data)}}
	}
	hb, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	// Real files pad the header to 8 bytes with spaces; do the same.
	for len(hb)%8 != 0 {
		hb = append(hb, ' ')
	}
	var buf []byte
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(hb)))
	buf = append(buf, hb...)
	buf = append(buf, data...)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

func f32bytes(v ...float32) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}
	return b
}

func u16bytes(v ...uint16) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint16(b, x)
	}
	return b
}

func TestReadDtypes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	f64 := binary.LittleEndian.AppendUint64(nil, math.Float64bits(1.5))
	f64 = binary.LittleEndian.AppendUint64(f64, math.Float64bits(-2.25))
	writeFile(t, path, map[string]string{"format": "pt"},
		rawTensor{"w.f32", F32, []int{2, 3}, f32bytes(1, 2, 3, 4, 5, 6)},
		// bf16: top halves of 1.0 (0x3f80), -2.0 (0xc000), 3.140625 (0x4049), 0
		rawTensor{"w.bf16", BF16, []int{4}, u16bytes(0x3f80, 0xc000, 0x4049, 0)},
		// f16: 1.0, -2.0, 65504 (max), smallest subnormal, a subnormal, +Inf, -Inf, NaN, -0
		rawTensor{"w.f16", F16, []int{9}, u16bytes(0x3c00, 0xc000, 0x7bff, 0x0001, 0x03ff, 0x7c00, 0xfc00, 0x7e00, 0x8000)},
		rawTensor{"w.f64", F64, []int{2}, f64},
		rawTensor{"w.i64", I64, []int{1}, make([]byte, 8)},
		rawTensor{"w.empty", F32, []int{0, 4}, nil},
		rawTensor{"w.scalar", F32, []int{}, f32bytes(7)},
	)
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if got := f.Names(); strings.Join(got, ",") != "w.bf16,w.empty,w.f16,w.f32,w.f64,w.i64,w.scalar" {
		t.Fatalf("names %v", got)
	}
	if f.Metadata()["format"] != "pt" {
		t.Fatalf("metadata %v", f.Metadata())
	}
	info, ok := f.Info("w.f32")
	if !ok || info.Dtype != F32 || len(info.Shape) != 2 || info.Elements() != 6 {
		t.Fatalf("info %+v", info)
	}

	x, err := f.Tensor("w.f32")
	if err != nil {
		t.Fatal(err)
	}
	if got := x.Float32s(); got[0] != 1 || got[5] != 6 || x.Dims() != 2 {
		t.Fatalf("f32 %v", got)
	}
	x, _ = f.Tensor("w.bf16")
	if got := x.Float32s(); got[0] != 1 || got[1] != -2 || got[2] != 3.140625 || got[3] != 0 {
		t.Fatalf("bf16 %v", got)
	}
	x, _ = f.Tensor("w.f16")
	got := x.Float32s()
	want := []float32{1, -2, 65504, float32(math.Pow(2, -24)), float32(1023 * math.Pow(2, -24))}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("f16[%d] = %v, want %v", i, got[i], w)
		}
	}
	if !math.IsInf(float64(got[5]), 1) || !math.IsInf(float64(got[6]), -1) || !math.IsNaN(float64(got[7])) {
		t.Fatalf("f16 specials %v", got[5:8])
	}
	if math.Float32bits(got[8]) != 0x80000000 {
		t.Fatalf("f16 -0 = %#x", math.Float32bits(got[8]))
	}
	x, _ = f.Tensor("w.f64")
	if got := x.Float32s(); got[0] != 1.5 || got[1] != -2.25 {
		t.Fatalf("f64 %v", got)
	}
	if _, err := f.Tensor("w.i64"); err == nil || !strings.Contains(err.Error(), "Raw") {
		t.Fatalf("i64 error %v", err)
	}
	if raw, err := f.Raw("w.i64"); err != nil || len(raw) != 8 {
		t.Fatalf("raw %v %v", raw, err)
	}
	x, err = f.Tensor("w.empty")
	if err != nil || x.Size() != 0 || x.Shape()[1] != 4 {
		t.Fatalf("empty %v %v", x, err)
	}
	x, _ = f.Tensor("w.scalar")
	if x.Dims() != 0 || x.Item() != 7 {
		t.Fatalf("scalar %v", x)
	}
	if _, err := f.Tensor("layers.0.w.f32"); err == nil || !strings.Contains(err.Error(), `did you mean "w.f32"`) {
		t.Fatalf("missing hint: %v", err)
	}
}

func TestF16Exhaustive(t *testing.T) {
	// Every finite half value must round-trip through float32 exactly:
	// convert, then re-encode by the reference formula and compare.
	for h := 0; h < 1<<16; h++ {
		v := f16to32(uint16(h))
		exp := (h >> 10) & 0x1f
		if exp == 0x1f {
			continue
		}
		mant := float64(h & 0x3ff)
		var want float64
		if exp == 0 {
			want = mant * math.Pow(2, -24)
		} else {
			want = (1 + mant/1024) * math.Pow(2, float64(exp-15))
		}
		if h&0x8000 != 0 {
			want = -want
		}
		if float64(v) != want {
			t.Fatalf("f16 %#04x = %v, want %v", h, v, want)
		}
	}
}

func TestShardsAndIndex(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), nil,
		rawTensor{"a", F32, []int{1}, f32bytes(1)})
	writeFile(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), nil,
		rawTensor{"b", F32, []int{1}, f32bytes(2)})
	index := map[string]any{"weight_map": map[string]string{
		"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
	ib, _ := json.Marshal(index)
	os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o644)

	f, err := OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info, _ := f.Info("b"); filepath.Base(info.File) != "model-00002-of-00002.safetensors" {
		t.Fatalf("shard %v", info.File)
	}
	if x, _ := f.Tensor("b"); x.Item() != 2 {
		t.Fatal("b")
	}
	f.Close()

	// Index that names a tensor no shard holds.
	index = map[string]any{"weight_map": map[string]string{"c": "model-00001-of-00002.safetensors"}}
	ib, _ = json.Marshal(index)
	os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o644)
	if _, err := OpenDir(dir); err == nil || !strings.Contains(err.Error(), `"c"`) {
		t.Fatalf("index error %v", err)
	}
	os.Remove(filepath.Join(dir, "model.safetensors.index.json"))

	// Duplicate name across shards.
	writeFile(t, filepath.Join(dir, "extra.safetensors"), nil, rawTensor{"a", F32, []int{1}, f32bytes(3)})
	if _, err := OpenDir(dir); err == nil || !strings.Contains(err.Error(), "more than one file") {
		t.Fatalf("duplicate error %v", err)
	}
}

func TestBadFiles(t *testing.T) {
	dir := t.TempDir()
	cases := map[string]struct {
		build func(path string)
		want  string
	}{
		"short": {func(p string) { os.WriteFile(p, []byte{1, 2, 3}, 0o644) }, "too short"},
		"header length": {func(p string) {
			os.WriteFile(p, binary.LittleEndian.AppendUint64(nil, 1<<40), 0o644)
		}, "exceeds file size"},
		"bad json": {func(p string) {
			b := binary.LittleEndian.AppendUint64(nil, 8)
			os.WriteFile(p, append(b, []byte("{notjson")...), 0o644)
		}, "header"},
		"unknown dtype": {func(p string) {
			writeFile(t, p, nil, rawTensor{"x", "Q4", []int{1}, []byte{1}})
		}, `unknown dtype "Q4"`},
		"size mismatch": {func(p string) {
			writeFile(t, p, nil, rawTensor{"x", F32, []int{2}, f32bytes(1)})
		}, "4 bytes for shape [2] of F32, want 8"},
		"offsets outside": {func(p string) {
			// Hand-build a header whose offsets run past the data.
			hb := []byte(`{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`)
			b := binary.LittleEndian.AppendUint64(nil, uint64(len(hb)))
			os.WriteFile(p, append(b, hb...), 0o644)
		}, "outside the data section"},
	}
	for name, c := range cases {
		p := filepath.Join(dir, strings.ReplaceAll(name, " ", "_")+".safetensors")
		c.build(p)
		_, err := Open(p)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want %q", name, err, c.want)
		}
	}
	if _, err := Open(filepath.Join(dir, "nope.safetensors")); err == nil {
		t.Error("missing file")
	}
	if _, err := OpenDir(dir + "/empty"); err == nil {
		t.Error("empty dir")
	}
}

// TestRealModel opens the EmbeddingGemma weights when FIBERAI_MODELS is set
// and materialises every tensor, checking dtype handling and load time on
// a real file.
func TestRealModel(t *testing.T) {
	dir := os.Getenv("FIBERAI_MODELS")
	if dir == "" {
		t.Skip("FIBERAI_MODELS not set")
	}
	path := filepath.Join(dir, "embeddinggemma-300m", "model.safetensors")
	if _, err := os.Stat(path); err != nil {
		t.Skip(err)
	}
	start := time.Now()
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var elems int
	dtypes := map[Dtype]int{}
	for _, name := range f.Names() {
		info, _ := f.Info(name)
		dtypes[info.Dtype]++
		x, err := f.Tensor(name)
		if err != nil {
			t.Fatal(err)
		}
		elems += x.Size()
		x.Release()
	}
	el := time.Since(start)
	t.Logf("%d tensors, %d parameters, dtypes %v, metadata %v, %v", len(f.Names()), elems, dtypes, f.Metadata(), el)
	if elems < 300_000_000 {
		t.Fatalf("expected about 308M parameters, got %d", elems)
	}
	if el > 2*time.Second {
		t.Errorf("materialising all tensors took %v, want under 2 s", el)
	}
}
