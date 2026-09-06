package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// modelTokenizer returns the path to the real tokenizer.json, or skips.
func modelTokenizer(t testing.TB) string {
	dir := os.Getenv("FIBERAI_MODELS")
	if dir == "" {
		t.Skip("FIBERAI_MODELS not set")
	}
	p := filepath.Join(dir, "embeddinggemma-300m", "tokenizer.json")
	if _, err := os.Stat(p); err != nil {
		t.Skip(err)
	}
	return p
}

type fixture struct {
	S   string `json:"s"`
	IDs []int  `json:"ids"`
	Raw []int  `json:"raw"`
}

func loadFixtures(t testing.TB) []fixture {
	b, err := os.ReadFile(filepath.Join("testdata", "tokens.json"))
	if err != nil {
		t.Skip("no fixtures: " + err.Error())
	}
	var fx []fixture
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}
	return fx
}

func equal(a []int, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestParity(t *testing.T) {
	tok, err := Load(modelTokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	fx := loadFixtures(t)
	if len(fx) == 0 {
		t.Fatal("no fixtures")
	}
	if tok.BOS() != 2 || tok.EOS() != 1 {
		t.Fatalf("bos/eos = %d/%d, want 2/1", tok.BOS(), tok.EOS())
	}
	fails := 0
	for _, f := range fx {
		if got := tok.Encode(f.S); !equal(got, f.IDs) {
			t.Errorf("Encode(%q)\n got %v\nwant %v", f.S, got, f.IDs)
			if fails++; fails > 12 {
				t.Fatal("too many mismatches")
			}
		}
		if got := tok.EncodeRaw(f.S); !equal(got, f.Raw) {
			t.Errorf("EncodeRaw(%q)\n got %v\nwant %v", f.S, got, f.Raw)
			if fails++; fails > 12 {
				t.Fatal("too many mismatches")
			}
		}
	}
}

func TestRoundTrip(t *testing.T) {
	tok, err := Load(modelTokenizer(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range loadFixtures(t) {
		// The normalizer maps space<->metaspace reversibly and byte
		// fallback is exact, so raw text round-trips, except that a literal
		// metaspace character in the input is indistinguishable from a
		// space after tokenizing and comes back as a space.
		if strings.Contains(f.S, tok.normTo) {
			continue
		}
		got := tok.Decode(tok.EncodeRaw(f.S))
		if got != f.S {
			t.Errorf("round trip %q -> %q", f.S, got)
		}
	}
}

func TestLoadTime(t *testing.T) {
	path := modelTokenizer(t)
	start := time.Now()
	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	el := time.Since(start)
	t.Logf("load %v, vocab %d", el, tok.VocabSize())
	if el > time.Second {
		t.Errorf("load took %v, want under 1 s", el)
	}
	if tok.VocabSize() != 262144 {
		t.Errorf("vocab size %d", tok.VocabSize())
	}
}

func BenchmarkEncode(b *testing.B) {
	tok, err := Load(modelTokenizer(b))
	if err != nil {
		b.Fatal(err)
	}
	line := "Sep  8 09:15:42 fw01 %ASA-6-302013: Built outbound TCP connection 12345 for outside:203.0.113.7/443 to inside:10.0.0.5/52918"
	chars := len([]rune(line))
	b.SetBytes(int64(chars))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok.EncodeRaw(line)
	}
	b.ReportMetric(float64(chars)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Mchar/s")
}
