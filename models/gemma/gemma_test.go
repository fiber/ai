package gemma

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func modelDir(t testing.TB) string {
	dir := os.Getenv("FIBERAI_MODELS")
	if dir == "" {
		t.Skip("FIBERAI_MODELS not set")
	}
	d := filepath.Join(dir, "embeddinggemma-300m")
	if _, err := os.Stat(filepath.Join(d, "config.json")); err != nil {
		t.Skip(err)
	}
	return d
}

func readMatrix(t testing.TB, f *os.File) [][]float32 {
	var hdr [2]int32
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil {
		t.Fatal(err)
	}
	n, dim := int(hdr[0]), int(hdr[1])
	buf := make([]float32, n*dim)
	if err := binary.Read(f, binary.LittleEndian, &buf); err != nil {
		t.Fatal(err)
	}
	m := make([][]float32, n)
	for i := range m {
		m[i] = buf[i*dim : (i+1)*dim]
	}
	return m
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
}

func loadReference(t testing.TB) (sentences [][]float32, qd [][]float32) {
	f, err := os.Open(filepath.Join("testdata", "reference.bin"))
	if err != nil {
		t.Skip("no reference.bin: " + err.Error())
	}
	defer f.Close()
	return readMatrix(t, f), readMatrix(t, f)
}

func TestParity(t *testing.T) {
	m, err := Load(modelDir(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	var sc struct {
		Sentences        []string `json:"sentences"`
		QueryPrompted    string   `json:"query_prompted"`
		DocumentPrompted string   `json:"document_prompted"`
	}
	b, _ := os.ReadFile(filepath.Join("testdata", "sentences.json"))
	if err := json.Unmarshal(b, &sc); err != nil {
		t.Fatal(err)
	}
	ref, qdRef := loadReference(t)

	got, err := m.Embed(sc.Sentences)
	if err != nil {
		t.Fatal(err)
	}
	var sum, worst float64 = 0, 1
	worstAt := -1
	for i := range got {
		c := cosine(got[i], ref[i])
		sum += c
		if c < worst {
			worst, worstAt = c, i
		}
	}
	mean := sum / float64(len(got))
	t.Logf("cosine to fp32 reference: mean %.6f, min %.6f at %q", mean, worst, sc.Sentences[worstAt])
	if mean < 0.9995 {
		t.Errorf("mean cosine %.6f below 0.9995", mean)
	}
	if worst < 0.999 {
		t.Errorf("min cosine %.6f below 0.999", worst)
	}

	// Prompted query/document embeddings.
	qEmb, _ := m.Embed([]string{sc.QueryPrompted}, PromptText(""))
	q2, _ := m.Embed([]string{"what is the mtu of the uplink"}, PromptText("task: search result | query: "))
	if c := cosine(qEmb[0], q2[0]); c < 0.9999 {
		t.Errorf("literal vs split prompt differ: %.6f", c)
	}
	qd, _ := m.Embed([]string{sc.QueryPrompted, sc.DocumentPrompted}, PromptText(""))
	for i := range qd {
		if c := cosine(qd[i], qdRef[i]); c < 0.999 {
			t.Errorf("prompted[%d] cosine %.6f", i, c)
		}
	}
}

func TestBatchInvariance(t *testing.T) {
	m, err := Load(modelDir(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	texts := []string{
		"short",
		"a considerably longer sentence that will pad the shorter ones in the batch",
		"medium length input here",
	}
	together, _ := m.Embed(texts)
	for i, s := range texts {
		alone, _ := m.Embed([]string{s})
		if c := cosine(together[i], alone[0]); c < 0.9999 {
			t.Errorf("padding changed %q: cosine %.6f", s, c)
		}
	}
}

func TestMatryoshka(t *testing.T) {
	m, err := Load(modelDir(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	full, _ := m.Embed([]string{"network anomaly detected on core switch"})
	trunc, _ := m.Embed([]string{"network anomaly detected on core switch"}, Dim(256))
	if len(trunc[0]) != 256 {
		t.Fatalf("dim = %d, want 256", len(trunc[0]))
	}
	// The truncated-and-renormalised full vector should match Dim(256).
	ref := append([]float32(nil), full[0][:256]...)
	var n float64
	for _, v := range ref {
		n += float64(v) * float64(v)
	}
	n = math.Sqrt(n)
	for i := range ref {
		ref[i] /= float32(n)
	}
	if c := cosine(trunc[0], ref); c < 0.9999 {
		t.Errorf("Matryoshka cosine %.6f", c)
	}
}

func TestLoadAndEmbedTime(t *testing.T) {
	dir := modelDir(t)
	start := time.Now()
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	t.Logf("load %v", time.Since(start))
	texts := make([]string, 32)
	for i := range texts {
		texts[i] = "the quick brown fox jumps over the lazy dog again and again"
	}
	start = time.Now()
	if _, err := m.Embed(texts); err != nil {
		t.Fatal(err)
	}
	el := time.Since(start)
	t.Logf("32 sentences in %v (%.0f sent/s)", el, 32/el.Seconds())
}

// TestParityLong checks inputs longer than the 512-token sliding window,
// where the sliding and full-attention layers stop being equivalent and the
// window mask path is exercised.
func TestParityLong(t *testing.T) {
	m, err := Load(modelDir(t))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	var lc struct {
		Texts  []string `json:"texts"`
		Tokens []int    `json:"tokens"`
	}
	b, err := os.ReadFile(filepath.Join("testdata", "long_sentences.json"))
	if err != nil {
		t.Skip(err)
	}
	if err := json.Unmarshal(b, &lc); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join("testdata", "reference_long.bin"))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	ref := readMatrix(t, f)
	got, err := m.Embed(lc.Texts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		n := len(m.tok.Encode(lc.Texts[i]))
		c := cosine(got[i], ref[i])
		t.Logf("long input %d: %d tokens (reference %d), cosine %.6f", i, n, lc.Tokens[i], c)
		if c < 0.999 {
			t.Errorf("long input %d: cosine %.6f below 0.999", i, c)
		}
	}
}
