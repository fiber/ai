package gemma

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/fiber/ai/safetensors"
	"github.com/fiber/ai/tensor"
)

// EmbeddingGemma 2 tests. The weight-dependent ones need
// $FIBERAI_MODELS/embeddinggemma-2; the references come from
// testdata/make_reference_eg2.py (transformers main + sentence-transformers
// 6.1, float32, CPU).

func modelDir2(t testing.TB) string {
	dir := os.Getenv("FIBERAI_MODELS")
	if dir == "" {
		t.Skip("FIBERAI_MODELS not set")
	}
	d := filepath.Join(dir, "embeddinggemma-2")
	if _, err := os.Stat(filepath.Join(d, "config.json")); err != nil {
		t.Skip(err)
	}
	return d
}

func load2(t testing.TB) *Model {
	m, err := Load(modelDir2(t))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// The architecture is read from the nested text_config, with the per-layer
// geometry and rotary bases the reference configuration defines. No
// weights needed.
func TestConfig2(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"model_type": "embedding_gemma2", "vision_config": {"hidden_size": 768},
	  "text_config": {"hidden_size": 512, "intermediate_size": 2048, "num_hidden_layers": 12,
	    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256, "vocab_size": 262144,
	    "sliding_window": 512, "rms_norm_eps": 1e-6, "hidden_activation": "gelu_pytorch_tanh",
	    "hidden_size_per_layer_input": 512, "embedding_dim": 768, "max_position_embeddings": 262144,
	    "per_layer_config": {"05": {"head_dim": 512, "num_key_value_heads": 1},
	                         "11": {"head_dim": 512, "num_key_value_heads": 1}},
	    "rope_parameters": {"full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
	                        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if c.Variant != 2 || c.HiddenSize != 512 || c.EmbeddingDim != 768 {
		t.Fatalf("variant %d, hidden %d, embedding %d", c.Variant, c.HiddenSize, c.EmbeddingDim)
	}
	for _, tc := range []struct {
		layer, headDim, kv int
		sliding            bool
		base               float64
	}{
		{0, 256, 2, true, 1e4},
		{4, 256, 2, true, 1e4},
		{5, 512, 1, false, 1e6},
		{11, 512, 1, false, 1e6},
	} {
		if got := c.headDim(tc.layer); got != tc.headDim {
			t.Errorf("layer %d head_dim %d, want %d", tc.layer, got, tc.headDim)
		}
		if got := c.kvHeads(tc.layer); got != tc.kv {
			t.Errorf("layer %d kv heads %d, want %d", tc.layer, got, tc.kv)
		}
		if got := c.isSliding(tc.layer); got != tc.sliding {
			t.Errorf("layer %d sliding %v, want %v", tc.layer, got, tc.sliding)
		}
		if got := c.ropeBase(tc.layer); got != tc.base {
			t.Errorf("layer %d rope base %g, want %g", tc.layer, got, tc.base)
		}
	}
	// An inclusive radius of 512: attend while |i-j| < 513.
	if got := c.window(); got != 513 {
		t.Errorf("window bound %d, want 513", got)
	}
}

// RMSNorm weights in this checkpoint multiply the normalised value
// directly. Loading them with Gemma 3's (1 + w) offset would not fail, it
// would just be wrong, so compare a loaded norm with the stored tensor.
func TestNormWeightsAreScales2(t *testing.T) {
	dir := modelDir2(t)
	m := load2(t)
	defer m.Close()
	f, err := safetensors.OpenDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for l := range m.layers {
		stored, err := f.Tensor(fmt.Sprintf("language_model.layers.%d.post_attention_layernorm.weight", l))
		if err != nil {
			t.Fatal(err)
		}
		want, got := stored.Float32s(), m.layers[l].postAttnNorm.Float32s()
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("layer %d norm[%d] = %v, stored %v: an offset was applied", l, i, got[i], want[i])
			}
		}
		if s := m.layers[l].scalar; s <= 0 || s > 2 {
			t.Errorf("layer %d scalar %v", l, s)
		}
	}
}

func TestParity2(t *testing.T) {
	m := load2(t)
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
	f, err := os.Open(filepath.Join("testdata", "eg2_reference.bin"))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	ref, qdRef := readMatrix(t, f), readMatrix(t, f)

	got, err := m.Embed(sc.Sentences)
	if err != nil {
		t.Fatal(err)
	}
	if len(got[0]) != 768 {
		t.Fatalf("dimension %d, want 768", len(got[0]))
	}
	var sum, worst float64 = 0, 1
	worstAt := 0
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

	qd, _ := m.Embed([]string{sc.QueryPrompted, sc.DocumentPrompted})
	for i := range qd {
		c := cosine(qd[i], qdRef[i])
		t.Logf("prompted[%d] cosine %.6f", i, c)
		if c < 0.999 {
			t.Errorf("prompted[%d] cosine %.6f", i, c)
		}
	}
	// The named prompts come from config_sentence_transformers.json.
	q2, err := m.Embed([]string{"what is the mtu of the uplink"}, Prompt("query"))
	if err != nil {
		t.Fatal(err)
	}
	if c := cosine(q2[0], qd[0]); c < 0.9999 {
		t.Errorf("Prompt(\"query\") differs from the literal prompt: %.6f", c)
	}
}

// Inputs longer than the sliding window: the window rule decides these.
func TestParityLong2(t *testing.T) {
	m := load2(t)
	defer m.Close()
	var lc struct {
		Texts []string `json:"texts"`
	}
	b, err := os.ReadFile(filepath.Join("testdata", "long_sentences.json"))
	if err != nil {
		t.Skip(err)
	}
	if err := json.Unmarshal(b, &lc); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join("testdata", "eg2_reference_long.bin"))
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
		c := cosine(got[i], ref[i])
		t.Logf("long input %d: %d tokens, cosine %.6f", i, len(m.tok.Encode(lc.Texts[i])), c)
		if c < 0.999 {
			t.Errorf("long input %d: cosine %.6f below 0.999", i, c)
		}
	}
}

func TestBatchInvariance2(t *testing.T) {
	m := load2(t)
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

// The model card lists 512, 256 and 128 as truncation sizes.
func TestMatryoshka2(t *testing.T) {
	m := load2(t)
	defer m.Close()
	s := []string{"network anomaly detected on core switch"}
	full, _ := m.Embed(s)
	for _, d := range []int{512, 256, 128} {
		trunc, err := m.Embed(s, Dim(d))
		if err != nil {
			t.Fatal(err)
		}
		if len(trunc[0]) != d {
			t.Fatalf("dim %d, want %d", len(trunc[0]), d)
		}
		ref := append([]float32(nil), full[0][:d]...)
		var n float64
		for _, v := range ref {
			n += float64(v) * float64(v)
		}
		for i := range ref {
			ref[i] /= float32(math.Sqrt(n))
		}
		if c := cosine(trunc[0], ref); c < 0.9999 {
			t.Errorf("Dim(%d) cosine %.6f against the truncated full vector", d, c)
		}
	}
}

func TestEmbedDoesNotPinMemory2(t *testing.T) {
	m := load2(t)
	defer m.Close()
	texts := make([]string, 16)
	for i := range texts {
		texts[i] = "interface GigabitEthernet0/1 changed state to down on switch access-7"
	}
	const limit = 128 << 20
	tensor.SetMappedLimit(limit)
	defer tensor.SetMappedLimit(512 << 20)
	run := func(n int) {
		for range n {
			if _, err := m.Embed(texts); err != nil {
				t.Fatal(err)
			}
		}
	}
	run(20)
	_, _, retained0, pinned0 := tensor.MappedStats()
	run(20)
	_, _, retained1, pinned1 := tensor.MappedStats()
	t.Logf("steady state: pinned %d -> %d bytes, retained %d -> %d bytes", pinned0, pinned1, retained0, retained1)
	if pinned1 > pinned0 {
		t.Errorf("pinned memory grew by %d bytes over 20 calls", pinned1-pinned0)
	}
	if retained1 > limit+8<<20 {
		t.Errorf("retained memory %d bytes exceeds the %d byte limit", retained1, limit)
	}
}

func TestLoadAndEmbedTime2(t *testing.T) {
	dir := modelDir2(t)
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
	if _, err := m.Embed(texts[:4]); err != nil { // warm-up
		t.Fatal(err)
	}
	start = time.Now()
	if _, err := m.Embed(texts); err != nil {
		t.Fatal(err)
	}
	el := time.Since(start)
	t.Logf("32 sentences in %v (%.0f sent/s)", el, 32/el.Seconds())
}

// Image and audio placeholders in the text would be embedded as ordinary
// tokens by a text-only encoder; they must be refused instead.
func TestRejectsMultimodalPlaceholders2(t *testing.T) {
	m := load2(t)
	defer m.Close()
	for _, s := range []string{"look at this <|image|> picture", "listen: <|audio|>"} {
		if _, err := m.Embed([]string{"fine", s}); err == nil {
			t.Errorf("%q: no error", s)
		} else {
			t.Log(err)
		}
	}
	if _, err := m.Embed([]string{"plain text mentioning an image"}); err != nil {
		t.Errorf("plain text refused: %v", err)
	}
}

// benchTexts are syslog-length inputs, the workload the manual quotes.
func benchTexts() []string {
	base := []string{
		"interface GigabitEthernet0/1 changed state to down",
		"%BGP-5-ADJCHANGE: neighbor 10.0.0.1 Up",
		"Authentication failure for user admin from 203.0.113.7 port 22",
		"disk usage on /var reached 92 percent on host web-3",
	}
	out := make([]string, 0, 64)
	for i := range 64 {
		out = append(out, fmt.Sprintf("%s (event %d)", base[i%len(base)], i))
	}
	return out
}

func benchEmbed(b *testing.B, dir string) {
	m, err := Load(dir)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	texts := benchTexts()
	if _, err := m.Embed(texts); err != nil { // warm-up
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if _, err := m.Embed(texts); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(texts)*b.N)/b.Elapsed().Seconds(), "texts/s")
}

func BenchmarkEmbed300m(b *testing.B) { benchEmbed(b, modelDir(b)) }
func BenchmarkEmbed2(b *testing.B)    { benchEmbed(b, modelDir2(b)) }
