package gemma

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/fiber/ai/tensor"
)

// EmbeddingGemma 2 audio tests. References come from
// testdata/make_reference_eg2_audio.py; the speech is public-domain
// LibriVox, see testdata/audio/README.md.

func readWAVFile(t testing.TB, name string) []float32 {
	f, err := os.Open(filepath.Join("testdata", "audio", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s, rate, err := ReadWAV(f)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 {
		t.Fatalf("%s: %d Hz", name, rate)
	}
	return s
}

// audioClips builds the clips exactly as the reference script does.
func audioClips(t testing.TB) map[string][]float32 {
	speech := readWAVFile(t, "librivox_20s.wav")
	return map[string][]float32{
		"speech_3s":            speech[:3*16000],
		"speech_20s":           speech,
		"speech_40s_truncated": append(append([]float32(nil), speech...), speech...),
		"tone_440hz":           readWAVFile(t, "tone_440hz.wav"),
		"noise_2s":             readWAVFile(t, "noise_2s.wav"),
		"quiet_1s":             readWAVFile(t, "quiet_1s.wav"),
		"speech_50ms":          speech[:800],
	}
}

type audioRefs struct {
	Clips     []string `json:"clips"`
	MixedText string   `json:"mixed_text"`
}

func TestParityAudio2(t *testing.T) {
	var refs audioRefs
	b, err := os.ReadFile(filepath.Join("testdata", "eg2_audio.json"))
	if err != nil {
		t.Skip(err)
	}
	if err := json.Unmarshal(b, &refs); err != nil {
		t.Fatal(err)
	}
	ref := readRefMatrix(t, "eg2_audio.bin")
	m := load2(t)
	defer m.Close()
	clips := audioClips(t)
	in := make([][]float32, len(refs.Clips))
	for i, n := range refs.Clips {
		in[i] = clips[n]
	}
	start := time.Now()
	got, err := m.EmbedAudio(in)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d clips in %v (audio tower loaded on first use)", len(in), time.Since(start))
	var sum, worst float64 = 0, 1
	for i := range got {
		c := cosine(got[i], ref[i])
		t.Logf("%-22s cosine %.6f", refs.Clips[i], c)
		sum += c
		worst = min(worst, c)
	}
	mean := sum / float64(len(got))
	t.Logf("mean %.6f, min %.6f", mean, worst)
	if mean < 0.9995 || worst < 0.999 {
		t.Errorf("mean %.6f (want ≥ 0.9995), min %.6f (want ≥ 0.999)", mean, worst)
	}

	mixed, err := m.EmbedInputs([]Input{{Text: refs.MixedText, Audio: [][]float32{clips["speech_3s"]}}})
	if err != nil {
		t.Fatal(err)
	}
	c := cosine(mixed[0], ref[len(refs.Clips)])
	t.Logf("mixed text and audio cosine %.6f", c)
	if c < 0.999 {
		t.Errorf("mixed input cosine %.6f", c)
	}
}

// Spoken sentences, synthesised with macOS's say at test time (the audio
// is never committed), must each be nearest to their own transcript.
func TestSpokenSentenceRetrieval2(t *testing.T) {
	if _, err := exec.LookPath("say"); err != nil {
		t.Skip("needs macOS say and afconvert")
	}
	m := load2(t)
	defer m.Close()
	texts := []string{
		"the network link went down",
		"someone failed to log in to the server",
		"the disk on the database host is almost full",
		"the weather will be sunny and warm tomorrow",
	}
	dir := t.TempDir()
	clips := make([][]float32, len(texts))
	for i, s := range texts {
		aiff := filepath.Join(dir, "s.aiff")
		wav := filepath.Join(dir, "s.wav")
		if out, err := exec.Command("say", "-o", aiff, s).CombinedOutput(); err != nil {
			t.Skipf("say: %v %s", err, out)
		}
		if out, err := exec.Command("afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav).CombinedOutput(); err != nil {
			t.Skipf("afconvert: %v %s", err, out)
		}
		clips[i] = readWAVPath(t, wav)
	}
	av, err := m.EmbedAudio(clips)
	if err != nil {
		t.Fatal(err)
	}
	tv, err := m.Embed(texts, Prompt("query"))
	if err != nil {
		t.Fatal(err)
	}
	for i := range av {
		best, bestC := -1, -2.0
		for j := range tv {
			if c := cosine(av[i], tv[j]); c > bestC {
				best, bestC = j, c
			}
		}
		t.Logf("spoken %q: nearest %q (%.3f)", texts[i], texts[best], bestC)
		if best != i {
			t.Errorf("spoken %q is nearest to %q", texts[i], texts[best])
		}
	}
}

func readWAVPath(t testing.TB, path string) []float32 {
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	s, rate, err := ReadWAV(f)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 {
		t.Fatalf("%s: %d Hz", path, rate)
	}
	return s
}

func TestAudioInputErrors2(t *testing.T) {
	m := load2(t)
	defer m.Close()
	clip := readWAVFile(t, "tone_440hz.wav")
	for name, in := range map[string]Input{
		"placeholder without audio": {Text: "listen: <|audio|>"},
		"audio without placeholder": {Text: "listen", Audio: [][]float32{clip}},
		"too short":                 {Text: "<|audio|>", Audio: [][]float32{clip[:10]}},
		"video placeholder":         {Text: "watch <|video|>"},
	} {
		if _, err := m.EmbedInputs([]Input{in}); err == nil {
			t.Errorf("%s: no error", name)
		} else {
			t.Logf("%s: %v", name, err)
		}
	}
}

func TestEmbedAudioDoesNotPinMemory2(t *testing.T) {
	m := load2(t)
	defer m.Close()
	clips := [][]float32{readWAVFile(t, "tone_440hz.wav"), readWAVFile(t, "noise_2s.wav")}
	const limit = 256 << 20
	tensor.SetMappedLimit(limit)
	defer tensor.SetMappedLimit(512 << 20)
	run := func(n int) {
		for range n {
			if _, err := m.EmbedAudio(clips); err != nil {
				t.Fatal(err)
			}
		}
	}
	run(3)
	_, _, retained0, pinned0 := tensor.MappedStats()
	run(3)
	_, _, retained1, pinned1 := tensor.MappedStats()
	t.Logf("steady state: pinned %d -> %d bytes, retained %d -> %d bytes", pinned0, pinned1, retained0, retained1)
	if pinned1 > pinned0 {
		t.Errorf("pinned memory grew by %d bytes over 3 calls", pinned1-pinned0)
	}
	if retained1 > limit+8<<20 {
		t.Errorf("retained memory %d bytes exceeds the %d byte limit", retained1, limit)
	}
}
