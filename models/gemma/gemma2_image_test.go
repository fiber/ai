package gemma

import (
	"encoding/json"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/fiber/ai/tensor"
)

// EmbeddingGemma 2 image tests. References come from
// testdata/make_reference_eg2_images.py (sentence-transformers 6.1,
// float32, CPU); the images are in testdata/images.

type imageRefs struct {
	Images    []string `json:"images"`
	MixedText string   `json:"mixed_text"`
	Prompt    string   `json:"prompt"`
}

func readImageRefs(t testing.TB) imageRefs {
	var r imageRefs
	b, err := os.ReadFile(filepath.Join("testdata", "eg2_images.json"))
	if err != nil {
		t.Skip(err)
	}
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func loadImage(t testing.TB, name string) image.Image {
	f, err := os.Open(filepath.Join("testdata", "images", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

func readRefMatrix(t testing.TB, name string) [][]float32 {
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Skip(err)
	}
	defer f.Close()
	return readMatrix(t, f)
}

// Square, landscape, portrait, a grayscale image, a tiny image upscaled
// thirty-fold, an extreme aspect ratio, an image with alpha, and a photo.
func TestParityImages2(t *testing.T) {
	refs := readImageRefs(t)
	ref := readRefMatrix(t, "eg2_images.bin")
	m := load2(t)
	defer m.Close()
	imgs := make([]image.Image, len(refs.Images))
	for i, n := range refs.Images {
		imgs[i] = loadImage(t, n)
	}
	start := time.Now()
	got, err := m.EmbedImages(imgs)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%d images in %v (vision tower loaded on first use)", len(imgs), time.Since(start))
	var sum, worst float64 = 0, 1
	for i := range got {
		c := cosine(got[i], ref[i])
		t.Logf("%-20s cosine %.6f", refs.Images[i], c)
		sum += c
		worst = min(worst, c)
	}
	mean := sum / float64(len(got))
	t.Logf("mean %.6f, min %.6f", mean, worst)
	if mean < 0.9995 || worst < 0.999 {
		t.Errorf("mean %.6f (want ≥ 0.9995), min %.6f (want ≥ 0.999)", mean, worst)
	}
}

// Text with an image in the middle, and an image under a prompt.
func TestParityMixed2(t *testing.T) {
	refs := readImageRefs(t)
	ref := readRefMatrix(t, "eg2_mixed.bin")
	m := load2(t)
	defer m.Close()
	mixed, err := m.EmbedInputs([]Input{{Text: refs.MixedText, Images: []image.Image{loadImage(t, "red_circle.png")}}})
	if err != nil {
		t.Fatal(err)
	}
	if c := cosine(mixed[0], ref[0]); c < 0.999 {
		t.Errorf("mixed input cosine %.6f", c)
	} else {
		t.Logf("mixed input cosine %.6f", c)
	}
	prompted, err := m.EmbedImages([]image.Image{loadImage(t, "blue_square.png")}, PromptText(refs.Prompt))
	if err != nil {
		t.Fatal(err)
	}
	if c := cosine(prompted[0], ref[1]); c < 0.999 {
		t.Errorf("prompted image cosine %.6f", c)
	} else {
		t.Logf("prompted image cosine %.6f", c)
	}
}

// Without the reference: each synthetic image must be nearest to its own
// description among the set, through the shared embedding space.
func TestImageTextRetrieval2(t *testing.T) {
	m := load2(t)
	defer m.Close()
	names := []string{"red_circle.png", "blue_square.png", "green_triangle.png", "earth.png"}
	texts := []string{"a red circle", "a blue square", "a green triangle", "a photo of the Earth seen from space"}
	imgs := make([]image.Image, len(names))
	for i, n := range names {
		imgs[i] = loadImage(t, n)
	}
	iv, err := m.EmbedImages(imgs)
	if err != nil {
		t.Fatal(err)
	}
	tv, err := m.Embed(texts, Prompt("query"))
	if err != nil {
		t.Fatal(err)
	}
	for i := range iv {
		best, bestC := -1, -2.0
		for j := range tv {
			if c := cosine(iv[i], tv[j]); c > bestC {
				best, bestC = j, c
			}
		}
		t.Logf("%-20s nearest %q (%.3f)", names[i], texts[best], bestC)
		if best != i {
			t.Errorf("%s: nearest text is %q, want %q", names[i], texts[best], texts[i])
		}
	}
}

// Go's JPEG decoder and libjpeg do not produce identical pixels; how much
// that moves the embedding is measured here and reported, not asserted
// as parity.
func TestJPEGDecodeEffect2(t *testing.T) {
	m := load2(t)
	defer m.Close()
	v, err := m.EmbedImages([]image.Image{loadImage(t, "earth.png"), loadImage(t, "earth.jpg")})
	if err != nil {
		t.Fatal(err)
	}
	c := cosine(v[0], v[1])
	t.Logf("same photo decoded by libjpeg (PNG) and by Go (JPEG): cosine %.6f", c)
	if c < 0.99 {
		t.Errorf("JPEG decoding moved the embedding too far: cosine %.6f", c)
	}
}

func TestImageInputErrors2(t *testing.T) {
	m := load2(t)
	defer m.Close()
	img := loadImage(t, "red_circle.png")
	for name, in := range map[string]Input{
		"placeholder without image": {Text: "look: <|image|>"},
		"image without placeholder": {Text: "look", Images: []image.Image{img}},
		"audio placeholder":         {Text: "listen <|audio|>"},
	} {
		if _, err := m.EmbedInputs([]Input{in}); err == nil {
			t.Errorf("%s: no error", name)
		} else {
			t.Logf("%s: %v", name, err)
		}
	}
	if _, err := m.Embed([]string{"look: <|image|>"}); err == nil {
		t.Error("Embed accepted an image placeholder")
	}
}

// Repeated image calls must not pin memory or grow the mapped allocator
// past its limit, as for text (B-005).
func TestEmbedImagesDoesNotPinMemory2(t *testing.T) {
	m := load2(t)
	defer m.Close()
	imgs := []image.Image{loadImage(t, "red_circle.png"), loadImage(t, "earth.png")}
	const limit = 256 << 20
	tensor.SetMappedLimit(limit)
	defer tensor.SetMappedLimit(512 << 20)
	run := func(n int) {
		for range n {
			if _, err := m.EmbedImages(imgs); err != nil {
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
