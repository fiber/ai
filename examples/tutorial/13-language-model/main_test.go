package main

import (
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/fiber/ai-data/shakespeare"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

// A model small enough to train in a second: the point is that the loop
// learns at all, not that it learns well.
func tiny(t *testing.T) (*model, []byte, [256]int, []int) {
	t.Helper()
	text, err := shakespeare.Text()
	if err != nil {
		t.Fatal(err)
	}
	symbols, index := shakespeare.Vocabulary(text)
	ids := shakespeare.Encode(text[:60000], index)
	tensor.Seed(3)
	return newModel(len(symbols), 64, 2, 2, 32), symbols, index, ids
}

func TestTrainingReducesLoss(t *testing.T) {
	m, _, _, ids := tiny(t)
	const batch, ctx = 8, 32
	r := rand.New(rand.NewPCG(3, 0))
	batchOf := func() (in, targets []int) {
		for range batch {
			s := r.IntN(len(ids) - ctx - 1)
			in = append(in, ids[s:s+ctx]...)
			targets = append(targets, ids[s+1:s+ctx+1]...)
		}
		return in, targets
	}

	opt := optim.NewAdamW(m.Params(), 3e-3, 0.01)
	var first, last float32
	for step := range 60 {
		in, targets := batchOf()
		loss := tensor.CrossEntropy(m.Forward(in, batch, ctx), targets)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step == 0 {
			first = loss.Item()
		}
		last = loss.Item()
	}
	// ln(65) = 4.17 is the loss of a model that has learned nothing.
	if first < 3.5 {
		t.Errorf("the untrained model already scores %.2f; ln(65) is 4.17", first)
	}
	if last >= first-0.8 {
		t.Errorf("loss went from %.2f to %.2f in 60 steps, which is not learning", first, last)
	}
}

// Generation must produce the requested number of characters, all of
// them from the vocabulary, and must not depend on a full context: it
// starts from one character and has to work while the prefix is short.
func TestGenerateFromOneCharacter(t *testing.T) {
	m, symbols, index, _ := tiny(t)
	r := rand.New(rand.NewPCG(4, 0))
	out := generate(m, symbols, index, '\n', 50, 32, 0.8, r)
	if len(out) != 50 {
		t.Fatalf("generated %d characters, want 50", len(out))
	}
	for i := 0; i < len(out); i++ {
		if index[out[i]] < 0 {
			t.Fatalf("character %d (%q) is not in the vocabulary", i, out[i])
		}
	}
	// Longer than the context, so the prefix has to be trimmed.
	if long := generate(m, symbols, index, '\n', 80, 32, 0.8, r); len(long) != 80 {
		t.Errorf("generated %d characters past the context length, want 80", len(long))
	}
}

// Temperature 0 would divide by zero; the example documents 0.8 and the
// low end should still be usable. At a very low temperature sampling is
// effectively greedy, so two draws from the same state must agree.
func TestLowTemperatureIsNearlyGreedy(t *testing.T) {
	logits := []float32{0.1, 3.0, 0.2, 2.9}
	r := rand.New(rand.NewPCG(5, 0))
	for range 20 {
		if got := sample(logits, 0.01, r); got != 1 {
			t.Fatalf("at temperature 0.01 the highest score should win, got index %d", got)
		}
	}
}

func TestVocabularyIsTheCorpusAlphabet(t *testing.T) {
	_, symbols, _, _ := tiny(t)
	if len(symbols) != 65 {
		t.Errorf("vocabulary has %d symbols, want 65", len(symbols))
	}
	if !strings.Contains(string(symbols), "\n") {
		t.Error("the newline is not in the vocabulary, so the model cannot end a line")
	}
}
