package main

import (
	"testing"

	"github.com/fiber/ai-data/shakespeare"
)

// The chapter's claims about the encoding itself: more merges mean
// fewer tokens, the corpus survives a round trip byte for byte, and
// the pieces are shorter than the characters they replace.
func TestBPE(t *testing.T) {
	text, err := shakespeare.Text()
	if err != nil {
		t.Fatal(err)
	}
	symbols, index := shakespeare.Vocabulary(text)

	var counts []int
	b := trainBPE(text, symbols, index, 256, []int{0, 16, 64, 256}, func(done, tokens int, last string) {
		counts = append(counts, tokens)
		if done > 0 && len(last) < 2 {
			t.Errorf("merge %d produced piece %q, want at least two bytes", done, last)
		}
	})
	if len(counts) != 4 {
		t.Fatalf("reported %d milestones, want 4", len(counts))
	}
	if counts[0] != len(text) {
		t.Errorf("before any merge the corpus is %d tokens, want %d characters", counts[0], len(text))
	}
	for i := 1; i < len(counts); i++ {
		if counts[i] >= counts[i-1] {
			t.Errorf("token count %d after milestone %d is not below %d", counts[i], i, counts[i-1])
		}
	}
	if len(b.pieces) != len(symbols)+256 {
		t.Errorf("vocabulary %d, want %d", len(b.pieces), len(symbols)+256)
	}

	ids := b.Encode(text, index)
	if got := b.Decode(ids); got != text {
		t.Fatalf("round trip changed the text (%d bytes back, %d in)", len(got), len(text))
	}
	if len(ids) != counts[len(counts)-1] {
		t.Errorf("encoder produced %d tokens, trainer counted %d", len(ids), counts[len(counts)-1])
	}
	if len(ids) >= len(text) {
		t.Errorf("%d tokens for %d characters: the encoding did not shorten the corpus", len(ids), len(text))
	}
}

// chunks must keep the space with the word that follows it and put
// every other byte on its own.
func TestChunks(t *testing.T) {
	got := chunks("To be, or\nnot")
	want := []string{"To", " be", ",", " or", "\n", "not"}
	if len(got) != len(want) {
		t.Fatalf("chunks = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("chunks = %q, want %q", got, want)
		}
	}
}
