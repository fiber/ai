package main

import (
	"math/rand/v2"
	"testing"
)

// The generator's promises, which the chapter's numbers rest on: every
// category has held-out shapes, no line of a held-out shape is in the
// training set, and the bag of words only knows training words.
func TestGeneratorSplit(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 0))
	train, seen, unseen := generate(r, 10)

	if len(shapes) != len(categories) {
		t.Fatalf("%d shape lists for %d categories", len(shapes), len(categories))
	}
	for c, sh := range shapes {
		if len(sh) <= heldOut {
			t.Errorf("%s has %d shapes, not enough to hold out %d", categories[c], len(sh), heldOut)
		}
	}
	perCat := make([]int, len(categories))
	for _, l := range unseen.labels {
		perCat[l]++
	}
	for c, n := range perCat {
		if n != heldOut*10 {
			t.Errorf("%s: %d held-out lines, want %d", categories[c], n, heldOut*10)
		}
	}

	// A held-out shape's fixed text must not occur in any training line.
	fixed := func(shape string) string {
		// The longest run without a placeholder identifies the shape.
		best := ""
		cur := ""
		for i := 0; i < len(shape); i++ {
			if shape[i] == '%' {
				if len(cur) > len(best) {
					best = cur
				}
				cur = ""
				i++
				continue
			}
			cur += string(shape[i])
		}
		if len(cur) > len(best) {
			best = cur
		}
		return best
	}
	for _, sh := range shapes {
		for _, s := range sh[len(sh)-heldOut:] {
			key := fixed(s)
			for _, set := range []set{train, seen} {
				for _, l := range set.lines {
					if contains(l, key) {
						t.Errorf("held-out shape text %q appears in %q", key, l)
					}
				}
			}
		}
	}

	v := vocabulary(train.lines)
	for w := range v {
		found := false
		for _, l := range train.lines {
			for _, lw := range words(l) {
				if lw == w {
					found = true
				}
			}
		}
		if !found {
			t.Errorf("vocabulary word %q is in no training line", w)
		}
	}
	x := bag(unseen.lines, v)
	if x.Dim(1) != len(v) {
		t.Errorf("bag has %d columns for a vocabulary of %d", x.Dim(1), len(v))
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestWordsDropNumbers(t *testing.T) {
	got := words("sshd[1234]: Accepted publickey for root from 10.1.2.3 port 22 ssh2")
	want := []string{"sshd", "accepted", "publickey", "for", "root", "from", "port"}
	if len(got) != len(want) {
		t.Fatalf("words = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("words = %q, want %q", got, want)
		}
	}
}
