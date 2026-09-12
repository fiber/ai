package main

import (
	"math/rand/v2"
	"testing"

	"github.com/fiber/ai-data/shakespeare"
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/tensor"
)

// While the whole history fits the window the two paths must agree
// exactly: the cache is an optimisation, not an approximation. Past the
// window they compute the same thing by different arithmetic and drift,
// which chapter 14 explains — so this checks the exact regime.
func TestCachedAndWholeAgreeWithinTheWindow(t *testing.T) {
	text, err := shakespeare.Text()
	if err != nil {
		t.Fatal(err)
	}
	symbols, index := shakespeare.Vocabulary(text)
	tensor.Seed(21)
	m := newModel(len(symbols), 64, 2, 2)
	const ctx, n = 64, 40 // n < ctx, so nothing is trimmed

	whole := generateWhole(m, symbols, index['\n'], n, ctx, 0.8, rand.New(rand.NewPCG(9, 0)))
	cached := generateCached(m, symbols, index['\n'], n, ctx, 0.8, rand.New(rand.NewPCG(9, 0)))
	if whole != cached {
		t.Errorf("within the window the paths differ:\n uncached %q\n cached   %q", whole, cached)
	}
	if len(cached) != n {
		t.Errorf("generated %d characters, want %d", len(cached), n)
	}
}

// Trim must bound the memory without moving the tokens that remain:
// Pos keeps counting while Len stops growing.
func TestCacheTrimBoundsMemory(t *testing.T) {
	tensor.Seed(22)
	m := newModel(40, 32, 2, 1)
	c := &nn.KVCache{}
	tensor.NoGrad(func() {
		for i := range 20 {
			m.blocks[0].attn.Step(tensor.Randn(1, 1, 32), c)
			c.Trim(8)
			if got := c.Len(); got > 8 {
				t.Fatalf("after %d tokens the cache holds %d, want at most 8", i+1, got)
			}
		}
	})
	if c.Pos() != 20 {
		t.Errorf("cache counted %d positions, want 20", c.Pos())
	}
	if c.Len() != 8 {
		t.Errorf("cache holds %d rows, want 8", c.Len())
	}
}
