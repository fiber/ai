// Example: a two-layer transformer trained to continue a repeating token
// pattern. Demonstrates Embedding.Lookup, MultiHeadAttention with a causal
// mask, RMSNorm, and CrossEntropy over every position of a sequence.
package main

import (
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

const (
	vocab  = 12
	seqLen = 24
	dim    = 64
	heads  = 4
	layers = 2
	batch  = 32
)

// block is one pre-norm transformer layer: x + attn(norm(x)), then
// x + mlp(norm(x)).
type block struct {
	n1, n2 *nn.RMSNorm
	attn   *nn.MultiHeadAttention
	mlp    nn.Sequential
}

func newBlock() *block {
	b := &block{n1: nn.NewRMSNorm(dim), n2: nn.NewRMSNorm(dim), attn: nn.NewMultiHeadAttention(dim, heads),
		mlp: nn.Sequential{nn.NewLinear(dim, 4*dim), nn.GELU{}, nn.NewLinear(4*dim, dim)}}
	b.attn.Mask = tensor.CausalMask(seqLen)
	return b
}

func (b *block) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = x.Add(b.attn.Forward(b.n1.Forward(x)))
	return x.Add(b.mlp.Forward(b.n2.Forward(x)))
}

func (b *block) Params() []*tensor.Tensor {
	ps := append(b.n1.Params(), b.n2.Params()...)
	ps = append(ps, b.attn.Params()...)
	return append(ps, b.mlp.Params()...)
}

type model struct {
	tok, pos *nn.Embedding
	blocks   []*block
	norm     *nn.RMSNorm
	head     *nn.Linear
}

func newModel() *model {
	m := &model{tok: nn.NewEmbedding(vocab, dim), pos: nn.NewEmbedding(seqLen, dim), norm: nn.NewRMSNorm(dim), head: nn.NewLinear(dim, vocab)}
	for i := 0; i < layers; i++ {
		m.blocks = append(m.blocks, newBlock())
	}
	return m
}

// Forward maps [batch×seqLen] token ids to [batch·seqLen × vocab] logits.
func (m *model) Forward(ids [][]int) *tensor.Tensor {
	positions := make([]int, seqLen)
	for i := range positions {
		positions[i] = i
	}
	rows := make([]*tensor.Tensor, len(ids))
	pe := m.pos.Lookup(positions)
	for b, seq := range ids {
		rows[b] = m.tok.Lookup(seq).Add(pe)
	}
	x := tensor.Stack(0, rows...) // [batch, seqLen, dim]
	for _, blk := range m.blocks {
		x = blk.Forward(x)
	}
	return m.head.Forward(m.norm.Forward(x)).Reshape(-1, vocab)
}

func (m *model) Params() []*tensor.Tensor {
	ps := append(m.tok.Params(), m.pos.Params()...)
	for _, b := range m.blocks {
		ps = append(ps, b.Params()...)
	}
	ps = append(ps, m.norm.Params()...)
	return append(ps, m.head.Params()...)
}

// sample makes a sequence that repeats a random pattern of length 3–6;
// the task is to predict each next token, which is trivial once the
// model can look back far enough to find the period.
func sample(r *rand.Rand) (ids, next []int) {
	p := 3 + r.IntN(4)
	pattern := make([]int, p)
	for i := range pattern {
		pattern[i] = r.IntN(vocab)
	}
	ids = make([]int, seqLen)
	next = make([]int, seqLen)
	for i := range ids {
		ids[i] = pattern[i%p]
		next[i] = pattern[(i+1)%p]
	}
	return ids, next
}

func main() {
	tensor.Seed(3)
	r := rand.New(rand.NewPCG(3, 0))
	m := newModel()
	opt := optim.NewAdamW(m.Params(), 2e-3, 0.01)
	n := 0
	for _, p := range m.Params() {
		n += p.Size()
	}
	fmt.Printf("transformer with %d parameters, %d layers, %d heads\n", n, layers, heads)

	start := time.Now()
	for step := 1; step <= 400; step++ {
		ids := make([][]int, batch)
		targets := make([]int, 0, batch*seqLen)
		for b := range ids {
			var next []int
			ids[b], next = sample(r)
			targets = append(targets, next...)
		}
		logits := m.Forward(ids)
		loss := tensor.CrossEntropy(logits, targets)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step%100 == 0 || step == 1 {
			// accuracy from position 6 on: before one full period has passed
			// the next token cannot be known, whatever the model
			correct, counted := 0, 0
			tensor.NoGrad(func() {
				for i, p := range m.Forward(ids).Argmax(1) {
					if i%seqLen < 6 {
						continue
					}
					counted++
					if p == targets[i] {
						correct++
					}
				}
			})
			fmt.Printf("step %3d  loss %.3f  accuracy after one period %.1f%%\n", step, loss.Item(), 100*float64(correct)/float64(counted))
		}
	}
	fmt.Printf("trained in %.1fs\n", time.Since(start).Seconds())

	// Continue a pattern the model has never seen.
	ids, next := sample(r)
	tensor.NoGrad(func() {
		pred := m.Forward([][]int{ids}).Argmax(1)
		fmt.Println("\npattern :", ids[:12])
		fmt.Println("truth   :", next[:12])
		fmt.Println("predicted:", pred[:12])
	})
}
