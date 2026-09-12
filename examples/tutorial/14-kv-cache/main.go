// Tutorial chapter 14: the same decoder as chapter 13, generating with
// and without a KV cache, so the difference can be measured rather than
// asserted.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"time"

	"github.com/fiber/ai-data/shakespeare"
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

const ropeBase = 1e4

type block struct {
	norm1 *nn.RMSNorm
	attn  *nn.MultiHeadAttention
	norm2 *nn.RMSNorm
	up    *nn.Linear
	down  *nn.Linear
}

func newBlock(dim, heads int) *block {
	a := nn.NewMultiHeadAttention(dim, heads)
	a.RoPEBase = ropeBase
	return &block{norm1: nn.NewRMSNorm(dim), attn: a, norm2: nn.NewRMSNorm(dim),
		up: nn.NewLinear(dim, 4*dim), down: nn.NewLinear(4*dim, dim)}
}

// forward runs the block. With a cache the attention only projects the
// new tokens and reads the rest from the cache; without one it attends
// over the whole input under a causal mask.
func (b *block) forward(x *tensor.Tensor, c *nn.KVCache, mask *tensor.Tensor) *tensor.Tensor {
	h := b.norm1.Forward(x)
	if c != nil {
		x = x.Add(b.attn.Step(h, c))
	} else {
		b.attn.Mask = mask
		x = x.Add(b.attn.Forward(h))
	}
	B, T, D := x.Dim(0), x.Dim(1), x.Dim(2)
	f := b.norm2.Forward(x).Reshape(B*T, D)
	return x.Add(b.down.Forward(b.up.Forward(f).GELU()).Reshape(B, T, D))
}

func (b *block) Params() []*tensor.Tensor {
	var p []*tensor.Tensor
	for _, m := range []nn.Module{b.norm1, b.attn, b.norm2, b.up, b.down} {
		p = append(p, m.Params()...)
	}
	return p
}

type model struct {
	tok    *nn.Embedding
	blocks []*block
	norm   *nn.RMSNorm
	head   *nn.Linear
	dim    int
}

func newModel(vocab, dim, heads, layers int) *model {
	m := &model{tok: nn.NewEmbedding(vocab, dim), norm: nn.NewRMSNorm(dim),
		head: nn.NewLinearNoBias(dim, vocab), dim: dim}
	tensor.NoGrad(func() { m.tok.W.MulScalarInPlace(0.02) })
	for range layers {
		m.blocks = append(m.blocks, newBlock(dim, heads))
	}
	return m
}

// forward maps ids to scores. caches is nil during training and holds
// one cache per block during cached generation.
func (m *model) forward(ids []int, B, T int, caches []*nn.KVCache, mask *tensor.Tensor) *tensor.Tensor {
	x := m.tok.Lookup(ids).Reshape(B, T, m.dim)
	for i, b := range m.blocks {
		var c *nn.KVCache
		if caches != nil {
			c = caches[i]
		}
		x = b.forward(x, c, mask)
	}
	return m.head.Forward(m.norm.Forward(x).Reshape(B*T, m.dim))
}

func (m *model) Params() []*tensor.Tensor {
	p := m.tok.Params()
	for _, b := range m.blocks {
		p = append(p, b.Params()...)
	}
	return append(p, append(m.norm.Params(), m.head.Params()...)...)
}

func sample(logits []float32, temp float32, r *rand.Rand) int {
	best := float32(math.Inf(-1))
	for _, v := range logits {
		best = max(best, v)
	}
	sum, p := float32(0), make([]float32, len(logits))
	for i, v := range logits {
		p[i] = float32(math.Exp(float64((v - best) / temp)))
		sum += p[i]
	}
	t := r.Float32() * sum
	for i, v := range p {
		if t -= v; t <= 0 {
			return i
		}
	}
	return len(p) - 1
}

// generateWhole is chapter 13's loop: the entire prefix goes through the
// model for every character produced.
func generateWhole(m *model, symbols []byte, start, n, ctx int, temp float32, r *rand.Rand) string {
	ids, out := []int{start}, make([]byte, 0, n)
	for range n {
		in := ids
		if len(in) > ctx {
			in = in[len(in)-ctx:]
		}
		var logits []float32
		tensor.NoGrad(func() {
			rows := m.forward(in, 1, len(in), nil, tensor.CausalMask(len(in)))
			logits = rows.Rows([]int{len(in) - 1}).Float32s()
		})
		next := sample(logits, temp, r)
		ids = append(ids, next)
		out = append(out, symbols[next])
	}
	return string(out)
}

// generateCached feeds one character per step and lets the cache supply
// everything before it.
func generateCached(m *model, symbols []byte, start, n, ctx int, temp float32, r *rand.Rand) string {
	caches := make([]*nn.KVCache, len(m.blocks))
	for i := range caches {
		caches[i] = &nn.KVCache{}
	}
	id, out := start, make([]byte, 0, n)
	for range n {
		var logits []float32
		tensor.NoGrad(func() {
			logits = m.forward([]int{id}, 1, 1, caches, nil).Float32s()
		})
		id = sample(logits, temp, r)
		out = append(out, symbols[id])
		// The uncached path slides a window of ctx characters; the cache
		// has to be bounded the same way, or the two stop computing the
		// same thing — and the cache would grow without limit.
		for _, c := range caches {
			c.Trim(ctx - 1)
		}
	}
	return string(out)
}

func main() {
	var (
		dim    = flag.Int("dim", 256, "model width")
		heads  = flag.Int("heads", 4, "attention heads")
		layers = flag.Int("layers", 4, "transformer blocks")
		ctx    = flag.Int("ctx", 128, "context length for the uncached path")
		batch  = flag.Int("batch", 32, "sequences per training step")
		steps  = flag.Int("steps", 400, "training steps — enough for word-shaped output")
		lr     = flag.Float64("lr", 3e-4, "AdamW learning rate")
		gen    = flag.Int("gen", 400, "characters to generate")
		temp   = flag.Float64("temp", 0.8, "sampling temperature")
		seed   = flag.Uint64("seed", 1, "random seed")
	)
	flag.Parse()

	text, err := shakespeare.Text()
	if err != nil {
		log.Fatal(err)
	}
	symbols, index := shakespeare.Vocabulary(text)
	data := shakespeare.Encode(text, index)

	tensor.Seed(*seed)
	r := rand.New(rand.NewPCG(*seed, 0))
	m := newModel(len(symbols), *dim, *heads, *layers)
	opt := optim.NewAdamW(m.Params(), float32(*lr), 0.01)
	mask := tensor.CausalMask(*ctx)

	fmt.Printf("training %d steps for a model that writes something worth timing\n", *steps)
	start := time.Now()
	for step := 1; step <= *steps; step++ {
		ids := make([]int, 0, *batch**ctx)
		targets := make([]int, 0, *batch**ctx)
		for range *batch {
			s := r.IntN(len(data) - *ctx - 1)
			ids = append(ids, data[s:s+*ctx]...)
			targets = append(targets, data[s+1:s+*ctx+1]...)
		}
		loss := tensor.CrossEntropy(m.forward(ids, *batch, *ctx, nil, mask), targets)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step%100 == 0 {
			fmt.Printf("  step %4d  loss %.3f  %.0fs\n", step, loss.Item(), time.Since(start).Seconds())
		}
	}

	// Both paths from the same seed must produce the same text: the cache
	// is an optimisation, not an approximation.
	begin := index['\n']
	r1 := rand.New(rand.NewPCG(7, 0))
	t0 := time.Now()
	whole := generateWhole(m, symbols, begin, *gen, *ctx, float32(*temp), r1)
	wholeTook := time.Since(t0)

	r2 := rand.New(rand.NewPCG(7, 0))
	t1 := time.Now()
	cached := generateCached(m, symbols, begin, *gen, *ctx, float32(*temp), r2)
	cachedTook := time.Since(t1)

	fmt.Printf("\n%s\n", cached)
	agree := 0
	for agree < len(whole) && agree < len(cached) && whole[agree] == cached[agree] {
		agree++
	}
	fmt.Printf("\nthe two paths agree for the first %d of %d characters\n", agree, *gen)
	fmt.Printf("whole prefix each step: %6.2fs  %6.0f characters/s\n",
		wholeTook.Seconds(), float64(*gen)/wholeTook.Seconds())
	fmt.Printf("with a KV cache:       %6.2fs  %6.0f characters/s  (%.1fx)\n",
		cachedTook.Seconds(), float64(*gen)/cachedTook.Seconds(),
		float64(wholeTook)/float64(cachedTook))

	// What the cache costs: two tensors per layer, growing by one row per
	// token, at four bytes a float.
	bytes := 2 * *layers * *gen * *dim * 4
	fmt.Printf("cache after %d characters: %.1f MB (2 x %d layers x %d tokens x %d wide x 4 bytes)\n",
		*gen, float64(bytes)/1e6, *layers, *gen, *dim)
}
