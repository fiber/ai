// Tutorial chapter 13: a character-level language model — the decoder
// architecture every current model is built from, trained from scratch
// on 1.1 MB of Shakespeare and then sampled from.
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

// ropeBase is the theta of the rotary positions. 10 000 is the value
// almost every model uses at short context lengths; Gemma raises it to
// 1e6 for its long-context layers, because a larger base turns the
// rotation more slowly and so distinguishes positions further apart.
const ropeBase = 1e4

// block is one transformer layer, pre-norm: the input is normalised
// before each sub-layer and added back afterwards, so the residual
// stream runs unchanged from the embedding to the output and the
// gradient has a short path to every layer.
type block struct {
	norm1 *nn.RMSNorm
	attn  *nn.MultiHeadAttention
	norm2 *nn.RMSNorm
	up    *nn.Linear
	down  *nn.Linear
}

func newBlock(dim, heads, ctx int) *block {
	a := nn.NewMultiHeadAttention(dim, heads)
	a.Mask = tensor.CausalMask(ctx) // a token may attend to itself and earlier
	a.RoPEBase = ropeBase           // positions by rotation, not a lookup table
	return &block{
		norm1: nn.NewRMSNorm(dim), attn: a,
		norm2: nn.NewRMSNorm(dim),
		up:    nn.NewLinear(dim, 4*dim), down: nn.NewLinear(4*dim, dim),
	}
}

func (b *block) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = x.Add(b.attn.Forward(b.norm1.Forward(x)))
	B, T, D := x.Dim(0), x.Dim(1), x.Dim(2)
	// The feed-forward part is per token, so the tokens are folded into
	// the row dimension and unfolded again.
	h := b.norm2.Forward(x).Reshape(B*T, D)
	h = b.down.Forward(b.up.Forward(h).GELU()).Reshape(B, T, D)
	return x.Add(h)
}

func (b *block) Params() []*tensor.Tensor {
	var p []*tensor.Tensor
	for _, m := range []nn.Module{b.norm1, b.attn, b.norm2, b.up, b.down} {
		p = append(p, m.Params()...)
	}
	return p
}

// model is the whole network: a table of token vectors, a stack of
// blocks, a final normalisation and a projection back to one score per
// symbol in the vocabulary.
type model struct {
	tok    *nn.Embedding
	blocks []*block
	norm   *nn.RMSNorm
	head   *nn.Linear
	dim    int
}

func newModel(vocab, dim, heads, layers, ctx int) *model {
	m := &model{tok: nn.NewEmbedding(vocab, dim), norm: nn.NewRMSNorm(dim),
		head: nn.NewLinearNoBias(dim, vocab), dim: dim}
	// nn.NewEmbedding draws from N(0,1), which is far too wide for a
	// residual stream that a dozen layers add into: the first block
	// would see inputs an order of magnitude larger than anything it
	// produces. 0.02 is what GPT-2 and Gemma use.
	tensor.NoGrad(func() { m.tok.W.MulScalarInPlace(0.02) })
	for range layers {
		m.blocks = append(m.blocks, newBlock(dim, heads, ctx))
	}
	return m
}

// Forward maps B*T token ids to B*T rows of vocabulary scores. There is
// no position input: the rotation inside attention supplies it.
func (m *model) Forward(ids []int, B, T int) *tensor.Tensor {
	x := m.tok.Lookup(ids).Reshape(B, T, m.dim)
	for _, b := range m.blocks {
		x = b.Forward(x)
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

// sample draws one symbol from the scores. Dividing by the temperature
// before the softmax sharpens the distribution below 1 and flattens it
// above: at 0 it is the most likely symbol every time, which loops; at 2
// it is nearly noise.
func sample(logits []float32, temp float32, r *rand.Rand) int {
	best := float32(math.Inf(-1))
	for _, v := range logits {
		best = max(best, v)
	}
	sum := float32(0)
	p := make([]float32, len(logits))
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

// generate continues from a one-character prompt, feeding each new
// symbol back in. Every step re-runs the whole prefix, because nothing
// keeps the keys and values of the earlier tokens — that is what a
// KV cache is for, and it is chapter 14.
func generate(m *model, symbols []byte, index [256]int, start byte, n, ctx int, temp float32, r *rand.Rand) string {
	ids := []int{index[start]}
	out := make([]byte, 0, n)
	for range n {
		in := ids
		if len(in) > ctx {
			in = in[len(in)-ctx:]
		}
		var logits []float32
		tensor.NoGrad(func() {
			rows := m.Forward(in, 1, len(in))
			logits = rows.Rows([]int{len(in) - 1}).Float32s()
		})
		next := sample(logits, temp, r)
		ids = append(ids, next)
		out = append(out, symbols[next])
	}
	return string(out)
}

func main() {
	var (
		dim    = flag.Int("dim", 256, "model width")
		heads  = flag.Int("heads", 4, "attention heads")
		layers = flag.Int("layers", 4, "transformer blocks")
		ctx    = flag.Int("ctx", 128, "context length in characters")
		batch  = flag.Int("batch", 32, "sequences per step")
		steps  = flag.Int("steps", 2000, "training steps")
		lr     = flag.Float64("lr", 3e-4, "AdamW learning rate")
		gen    = flag.Int("gen", 400, "characters to generate at the end")
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
	split := len(data) * 9 / 10
	train, val := data[:split], data[split:]
	fmt.Printf("corpus %d characters, vocabulary %d symbols, %d train / %d validation\n",
		len(text), len(symbols), len(train), len(val))

	tensor.Seed(*seed)
	r := rand.New(rand.NewPCG(*seed, 0))
	m := newModel(len(symbols), *dim, *heads, *layers, *ctx)
	params := 0
	for _, p := range m.Params() {
		params += p.Size()
	}
	fmt.Printf("model %d blocks, width %d, %d heads, context %d: %.2fM parameters\n\n",
		*layers, *dim, *heads, *ctx, float64(params)/1e6)

	// A batch is a set of random windows: the input is ctx characters and
	// the target is the same window shifted by one, so every position
	// predicts its successor and one pass trains on ctx examples at once.
	batchOf := func(src []int) (ids, targets []int) {
		ids = make([]int, 0, *batch**ctx)
		targets = make([]int, 0, *batch**ctx)
		for range *batch {
			s := r.IntN(len(src) - *ctx - 1)
			ids = append(ids, src[s:s+*ctx]...)
			targets = append(targets, src[s+1:s+*ctx+1]...)
		}
		return ids, targets
	}

	opt := optim.NewAdamW(m.Params(), float32(*lr), 0.01)
	start := time.Now()
	for step := 1; step <= *steps; step++ {
		ids, targets := batchOf(train)
		loss := tensor.CrossEntropy(m.Forward(ids, *batch, *ctx), targets)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step == 1 || step%250 == 0 {
			var vl float32
			tensor.NoGrad(func() {
				vi, vt := batchOf(val)
				vl = tensor.CrossEntropy(m.Forward(vi, *batch, *ctx), vt).Item()
			})
			fmt.Printf("  step %4d  train %.3f  validation %.3f  %.0fs\n",
				step, loss.Item(), vl, time.Since(start).Seconds())
		}
	}
	took := time.Since(start)
	fmt.Printf("\n%d steps in %.0fs — %.0f ms/step, %.0f tokens/s\n",
		*steps, took.Seconds(), took.Seconds()*1000/float64(*steps),
		float64(*steps**batch**ctx)/took.Seconds())

	genStart := time.Now()
	out := generate(m, symbols, index, '\n', *gen, *ctx, float32(*temp), r)
	fmt.Printf("\nsample at temperature %.1f:\n%s\n", *temp, out)
	fmt.Printf("\n%d characters in %.1fs — %.0f characters/s, re-running the whole prefix each time\n",
		*gen, time.Since(genStart).Seconds(), float64(*gen)/time.Since(genStart).Seconds())
}
