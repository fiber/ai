// Tutorial chapter 17: tokens. A byte-pair encoding trained on the
// Shakespeare corpus in a few dozen lines, and chapter 15's decoder
// trained twice, on characters and on the learned pieces, so that the
// trade between vocabulary size and sequence length can be measured
// on the one scale that allows a comparison: bits per character.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/fiber/ai-data/shakespeare"
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
	"github.com/fiber/ai/tokenizer"
)

// ---------------------------------------------------------------------
// Byte-pair encoding
// ---------------------------------------------------------------------

// bpe is a vocabulary of pieces and the ordered list of merges that
// built it. Piece 0..base-1 are the single bytes of the corpus; every
// later piece is the concatenation of an earlier pair.
type bpe struct {
	pieces []string
	rank   map[[2]int]int // pair -> position in the merge order
	merged map[[2]int]int // pair -> id of the piece it becomes
	cache  map[string][]int
}

// chunks splits text the way every production tokenizer does before
// merging: a run of letters keeps the single space in front of it, and
// every other byte stands alone. Merges never cross a chunk boundary,
// so a piece is at most one word with its leading space and the
// vocabulary cannot fill up with fragments like "e, or".
func chunks(text string) []string {
	var out []string
	isLetter := func(c byte) bool { return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' }
	for i := 0; i < len(text); {
		start := i
		if text[i] == ' ' && i+1 < len(text) && isLetter(text[i+1]) {
			i++
		}
		if isLetter(text[i]) {
			for i < len(text) && isLetter(text[i]) {
				i++
			}
		} else {
			i++
		}
		out = append(out, text[start:i])
	}
	return out
}

// trainBPE learns merges from the text. At every count in report it
// calls fn with the number of merges so far and the number of tokens
// the corpus then encodes to.
func trainBPE(text string, symbols []byte, index [256]int, merges int, report []int, fn func(done, tokens int, last string)) *bpe {
	b := &bpe{rank: map[[2]int]int{}, merged: map[[2]int]int{}, cache: map[string][]int{}}
	for _, s := range symbols {
		b.pieces = append(b.pieces, string(s))
	}

	// The corpus as distinct chunks with their frequencies: 1.1 MB of
	// text is a few thousand different words, and a merge only needs to
	// know how often a pair occurs, not where.
	type word struct {
		syms []int
		freq int
	}
	freq := map[string]int{}
	for _, c := range chunks(text) {
		freq[c]++
	}
	var words []word
	for c, f := range freq {
		words = append(words, word{shakespeare.Encode(c, index), f})
	}
	sort.Slice(words, func(i, j int) bool { return words[i].freq > words[j].freq })

	tokens := func() int {
		n := 0
		for _, w := range words {
			n += len(w.syms) * w.freq
		}
		return n
	}
	next := 0
	if len(report) > 0 && report[0] == 0 {
		fn(0, tokens(), "")
		next = 1
	}

	counts := map[[2]int]int{}
	for done := 1; done <= merges; done++ {
		clear(counts)
		for _, w := range words {
			for i := 0; i+1 < len(w.syms); i++ {
				counts[[2]int{w.syms[i], w.syms[i+1]}] += w.freq
			}
		}
		var best [2]int
		bestN := 0
		for p, n := range counts {
			// Ties broken by the pair's ids so that the result does not
			// depend on map iteration order.
			if n > bestN || n == bestN && (p[0] < best[0] || p[0] == best[0] && p[1] < best[1]) {
				best, bestN = p, n
			}
		}
		if bestN < 2 {
			break
		}
		id := len(b.pieces)
		b.pieces = append(b.pieces, b.pieces[best[0]]+b.pieces[best[1]])
		b.rank[best] = done
		b.merged[best] = id
		for wi := range words {
			words[wi].syms = replacePair(words[wi].syms, best, id)
		}
		if next < len(report) && report[next] == done {
			fn(done, tokens(), b.pieces[id])
			next++
		}
	}
	return b
}

// replacePair rewrites every occurrence of the pair in place.
func replacePair(syms []int, pair [2]int, id int) []int {
	out := syms[:0]
	for i := 0; i < len(syms); i++ {
		if i+1 < len(syms) && syms[i] == pair[0] && syms[i+1] == pair[1] {
			out = append(out, id)
			i++
		} else {
			out = append(out, syms[i])
		}
	}
	return out
}

// encodeChunk applies the merges to one chunk in the order they were
// learned: always the lowest-ranked pair present, until none is left.
// This is what makes the encoding deterministic and what makes the
// encoder and the trainer agree.
func (b *bpe) encodeChunk(c string, index [256]int) []int {
	if ids, ok := b.cache[c]; ok {
		return ids
	}
	syms := shakespeare.Encode(c, index)
	for {
		bestRank, bestAt := math.MaxInt, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := b.rank[[2]int{syms[i], syms[i+1]}]; ok && r < bestRank {
				bestRank, bestAt = r, i
			}
		}
		if bestAt < 0 {
			break
		}
		p := [2]int{syms[bestAt], syms[bestAt+1]}
		syms = append(syms[:bestAt], append([]int{b.merged[p]}, syms[bestAt+2:]...)...)
	}
	b.cache[c] = syms
	return syms
}

func (b *bpe) Encode(text string, index [256]int) []int {
	var ids []int
	for _, c := range chunks(text) {
		ids = append(ids, b.encodeChunk(c, index)...)
	}
	return ids
}

func (b *bpe) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		sb.WriteString(b.pieces[id])
	}
	return sb.String()
}

// ---------------------------------------------------------------------
// The decoder of chapter 15, unchanged
// ---------------------------------------------------------------------

const ropeBase = 1e4

type block struct {
	norm1 *nn.RMSNorm
	attn  *nn.MultiHeadAttention
	norm2 *nn.RMSNorm
	up    *nn.Linear
	down  *nn.Linear
}

func newBlock(dim, heads, ctx int) *block {
	a := nn.NewMultiHeadAttention(dim, heads)
	a.Mask = tensor.CausalMask(ctx)
	a.RoPEBase = ropeBase
	return &block{
		norm1: nn.NewRMSNorm(dim), attn: a,
		norm2: nn.NewRMSNorm(dim),
		up:    nn.NewLinear(dim, 4*dim), down: nn.NewLinear(4*dim, dim),
	}
}

func (b *block) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = x.Add(b.attn.Forward(b.norm1.Forward(x)))
	B, T, D := x.Dim(0), x.Dim(1), x.Dim(2)
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
	tensor.NoGrad(func() { m.tok.W.MulScalarInPlace(0.02) })
	for range layers {
		m.blocks = append(m.blocks, newBlock(dim, heads, ctx))
	}
	return m
}

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

// ---------------------------------------------------------------------
// Training and measuring on either vocabulary
// ---------------------------------------------------------------------

// vocab is what the model needs to know about its symbols: how to print
// one, and how many characters it stands for.
type vocab struct {
	name   string
	pieces []string
	train  []int
	val    []int
}

type result struct {
	msPerStep   float64
	valLoss     float64 // nats per token, the number every framework prints
	bitsPerChar float64 // the number that can be compared
	charsPerSec float64
	sample      string
}

func run(v vocab, dim, heads, layers, ctx, batch, steps int, lr float64, gen int, temp float32, seed uint64) result {
	tensor.Seed(seed)
	r := rand.New(rand.NewPCG(seed, 0))
	m := newModel(len(v.pieces), dim, heads, layers, ctx)

	batchOf := func(src []int) (ids, targets []int) {
		for range batch {
			s := r.IntN(len(src) - ctx - 1)
			ids = append(ids, src[s:s+ctx]...)
			targets = append(targets, src[s+1:s+ctx+1]...)
		}
		return ids, targets
	}

	opt := optim.NewAdamW(m.Params(), float32(lr), 0.01)
	start := time.Now()
	for step := 1; step <= steps; step++ {
		ids, targets := batchOf(v.train)
		loss := tensor.CrossEntropy(m.Forward(ids, batch, ctx), targets)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		if step == 1 || step%250 == 0 {
			fmt.Printf("  step %4d  train %.3f  %.0fs\n", step, loss.Item(), time.Since(start).Seconds())
		}
	}
	res := result{msPerStep: time.Since(start).Seconds() * 1000 / float64(steps)}

	// The held-out text, in consecutive windows, every one scored: the
	// cross-entropy is a mean per token, so it is scaled back up to a
	// total in nats and then divided by the characters those tokens
	// spell. A character model has one character per token and the two
	// numbers coincide; a piece model does not.
	tensor.NoGrad(func() {
		nats, tokens, chars := 0.0, 0, 0
		var ids, targets []int
		flush := func() {
			if len(ids) == 0 {
				return
			}
			B := len(ids) / ctx
			nats += float64(tensor.CrossEntropy(m.Forward(ids, B, ctx), targets).Item()) * float64(len(targets))
			for _, t := range targets {
				chars += len(v.pieces[t])
			}
			tokens += len(targets)
			ids, targets = ids[:0], targets[:0]
		}
		for s := 0; s+ctx+1 <= len(v.val); s += ctx {
			ids = append(ids, v.val[s:s+ctx]...)
			targets = append(targets, v.val[s+1:s+ctx+1]...)
			if len(ids) == batch*ctx {
				flush()
			}
		}
		flush()
		res.valLoss = nats / float64(tokens)
		res.bitsPerChar = nats / math.Ln2 / float64(chars)
	})

	// Generate until gen characters are out, whole prefix each step as
	// in chapter 15, so the two models pay the same per step and the
	// difference is how many characters a step buys.
	gr := rand.New(rand.NewPCG(seed, 99))
	ids := []int{v.train[0]}
	var out strings.Builder
	genStart := time.Now()
	for out.Len() < gen {
		in := ids
		if len(in) > ctx {
			in = in[len(in)-ctx:]
		}
		var logits []float32
		tensor.NoGrad(func() {
			logits = m.Forward(in, 1, len(in)).Rows([]int{len(in) - 1}).Float32s()
		})
		next := sample(logits, temp, gr)
		ids = append(ids, next)
		out.WriteString(v.pieces[next])
	}
	res.charsPerSec = float64(out.Len()) / time.Since(genStart).Seconds()
	res.sample = out.String()
	return res
}

func main() {
	var (
		merges = flag.Int("merges", 1024, "byte-pair merges to learn")
		dim    = flag.Int("dim", 256, "model width")
		heads  = flag.Int("heads", 4, "attention heads")
		layers = flag.Int("layers", 4, "transformer blocks")
		ctx    = flag.Int("ctx", 128, "context length in tokens")
		batch  = flag.Int("batch", 32, "sequences per step")
		steps  = flag.Int("steps", 1000, "training steps per model")
		lr     = flag.Float64("lr", 3e-4, "AdamW learning rate")
		gen    = flag.Int("gen", 400, "characters to generate from each model")
		temp   = flag.Float64("temp", 0.8, "sampling temperature")
		seed   = flag.Uint64("seed", 1, "random seed")
		gemma  = flag.String("gemma", "", "path to the embeddinggemma-300m directory (default $FIBERAI_MODELS/embeddinggemma-300m)")
	)
	flag.Parse()

	text, err := shakespeare.Text()
	if err != nil {
		log.Fatal(err)
	}
	symbols, index := shakespeare.Vocabulary(text)

	// Three views of one sentence.
	sentence := "To be, or not to be, that is the question:"
	fmt.Printf("%q\n", sentence)
	fmt.Printf("  %3d characters  %v\n", len(sentence), shakespeare.Encode(sentence, index)[:12])
	fmt.Printf("  %3d words       %q\n", len(strings.Fields(sentence)), strings.Fields(sentence)[:4])
	if *gemma == "" {
		if root := os.Getenv("FIBERAI_MODELS"); root != "" {
			*gemma = filepath.Join(root, "embeddinggemma-300m")
		}
	}
	var gt *tokenizer.Tokenizer
	if *gemma != "" {
		if gt, err = tokenizer.Load(filepath.Join(*gemma, "tokenizer.json")); err != nil {
			fmt.Printf("  (Gemma tokenizer not loaded: %v)\n", err)
			gt = nil
		}
	}
	if gt != nil {
		ids := gt.EncodeRaw(sentence)
		var ps []string
		for _, id := range ids {
			ps = append(ps, gt.Piece(id))
		}
		fmt.Printf("  %3d Gemma pieces %q  (vocabulary %d)\n", len(ids), ps, gt.VocabSize())
	} else {
		fmt.Println("  (set FIBERAI_MODELS to see the same sentence in Gemma's pieces)")
	}

	// Train the BPE and watch the corpus shrink.
	fmt.Printf("\nlearning %d merges on %d characters\n", *merges, len(text))
	fmt.Println("  merges   vocabulary    corpus tokens   chars/token   last piece")
	start := time.Now()
	report := []int{0, 1, 2, 3, 4, 8, 16, 64, 256, 512, 1024, 2048, 4096}
	b := trainBPE(text, symbols, index, *merges, report, func(done, tokens int, last string) {
		fmt.Printf("  %6d   %10d   %13d   %11.2f   %q\n", done, len(symbols)+done, tokens,
			float64(len(text))/float64(tokens), last)
	})
	fmt.Printf("  learned in %.1fs\n", time.Since(start).Seconds())
	if gt != nil {
		n := len(gt.EncodeRaw(text))
		fmt.Printf("  Gemma's %d pieces, trained on other text: %d tokens, %.2f chars/token\n",
			gt.VocabSize(), n, float64(len(text))/float64(n))
	}

	// The same text on both vocabularies, split at a chunk boundary so
	// that the two models are validated on exactly the same characters.
	cut := len(text) * 9 / 10
	for cut < len(text) && text[cut] != '\n' {
		cut++
	}
	head, tail := text[:cut], text[cut:]
	pieceIDs := b.Encode(text, index)
	if b.Decode(pieceIDs) != text {
		log.Fatal("encode/decode does not round-trip")
	}
	var pieces []string
	for _, s := range symbols {
		pieces = append(pieces, string(s))
	}
	vocabs := []vocab{
		{"characters", pieces, shakespeare.Encode(head, index), shakespeare.Encode(tail, index)},
		{fmt.Sprintf("%d pieces", len(b.pieces)), b.pieces, b.Encode(head, index), b.Encode(tail, index)},
	}
	fmt.Printf("\nvalidation text: %d characters = %d character tokens = %d piece tokens\n",
		len(tail), len(vocabs[0].val), len(vocabs[1].val))

	var results []result
	for _, v := range vocabs {
		fmt.Printf("\ntraining on %s: %d steps of %d x %d tokens\n", v.name, *steps, *batch, *ctx)
		results = append(results, run(v, *dim, *heads, *layers, *ctx, *batch, *steps, *lr, *gen, float32(*temp), *seed))
	}

	fmt.Printf("\n%-18s %10s %14s %12s %12s\n", "", "ms/step", "loss/token", "bits/char", "chars/s")
	for i, v := range vocabs {
		r := results[i]
		fmt.Printf("%-18s %10.0f %14.3f %12.3f %12.0f\n", v.name, r.msPerStep, r.valLoss, r.bitsPerChar, r.charsPerSec)
	}
	for i, v := range vocabs {
		fmt.Printf("\nsample from the %s model:\n%s\n", v.name, results[i].sample)
	}
}
