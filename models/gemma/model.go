package gemma

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/fiber/ai/safetensors"
	"github.com/fiber/ai/tensor"
	"github.com/fiber/ai/tokenizer"
)

// layer holds one transformer block's parameters, already transposed for
// x·W and with the RMSNorm (1+w) offset folded in.
type layer struct {
	inputNorm, postAttnNorm, preFFNNorm, postFFNNorm *tensor.Tensor
	qNorm, kNorm                                     *tensor.Tensor
	wq, wk, wv, wo                                   *tensor.Tensor
	wgate, wup, wdown                                *tensor.Tensor
}

// dense is a bias-free linear layer of the sentence-transformers head.
type dense struct {
	w    *tensor.Tensor // [in, out]
	tanh bool
}

// Model is a loaded Gemma text encoder plus its sentence-embedding head.
type Model struct {
	cfg    Config
	tok    *tokenizer.Tokenizer
	embed  *tensor.Tensor // [vocab, hidden], also released with the model
	layers []layer
	norm   *tensor.Tensor // final RMSNorm

	poolMean    bool
	includeProm bool
	dense       []dense
	l2          bool

	prompts     map[string]string
	maxTokens   int
	batchTokens int
	scale       float32
}

// Load reads a model directory in the Hugging Face / sentence-transformers
// layout: config.json, tokenizer.json, the weight safetensors, and the
// pooling and Dense module configs.
func Load(dir string, opts ...Option) (*Model, error) {
	cfg, err := LoadConfig(dir)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	f, err := safetensors.OpenDir(dir)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	m := &Model{
		cfg:         cfg,
		tok:         tok,
		poolMean:    true,
		includeProm: true,
		l2:          true,
		maxTokens:   cfg.MaxPositionEmbeddings,
		batchTokens: 8192,
		scale:       float32(1 / math.Sqrt(cfg.QueryPreAttnScalar)),
	}
	if m.maxTokens == 0 || m.maxTokens > 2048 {
		m.maxTokens = 2048
	}

	get := func(name string) (*tensor.Tensor, error) {
		for _, n := range []string{name, "model." + name} {
			if _, ok := f.Info(n); ok {
				return f.Tensor(n)
			}
		}
		return nil, fmt.Errorf("gemma: weight %q not found", name)
	}
	// transpose loads a [out, in] weight as [in, out] for x·W.
	transpose := func(name string) (*tensor.Tensor, error) {
		w, err := get(name)
		if err != nil {
			return nil, err
		}
		return w.Transpose(0, 1).Contiguous(), nil
	}
	// norm loads an RMSNorm weight with Gemma's (1 + w) offset folded in.
	norm := func(name string) (*tensor.Tensor, error) {
		w, err := get(name)
		if err != nil {
			return nil, err
		}
		return w.AddScalar(1), nil
	}

	if m.embed, err = get("embed_tokens.weight"); err != nil {
		return nil, err
	}
	m.layers = make([]layer, cfg.NumHiddenLayers)
	for i := range m.layers {
		p := fmt.Sprintf("layers.%d.", i)
		set := []struct {
			dst **tensor.Tensor
			fn  func(string) (*tensor.Tensor, error)
			key string
		}{
			{&m.layers[i].inputNorm, norm, p + "input_layernorm.weight"},
			{&m.layers[i].postAttnNorm, norm, p + "post_attention_layernorm.weight"},
			{&m.layers[i].preFFNNorm, norm, p + "pre_feedforward_layernorm.weight"},
			{&m.layers[i].postFFNNorm, norm, p + "post_feedforward_layernorm.weight"},
			{&m.layers[i].qNorm, norm, p + "self_attn.q_norm.weight"},
			{&m.layers[i].kNorm, norm, p + "self_attn.k_norm.weight"},
			{&m.layers[i].wq, transpose, p + "self_attn.q_proj.weight"},
			{&m.layers[i].wk, transpose, p + "self_attn.k_proj.weight"},
			{&m.layers[i].wv, transpose, p + "self_attn.v_proj.weight"},
			{&m.layers[i].wo, transpose, p + "self_attn.o_proj.weight"},
			{&m.layers[i].wgate, transpose, p + "mlp.gate_proj.weight"},
			{&m.layers[i].wup, transpose, p + "mlp.up_proj.weight"},
			{&m.layers[i].wdown, transpose, p + "mlp.down_proj.weight"},
		}
		for _, s := range set {
			if *s.dst, err = s.fn(s.key); err != nil {
				return nil, err
			}
		}
	}
	if m.norm, err = norm("norm.weight"); err != nil {
		return nil, err
	}
	// Fold the pre-norms into the weights they feed: RMSNorm(x)·W equals
	// (x/rms(x))·(diag(g)·W), so the encoder scales rows of the product
	// instead of normalising the input in a separate pass. W is [in, out];
	// scaling its rows by g is diag(g)·W.
	fold := func(w, g *tensor.Tensor) *tensor.Tensor {
		out := w.Mul(g.Reshape(g.Size(), 1))
		w.Release()
		return out
	}
	for i := range m.layers {
		ly := &m.layers[i]
		ly.wq = fold(ly.wq, ly.inputNorm)
		ly.wk = fold(ly.wk, ly.inputNorm)
		ly.wv = fold(ly.wv, ly.inputNorm)
		ly.wgate = fold(ly.wgate, ly.preFFNNorm)
		ly.wup = fold(ly.wup, ly.preFFNNorm)
	}

	if err := m.loadHead(dir); err != nil {
		return nil, err
	}
	m.loadSentenceConfig(dir)
	for _, o := range opts {
		o(m)
	}
	return m, nil
}

// loadHead reads the pooling mode and the Dense layers of the
// sentence-transformers head from modules.json and the per-module configs.
func (m *Model) loadHead(dir string) error {
	modules := []struct {
		Type string `json:"type"`
		Path string `json:"path"`
	}{}
	if b, err := os.ReadFile(filepath.Join(dir, "modules.json")); err == nil {
		if err := json.Unmarshal(b, &modules); err != nil {
			return fmt.Errorf("gemma: modules.json: %w", err)
		}
	}
	for _, mod := range modules {
		switch {
		case mod.Type == "sentence_transformers.models.Pooling":
			var pc struct {
				Mean          bool `json:"pooling_mode_mean_tokens"`
				CLS           bool `json:"pooling_mode_cls_token"`
				Last          bool `json:"pooling_mode_lasttoken"`
				IncludePrompt bool `json:"include_prompt"`
			}
			readJSON(filepath.Join(dir, mod.Path, "config.json"), &pc)
			m.poolMean, m.includeProm = pc.Mean, pc.IncludePrompt
			if pc.CLS || pc.Last {
				return fmt.Errorf("gemma: only mean pooling is supported")
			}
		case mod.Type == "sentence_transformers.models.Dense":
			var dc struct {
				In         int    `json:"in_features"`
				Out        int    `json:"out_features"`
				Bias       bool   `json:"bias"`
				Activation string `json:"activation_function"`
			}
			if err := readJSON(filepath.Join(dir, mod.Path, "config.json"), &dc); err != nil {
				return err
			}
			f, err := safetensors.OpenDir(filepath.Join(dir, mod.Path))
			if err != nil {
				return err
			}
			w, err := f.Tensor("linear.weight")
			f.Close()
			if err != nil {
				return err
			}
			d := dense{w: w.Transpose(0, 1).Contiguous()}
			switch dc.Activation {
			case "torch.nn.modules.linear.Identity", "":
			case "torch.nn.modules.activation.Tanh":
				d.tanh = true
			default:
				return fmt.Errorf("gemma: Dense activation %q not supported", dc.Activation)
			}
			m.dense = append(m.dense, d)
		case mod.Type == "sentence_transformers.models.Normalize":
			m.l2 = true
		}
	}
	return nil
}

func (m *Model) loadSentenceConfig(dir string) {
	var sc struct {
		Prompts map[string]string `json:"prompts"`
	}
	readJSON(filepath.Join(dir, "config_sentence_transformers.json"), &sc)
	m.prompts = sc.Prompts
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// Config returns the model's hyper-parameters.
func (m *Model) Config() Config { return m.cfg }

// Tokenizer returns the model's tokenizer.
func (m *Model) Tokenizer() *tokenizer.Tokenizer { return m.tok }

// Dim returns the embedding dimension the model produces before any
// Matryoshka truncation.
func (m *Model) Dim() int {
	if len(m.dense) > 0 {
		return m.dense[len(m.dense)-1].outFeatures()
	}
	return m.cfg.HiddenSize
}

func (d dense) outFeatures() int { return d.w.Shape()[1] }

// Option configures a model at load time.
type Option func(*Model)

// Threads caps the cores a single forward pass may use.
func Threads(n int) Option { return func(m *Model) { tensor.SetThreads(n) } }

// MaxTokens truncates inputs to at most n tokens (default the model's
// context length, capped at 2048).
func MaxTokens(n int) Option { return func(m *Model) { m.maxTokens = n } }

// BatchTokens sets the padded-token budget per forward pass (default 8192).
func BatchTokens(n int) Option { return func(m *Model) { m.batchTokens = n } }

// EmbedOption configures a single Embed call.
type EmbedOption func(*embedOpts)

type embedOpts struct {
	prompt string
	dim    int
}

// Prompt prepends the named task prompt from
// config_sentence_transformers.json (for example "query", "document",
// "Retrieval-query", "Clustering").
func Prompt(name string) EmbedOption { return func(o *embedOpts) { o.prompt = "name:" + name } }

// PromptText prepends a literal prefix such as "task: search result | query: ".
func PromptText(s string) EmbedOption { return func(o *embedOpts) { o.prompt = "text:" + s } }

// Dim truncates each embedding to the first d dimensions and renormalises
// (Matryoshka representation). d must not exceed the model dimension.
func Dim(d int) EmbedOption { return func(o *embedOpts) { o.dim = d } }

// Close releases the model's weights.
func (m *Model) Close() {
	release := func(t *tensor.Tensor) {
		if t != nil {
			defer func() { recover() }()
			t.Release()
		}
	}
	release(m.embed)
	release(m.norm)
	for _, l := range m.layers {
		for _, t := range []*tensor.Tensor{l.inputNorm, l.postAttnNorm, l.preFFNNorm, l.postFFNNorm, l.qNorm, l.kNorm, l.wq, l.wk, l.wv, l.wo, l.wgate, l.wup, l.wdown} {
			release(t)
		}
	}
	for _, d := range m.dense {
		release(d.w)
	}
}

// sortIndex returns the indices that sort lengths ascending, and the inverse
// permutation to restore the original order.
func sortIndex(lengths []int) (order, inverse []int) {
	order = make([]int, len(lengths))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return lengths[order[a]] < lengths[order[b]] })
	inverse = make([]int, len(order))
	for rank, i := range order {
		inverse[i] = rank
	}
	return order, inverse
}
