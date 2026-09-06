package gemma

import (
	"fmt"
	"math"
	"strings"

	"github.com/fiber/ai/tensor"
)

// Embed turns texts into sentence embeddings. It tokenises with the given
// prompt, batches inputs of similar length under a token budget, runs each
// batch through the encoder and head under NoGrad, and restores the input
// order. Each row is one embedding.
func (m *Model) Embed(texts []string, opts ...EmbedOption) ([][]float32, error) {
	var o embedOpts
	for _, fn := range opts {
		fn(&o)
	}
	prefix, err := m.promptPrefix(o.prompt)
	if err != nil {
		return nil, err
	}
	dim := m.Dim()
	if o.dim != 0 {
		if o.dim < 1 || o.dim > dim {
			return nil, fmt.Errorf("gemma: Dim(%d) outside 1..%d", o.dim, dim)
		}
		dim = o.dim
	}

	// Tokenise everything first so batches can be formed by length.
	ids := make([][]int, len(texts))
	lengths := make([]int, len(texts))
	for i, t := range texts {
		seq := m.tok.Encode(prefix + t)
		if len(seq) > m.maxTokens {
			seq = seq[:m.maxTokens]
		}
		ids[i], lengths[i] = seq, len(seq)
	}
	order, inverse := sortIndex(lengths)

	out := make([][]float32, len(texts))
	var batch []int
	budget := m.batchTokens
	flush := func() error {
		if len(batch) == 0 {
			return nil
		}
		emb := m.embedBatch(ids, batch, dim)
		for bi, idx := range batch {
			out[idx] = emb[bi]
		}
		batch = batch[:0]
		return nil
	}
	maxLen := 0
	for _, idx := range order {
		l := lengths[idx]
		newMax := max(maxLen, l)
		if len(batch) > 0 && newMax*(len(batch)+1) > budget {
			if err := flush(); err != nil {
				return nil, err
			}
			maxLen = 0
			newMax = l
		}
		batch = append(batch, idx)
		maxLen = newMax
	}
	if err := flush(); err != nil {
		return nil, err
	}
	_ = inverse
	return out, nil
}

func (m *Model) promptPrefix(spec string) (string, error) {
	switch {
	case spec == "":
		return "", nil
	case strings.HasPrefix(spec, "text:"):
		return strings.TrimPrefix(spec, "text:"), nil
	case strings.HasPrefix(spec, "name:"):
		name := strings.TrimPrefix(spec, "name:")
		p, ok := m.prompts[name]
		if !ok {
			return "", fmt.Errorf("gemma: no prompt named %q", name)
		}
		return p, nil
	}
	return "", nil
}

// embedBatch runs the encoder and head for the sequences named by idx and
// returns one dim-length embedding per sequence, in idx order.
func (m *Model) embedBatch(ids [][]int, idx []int, dim int) [][]float32 {
	var result [][]float32
	tensor.NoGrad(func() {
		B := len(idx)
		T := 0
		for _, i := range idx {
			T = max(T, len(ids[i]))
		}
		lengths := make([]int, B)
		flat := make([]int, B*T)
		for b, i := range idx {
			seq := ids[i]
			lengths[b] = len(seq)
			for t := 0; t < T; t++ {
				if t < len(seq) {
					flat[b*T+t] = seq[t]
				} else {
					flat[b*T+t] = 0 // pad id
				}
			}
		}
		h := m.encode(flat, lengths, B, T) // [B, T, hidden]
		pooled := m.pool(h, lengths, B, T) // [B, hidden]
		h.Release()
		emb := m.applyHead(pooled, dim) // [B, dim]
		data := emb.Float32s()
		emb.Release()
		result = make([][]float32, B)
		for b := 0; b < B; b++ {
			result[b] = append([]float32(nil), data[b*dim:(b+1)*dim]...)
		}
	})
	return result
}

// encode runs the transformer stack and returns the final hidden states.
func (m *Model) encode(flat, lengths []int, B, T int) *tensor.Tensor {
	cfg := m.cfg
	H, Hkv, D := cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim

	positions := make([]int, T)
	for t := range positions {
		positions[t] = t
	}
	pad := tensor.PaddingMask(lengths, T) // [B,1,1,T]
	var window *tensor.Tensor
	if cfg.SlidingWindow > 0 && T > cfg.SlidingWindow {
		window = tensor.WindowMask(T, cfg.SlidingWindow).Reshape(1, 1, T, T)
	}

	eps := float32(cfg.RMSNormEps)
	emb := m.embed.Rows(flat) // root [B*T, hidden]
	x := emb.Reshape(B, T, cfg.HiddenSize).MulScalar(float32(math.Sqrt(float64(cfg.HiddenSize))))
	rec(emb)

	for l := range m.layers {
		ly := &m.layers[l]
		mask := pad
		var maskTmp *tensor.Tensor
		if window != nil && cfg.isSliding(l) {
			maskTmp = pad.Add(window)
			mask = maskTmp
		}

		// Attention. rel() frees each intermediate's off-heap storage as
		// soon as its last reader is done, so the next layer reuses the
		// buffers instead of faulting fresh pages (66% of the time before).
		h := tensor.RMSNorm(x, ly.inputNorm, eps)
		q0 := h.MatMul(ly.wq)
		k0 := h.MatMul(ly.wk)
		v0 := h.MatMul(ly.wv)
		rec(h)
		q := tensor.RMSNorm(q0.Reshape(B, T, H, D), ly.qNorm, eps)
		k := tensor.RMSNorm(k0.Reshape(B, T, Hkv, D), ly.kNorm, eps)
		rec(q0, k0)
		qc := q.Permute(0, 2, 1, 3).Contiguous()
		kc := k.Permute(0, 2, 1, 3).Contiguous()
		vc := v0.Reshape(B, T, Hkv, D).Permute(0, 2, 1, 3).Contiguous()
		rec(q)
		rel(k, v0)
		qr := tensor.RoPE(qc, cfg.ropeBase(l), positions) // [B,H,T,D]
		kr := tensor.RoPE(kc, cfg.ropeBase(l), positions) // [B,Hkv,T,D]
		rec(qc, kc)
		att := tensor.AttentionScaled(qr, kr.Expand(B, H, T, D), vc.Expand(B, H, T, D), mask, m.scale)
		rec(qr, kr, vc)
		rel(maskTmp)
		attc := att.Permute(0, 2, 1, 3).Contiguous()
		rec(att)
		o0 := attc.Reshape(B, T, H*D).MatMul(ly.wo)
		rec(attc)
		o := tensor.RMSNorm(o0, ly.postAttnNorm, eps)
		rec(o0)
		x2 := x.Add(o)
		rec(x, o)
		x = x2

		// Feed-forward: down(gelu(gate(x)) * up(x)).
		h2 := tensor.RMSNorm(x, ly.preFFNNorm, eps)
		gate := h2.MatMul(ly.wgate).GELU()
		up := h2.MatMul(ly.wup)
		rec(h2)
		gu := gate.Mul(up)
		rec(gate, up)
		mlp0 := gu.MatMul(ly.wdown)
		rec(gu)
		mlp := tensor.RMSNorm(mlp0, ly.postFFNNorm, eps)
		rec(mlp0)
		x3 := x.Add(mlp)
		rec(x, mlp)
		x = x3
	}
	out := tensor.RMSNorm(x, m.norm, eps)
	rec(x)
	return out
}

// pool averages the token states of each sequence over its non-padding
// positions.
func (m *Model) pool(h *tensor.Tensor, lengths []int, B, T int) *tensor.Tensor {
	hidden := m.cfg.HiddenSize
	data := h.Data()
	out := make([]float32, B*hidden)
	for b := 0; b < B; b++ {
		n := lengths[b]
		if n == 0 {
			n = 1
		}
		base := b * T * hidden
		dst := out[b*hidden : (b+1)*hidden]
		for t := 0; t < n; t++ {
			row := data[base+t*hidden : base+(t+1)*hidden]
			for i, v := range row {
				dst[i] += v
			}
		}
		inv := 1 / float32(n)
		for i := range dst {
			dst[i] *= inv
		}
	}
	return tensor.New(out, B, hidden)
}

// applyHead runs the Dense layers, optional L2 normalisation and Matryoshka
// truncation, returning [B, dim].
func (m *Model) applyHead(x *tensor.Tensor, dim int) *tensor.Tensor {
	for _, d := range m.dense {
		x = x.MatMul(d.w)
		if d.tanh {
			x = x.Tanh()
		}
	}
	full := x.Shape()[1]
	if dim < full {
		x = x.Narrow(1, 0, dim).Contiguous()
	}
	if m.l2 {
		x = l2normalize(x)
	}
	return x
}

// rel releases each tensor's off-heap storage, ignoring the ones that
// cannot be released (views and aliases): under NoGrad every tensor here is
// a fresh contiguous result or a view of one, and releasing the roots as
// soon as they are consumed keeps the mapped free list full so later
// allocations reuse buffers instead of faulting new pages.
func rel(ts ...*tensor.Tensor) {
	for _, t := range ts {
		if t == nil {
			continue
		}
		func() {
			defer func() { recover() }()
			t.Release()
		}()
	}
}

// rec recycles roots that no live view aliases, forcing their storage back
// to the free list even though an already-consumed view marked them shared.
func rec(ts ...*tensor.Tensor) {
	for _, t := range ts {
		if t == nil {
			continue
		}
		func() {
			defer func() { recover() }()
			t.Recycle()
		}()
	}
}

// l2normalize scales each row to unit length.
func l2normalize(x *tensor.Tensor) *tensor.Tensor {
	norm := x.Square().Sum(-1).AddScalar(1e-12).Sqrt().Unsqueeze(-1)
	return x.Div(norm)
}
