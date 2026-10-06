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
		if err := m.checkTextOnly(seq); err != nil {
			return nil, fmt.Errorf("text %d: %w", i, err)
		}
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

// checkTextOnly refuses input that carries EmbeddingGemma 2's image,
// audio or video placeholder tokens. The reference model replaces those
// positions with the output of a vision or audio tower; this package runs
// neither, and embedding the placeholders as ordinary tokens would give a
// vector that looks valid and means something else.
func (m *Model) checkTextOnly(seq []int) error {
	for _, id := range seq {
		if name, ok := m.cfg.MultimodalTokens[id]; ok {
			return fmt.Errorf("gemma: input contains the %s placeholder token; EmbeddingGemma 2 image, audio and video input are not supported, only text", name)
		}
	}
	return nil
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
		var h *tensor.Tensor // [B, T, hidden]
		if m.cfg.Variant == 2 {
			h = m.encode2(flat, lengths, B, T)
		} else {
			h = m.encode(flat, lengths, B, T)
		}
		pooled := m.pool(h, lengths, B, T) // [B, hidden]
		h.Recycle()                        // pool read it through Data(): Release would refuse (B-005)
		emb := m.applyHead(pooled, dim)    // [B, dim]
		pooled.Recycle()
		data := emb.Float32s()
		emb.Recycle()
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
	if w := cfg.window(); cfg.SlidingWindow > 0 && T > w {
		window = tensor.WindowMask(T, w).Reshape(1, 1, T, T)
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
		// Pre-norm folded into the weights (see Load): one read pass for the
		// row statistics, the scale applied in each product's epilogue.
		xf := x.Reshape(B*T, cfg.HiddenSize)
		scale := rmsScale(xf.RowSumSquares(), cfg.HiddenSize, eps)
		q0 := tensor.MatMulFused(xf, ly.wq, tensor.Fused{RowScale: scale})
		k0 := tensor.MatMulFused(xf, ly.wk, tensor.Fused{RowScale: scale})
		v0 := tensor.MatMulFused(xf, ly.wv, tensor.Fused{RowScale: scale})
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

		// Feed-forward: down(gelu(gate(x)) * up(x)), with the pre-norm folded
		// and GELU and the gated product applied in the gate epilogue.
		xf = x.Reshape(B*T, cfg.HiddenSize)
		scale = rmsScale(xf.RowSumSquares(), cfg.HiddenSize, eps)
		up := tensor.MatMulFused(xf, ly.wup, tensor.Fused{RowScale: scale})
		gu := tensor.MatMulFused(xf, ly.wgate, tensor.Fused{RowScale: scale, Act: tensor.GELUAct, Mul: up})
		rec(up)
		mlp0 := gu.MatMul(ly.wdown).Reshape(B, T, cfg.HiddenSize)
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

// encode2 runs EmbeddingGemma 2's text tower. It follows encode, with the
// differences the architecture brings: per-layer head geometry, a weight-
// free RMSNorm on the values, attention scores not scaled, a third
// residual block per layer that mixes in that layer's slice of the
// per-layer embeddings, and a stored scalar on every layer's output.
func (m *Model) encode2(flat, lengths []int, B, T int) *tensor.Tensor {
	cfg := m.cfg
	H, hidden := cfg.NumAttentionHeads, cfg.HiddenSize
	L, P := cfg.NumHiddenLayers, cfg.HiddenSizePerLayerInput

	positions := make([]int, T)
	for t := range positions {
		positions[t] = t
	}
	pad := tensor.PaddingMask(lengths, T) // [B,1,1,T]
	var window *tensor.Tensor
	if w := cfg.window(); cfg.SlidingWindow > 0 && T > w {
		window = tensor.WindowMask(T, w).Reshape(1, 1, T, T)
	}

	eps := float32(cfg.RMSNormEps)
	emb := m.embed.Rows(flat) // root [B*T, hidden]
	x := emb.Reshape(B, T, hidden).MulScalar(float32(math.Sqrt(float64(hidden))))
	rec(emb)

	// Per-layer embeddings, once per pass: the scaled token embeddings
	// projected to one P-vector per layer per token, scaled by hidden^-½
	// and RMS-normalised. Layer l reads its slice in the PLE block below.
	pl := x.Reshape(B*T, hidden).MatMul(m.pleProj).MulScalar(float32(1 / math.Sqrt(float64(hidden))))
	ple := tensor.RMSNorm(pl.Reshape(B*T, L, P), m.pleNorm, eps) // [B*T, L, P]
	rec(pl)

	for l := range m.layers {
		ly := &m.layers[l]
		D, Hkv := cfg.headDim(l), cfg.kvHeads(l)
		mask := pad
		var maskTmp *tensor.Tensor
		if window != nil && cfg.isSliding(l) {
			maskTmp = pad.Add(window)
			mask = maskTmp
		}

		// Attention, pre-norm folded into the weights as in encode.
		xf := x.Reshape(B*T, hidden)
		scale := rmsScale(xf.RowSumSquares(), hidden, eps)
		q0 := tensor.MatMulFused(xf, ly.wq, tensor.Fused{RowScale: scale})
		k0 := tensor.MatMulFused(xf, ly.wk, tensor.Fused{RowScale: scale})
		v0 := tensor.MatMulFused(xf, ly.wv, tensor.Fused{RowScale: scale})
		q := tensor.RMSNorm(q0.Reshape(B, T, H, D), ly.qNorm, eps)
		k := tensor.RMSNorm(k0.Reshape(B, T, Hkv, D), ly.kNorm, eps)
		v := tensor.RMSNorm(v0.Reshape(B, T, Hkv, D), ly.vNorm, eps)
		rec(q0, k0, v0)
		qc := q.Permute(0, 2, 1, 3).Contiguous()
		kc := k.Permute(0, 2, 1, 3).Contiguous()
		vc := v.Permute(0, 2, 1, 3).Contiguous()
		// With one head, [B,T,1,D] permuted to [B,1,T,D] is already
		// contiguous and Contiguous returns a view of the same storage:
		// recycling the source then would free what attention still reads.
		// rel refuses in that case; rec is only safe after a real copy.
		freeSrc := func(t *tensor.Tensor, heads int) {
			if heads > 1 {
				rec(t)
			} else {
				rel(t)
			}
		}
		freeSrc(q, H)
		freeSrc(k, Hkv)
		freeSrc(v, Hkv)
		qr := tensor.RoPE(qc, cfg.ropeBase(l), positions) // [B,H,T,D]
		kr := tensor.RoPE(kc, cfg.ropeBase(l), positions) // [B,Hkv,T,D]
		rec(qc, kc)
		// Grouped-query attention with more than one key/value head: the
		// query heads are grouped as [B, Hkv, rep, T, D], so query head h
		// reads key/value head h/rep as the reference's repeat_kv does,
		// and keys and values broadcast over rep with stride 0. The mask
		// gains the same axis.
		rep := H / Hkv
		att := tensor.AttentionScaled(
			qr.Reshape(B, Hkv, rep, T, D),
			kr.Reshape(B, Hkv, 1, T, D).Expand(B, Hkv, rep, T, D),
			vc.Reshape(B, Hkv, 1, T, D).Expand(B, Hkv, rep, T, D),
			mask.Unsqueeze(1), m.scale)
		rec(qr, kr, vc)
		rel(maskTmp)
		attc := att.Reshape(B, H, T, D).Permute(0, 2, 1, 3).Contiguous()
		rec(att)
		o0 := attc.Reshape(B, T, H*D).MatMul(ly.wo)
		rec(attc)
		o := tensor.RMSNorm(o0, ly.postAttnNorm, eps)
		rec(o0)
		x2 := x.Add(o)
		rec(x, o)
		x = x2

		// Feed-forward, as in encode.
		xf = x.Reshape(B*T, hidden)
		scale = rmsScale(xf.RowSumSquares(), hidden, eps)
		up := tensor.MatMulFused(xf, ly.wup, tensor.Fused{RowScale: scale})
		gu := tensor.MatMulFused(xf, ly.wgate, tensor.Fused{RowScale: scale, Act: tensor.GELUAct, Mul: up})
		rec(up)
		mlp0 := gu.MatMul(ly.wdown).Reshape(B, T, hidden)
		rec(gu)
		mlp := tensor.RMSNorm(mlp0, ly.postFFNNorm, eps)
		rec(mlp0)
		x3 := x.Add(mlp)
		rec(x, mlp)
		x = x3

		// PLE block: x + norm(proj(gelu(gate(x)) ⊙ ple_l)), with GELU and
		// the gating product in the gate's epilogue. No norm before it.
		pleL := ple.Select(1, l).Contiguous() // [B*T, P]
		g := tensor.MatMulFused(x.Reshape(B*T, hidden), ly.pleGate, tensor.Fused{Act: tensor.GELUAct, Mul: pleL})
		rec(pleL)
		y0 := g.MatMul(ly.pleProj).Reshape(B, T, hidden)
		rec(g)
		y := tensor.RMSNorm(y0, ly.plePostNorm, eps)
		rec(y0)
		x4 := x.Add(y)
		rec(x, y)
		x = x4.MulScalar(ly.scalar)
		rec(x4)
	}
	rec(ple)
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
	// EmbeddingGemma 2 projects the hidden states to the embedding size
	// per token; the projection is linear, so applying it to the pooled
	// vector is the same result for a fraction of the work.
	if m.embProj != nil {
		x = x.MatMul(m.embProj)
	}
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

// rmsScale turns row sums of squares into the RMSNorm factors 1/√(ss/n+ε).
func rmsScale(ss []float32, n int, eps float32) []float32 {
	out := make([]float32, len(ss))
	for i, v := range ss {
		out[i] = float32(1 / math.Sqrt(float64(v/float32(n)+eps)))
	}
	return out
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
