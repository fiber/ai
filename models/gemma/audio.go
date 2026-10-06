package gemma

import (
	"fmt"
	"math"

	"github.com/fiber/ai/internal/parallel"
	"github.com/fiber/ai/safetensors"
	"github.com/fiber/ai/tensor"
)

// EmbeddingGemma 2's audio tower, a Gemma 4 conformer: log-mel frames are
// subsampled 4× by two strided convolutions, run through conformer layers
// with chunked local attention, projected, and taken into the text
// model's width, where they replace <|audio|> placeholders.

// clipped is a bias-free projection whose input and output are clamped to
// stored bounds (Gemma 4's "clippable linear").
type clipped struct {
	w                            *tensor.Tensor // [in, out]
	inMin, inMax, outMin, outMax float32
}

func (c clipped) apply(x *tensor.Tensor) *tensor.Tensor {
	xc := x.Clamp(c.inMin, c.inMax)
	y := xc.MatMul(c.w)
	rec(xc)
	out := y.Clamp(c.outMin, c.outMax)
	rec(y)
	return out
}

type audioFFN struct {
	preNorm, postNorm *tensor.Tensor
	l1, l2            clipped
}

type audioLayer struct {
	ffn1, ffn2                         audioFFN
	normPreAttn, normPostAttn, normOut *tensor.Tensor
	q, k, v, post                      clipped
	relK                               *tensor.Tensor // [hidden, hidden]
	qScale                             []float32      // per head dimension: d^-½/ln 2 · softplus(per_dim_scale)
	convPre, convNorm                  *tensor.Tensor
	convStart, convEnd                 clipped
	depthwise                          []float32 // [hidden][kernel]
}

type audioTower struct {
	cfg              AudioConfig
	mel              *melTables
	conv0, conv1     *tensor.Tensor // [out, in, 3, 3]
	norm0, norm1     *tensor.Tensor // LayerNorm weights (no bias)
	zero0, zero1     *tensor.Tensor // the absent LayerNorm biases
	inProj           *tensor.Tensor // [1024, hidden]
	layers           []audioLayer
	outProj, outBias *tensor.Tensor // [hidden, out], [out]
	embProj          *tensor.Tensor // [out, text hidden], after a weight-free RMSNorm
	posEmb           *tensor.Tensor // [left context + 1, hidden], distances left..0
}

// loadAudio reads the audio tower on first use, like the vision tower.
func (m *Model) loadAudio() error {
	m.audioOnce.Do(func() {
		ac := m.cfg.Audio
		if ac == nil {
			m.audioErr = fmt.Errorf("gemma: this checkpoint has no audio tower")
			return
		}
		f, err := safetensors.OpenDir(m.dir)
		if err != nil {
			m.audioErr = err
			return
		}
		defer f.Close()
		var firstErr error
		get := func(name string) *tensor.Tensor {
			if firstErr != nil {
				return nil
			}
			if _, ok := f.Info(name); !ok {
				firstErr = fmt.Errorf("gemma: weight %q not found", name)
				return nil
			}
			t, err := f.Tensor(name)
			if err != nil {
				firstErr = err
			}
			return t
		}
		tr := func(name string) *tensor.Tensor {
			w := get(name)
			if w == nil {
				return nil
			}
			return w.Transpose(0, 1).Contiguous()
		}
		scalar := func(name string) float32 {
			t := get(name)
			if t == nil {
				return 0
			}
			return t.Float32s()[0]
		}
		clip := func(p string) clipped {
			c := clipped{w: tr(p + ".linear.weight")}
			if ac.UseClippedLinears {
				c.inMin, c.inMax = scalar(p+".input_min"), scalar(p+".input_max")
				c.outMin, c.outMax = scalar(p+".output_min"), scalar(p+".output_max")
			} else {
				inf := float32(math.Inf(1))
				c.inMin, c.inMax, c.outMin, c.outMax = -inf, inf, -inf, inf
			}
			return c
		}
		a := &audioTower{cfg: *ac, mel: newMelTables()}
		const sp = "audio_tower.subsample_conv_projection."
		a.conv0, a.conv1 = get(sp+"layer0.conv.weight"), get(sp+"layer1.conv.weight")
		a.norm0, a.norm1 = get(sp+"layer0.norm.weight"), get(sp+"layer1.norm.weight")
		a.zero0, a.zero1 = tensor.Zeros(ac.SubsamplingConvChannels[0]), tensor.Zeros(ac.SubsamplingConvChannels[1])
		a.inProj = tr(sp + "input_proj_linear.weight")
		a.outProj, a.outBias = tr("audio_tower.output_proj.weight"), get("audio_tower.output_proj.bias")
		a.embProj = tr("embed_audio.embedding_projection.weight")
		hd := ac.HiddenSize / ac.NumAttentionHeads
		a.layers = make([]audioLayer, ac.NumHiddenLayers)
		for i := range a.layers {
			p := fmt.Sprintf("audio_tower.layers.%d.", i)
			ly := &a.layers[i]
			for _, ff := range []struct {
				dst  *audioFFN
				name string
			}{{&ly.ffn1, "feed_forward1"}, {&ly.ffn2, "feed_forward2"}} {
				ff.dst.preNorm = get(p + ff.name + ".pre_layer_norm.weight")
				ff.dst.postNorm = get(p + ff.name + ".post_layer_norm.weight")
				ff.dst.l1, ff.dst.l2 = clip(p+ff.name+".ffw_layer_1"), clip(p+ff.name+".ffw_layer_2")
			}
			ly.normPreAttn, ly.normPostAttn, ly.normOut = get(p+"norm_pre_attn.weight"), get(p+"norm_post_attn.weight"), get(p+"norm_out.weight")
			ly.q, ly.k, ly.v, ly.post = clip(p+"self_attn.q_proj"), clip(p+"self_attn.k_proj"), clip(p+"self_attn.v_proj"), clip(p+"self_attn.post")
			ly.relK = tr(p + "self_attn.relative_k_proj.weight")
			if pds := get(p + "self_attn.per_dim_scale"); pds != nil {
				ly.qScale = make([]float32, hd)
				base := math.Pow(float64(hd), -0.5) / math.Ln2
				for j, v := range pds.Float32s() {
					ly.qScale[j] = float32(base * math.Log1p(math.Exp(float64(v)))) // softplus
				}
			}
			ly.convPre, ly.convNorm = get(p+"lconv1d.pre_layer_norm.weight"), get(p+"lconv1d.conv_norm.weight")
			ly.convStart, ly.convEnd = clip(p+"lconv1d.linear_start"), clip(p+"lconv1d.linear_end")
			if dw := get(p + "lconv1d.depthwise_conv1d.weight"); dw != nil {
				ly.depthwise = dw.Float32s() // [hidden, 1, kernel]
			}
		}
		if firstErr != nil {
			m.audioErr = firstErr
			return
		}
		a.posEmb = audioPositions(ac.HiddenSize, ac.AttentionContextLeft-1)
		m.audio = a
	})
	return m.audioErr
}

// audioPositions is the sinusoidal relative position table for distances
// left, left−1, …, 0: sines then cosines over geometric timescales.
func audioPositions(hidden, left int) *tensor.Tensor {
	half := hidden / 2
	inc := math.Log(10000) / float64(max(half-1, 1))
	out := make([]float32, (left+1)*hidden)
	for p := 0; p <= left; p++ {
		d := float64(left - p)
		row := out[p*hidden:]
		for i := range half {
			a := d * math.Exp(-float64(i)*inc)
			row[i], row[half+i] = float32(math.Sin(a)), float32(math.Cos(a))
		}
	}
	return tensor.New(out, left+1, hidden)
}

// subsample runs the two strided convolutions with LayerNorm and ReLU, and
// the projection. feats is [frames, 128]; it returns [frames/4, hidden]
// and the subsampled validity mask.
func (a *audioTower) subsample(feats []float32, valid []bool) (*tensor.Tensor, []bool) {
	frames := len(valid)
	x := tensor.New(feats, 1, 1, frames, melBins)
	layer := func(x, w, g, b *tensor.Tensor, mask []bool) (*tensor.Tensor, []bool) {
		// Masked frames are zeroed before each convolution.
		xd := x.Data()
		c, t, f := x.Dim(1), x.Dim(2), x.Dim(3)
		for ci := range c {
			for ti := range t {
				if !mask[ti] {
					row := xd[(ci*t+ti)*f : (ci*t+ti+1)*f]
					for i := range row {
						row[i] = 0
					}
				}
			}
		}
		y := tensor.Conv2D(x, w, nil, 2, 1) // [1, out, t', f']
		rec(x)
		yp := y.Permute(0, 2, 3, 1).Contiguous() // channels last for the norm
		rec(y)
		n := tensor.LayerNorm(yp, g, b, float32(a.cfg.RMSNormEps)).ReLU()
		rec(yp)
		out := n.Permute(0, 3, 1, 2).Contiguous()
		rec(n)
		sub := make([]bool, out.Dim(2))
		for i := range sub {
			sub[i] = mask[2*i]
		}
		return out, sub
	}
	h0, m0 := layer(x, a.conv0, a.norm0, a.zero0, valid)
	h1, m1 := layer(h0, a.conv1, a.norm1, a.zero1, m0)
	t := h1.Dim(2)
	// [1, C, T, F] → [T, F·C], frequency-major as the reference's reshape.
	hp := h1.Permute(0, 2, 3, 1).Contiguous().Reshape(t, h1.Dim(3)*h1.Dim(1))
	rec(h1)
	out := hp.MatMul(a.inProj)
	rec(hp)
	return out, m1
}

func silu(x *tensor.Tensor) *tensor.Tensor {
	s := x.Sigmoid()
	out := x.Mul(s)
	rec(s)
	return out
}

func (a *audioTower) ffn(x *tensor.Tensor, f *audioFFN) *tensor.Tensor {
	eps := float32(a.cfg.RMSNormEps)
	h0 := tensor.RMSNorm(x, f.preNorm, eps)
	h1 := f.l1.apply(h0)
	rec(h0)
	h2 := silu(h1)
	rec(h1)
	h3 := f.l2.apply(h2)
	rec(h2)
	h4 := tensor.RMSNorm(h3, f.postNorm, eps)
	rec(h3)
	h5 := h4.MulScalar(float32(a.cfg.ResidualWeight))
	rec(h4)
	out := x.Add(h5)
	rec(h5)
	return out
}

// attend is the chunked local attention: each frame t attends to the
// valid frames s with 0 ≤ t−s < left, with logit
// cap·tanh((q·k + q·r_{t−s})/cap). That is what the reference's
// block-and-shift construction computes once its sliding-window mask is
// applied; the relative table has one more entry (distance left), which
// the mask always removes.
func (a *audioTower) attend(q, k, v, rel []float32, valid []bool, T int) []float32 {
	c := a.cfg
	H := c.NumAttentionHeads
	D := c.HiddenSize / H
	left := c.AttentionContextLeft - 1
	limit := float32(c.AttentionLogitCap)
	out := make([]float32, T*H*D)
	parallel.Range(T*H, 16, func(lo, hi int) {
		logits := make([]float32, left+1)
		for job := lo; job < hi; job++ {
			t, h := job/H, job%H
			qv := q[(t*H+h)*D : (t*H+h+1)*D]
			first := max(0, t-left+1)
			mx := float32(math.Inf(-1))
			for s := first; s <= t; s++ {
				if !valid[s] {
					continue
				}
				kv := k[(s*H+h)*D : (s*H+h+1)*D]
				rv := rel[((left-(t-s))*H+h)*D : ((left-(t-s))*H+h+1)*D]
				var dot float32
				for i := range D {
					dot += qv[i] * (kv[i] + rv[i])
				}
				l := limit * float32(math.Tanh(float64(dot/limit)))
				logits[s-first] = l
				mx = max(mx, l)
			}
			o := out[(t*H+h)*D : (t*H+h+1)*D]
			if math.IsInf(float64(mx), -1) {
				continue // no valid key: a padding frame, dropped later
			}
			var sum float32
			for s := first; s <= t; s++ {
				if valid[s] {
					w := float32(math.Exp(float64(logits[s-first] - mx)))
					logits[s-first] = w
					sum += w
				}
			}
			for s := first; s <= t; s++ {
				if !valid[s] {
					continue
				}
				w := logits[s-first] / sum
				vv := v[(s*H+h)*D : (s*H+h+1)*D]
				for i := range D {
					o[i] += w * vv[i]
				}
			}
		}
	})
	return out
}

// depthwiseCausal applies a causal depthwise convolution over time to
// x [T, C] with weights [C][kernel]: out[t,c] = Σ_j w[c,j]·x[t−kernel+1+j, c].
func depthwiseCausal(x []float32, w []float32, T, C, K int) []float32 {
	out := make([]float32, T*C)
	parallel.Range(T, 32, func(lo, hi int) {
		for t := lo; t < hi; t++ {
			o := out[t*C : (t+1)*C]
			for j := range K {
				src := t - K + 1 + j
				if src < 0 {
					continue
				}
				in := x[src*C : (src+1)*C]
				for c := range C {
					o[c] += w[c*K+j] * in[c]
				}
			}
		}
	})
	return out
}

// encode turns 16 kHz samples into soft tokens in the text model's width,
// [valid tokens, text hidden]. It returns nil for audio too short to give
// a single frame.
func (a *audioTower) encode(samples []float32) *tensor.Tensor {
	c := a.cfg
	eps := float32(c.RMSNormEps)
	feats, valid := a.mel.logMel(samples)
	if len(valid) == 0 {
		return nil
	}
	x, mask := a.subsample(feats, valid)
	T, hidden := x.Dim(0), c.HiddenSize
	H := c.NumAttentionHeads
	D := hidden / H
	kScale := float32(math.Log(1+math.E) / math.Ln2)

	for l := range a.layers {
		ly := &a.layers[l]
		x1 := a.ffn(x, &ly.ffn1)
		rec(x)
		x = x1

		h := tensor.RMSNorm(x, ly.normPreAttn, eps)
		q, k, v := ly.q.apply(h), ly.k.apply(h), ly.v.apply(h)
		rec(h)
		qd, kd := q.Float32s(), k.Float32s()
		for t := range T * H {
			row := qd[t*D : (t+1)*D]
			for i := range row {
				row[i] *= ly.qScale[i]
			}
		}
		for i := range kd {
			kd[i] *= kScale
		}
		relT := a.posEmb.MatMul(ly.relK)
		att := a.attend(qd, kd, v.Float32s(), relT.Float32s(), mask, T)
		rec(q, k, v, relT)
		o := ly.post.apply(tensor.New(att, T, hidden))
		on := tensor.RMSNorm(o, ly.normPostAttn, eps)
		rec(o)
		x2 := x.Add(on)
		rec(x, on)
		x = x2

		// Light convolution: GLU, causal depthwise conv, norm, SiLU.
		hc := tensor.RMSNorm(x, ly.convPre, eps)
		st := ly.convStart.apply(hc)
		rec(hc)
		ga := st.Narrow(1, 0, hidden).Contiguous()
		gb := st.Narrow(1, hidden, hidden).Contiguous()
		rec(st)
		gs := gb.Sigmoid()
		glu := ga.Mul(gs)
		rec(ga, gb, gs)
		dw := depthwiseCausal(glu.Float32s(), ly.depthwise, T, hidden, c.ConvKernelSize)
		rec(glu)
		cn := tensor.RMSNorm(tensor.New(dw, T, hidden), ly.convNorm, eps)
		cs := silu(cn)
		rec(cn)
		ce := ly.convEnd.apply(cs)
		rec(cs)
		x3 := x.Add(ce)
		rec(x, ce)
		x = x3

		x4 := a.ffn(x, &ly.ffn2)
		rec(x)
		x5 := tensor.RMSNorm(x4, ly.normOut, eps)
		rec(x4)
		x = x5
	}

	// Output projection, keep the valid tokens, then embed_audio: a
	// weight-free RMSNorm and the projection into the text width.
	y := x.MatMul(a.outProj).Add(a.outBias)
	rec(x)
	var keep []int
	for t, ok := range mask {
		if ok && t < T {
			keep = append(keep, t)
		}
	}
	if len(keep) == 0 {
		rec(y)
		return nil
	}
	yk := y.Rows(keep)
	rec(y)
	out := c.OutputProjDims
	scale := rmsScale(yk.RowSumSquares(), out, eps)
	res := tensor.MatMulFused(yk, a.embProj, tensor.Fused{RowScale: scale})
	rec(yk)
	return res
}

func (a *audioTower) release(release func(*tensor.Tensor)) {
	for _, t := range []*tensor.Tensor{a.conv0, a.conv1, a.norm0, a.norm1, a.zero0, a.zero1, a.inProj, a.outProj, a.outBias, a.embProj, a.posEmb} {
		release(t)
	}
	for _, l := range a.layers {
		for _, t := range []*tensor.Tensor{l.ffn1.preNorm, l.ffn1.postNorm, l.ffn1.l1.w, l.ffn1.l2.w, l.ffn2.preNorm, l.ffn2.postNorm, l.ffn2.l1.w, l.ffn2.l2.w,
			l.normPreAttn, l.normPostAttn, l.normOut, l.q.w, l.k.w, l.v.w, l.post.w, l.relK, l.convPre, l.convNorm, l.convStart.w, l.convEnd.w} {
			release(t)
		}
	}
}
