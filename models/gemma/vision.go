package gemma

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"path/filepath"

	"github.com/fiber/ai/safetensors"
	"github.com/fiber/ai/tensor"
)

// EmbeddingGemma 2's vision tower, a Gemma 4 vision encoder: images are
// resized to a patch budget, cut into 16×16 patches, run through a
// bidirectional transformer with axial 2-D rotary positions, averaged over
// 3×3 patch blocks into soft tokens, and projected into the text model's
// width, where they take the place of <|image|> placeholders.

// visionLayer is one encoder block, weights transposed for x·W and the
// two pre-norms folded into the products they feed, as in the text tower.
type visionLayer struct {
	postAttnNorm, postFFNNorm *tensor.Tensor
	qNorm, kNorm, vNorm       *tensor.Tensor
	wq, wk, wv, wo            *tensor.Tensor
	wgate, wup, wdown         *tensor.Tensor
}

type visionTower struct {
	cfg     VisionConfig
	inProj  *tensor.Tensor // [3·p·p, hidden]
	posX    *tensor.Tensor // [positions, hidden]
	posY    *tensor.Tensor
	layers  []visionLayer
	embProj *tensor.Tensor // [hidden, text hidden], after a weight-free RMSNorm
}

// loadVision reads the vision tower's weights. It is called on the first
// image, so a text-only user never pays the tower's 680 MB.
func (m *Model) loadVision() error {
	m.visionOnce.Do(func() {
		vc := m.cfg.Vision
		if vc == nil {
			m.visionErr = fmt.Errorf("gemma: this checkpoint has no vision tower")
			return
		}
		f, err := safetensors.OpenDir(m.dir)
		if err != nil {
			m.visionErr = err
			return
		}
		defer f.Close()
		get := func(name string) (*tensor.Tensor, error) {
			if _, ok := f.Info(name); !ok {
				return nil, fmt.Errorf("gemma: weight %q not found in %s", name, filepath.Base(m.dir))
			}
			return f.Tensor(name)
		}
		tr := func(name string) (*tensor.Tensor, error) {
			w, err := get(name)
			if err != nil {
				return nil, err
			}
			return w.Transpose(0, 1).Contiguous(), nil
		}
		v := &visionTower{cfg: *vc}
		if v.inProj, err = tr("vision_tower.patch_embedder.input_proj.weight"); err != nil {
			m.visionErr = err
			return
		}
		table, err := get("vision_tower.patch_embedder.position_embedding_table")
		if err != nil {
			m.visionErr = err
			return
		}
		v.posX = table.Select(0, 0).Contiguous()
		v.posY = table.Select(0, 1).Contiguous()
		if v.embProj, err = tr("embed_vision.embedding_projection.weight"); err != nil {
			m.visionErr = err
			return
		}
		fold := func(w, g *tensor.Tensor) *tensor.Tensor {
			out := w.Mul(g.Reshape(g.Size(), 1))
			w.Release()
			return out
		}
		v.layers = make([]visionLayer, vc.NumHiddenLayers)
		for i := range v.layers {
			p := fmt.Sprintf("vision_tower.encoder.layers.%d.", i)
			ly := &v.layers[i]
			var inNorm, preFFN *tensor.Tensor
			for _, s := range []struct {
				dst  **tensor.Tensor
				name string
				t    bool
			}{
				{&inNorm, "input_layernorm.weight", false},
				{&ly.postAttnNorm, "post_attention_layernorm.weight", false},
				{&preFFN, "pre_feedforward_layernorm.weight", false},
				{&ly.postFFNNorm, "post_feedforward_layernorm.weight", false},
				{&ly.qNorm, "self_attn.q_norm.weight", false},
				{&ly.kNorm, "self_attn.k_norm.weight", false},
				{&ly.wq, "self_attn.q_proj.linear.weight", true},
				{&ly.wk, "self_attn.k_proj.linear.weight", true},
				{&ly.wv, "self_attn.v_proj.linear.weight", true},
				{&ly.wo, "self_attn.o_proj.linear.weight", true},
				{&ly.wgate, "mlp.gate_proj.linear.weight", true},
				{&ly.wup, "mlp.up_proj.linear.weight", true},
				{&ly.wdown, "mlp.down_proj.linear.weight", true},
			} {
				var err error
				if s.t {
					*s.dst, err = tr(p + s.name)
				} else {
					*s.dst, err = get(p + s.name)
				}
				if err != nil {
					m.visionErr = err
					return
				}
			}
			ly.vNorm = tensor.Ones(vc.HeadDim)
			ly.wq, ly.wk, ly.wv = fold(ly.wq, inNorm), fold(ly.wk, inNorm), fold(ly.wv, inNorm)
			ly.wgate, ly.wup = fold(ly.wgate, preFFN), fold(ly.wup, preFFN)
		}
		m.vision = v
	})
	return m.visionErr
}

// patches is a preprocessed image: rows of p·p·3 values in [0, 1], their
// (x, y) grid positions, and the grid size in patches.
type patches struct {
	data       []float32
	xs, ys     []int
	gw, gh     int
	softTokens int
}

// visionSize returns the resized height and width: the largest sizes that
// are multiples of patch·pooling and fit maxPatches patches, keeping the
// aspect ratio (the reference's get_aspect_ratio_preserving_size).
func visionSize(h, w, patch, maxPatches, pool int) (int, int, error) {
	targetPx := float64(maxPatches * patch * patch)
	factor := math.Sqrt(targetPx / float64(h*w))
	side := pool * patch
	th := int(math.Floor(factor*float64(h)/float64(side))) * side
	tw := int(math.Floor(factor*float64(w)/float64(side))) * side
	if th == 0 && tw == 0 {
		return 0, 0, fmt.Errorf("gemma: image %d×%d cannot be resized to a multiple of %d", w, h, side)
	}
	maxSide := (maxPatches / (pool * pool)) * side
	switch {
	case th == 0:
		th = side
		tw = min(int(math.Floor(float64(w)/float64(h)))*side, maxSide)
	case tw == 0:
		tw = side
		th = min(int(math.Floor(float64(h)/float64(w)))*side, maxSide)
	}
	if float64(th*tw) > targetPx {
		return 0, 0, fmt.Errorf("gemma: image %d×%d would need more than %d patches", w, h, maxPatches)
	}
	return th, tw, nil
}

// rgb8 returns the image as interleaved 8-bit RGB. Alpha is dropped
// without compositing, as PIL's convert("RGB") does.
func rgb8(img image.Image) ([]uint8, int, int) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]uint8, w*h*3)
	for y := range h {
		for x := range w {
			o := out[(y*w+x)*3:]
			switch c := img.At(b.Min.X+x, b.Min.Y+y).(type) {
			case color.Gray:
				o[0], o[1], o[2] = c.Y, c.Y, c.Y
			default:
				n := color.NRGBAModel.Convert(c).(color.NRGBA)
				o[0], o[1], o[2] = n.R, n.G, n.B
			}
		}
	}
	return out, w, h
}

// preprocess turns an image into patches for the vision tower.
func (v *visionTower) preprocess(img image.Image) (*patches, error) {
	c := v.cfg
	px, w, h := rgb8(img)
	soft := c.DefaultOutputLength
	if soft == 0 {
		soft = 280
	}
	maxPatches := soft * c.PoolingKernelSize * c.PoolingKernelSize
	th, tw, err := visionSize(h, w, c.PatchSize, maxPatches, c.PoolingKernelSize)
	if err != nil {
		return nil, err
	}
	px = resizeRGB(px, w, h, tw, th)
	p := c.PatchSize
	gw, gh := tw/p, th/p
	n := gw * gh
	out := &patches{data: make([]float32, n*p*p*3), xs: make([]int, n), ys: make([]int, n), gw: gw, gh: gh,
		softTokens: n / (c.PoolingKernelSize * c.PoolingKernelSize)}
	// Patch rows are in grid order, row by row; within a patch the values
	// run over pixel row, pixel column, channel.
	const inv255 = 1.0 / 255
	for gy := range gh {
		for gx := range gw {
			i := gy*gw + gx
			out.xs[i], out.ys[i] = gx, gy
			row := out.data[i*p*p*3:]
			for py := range p {
				for pxx := range p {
					src := ((gy*p+py)*tw + gx*p + pxx) * 3
					dst := (py*p + pxx) * 3
					for ch := range 3 {
						row[dst+ch] = float32(px[src+ch]) * inv255
					}
				}
			}
		}
	}
	return out, nil
}

// axialRoPE rotates the first half of each head by the patches' x
// positions and the second half by their y positions. x is [1,H,N,D].
func axialRoPE(x *tensor.Tensor, base float64, xs, ys []int) *tensor.Tensor {
	d := x.Dim(3)
	a := x.Narrow(3, 0, d/2).Contiguous()
	b := x.Narrow(3, d/2, d/2).Contiguous()
	ra := tensor.RoPE(a, base, xs)
	rb := tensor.RoPE(b, base, ys)
	rec(a, b)
	out := tensor.Cat(3, ra, rb)
	rec(ra, rb)
	return out
}

// encode runs one image through the tower and returns its soft tokens in
// the text model's width, [softTokens, text hidden].
func (v *visionTower) encode(p *patches) *tensor.Tensor {
	c := v.cfg
	hidden, H, D := c.HiddenSize, c.NumAttentionHeads, c.HeadDim
	N := len(p.xs)
	eps := float32(c.RMSNormEps)
	base := c.RopeParameters.RopeTheta

	// Patch embedding: 2·(p − 0.5), projected, plus the x and y embeddings.
	pv := tensor.New(p.data, N, len(p.data)/N).MulScalar(2).AddScalar(-1)
	h0 := pv.MatMul(v.inProj)
	rec(pv)
	ex, ey := v.posX.Rows(p.xs), v.posY.Rows(p.ys)
	h1 := h0.Add(ex)
	rec(h0, ex)
	x := h1.Add(ey)
	rec(h1, ey)

	for l := range v.layers {
		ly := &v.layers[l]
		scale := rmsScale(x.RowSumSquares(), hidden, eps)
		q0 := tensor.MatMulFused(x, ly.wq, tensor.Fused{RowScale: scale})
		k0 := tensor.MatMulFused(x, ly.wk, tensor.Fused{RowScale: scale})
		v0 := tensor.MatMulFused(x, ly.wv, tensor.Fused{RowScale: scale})
		q := tensor.RMSNorm(q0.Reshape(1, N, H, D), ly.qNorm, eps)
		k := tensor.RMSNorm(k0.Reshape(1, N, H, D), ly.kNorm, eps)
		vv := tensor.RMSNorm(v0.Reshape(1, N, H, D), ly.vNorm, eps)
		rec(q0, k0, v0)
		qc := q.Permute(0, 2, 1, 3).Contiguous()
		kc := k.Permute(0, 2, 1, 3).Contiguous()
		vc := vv.Permute(0, 2, 1, 3).Contiguous()
		rec(q, k, vv) // H > 1: the permutes copied
		qr := axialRoPE(qc, base, p.xs, p.ys)
		kr := axialRoPE(kc, base, p.xs, p.ys)
		rec(qc, kc)
		att := tensor.AttentionScaled(qr, kr, vc, nil, 1)
		rec(qr, kr, vc)
		attc := att.Permute(0, 2, 1, 3).Contiguous()
		rec(att)
		o0 := attc.Reshape(N, H*D).MatMul(ly.wo)
		rec(attc)
		o := tensor.RMSNorm(o0, ly.postAttnNorm, eps)
		rec(o0)
		x2 := x.Add(o)
		rec(x, o)
		x = x2

		scale = rmsScale(x.RowSumSquares(), hidden, eps)
		up := tensor.MatMulFused(x, ly.wup, tensor.Fused{RowScale: scale})
		gu := tensor.MatMulFused(x, ly.wgate, tensor.Fused{RowScale: scale, Act: tensor.GELUAct, Mul: up})
		rec(up)
		mlp0 := gu.MatMul(ly.wdown)
		rec(gu)
		mlp := tensor.RMSNorm(mlp0, ly.postFFNNorm, eps)
		rec(mlp0)
		x3 := x.Add(mlp)
		rec(x, mlp)
		x = x3
	}

	// Average 3×3 blocks of patches by position, in pooled-grid row order,
	// and scale by √hidden.
	k := c.PoolingKernelSize
	pw := p.gw / k
	soft := p.softTokens
	xd := x.Float32s()
	rec(x)
	pooled := make([]float32, soft*hidden)
	inv := float32(math.Sqrt(float64(hidden))) / float32(k*k)
	for i := range N {
		dst := pooled[((p.ys[i]/k)*pw+p.xs[i]/k)*hidden:]
		row := xd[i*hidden : (i+1)*hidden]
		for j, val := range row {
			dst[j] += val
		}
	}
	for j := range pooled {
		pooled[j] *= inv
	}
	// embed_vision: weight-free RMSNorm, then into the text width.
	pt := tensor.New(pooled, soft, hidden)
	ps := rmsScale(pt.RowSumSquares(), hidden, eps)
	out := tensor.MatMulFused(pt, v.embProj, tensor.Fused{RowScale: ps})
	rec(pt)
	return out
}

func (v *visionTower) release(release func(*tensor.Tensor)) {
	for _, t := range []*tensor.Tensor{v.inProj, v.posX, v.posY, v.embProj} {
		release(t)
	}
	for _, l := range v.layers {
		for _, t := range []*tensor.Tensor{l.postAttnNorm, l.postFFNNorm, l.qNorm, l.kNorm, l.vNorm, l.wq, l.wk, l.wv, l.wo, l.wgate, l.wup, l.wdown} {
			release(t)
		}
	}
}
