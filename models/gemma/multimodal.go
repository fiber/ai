package gemma

import (
	"fmt"
	"image"
	"math"

	"github.com/fiber/ai/tensor"
)

// Input is one item for EmbedInputs: text that may contain <|image|>
// placeholders, and the images that fill them, in order. The result is
// one embedding for the whole item, text and images together.
type Input struct {
	Text   string
	Images []image.Image
}

// EmbedImages embeds each image on its own, one vector per image, in the
// space the text embeddings live in. It loads the vision tower on first
// use.
func (m *Model) EmbedImages(images []image.Image, opts ...EmbedOption) ([][]float32, error) {
	in := make([]Input, len(images))
	for i, img := range images {
		in[i] = Input{Text: imagePlaceholder, Images: []image.Image{img}}
	}
	return m.EmbedInputs(in, opts...)
}

// imagePlaceholder is how text marks the position of an image.
const imagePlaceholder = "<|image|>"

// EmbedInputs embeds items that mix text and images. Each <|image|> in an
// item's text is replaced by the next image's soft tokens between the
// begin- and end-of-image markers, and the text tower runs over the whole
// sequence, exactly as for text. Prompts and Dim apply as in Embed.
func (m *Model) EmbedInputs(inputs []Input, opts ...EmbedOption) ([][]float32, error) {
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
	needVision := false
	for _, in := range inputs {
		needVision = needVision || len(in.Images) > 0
	}
	if needVision {
		if err := m.loadVision(); err != nil {
			return nil, err
		}
	}

	// Each item becomes a row of input embeddings: scaled token embeddings
	// for text, soft tokens in the image positions.
	seqs := make([]*tensor.Tensor, len(inputs))
	lengths := make([]int, len(inputs))
	defer func() {
		for _, s := range seqs {
			rec(s)
		}
	}()
	for i, in := range inputs {
		seq, err := m.inputEmbeddings(prefix+in.Text, in.Images)
		if err != nil {
			return nil, fmt.Errorf("input %d: %w", i, err)
		}
		seqs[i], lengths[i] = seq, seq.Dim(0)
	}

	out := make([][]float32, len(inputs))
	order, _ := sortIndex(lengths)
	var batch []int
	maxLen := 0
	flush := func() {
		if len(batch) == 0 {
			return
		}
		emb := m.embedBatchFrom(seqs, lengths, batch, dim)
		for bi, idx := range batch {
			out[idx] = emb[bi]
		}
		batch, maxLen = batch[:0], 0
	}
	for _, idx := range order {
		newMax := max(maxLen, lengths[idx])
		if len(batch) > 0 && newMax*(len(batch)+1) > m.batchTokens {
			flush()
			newMax = lengths[idx]
		}
		batch = append(batch, idx)
		maxLen = newMax
	}
	flush()
	return out, nil
}

// inputEmbeddings tokenises text, checks its placeholders against the
// images, runs the images through the vision tower and assembles the
// input embeddings [length, hidden].
func (m *Model) inputEmbeddings(text string, images []image.Image) (*tensor.Tensor, error) {
	cfg := m.cfg
	ids := m.tok.Encode(text)
	nImages := 0
	for _, id := range ids {
		if name, ok := cfg.MultimodalTokens[id]; ok {
			if id == cfg.ImageToken && cfg.Vision != nil {
				nImages++
				continue
			}
			return nil, fmt.Errorf("gemma: input contains the %s placeholder token; EmbeddingGemma 2 audio and video input are not supported", name)
		}
	}
	if nImages != len(images) {
		return nil, fmt.Errorf("gemma: %d %s placeholders in the text for %d images", nImages, imagePlaceholder, len(images))
	}

	soft := make([]*tensor.Tensor, len(images))
	defer func() {
		for _, s := range soft {
			rec(s)
		}
	}()
	total := len(ids)
	for i, img := range images {
		p, err := m.vision.preprocess(img)
		if err != nil {
			return nil, fmt.Errorf("image %d: %w", i, err)
		}
		var st *tensor.Tensor
		tensor.NoGrad(func() { st = m.vision.encode(p) })
		soft[i] = st
		total += st.Dim(0) + 1 // the placeholder becomes begin, n soft tokens, end
	}
	if total > m.maxTokens {
		return nil, fmt.Errorf("gemma: input is %d tokens with its images, more than the %d the model takes", total, m.maxTokens)
	}

	hidden := cfg.HiddenSize
	scale := float32(math.Sqrt(float64(hidden)))
	embed := m.embed.Data()
	rows := make([]float32, 0, total*hidden)
	token := func(id int) {
		for _, v := range embed[id*hidden : (id+1)*hidden] {
			rows = append(rows, v*scale)
		}
	}
	next := 0
	for _, id := range ids {
		if id != cfg.ImageToken {
			token(id)
			continue
		}
		token(cfg.BOIToken)
		rows = append(rows, soft[next].Float32s()...)
		token(cfg.EOIToken)
		next++
	}
	return tensor.New(rows, total, hidden), nil
}

// embedBatchFrom runs the text tower over the sequences named by idx,
// padded to the longest, and returns one dim-length embedding each.
func (m *Model) embedBatchFrom(seqs []*tensor.Tensor, lengths, idx []int, dim int) [][]float32 {
	var result [][]float32
	tensor.NoGrad(func() {
		hidden := m.cfg.HiddenSize
		B, T := len(idx), 0
		for _, i := range idx {
			T = max(T, lengths[i])
		}
		x := make([]float32, B*T*hidden)
		bl := make([]int, B)
		for b, i := range idx {
			bl[b] = lengths[i]
			copy(x[b*T*hidden:], seqs[i].Float32s())
		}
		h := m.encode2From(tensor.New(x, B, T, hidden), bl, B, T)
		pooled := m.pool(h, bl, B, T)
		h.Recycle()
		emb := m.applyHead(pooled, dim)
		pooled.Recycle()
		data := emb.Float32s()
		emb.Recycle()
		result = make([][]float32, B)
		for b := range B {
			result[b] = append([]float32(nil), data[b*dim:(b+1)*dim]...)
		}
	})
	return result
}
