// Package gemma runs Google's EmbeddingGemma models natively on the
// fiber/ai stack: it loads the Hugging Face weights and tokenizer from a
// directory and turns text into sentence embeddings, with no Python and
// no external service. Two architectures are supported, chosen from the
// checkpoint's config.json: EmbeddingGemma-300m and other Gemma 3 text
// encoders, and EmbeddingGemma 2, whose images and audio (EmbedImages,
// EmbedAudio, EmbedInputs) land in the same space as its text.
package gemma

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config holds the Gemma 3 text-model hyper-parameters read from
// config.json. Only the fields the encoder needs are kept.
type Config struct {
	HiddenSize            int      `json:"hidden_size"`
	IntermediateSize      int      `json:"intermediate_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	HeadDim               int      `json:"head_dim"`
	VocabSize             int      `json:"vocab_size"`
	SlidingWindow         int      `json:"sliding_window"`
	SlidingWindowPattern  int      `json:"sliding_window_pattern"`
	LayerTypes            []string `json:"layer_types"`
	RopeTheta             float64  `json:"rope_theta"`
	RopeLocalBaseFreq     float64  `json:"rope_local_base_freq"`
	QueryPreAttnScalar    float64  `json:"query_pre_attn_scalar"`
	RMSNormEps            float64  `json:"rms_norm_eps"`
	HiddenActivation      string   `json:"hidden_activation"`
	UseBidirectional      bool     `json:"use_bidirectional_attention"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	RopeScaling           any      `json:"rope_scaling"`

	// EmbeddingGemma 2 only. Variant is 2 for that architecture and 0 for
	// Gemma 3 encoders; the remaining fields are zero for the latter.
	Variant                 int                       `json:"-"`
	HiddenSizePerLayerInput int                       `json:"hidden_size_per_layer_input"`
	EmbeddingDim            int                       `json:"embedding_dim"`
	PerLayerConfig          map[string]LayerOverride  `json:"per_layer_config"`
	RopeParameters          map[string]RopeParameters `json:"rope_parameters"`
	// MultimodalTokens maps the image, audio and video placeholder token
	// ids, and their begin/end markers, to a name for error messages.
	MultimodalTokens map[int]string `json:"-"`
	// Image, BOI and EOI are the image placeholder and its begin and end
	// markers; Vision is the vision tower's configuration, nil when the
	// checkpoint has none.
	ImageToken, BOIToken, EOIToken int           `json:"-"`
	Vision                         *VisionConfig `json:"-"`
	// AudioToken, BOAToken and EOAToken are the audio placeholder and its
	// markers; Audio is the audio tower's configuration, nil when absent.
	AudioToken, BOAToken, EOAToken int          `json:"-"`
	Audio                          *AudioConfig `json:"-"`
}

// AudioConfig holds the EmbeddingGemma 2 audio tower's hyper-parameters
// (a Gemma 4 conformer).
type AudioConfig struct {
	HiddenSize              int     `json:"hidden_size"`
	NumHiddenLayers         int     `json:"num_hidden_layers"`
	NumAttentionHeads       int     `json:"num_attention_heads"`
	AttentionChunkSize      int     `json:"attention_chunk_size"`
	AttentionContextLeft    int     `json:"attention_context_left"`
	AttentionContextRight   int     `json:"attention_context_right"`
	AttentionLogitCap       float64 `json:"attention_logit_cap"`
	ConvKernelSize          int     `json:"conv_kernel_size"`
	OutputProjDims          int     `json:"output_proj_dims"`
	ResidualWeight          float64 `json:"residual_weight"`
	RMSNormEps              float64 `json:"rms_norm_eps"`
	HiddenAct               string  `json:"hidden_act"`
	UseClippedLinears       bool    `json:"use_clipped_linears"`
	SubsamplingConvChannels []int   `json:"subsampling_conv_channels"`
}

func (a *AudioConfig) validate() error {
	switch {
	case a.HiddenSize == 0 || a.NumHiddenLayers == 0 || a.NumAttentionHeads == 0:
		return fmt.Errorf("gemma: audio_config is incomplete")
	case a.HiddenSize%a.NumAttentionHeads != 0:
		return fmt.Errorf("gemma: audio width %d not divisible by %d heads", a.HiddenSize, a.NumAttentionHeads)
	case a.HiddenAct != "silu":
		return fmt.Errorf("gemma: audio activation %q not supported", a.HiddenAct)
	case a.AttentionContextRight != 0:
		return fmt.Errorf("gemma: audio attention with right context not supported")
	case len(a.SubsamplingConvChannels) != 2:
		return fmt.Errorf("gemma: audio subsampling with %d convolutions not supported", len(a.SubsamplingConvChannels))
	}
	return nil
}

// VisionConfig holds the EmbeddingGemma 2 vision tower's hyper-parameters
// (a Gemma 4 vision encoder).
type VisionConfig struct {
	HiddenSize            int     `json:"hidden_size"`
	IntermediateSize      int     `json:"intermediate_size"`
	NumHiddenLayers       int     `json:"num_hidden_layers"`
	NumAttentionHeads     int     `json:"num_attention_heads"`
	NumKeyValueHeads      int     `json:"num_key_value_heads"`
	HeadDim               int     `json:"head_dim"`
	PatchSize             int     `json:"patch_size"`
	PoolingKernelSize     int     `json:"pooling_kernel_size"`
	PositionEmbeddingSize int     `json:"position_embedding_size"`
	DefaultOutputLength   int     `json:"default_output_length"`
	RMSNormEps            float64 `json:"rms_norm_eps"`
	HiddenActivation      string  `json:"hidden_activation"`
	UseClippedLinears     bool    `json:"use_clipped_linears"`
	Standardize           bool    `json:"standardize"`
	RopeParameters        struct {
		RopeTheta float64 `json:"rope_theta"`
		RopeType  string  `json:"rope_type"`
	} `json:"rope_parameters"`
}

// validate refuses vision configurations this package does not run.
func (v *VisionConfig) validate() error {
	switch {
	case v.HiddenSize == 0 || v.NumHiddenLayers == 0 || v.PatchSize == 0 || v.PoolingKernelSize == 0:
		return fmt.Errorf("gemma: vision_config is incomplete")
	case v.NumAttentionHeads != v.NumKeyValueHeads:
		return fmt.Errorf("gemma: vision tower with grouped-query attention not supported")
	case v.HeadDim%4 != 0:
		return fmt.Errorf("gemma: vision head_dim %d must be a multiple of 4 for axial RoPE", v.HeadDim)
	case v.RopeParameters.RopeType != "axial":
		return fmt.Errorf("gemma: vision rope_type %q not supported, only axial", v.RopeParameters.RopeType)
	case v.UseClippedLinears:
		return fmt.Errorf("gemma: vision tower with clipped linears not supported")
	case v.Standardize:
		return fmt.Errorf("gemma: vision tower with output standardisation not supported")
	case v.HiddenActivation != "gelu_pytorch_tanh":
		return fmt.Errorf("gemma: vision activation %q not supported", v.HiddenActivation)
	}
	return nil
}

// LayerOverride is EmbeddingGemma 2's per-layer attention geometry: the
// full-attention layers are wider than the sliding ones.
type LayerOverride struct {
	HeadDim          int `json:"head_dim"`
	NumKeyValueHeads int `json:"num_key_value_heads"`
}

// RopeParameters holds one layer type's rotary settings.
type RopeParameters struct {
	RopeTheta float64 `json:"rope_theta"`
	RopeType  string  `json:"rope_type"`
}

// LoadConfig reads config.json from a model directory.
func LoadConfig(dir string) (Config, error) {
	var c Config
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return c, err
	}
	// EmbeddingGemma 2 is a multimodal checkpoint whose text model sits
	// under text_config; the vision and audio configs beside it are not
	// read, because this package does not run those towers.
	var top struct {
		ModelType  string          `json:"model_type"`
		TextConfig json.RawMessage `json:"text_config"`
		Image      *int            `json:"image_token_id"`
		Audio      *int            `json:"audio_token_id"`
		Video      *int            `json:"video_token_id"`
		BOI        *int            `json:"boi_token_id"`
		EOI        *int            `json:"eoi_token_id"`
		BOA        *int            `json:"boa_token_id"`
		EOA        *int            `json:"eoa_token_index"`
		Vision     *VisionConfig   `json:"vision_config"`
		AudioCfg   *AudioConfig    `json:"audio_config"`
	}
	if err := json.Unmarshal(b, &top); err != nil {
		return c, fmt.Errorf("gemma: config.json: %w", err)
	}
	if top.ModelType == "embedding_gemma2" {
		if err := json.Unmarshal(top.TextConfig, &c); err != nil {
			return c, fmt.Errorf("gemma: config.json text_config: %w", err)
		}
		c.Variant = 2
		c.MultimodalTokens = map[int]string{}
		for _, t := range []struct {
			id   *int
			name string
		}{{top.Image, "image"}, {top.Audio, "audio"}, {top.Video, "video"},
			{top.BOI, "begin-of-image"}, {top.EOI, "end-of-image"},
			{top.BOA, "begin-of-audio"}, {top.EOA, "end-of-audio"}} {
			if t.id != nil {
				c.MultimodalTokens[*t.id] = t.name
			}
		}
		if top.Image != nil && top.BOI != nil && top.EOI != nil && top.Vision != nil {
			if err := top.Vision.validate(); err != nil {
				return c, err
			}
			c.ImageToken, c.BOIToken, c.EOIToken = *top.Image, *top.BOI, *top.EOI
			c.Vision = top.Vision
		}
		if top.Audio != nil && top.BOA != nil && top.EOA != nil && top.AudioCfg != nil {
			if err := top.AudioCfg.validate(); err != nil {
				return c, err
			}
			c.AudioToken, c.BOAToken, c.EOAToken = *top.Audio, *top.BOA, *top.EOA
			c.Audio = top.AudioCfg
		}
		return c, c.fillVariant2()
	}
	if err := json.Unmarshal(b, &c); err != nil {
		return c, fmt.Errorf("gemma: config.json: %w", err)
	}
	if c.SlidingWindowPattern == 0 {
		c.SlidingWindowPattern = 6
	}
	if len(c.LayerTypes) == 0 {
		c.LayerTypes = make([]string, c.NumHiddenLayers)
		for i := range c.LayerTypes {
			if (i+1)%c.SlidingWindowPattern == 0 {
				c.LayerTypes[i] = "full_attention"
			} else {
				c.LayerTypes[i] = "sliding_attention"
			}
		}
	}
	return c, c.validate()
}

func (c Config) validate() error {
	switch {
	case c.HiddenSize == 0 || c.NumHiddenLayers == 0:
		return fmt.Errorf("gemma: config is missing hidden_size or num_hidden_layers")
	case c.HeadDim%2 != 0:
		return fmt.Errorf("gemma: head_dim %d must be even for RoPE", c.HeadDim)
	case c.NumAttentionHeads%c.NumKeyValueHeads != 0:
		return fmt.Errorf("gemma: %d query heads not a multiple of %d key/value heads", c.NumAttentionHeads, c.NumKeyValueHeads)
	case c.HiddenActivation != "" && c.HiddenActivation != "gelu_pytorch_tanh":
		return fmt.Errorf("gemma: activation %q not supported (want gelu_pytorch_tanh)", c.HiddenActivation)
	case c.RopeScaling != nil:
		return fmt.Errorf("gemma: rope_scaling is set; not supported")
	case len(c.LayerTypes) != c.NumHiddenLayers:
		return fmt.Errorf("gemma: %d layer types for %d layers", len(c.LayerTypes), c.NumHiddenLayers)
	}
	return nil
}

// fillVariant2 applies EmbeddingGemma 2's defaults, as the reference
// configuration class does, and checks what this package can run.
func (c *Config) fillVariant2() error {
	if len(c.LayerTypes) == 0 {
		c.LayerTypes = make([]string, c.NumHiddenLayers)
		for i := range c.LayerTypes {
			if (i+1)%6 == 0 {
				c.LayerTypes[i] = "full_attention"
			} else {
				c.LayerTypes[i] = "sliding_attention"
			}
		}
	}
	if c.PerLayerConfig == nil {
		c.PerLayerConfig = map[string]LayerOverride{}
		for i, t := range c.LayerTypes {
			if t == "full_attention" {
				c.PerLayerConfig[fmt.Sprintf("%02d", i)] = LayerOverride{HeadDim: 512, NumKeyValueHeads: 1}
			}
		}
	}
	if c.RopeParameters == nil {
		c.RopeParameters = map[string]RopeParameters{
			"sliding_attention": {RopeTheta: 10000, RopeType: "default"},
			"full_attention":    {RopeTheta: 1000000, RopeType: "default"},
		}
	}
	for t, rp := range c.RopeParameters {
		if rp.RopeType != "" && rp.RopeType != "default" {
			return fmt.Errorf("gemma: rope_type %q for %s not supported", rp.RopeType, t)
		}
	}
	if c.EmbeddingDim == 0 {
		c.EmbeddingDim = 768
	}
	if c.HiddenSizePerLayerInput == 0 {
		return fmt.Errorf("gemma: embedding_gemma2 config without hidden_size_per_layer_input")
	}
	if err := c.validate(); err != nil {
		return err
	}
	for l := range c.LayerTypes {
		if c.headDim(l)%2 != 0 {
			return fmt.Errorf("gemma: layer %d head_dim %d must be even for RoPE", l, c.headDim(l))
		}
		if c.NumAttentionHeads%c.kvHeads(l) != 0 {
			return fmt.Errorf("gemma: layer %d: %d query heads not a multiple of %d key/value heads", l, c.NumAttentionHeads, c.kvHeads(l))
		}
	}
	return nil
}

// override returns layer l's geometry override, if any. The reference
// keys per_layer_config by zero-padded layer index ("05").
func (c Config) override(l int) (LayerOverride, bool) {
	o, ok := c.PerLayerConfig[fmt.Sprintf("%02d", l)]
	if !ok {
		o, ok = c.PerLayerConfig[fmt.Sprint(l)]
	}
	return o, ok
}

// headDim returns layer l's attention head dimension.
func (c Config) headDim(l int) int {
	if o, ok := c.override(l); ok && o.HeadDim != 0 {
		return o.HeadDim
	}
	return c.HeadDim
}

// kvHeads returns layer l's number of key/value heads.
func (c Config) kvHeads(l int) int {
	if o, ok := c.override(l); ok && o.NumKeyValueHeads != 0 {
		return o.NumKeyValueHeads
	}
	return c.NumKeyValueHeads
}

// window returns the effective sliding-window bound: positions attend when
// their distance is below it. For bidirectional Gemma 3 models transformers
// halves the configured value and adds one (config 512 → bound 257, so 256
// tokens to each side, a 512-wide window); causal models use the value as
// is. EmbeddingGemma 2 configures an inclusive radius instead — a sliding
// layer attends where |i − j| ≤ sliding_window — so its bound is one more.
func (c Config) window() int {
	if c.Variant == 2 {
		return c.SlidingWindow + 1
	}
	if c.UseBidirectional {
		return c.SlidingWindow/2 + 1
	}
	return c.SlidingWindow
}

// isSliding reports whether layer l uses sliding-window attention.
func (c Config) isSliding(l int) bool { return c.LayerTypes[l] == "sliding_attention" }

// ropeBase returns the RoPE base frequency for layer l: the local base for
// sliding layers, the global theta for full-attention layers.
func (c Config) ropeBase(l int) float64 {
	if c.Variant == 2 {
		if rp, ok := c.RopeParameters[c.LayerTypes[l]]; ok && rp.RopeTheta != 0 {
			return rp.RopeTheta
		}
	}
	if c.isSliding(l) {
		if c.RopeLocalBaseFreq != 0 {
			return c.RopeLocalBaseFreq
		}
		return 10000
	}
	if c.RopeTheta != 0 {
		return c.RopeTheta
	}
	return 10000
}
