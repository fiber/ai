// Package gemma runs Google's EmbeddingGemma and other Gemma 3 text
// encoders natively on the fiber/ai stack: it loads the Hugging Face
// weights and tokenizer from a directory and turns text into sentence
// embeddings, with no Python and no external service.
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
}

// LoadConfig reads config.json from a model directory.
func LoadConfig(dir string) (Config, error) {
	var c Config
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return c, err
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

// window returns the effective sliding-window bound: positions attend when
// their distance is below it. For bidirectional models transformers halves
// the configured value and adds one (config 512 → bound 257, so 256 tokens
// to each side, a 512-wide window); causal models use the value as is.
func (c Config) window() int {
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
