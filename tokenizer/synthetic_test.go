package tokenizer

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// buildTokenizer writes a minimal but complete tokenizer.json in memory: the
// 256 byte-fallback tokens, a handful of pieces, two merges, a space->metaspace
// normalizer and a [BOS] A [EOS] template. It exercises every path without the
// gated model download.
func buildTokenizer(t *testing.T, extra map[string]any) *Tokenizer {
	t.Helper()
	vocab := map[string]int{
		"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
		"▁": 4, "a": 5, "b": 6, "c": 7, "ab": 8, "abc": 9, "▁a": 10,
	}
	next := 11
	for b := 0; b < 256; b++ {
		vocab[fmt.Sprintf("<0x%02X>", b)] = next
		next++
	}
	doc := map[string]any{
		"added_tokens": []map[string]any{
			{"id": 2, "content": "<bos>", "special": true, "normalized": false},
			{"id": 1, "content": "<eos>", "special": true, "normalized": false},
			{"id": 0, "content": "<pad>", "special": true, "normalized": false},
		},
		"normalizer":    map[string]any{"type": "Replace", "pattern": map[string]any{"String": " "}, "content": "▁"},
		"pre_tokenizer": map[string]any{"type": "Split", "pattern": map[string]any{"String": " "}, "behavior": "MergedWithPrevious"},
		"post_processor": map[string]any{
			"type": "TemplateProcessing",
			"single": []map[string]any{
				{"SpecialToken": map[string]any{"id": "<bos>", "type_id": 0}},
				{"Sequence": map[string]any{"id": "A", "type_id": 0}},
				{"SpecialToken": map[string]any{"id": "<eos>", "type_id": 0}},
			},
		},
		"model": map[string]any{
			"type":          "BPE",
			"unk_token":     "<unk>",
			"byte_fallback": true,
			"ignore_merges": false,
			"vocab":         vocab,
			"merges":        [][2]string{{"a", "b"}, {"ab", "c"}, {"▁", "a"}},
		},
	}
	for k, v := range extra {
		doc[k] = v
	}
	b, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := parse(b)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestSyntheticBPE(t *testing.T) {
	tok := buildTokenizer(t, nil)
	cases := []struct {
		in  string
		raw []int
	}{
		{"abc", []int{9}},                  // a+b->ab, ab+c->abc
		{"ab", []int{8}},                   // a+b->ab
		{"ba", []int{6, 5}},                // no merge for b,a
		{"a", []int{5}},                    // single piece
		{" a", []int{10}},                  // space->metaspace, "▁a" is a piece
		{"", nil},                          // empty
		{"Z", []int{11 + 'Z'}},             // byte fallback for an unknown rune
		{"é", []int{11 + 0xC3, 11 + 0xA9}}, // 'é' -> two utf-8 bytes
	}
	for _, c := range cases {
		if got := tok.EncodeRaw(c.in); !equal(got, c.raw) {
			t.Errorf("EncodeRaw(%q) = %v, want %v", c.in, got, c.raw)
		}
	}
	// Template wraps with BOS/EOS.
	if got := tok.Encode("abc"); !equal(got, []int{2, 9, 1}) {
		t.Errorf("Encode(abc) = %v, want [2 9 1]", got)
	}
	// Added tokens are extracted from raw text and split the BPE chunks.
	if got := tok.EncodeRaw("a<bos>b"); !equal(got, []int{5, 2, 6}) {
		t.Errorf("added split = %v, want [5 2 6]", got)
	}
	// Decode round-trips text without a literal metaspace.
	if got := tok.Decode(tok.EncodeRaw("abc é")); got != "abc é" {
		t.Errorf("decode = %q", got)
	}
	if tok.BOS() != 2 || tok.EOS() != 1 {
		t.Errorf("bos/eos %d/%d", tok.BOS(), tok.EOS())
	}
}

func TestLoadErrors(t *testing.T) {
	cases := map[string]struct {
		doc  map[string]any
		want string
	}{
		"model type": {map[string]any{"model": map[string]any{"type": "WordPiece", "vocab": map[string]int{"a": 0}}}, "model type"},
		"normalizer": {map[string]any{
			"model":      map[string]any{"type": "BPE", "byte_fallback": false, "vocab": map[string]int{"a": 0}, "merges": [][2]string{}},
			"normalizer": map[string]any{"type": "NFKC"},
		}, "normalizer"},
		"pre_tokenizer": {map[string]any{
			"model":         map[string]any{"type": "BPE", "byte_fallback": false, "vocab": map[string]int{"a": 0}, "merges": [][2]string{}},
			"pre_tokenizer": map[string]any{"type": "ByteLevel"},
		}, "pre_tokenizer"},
		"missing byte token": {map[string]any{
			"model": map[string]any{"type": "BPE", "byte_fallback": true, "vocab": map[string]int{"a": 0}, "merges": [][2]string{}},
		}, "byte_fallback"},
		"empty vocab": {map[string]any{
			"model": map[string]any{"type": "BPE", "vocab": map[string]int{}, "merges": [][2]string{}},
		}, "empty vocab"},
	}
	for name, c := range cases {
		b, _ := json.Marshal(c.doc)
		_, err := parse(b)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want %q", name, err, c.want)
		}
	}
}
