// Package tokenizer implements the Hugging Face byte-level BPE tokenizer as
// used by Gemma-class models, read from a tokenizer.json file. It supports
// the pieces the Gemma files use and no more: a byte-fallback BPE model, a
// space-to-metaspace normalizer, added-token extraction and a BOS/EOS
// template. Anything else in the file is refused at load time rather than
// silently ignored, so a mismatch surfaces as an error and not as wrong
// token ids.
package tokenizer

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode/utf8"
)

// Tokenizer encodes text to token ids and back.
type Tokenizer struct {
	vocab    map[string]int32
	pieces   []string // id -> piece, index is the id
	merges   map[[2]int32]mergeInfo
	added    *trie
	addedIDs map[int32]string // added-token id -> content, for decoding

	bos, eos int32
	addBOS   bool
	addEOS   bool

	// normalizer: replace every occurrence of from with to.
	normFrom string
	normTo   string

	byteFallback bool
	byteTok      [256]int32 // id of "<0xNN>" for each byte value
	isByte       map[int32]byte
}

type mergeInfo struct {
	rank   int32
	merged int32
}

// Load reads a tokenizer.json file.
func Load(path string) (*Tokenizer, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parse(b)
}

type jsonFile struct {
	AddedTokens []struct {
		ID         int32  `json:"id"`
		Content    string `json:"content"`
		Special    bool   `json:"special"`
		Normalized bool   `json:"normalized"`
	} `json:"added_tokens"`
	Normalizer    json.RawMessage `json:"normalizer"`
	PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
	PostProcessor struct {
		Type   string `json:"type"`
		Single []map[string]struct {
			ID string `json:"id"`
		} `json:"single"`
	} `json:"post_processor"`
	Model struct {
		Type         string           `json:"type"`
		UnkToken     string           `json:"unk_token"`
		ByteFallback bool             `json:"byte_fallback"`
		IgnoreMerges bool             `json:"ignore_merges"`
		Vocab        map[string]int32 `json:"vocab"`
		Merges       json.RawMessage  `json:"merges"`
	} `json:"model"`
}

func parse(b []byte) (*Tokenizer, error) {
	var f jsonFile
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}
	if f.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer: model type %q not supported (want BPE)", f.Model.Type)
	}
	t := &Tokenizer{
		vocab:        f.Model.Vocab,
		merges:       make(map[[2]int32]mergeInfo, 1<<19),
		added:        newTrie(),
		addedIDs:     map[int32]string{},
		byteFallback: f.Model.ByteFallback,
		isByte:       map[int32]byte{},
		bos:          -1,
		eos:          -1,
	}
	if len(t.vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: empty vocab")
	}

	// id -> piece
	maxID := int32(-1)
	for _, id := range t.vocab {
		if id > maxID {
			maxID = id
		}
	}
	t.pieces = make([]string, maxID+1)
	for p, id := range t.vocab {
		t.pieces[id] = p
	}

	if err := t.parseNormalizer(f.Normalizer); err != nil {
		return nil, err
	}
	if err := t.parsePreTokenizer(f.PreTokenizer); err != nil {
		return nil, err
	}
	if err := t.parseMerges(f.Model.Merges); err != nil {
		return nil, err
	}

	// Byte-fallback tokens.
	if t.byteFallback {
		for b := 0; b < 256; b++ {
			key := fmt.Sprintf("<0x%02X>", b)
			id, ok := t.vocab[key]
			if !ok {
				return nil, fmt.Errorf("tokenizer: byte_fallback set but %s is missing from the vocab", key)
			}
			t.byteTok[b] = id
			t.isByte[id] = byte(b)
		}
	}

	// Added tokens: matched literally against raw text, longest first.
	for _, a := range f.AddedTokens {
		if a.Normalized {
			return nil, fmt.Errorf("tokenizer: added token %q is normalized; only raw added tokens are supported", a.Content)
		}
		t.added.insert(a.Content, a.ID)
		t.addedIDs[a.ID] = a.Content
	}

	// Post-processor: read the BOS/EOS template.
	if f.PostProcessor.Type != "" && f.PostProcessor.Type != "TemplateProcessing" {
		return nil, fmt.Errorf("tokenizer: post-processor %q not supported", f.PostProcessor.Type)
	}
	// The single-sequence template is [BOS] A [EOS]: the special token
	// before the "A" sequence is the prefix, the one after is the suffix.
	seenSeq := false
	for _, entry := range f.PostProcessor.Single {
		if _, ok := entry["Sequence"]; ok {
			seenSeq = true
			continue
		}
		st, ok := entry["SpecialToken"]
		if !ok {
			continue
		}
		id, ok := t.vocab[st.ID]
		if !ok {
			return nil, fmt.Errorf("tokenizer: template special token %q not in vocab", st.ID)
		}
		if !seenSeq {
			t.bos, t.addBOS = id, true
		} else {
			t.eos, t.addEOS = id, true
		}
	}
	return t, nil
}

func (t *Tokenizer) parseNormalizer(raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var n struct {
		Type    string `json:"type"`
		Pattern struct {
			String string `json:"String"`
		} `json:"pattern"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &n); err != nil {
		return fmt.Errorf("tokenizer: normalizer: %w", err)
	}
	if n.Type != "Replace" {
		return fmt.Errorf("tokenizer: normalizer %q not supported (want Replace)", n.Type)
	}
	t.normFrom, t.normTo = n.Pattern.String, n.Content
	return nil
}

func (t *Tokenizer) parsePreTokenizer(raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var p struct {
		Type    string `json:"type"`
		Pattern struct {
			String string `json:"String"`
		} `json:"pattern"`
	}
	if err := json.Unmarshal(raw, &p); err != nil {
		return fmt.Errorf("tokenizer: pre_tokenizer: %w", err)
	}
	// The Gemma file splits on a literal space. Because the normalizer has
	// already turned every space into the metaspace character, this split
	// never fires and BPE runs over the whole chunk. Accept it and rely on
	// that; refuse anything else so a file that needs real pre-tokenization
	// does not silently tokenize wrong.
	switch p.Type {
	case "Split":
		if p.Pattern.String != " " {
			return fmt.Errorf("tokenizer: pre_tokenizer Split on %q not supported", p.Pattern.String)
		}
	case "Metaspace", "":
		// nothing to do
	default:
		return fmt.Errorf("tokenizer: pre_tokenizer %q not supported", p.Type)
	}
	return nil
}

func (t *Tokenizer) parseMerges(raw json.RawMessage) error {
	if len(raw) == 0 {
		return fmt.Errorf("tokenizer: no merges")
	}
	// Merges are either ["a b", ...] (older) or [["a","b"], ...] (newer).
	var pairs [][2]string
	if raw[0] == '[' {
		// Peek at the first element.
		var probe []json.RawMessage
		if err := json.Unmarshal(raw, &probe); err != nil {
			return fmt.Errorf("tokenizer: merges: %w", err)
		}
		if len(probe) > 0 && len(probe[0]) > 0 && probe[0][0] == '[' {
			if err := json.Unmarshal(raw, &pairs); err != nil {
				return fmt.Errorf("tokenizer: merges: %w", err)
			}
		} else {
			var lines []string
			if err := json.Unmarshal(raw, &lines); err != nil {
				return fmt.Errorf("tokenizer: merges: %w", err)
			}
			for _, l := range lines {
				sp := strings.SplitN(l, " ", 2)
				if len(sp) != 2 {
					return fmt.Errorf("tokenizer: bad merge line %q", l)
				}
				pairs = append(pairs, [2]string{sp[0], sp[1]})
			}
		}
	}
	for rank, p := range pairs {
		l, ok1 := t.vocab[p[0]]
		r, ok2 := t.vocab[p[1]]
		m, ok3 := t.vocab[p[0]+p[1]]
		if !ok1 || !ok2 || !ok3 {
			// A merge referencing a piece outside the vocab cannot fire;
			// skip it rather than fail, matching HF's tolerance.
			continue
		}
		key := [2]int32{l, r}
		if _, exists := t.merges[key]; !exists {
			t.merges[key] = mergeInfo{rank: int32(rank), merged: m}
		}
	}
	return nil
}

// VocabSize returns the number of ids (highest id + 1).
func (t *Tokenizer) VocabSize() int { return len(t.pieces) }

// Piece returns the string for an id, or "" if the id is out of range.
func (t *Tokenizer) Piece(id int) string {
	if id < 0 || id >= len(t.pieces) {
		return ""
	}
	return t.pieces[id]
}

// ID returns the id of a piece.
func (t *Tokenizer) ID(piece string) (int, bool) {
	id, ok := t.vocab[piece]
	return int(id), ok
}

// BOS and EOS return the template's special ids, or -1 when the template
// does not add them.
func (t *Tokenizer) BOS() int { return int(t.bos) }
func (t *Tokenizer) EOS() int { return int(t.eos) }

// Encode tokenizes text with the file's BOS/EOS template applied.
func (t *Tokenizer) Encode(text string) []int {
	ids := t.encode(text)
	out := make([]int, 0, len(ids)+2)
	if t.addBOS {
		out = append(out, int(t.bos))
	}
	for _, id := range ids {
		out = append(out, int(id))
	}
	if t.addEOS {
		out = append(out, int(t.eos))
	}
	return out
}

// EncodeRaw tokenizes text without the BOS/EOS template.
func (t *Tokenizer) EncodeRaw(text string) []int {
	ids := t.encode(text)
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out
}

// encode extracts added tokens from the raw text, then normalizes and runs
// BPE on the chunks between them.
func (t *Tokenizer) encode(text string) []int32 {
	var out []int32
	var buf strings.Builder
	flush := func() {
		if buf.Len() == 0 {
			return
		}
		out = t.bpe(t.normalize(buf.String()), out)
		buf.Reset()
	}
	for i := 0; i < len(text); {
		if id, n, ok := t.added.longest(text[i:]); ok {
			flush()
			out = append(out, id)
			i += n
			continue
		}
		_, sz := utf8.DecodeRuneInString(text[i:])
		buf.WriteString(text[i : i+sz])
		i += sz
	}
	flush()
	return out
}

func (t *Tokenizer) normalize(s string) string {
	if t.normFrom == "" {
		return s
	}
	return strings.ReplaceAll(s, t.normFrom, t.normTo)
}

// symbol is one node of the BPE working list, a doubly linked list over a
// backing array so merges are O(1) and candidates index by position.
type symbol struct {
	id         int32
	prev, next int32
	alive      bool
}

type candidate struct {
	rank int32
	pos  int32
	l, r int32
}

type candHeap []candidate

func (h candHeap) Len() int { return len(h) }
func (h candHeap) Less(i, j int) bool {
	if h[i].rank != h[j].rank {
		return h[i].rank < h[j].rank
	}
	return h[i].pos < h[j].pos // stable: earliest position wins ties
}
func (h candHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *candHeap) Push(x any)   { *h = append(*h, x.(candidate)) }
func (h *candHeap) Pop() any     { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

// bpe tokenizes one normalized chunk and appends the ids to out.
func (t *Tokenizer) bpe(chunk string, out []int32) []int32 {
	if chunk == "" {
		return out
	}
	// Initial symbols: each rune's piece if present, else byte fallback.
	syms := make([]symbol, 0, len(chunk))
	push := func(id int32) {
		syms = append(syms, symbol{id: id, alive: true})
	}
	for _, r := range chunk {
		if id, ok := t.vocab[string(r)]; ok {
			push(id)
			continue
		}
		if t.byteFallback {
			var b [4]byte
			n := utf8.EncodeRune(b[:], r)
			for k := 0; k < n; k++ {
				push(t.byteTok[b[k]])
			}
			continue
		}
		// No fallback: emit unk if we have one.
		if id, ok := t.vocab["<unk>"]; ok {
			push(id)
		}
	}
	n := len(syms)
	if n == 0 {
		return out
	}
	for i := range syms {
		syms[i].prev = int32(i - 1)
		syms[i].next = int32(i + 1)
	}
	syms[n-1].next = -1

	h := &candHeap{}
	add := func(pos int32) {
		nx := syms[pos].next
		if nx < 0 {
			return
		}
		if m, ok := t.merges[[2]int32{syms[pos].id, syms[nx].id}]; ok {
			heap.Push(h, candidate{rank: m.rank, pos: pos, l: syms[pos].id, r: syms[nx].id})
		}
	}
	for i := int32(0); i < int32(n); i++ {
		add(i)
	}
	for h.Len() > 0 {
		c := heap.Pop(h).(candidate)
		s := &syms[c.pos]
		if !s.alive || s.id != c.l {
			continue
		}
		j := s.next
		if j < 0 || !syms[j].alive || syms[j].id != c.r {
			continue
		}
		m := t.merges[[2]int32{c.l, c.r}]
		s.id = m.merged
		// unlink j
		nj := syms[j].next
		s.next = nj
		if nj >= 0 {
			syms[nj].prev = c.pos
		}
		syms[j].alive = false
		// new candidates around the merged symbol
		add(c.pos)
		if p := s.prev; p >= 0 {
			add(p)
		}
	}
	for i := int32(0); i >= 0; i = syms[i].next {
		out = append(out, syms[i].id)
		if syms[i].next < 0 {
			break
		}
	}
	return out
}

// Decode turns ids back into text: byte-fallback tokens become their bytes,
// the metaspace character becomes a space, and added tokens render as their
// literal content.
func (t *Tokenizer) Decode(ids []int) string {
	var b []byte
	for _, id := range ids {
		i32 := int32(id)
		if bv, ok := t.isByte[i32]; ok {
			b = append(b, bv)
			continue
		}
		p := t.Piece(id)
		if t.normTo != "" {
			p = strings.ReplaceAll(p, t.normTo, t.normFrom)
		}
		b = append(b, p...)
	}
	return string(b)
}
