package tokenizer

import "unicode/utf8"

// trie matches added tokens against raw text, longest match first. Keys are
// stored by byte so matching does not allocate; only whole added tokens
// carry an id.
type trie struct {
	root *node
}

type node struct {
	children map[byte]*node
	id       int32
	term     bool
}

func newTrie() *trie { return &trie{root: &node{}} }

func (t *trie) insert(s string, id int32) {
	if s == "" {
		return
	}
	n := t.root
	for i := 0; i < len(s); i++ {
		if n.children == nil {
			n.children = map[byte]*node{}
		}
		c := n.children[s[i]]
		if c == nil {
			c = &node{}
			n.children[s[i]] = c
		}
		n = c
	}
	n.term = true
	n.id = id
}

// longest returns the id and byte length of the longest added token that is
// a prefix of s, and whether one was found. A match must end on a UTF-8
// boundary, which it always does because added tokens are whole strings.
func (t *trie) longest(s string) (int32, int, bool) {
	n := t.root
	bestID := int32(0)
	bestLen := 0
	found := false
	for i := 0; i < len(s); i++ {
		if n.children == nil {
			break
		}
		c := n.children[s[i]]
		if c == nil {
			break
		}
		n = c
		if n.term {
			bestID, bestLen, found = n.id, i+1, true
		}
	}
	if found && bestLen < len(s) {
		// Guard against splitting a multi-byte rune: the next byte must
		// start a new rune. Added tokens are whole strings, so a match
		// that stops mid-rune would be a false positive.
		if !utf8.RuneStart(s[bestLen]) {
			// walk back is unnecessary for the Gemma tokens; treat as no
			// match to stay safe.
			return 0, 0, false
		}
	}
	return bestID, bestLen, found
}
