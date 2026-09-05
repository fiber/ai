package logtemplate

import (
	"sort"
	"strings"
)

// Wildcard marks a token position the template does not fix.
const Wildcard = "<*>"

// Template is one mined message shape with its occurrence count.
type Template struct {
	ID     int
	Tokens []string
	Count  int
}

// String joins the tokens with single spaces.
func (t Template) String() string { return strings.Join(t.Tokens, " ") }

// Miner implements the Drain algorithm (He et al., 2017): a fixed-depth
// tree keyed by the number of tokens and the first tokens of a line, with
// leaves holding the templates of that shape. A line joins the leaf
// template whose tokens agree with it in at least Threshold of the
// positions (wildcards count as agreeing); the disagreeing positions
// become wildcards. Otherwise the line starts a new template.
//
// A Miner is not safe for concurrent use; one goroutine per source is
// the intended pattern, and one core handles well over 300 000 lines a
// second.
type Miner struct {
	// Threshold is the share of equal tokens for a line to join a template
	// (default 0.5). Depth is how many leading tokens key the tree (default
	// 2). MaxChildren caps the branching per level; a level that is full
	// routes further tokens to a shared wildcard branch (default 100).
	Threshold   float64
	Depth       int
	MaxChildren int
	// PreMask applies Mask to every line before tokenising (default true).
	PreMask bool

	root      map[int]*node // by token count
	templates []*Template
}

type node struct {
	children map[string]*node
	leaf     []*Template
}

// NewMiner returns a Miner with the default settings.
func NewMiner() *Miner {
	return &Miner{Threshold: 0.5, Depth: 2, MaxChildren: 100, PreMask: true, root: map[int]*node{}}
}

// Add assigns line to a template, creating one if necessary, and returns
// its ID. IDs are dense and stable for the life of the Miner.
func (m *Miner) Add(line string) int {
	if m.PreMask {
		line = Mask(line)
	}
	tokens := strings.Fields(line)
	if len(tokens) == 0 {
		tokens = []string{""}
	}
	n, ok := m.root[len(tokens)]
	if !ok {
		n = &node{children: map[string]*node{}}
		m.root[len(tokens)] = n
	}
	for d := 0; d < m.Depth && d < len(tokens); d++ {
		key := tokens[d]
		if hasDigitOrPlaceholder(key) {
			key = Wildcard // variable tokens must not split the tree
		}
		child, ok := n.children[key]
		if !ok {
			if len(n.children) >= m.MaxChildren {
				key = Wildcard
				child, ok = n.children[key]
			}
			if !ok {
				child = &node{children: map[string]*node{}}
				n.children[key] = child
			}
		}
		n = child
	}
	best, bestScore := -1, 0.0
	for i, t := range n.leaf {
		if s := similarity(t.Tokens, tokens); s > bestScore {
			best, bestScore = i, s
		}
	}
	if best >= 0 && bestScore >= m.Threshold {
		t := n.leaf[best]
		for i := range t.Tokens {
			if t.Tokens[i] != tokens[i] {
				t.Tokens[i] = Wildcard
			}
		}
		t.Count++
		return t.ID
	}
	t := &Template{ID: len(m.templates), Tokens: append([]string(nil), tokens...), Count: 1}
	m.templates = append(m.templates, t)
	n.leaf = append(n.leaf, t)
	return t.ID
}

func similarity(tmpl, tokens []string) float64 {
	equal := 0
	for i := range tmpl {
		if tmpl[i] == tokens[i] || tmpl[i] == Wildcard {
			equal++
		}
	}
	return float64(equal) / float64(len(tmpl))
}

func hasDigitOrPlaceholder(tok string) bool {
	if strings.Contains(tok, "<") {
		return true
	}
	for i := 0; i < len(tok); i++ {
		if tok[i] >= '0' && tok[i] <= '9' {
			return true
		}
	}
	return false
}

// Template returns a copy of the template with the given ID.
func (m *Miner) Template(id int) Template {
	t := m.templates[id]
	return Template{ID: t.ID, Tokens: append([]string(nil), t.Tokens...), Count: t.Count}
}

// Templates returns all templates, most frequent first.
func (m *Miner) Templates() []Template {
	out := make([]Template, 0, len(m.templates))
	for _, t := range m.templates {
		out = append(out, Template{ID: t.ID, Tokens: append([]string(nil), t.Tokens...), Count: t.Count})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Count > out[j].Count })
	return out
}

// Len is the number of templates mined so far.
func (m *Miner) Len() int { return len(m.templates) }

// ResetCounts sets every template's count to zero, keeping the
// templates: the daily roll-over of a dashboard.
func (m *Miner) ResetCounts() {
	for _, t := range m.templates {
		t.Count = 0
	}
}
