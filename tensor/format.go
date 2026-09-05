package tensor

import (
	"fmt"
	"strconv"
	"strings"
)

// PrintOptions controls String formatting.
var PrintOptions = struct {
	// Precision is the number of significant digits.
	Precision int
	// EdgeItems is how many leading/trailing entries of a dimension longer
	// than Threshold are shown.
	EdgeItems int
	// Threshold is the dimension size above which entries are elided.
	Threshold int
}{Precision: 4, EdgeItems: 3, Threshold: 8}

// String renders the tensor in a NumPy-like nested layout, e.g.
//
//	[[1 2 3]
//	 [4 5 6]]
func (t *Tensor) String() string {
	var b strings.Builder
	t.formatInto(&b)
	return b.String()
}

func (t *Tensor) formatInto(b *strings.Builder) {
	if t.size == 0 {
		fmt.Fprintf(b, "[] shape=%v", t.shape)
		return
	}
	if len(t.shape) == 0 {
		b.WriteString(formatValue(t.data[0]))
		return
	}
	width := 0
	tc := t.Contiguous()
	for _, v := range tc.data[:tc.size] {
		if w := len(formatValue(v)); w > width {
			width = w
		}
	}
	tc.formatDim(b, 0, 0, width, "")
}

func formatValue(v float32) string {
	return strconv.FormatFloat(float64(v), 'g', PrintOptions.Precision, 32)
}

// formatDim renders dimension d starting at storage offset off.
func (t *Tensor) formatDim(b *strings.Builder, d, off, width int, indent string) {
	n := t.shape[d]
	stride := t.strides[d]
	show := func(i int) bool {
		return n <= PrintOptions.Threshold || i < PrintOptions.EdgeItems || i >= n-PrintOptions.EdgeItems
	}
	b.WriteByte('[')
	if d == len(t.shape)-1 {
		elided := false
		for i := 0; i < n; i++ {
			if !show(i) {
				if !elided {
					b.WriteString(" ...")
					elided = true
				}
				continue
			}
			if i > 0 {
				b.WriteByte(' ')
			}
			s := formatValue(t.data[off+i*stride])
			b.WriteString(strings.Repeat(" ", width-len(s)))
			b.WriteString(s)
		}
		b.WriteByte(']')
		return
	}
	elided := false
	for i := 0; i < n; i++ {
		if !show(i) {
			if !elided {
				b.WriteString("\n" + indent + " ...")
				elided = true
			}
			continue
		}
		if i > 0 {
			b.WriteString("\n")
			if d < len(t.shape)-2 {
				b.WriteString("\n")
			}
			b.WriteString(indent + " ")
		}
		t.formatDim(b, d+1, off+i*stride, width, indent+" ")
	}
	b.WriteByte(']')
}

// GoString returns a debugging representation including shape and flags.
func (t *Tensor) GoString() string {
	var b strings.Builder
	fmt.Fprintf(&b, "tensor.Tensor{shape: %v", t.shape)
	if !t.IsContiguous() {
		fmt.Fprintf(&b, ", strides: %v", t.strides)
	}
	if t.requiresGrad {
		b.WriteString(", requiresGrad")
	}
	if t.node != nil {
		fmt.Fprintf(&b, ", op: %s", t.node.op)
	}
	b.WriteString("}")
	return b.String()
}
