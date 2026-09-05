// Package metrics judges predictions: a confusion matrix with per-class
// precision, recall and F1 for classifiers, and mean absolute / root mean
// squared error for regressions.
package metrics

import (
	"fmt"
	"math"
	"strings"

	"github.com/fiber/ai/tensor"
)

// Matrix is a confusion matrix: Counts[truth][predicted].
type Matrix struct {
	Counts [][]int
	Labels []string // optional names, one per class
}

// Confusion counts predicted against true classes over 0..classes-1.
func Confusion(pred, truth []int, classes int) Matrix {
	if len(pred) != len(truth) {
		panic("metrics: pred and truth have different lengths")
	}
	m := Matrix{Counts: make([][]int, classes)}
	for i := range m.Counts {
		m.Counts[i] = make([]int, classes)
	}
	for i := range pred {
		if pred[i] < 0 || pred[i] >= classes || truth[i] < 0 || truth[i] >= classes {
			panic(fmt.Sprintf("metrics: class out of range at %d: pred %d, truth %d", i, pred[i], truth[i]))
		}
		m.Counts[truth[i]][pred[i]]++
	}
	return m
}

// Total is the number of examples.
func (m Matrix) Total() int {
	n := 0
	for _, row := range m.Counts {
		for _, v := range row {
			n += v
		}
	}
	return n
}

// Accuracy is the share of examples on the diagonal.
func (m Matrix) Accuracy() float64 {
	n := m.Total()
	if n == 0 {
		return math.NaN()
	}
	d := 0
	for i := range m.Counts {
		d += m.Counts[i][i]
	}
	return float64(d) / float64(n)
}

// Precision of class c: of the examples predicted as c, how many are c.
func (m Matrix) Precision(c int) float64 {
	col := 0
	for i := range m.Counts {
		col += m.Counts[i][c]
	}
	if col == 0 {
		return math.NaN()
	}
	return float64(m.Counts[c][c]) / float64(col)
}

// Recall of class c: of the examples that are c, how many were found.
func (m Matrix) Recall(c int) float64 {
	row := 0
	for _, v := range m.Counts[c] {
		row += v
	}
	if row == 0 {
		return math.NaN()
	}
	return float64(m.Counts[c][c]) / float64(row)
}

// F1 of class c: the harmonic mean of precision and recall.
func (m Matrix) F1(c int) float64 {
	p, r := m.Precision(c), m.Recall(c)
	if p+r == 0 || math.IsNaN(p) || math.IsNaN(r) {
		return math.NaN()
	}
	return 2 * p * r / (p + r)
}

// String renders the matrix with per-class recall and precision.
func (m Matrix) String() string {
	k := len(m.Counts)
	name := func(i int) string {
		if i < len(m.Labels) {
			return m.Labels[i]
		}
		return fmt.Sprintf("class %d", i)
	}
	w := 8
	for i := 0; i < k; i++ {
		w = max(w, len(name(i))+1)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%*s", w, "truth \\ pred")
	for j := 0; j < k; j++ {
		fmt.Fprintf(&b, "%*s", w, name(j))
	}
	fmt.Fprintf(&b, "%*s\n", w, "recall")
	for i := 0; i < k; i++ {
		fmt.Fprintf(&b, "%*s", w, name(i))
		for j := 0; j < k; j++ {
			fmt.Fprintf(&b, "%*d", w, m.Counts[i][j])
		}
		fmt.Fprintf(&b, "%*.1f%%\n", w-1, 100*m.Recall(i))
	}
	fmt.Fprintf(&b, "%*s", w, "precision")
	for j := 0; j < k; j++ {
		fmt.Fprintf(&b, "%*.1f%%", w-1, 100*m.Precision(j))
	}
	fmt.Fprintf(&b, "%*.1f%%\n", w-1, 100*m.Accuracy())
	return b.String()
}

// MAE is the mean absolute difference between pred and truth.
func MAE(pred, truth *tensor.Tensor) float64 {
	var v float64
	tensor.NoGrad(func() { v = float64(pred.Sub(truth).Abs().Mean().Item()) })
	return v
}

// RMSE is the root of the mean squared difference between pred and truth.
func RMSE(pred, truth *tensor.Tensor) float64 {
	var v float64
	tensor.NoGrad(func() { v = math.Sqrt(float64(pred.Sub(truth).Square().Mean().Item())) })
	return v
}
