package metrics

import (
	"math"
	"strings"
	"testing"

	"github.com/fiber/ai/tensor"
)

func TestConfusion(t *testing.T) {
	truth := []int{0, 0, 0, 1, 1, 2, 2, 2, 2, 2}
	pred := []int{0, 0, 1, 1, 1, 2, 2, 2, 0, 1}
	m := Confusion(pred, truth, 3)
	m.Labels = []string{"printer", "camera", "server"}
	if m.Total() != 10 || m.Counts[0][1] != 1 || m.Counts[2][0] != 1 {
		t.Fatalf("counts %v", m.Counts)
	}
	if math.Abs(m.Accuracy()-0.7) > 1e-9 {
		t.Fatalf("accuracy %v", m.Accuracy())
	}
	if math.Abs(m.Recall(2)-0.6) > 1e-9 || math.Abs(m.Precision(1)-0.5) > 1e-9 {
		t.Fatalf("recall(2) %v precision(1) %v", m.Recall(2), m.Precision(1))
	}
	if f := m.F1(0); math.Abs(f-2*(2.0/3)*(2.0/3)/(4.0/3)) > 1e-9 {
		t.Fatalf("f1(0) %v", f)
	}
	s := m.String()
	if !strings.Contains(s, "camera") || !strings.Contains(s, "70.0%") {
		t.Fatalf("table:\n%s", s)
	}
}

func TestRegressionErrors(t *testing.T) {
	p := tensor.New([]float32{1, 2, 3, 4}, 4, 1)
	y := tensor.New([]float32{1, 4, 3, 0}, 4, 1)
	if MAE(p, y) != 1.5 {
		t.Fatalf("MAE %v", MAE(p, y))
	}
	if r := RMSE(p, y); math.Abs(r-math.Sqrt(5)) > 1e-6 {
		t.Fatalf("RMSE %v", r)
	}
}
