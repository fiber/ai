package nn

import (
	"math"
	"testing"

	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

func TestGlobalPool1D(t *testing.T) {
	// [1, 2, 3]: two channels of three steps each.
	x := tensor.New([]float32{1, 2, 6, -1, -4, 0}, 1, 2, 3).SetRequiresGrad(true)
	avg := GlobalAvgPool1D{}.Forward(x)
	if !avg.Shape().Equal(tensor.Shape{1, 2}) {
		t.Fatalf("average shape %v, want [1 2]", avg.Shape())
	}
	if got := avg.Float32s(); !approxAll(got, []float32{3, -5.0 / 3}) {
		t.Errorf("average = %v, want [3 -1.667]", got)
	}
	mx := GlobalMaxPool1D{}.Forward(x)
	if got := mx.Float32s(); !approxAll(got, []float32{6, 0}) {
		t.Errorf("maximum = %v, want [6 0]", got)
	}

	// The average spreads its gradient over the length, the maximum
	// gives all of it to the winning step.
	x.ZeroGrad()
	GlobalAvgPool1D{}.Forward(x).Sum().Backward()
	if got := x.Grad().Float32s(); !approxAll(got, []float32{1. / 3, 1. / 3, 1. / 3, 1. / 3, 1. / 3, 1. / 3}) {
		t.Errorf("average gradient = %v", got)
	}
	x.ZeroGrad()
	GlobalMaxPool1D{}.Forward(x).Sum().Backward()
	if got := x.Grad().Float32s(); !approxAll(got, []float32{0, 0, 1, 0, 0, 1}) {
		t.Errorf("maximum gradient = %v", got)
	}
}

// The layer's reason to exist: a Conv1D encoder that ends in a Linear
// without Flatten, so the model does not depend on the input length.
func TestGlobalPoolInSequential(t *testing.T) {
	tensor.Seed(4)
	m := Sequential{NewConv1D(2, 4, 3), ReLU{}, GlobalAvgPool1D{}, NewLinear(4, 1)}
	x, y := tensor.Randn(8, 2, 16), tensor.Randn(8, 1)
	opt := optim.NewAdamW(m.Params(), 1e-2, 0)
	first := tensor.MSELoss(m.Forward(x), y).Item()
	var last float32
	for range 50 {
		loss := tensor.MSELoss(m.Forward(x), y)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
		last = loss.Item()
	}
	if !(last < first) {
		t.Errorf("loss %v did not fall below the initial %v", last, first)
	}
	// The same model on a different length: the point of global pooling.
	if out := m.Forward(tensor.Randn(3, 2, 40)); !out.Shape().Equal(tensor.Shape{3, 1}) {
		t.Errorf("shape %v on a longer input, want [3 1]", out.Shape())
	}
}

func TestGlobalPoolRejectsRank(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("no panic on a 2-D input")
		}
	}()
	GlobalAvgPool1D{}.Forward(tensor.Zeros(2, 3))
}

func TestSnapshotRoundTrip(t *testing.T) {
	tensor.Seed(9)
	m := Sequential{NewLinear(3, 5), Tanh{}, NewLinear(5, 2)}
	best := NewSnapshot(m)
	want := make([][]float32, 0, len(m.Params()))
	for _, p := range m.Params() {
		want = append(want, p.Float32s())
	}

	// Train away from the captured state, then come back to it.
	x, y := tensor.Randn(16, 3), tensor.Randn(16, 2)
	opt := optim.NewAdamW(m.Params(), 1e-1, 0)
	for range 20 {
		loss := tensor.MSELoss(m.Forward(x), y)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	moved := false
	for i, p := range m.Params() {
		for j, v := range p.Float32s() {
			moved = moved || v != want[i][j]
		}
	}
	if !moved {
		t.Fatal("training did not change the parameters, the test proves nothing")
	}
	if err := best.Restore(m); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	for i, p := range m.Params() {
		for j, v := range p.Float32s() {
			if v != want[i][j] {
				t.Fatalf("parameter %d[%d] = %v after restore, want %v", i, j, v, want[i][j])
			}
		}
	}

	// The optimiser still holds the same tensors, so training continues.
	before := tensor.MSELoss(m.Forward(x), y).Item()
	for range 20 {
		loss := tensor.MSELoss(m.Forward(x), y)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	if after := tensor.MSELoss(m.Forward(x), y).Item(); !(after < before) {
		t.Errorf("loss %v after restoring and training on, was %v", after, before)
	}
}

// What Capture allocates must not grow with the model: the parameters
// go into buffers the snapshot already holds, and only the model's own
// Params() slices are built per call, one per module. Without that,
// capturing every epoch of a long run would allocate a copy of the whole
// model every time.
func TestSnapshotCaptureAllocationIsConstant(t *testing.T) {
	tensor.Seed(2)
	small := Sequential{NewLinear(8, 8), NewLinear(8, 8)}
	big := Sequential{NewLinear(256, 256), NewLinear(256, 256)}
	ss, bs := NewSnapshot(small), NewSnapshot(big)
	a := testing.AllocsPerRun(20, func() { _ = ss.Capture(small) })
	b := testing.AllocsPerRun(20, func() { _ = bs.Capture(big) })
	if a != b {
		t.Errorf("capturing the small model allocates %v times, the 1000x larger one %v", a, b)
	}
	// What is left is the model's own Params() slices, one per module.
	if a > 12 {
		t.Errorf("Capture allocates %v times for a two-layer model", a)
	}
}

func TestSnapshotMismatch(t *testing.T) {
	tensor.Seed(3)
	s := NewSnapshot(Sequential{NewLinear(3, 4)})
	if err := s.Restore(Sequential{NewLinear(3, 4), NewLinear(4, 2)}); err == nil {
		t.Error("restoring into a model with more parameters gave no error")
	}
	if err := s.Restore(Sequential{NewLinear(3, 5)}); err == nil {
		t.Error("restoring into a model with different shapes gave no error")
	}
	if err := s.Capture(Sequential{NewLinear(9, 9)}); err == nil {
		t.Error("capturing a different model gave no error")
	}
}

func approxAll(got, want []float32) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			return false
		}
	}
	return true
}
