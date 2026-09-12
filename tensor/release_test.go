package tensor

import (
	"math"
	"testing"
)

// convStackGradients trains nothing; it builds the two-stage convolution
// whose backward exposed B-009 and returns a fingerprint of every
// gradient. The result must not depend on where the storage came from.
func convStackGradients(t *testing.T) (uint64, float64) {
	t.Helper()
	Seed(3)
	const batch = 32
	x := Randn(batch, 1, 28, 28)
	w1 := Randn(16, 1, 3, 3).SetRequiresGrad(true)
	b1 := Zeros(16).SetRequiresGrad(true)
	w2 := Randn(32, 16, 3, 3).SetRequiresGrad(true)
	b2 := Zeros(32).SetRequiresGrad(true)
	head := Randn(32*7*7, 10).SetRequiresGrad(true)

	h := Conv2D(x, w1, b1, 1, 1).ReLU()
	h = MaxPool2D(h, 2, 2)
	h = Conv2D(h, w2, b2, 1, 1).ReLU()
	h = MaxPool2D(h, 2, 2)
	targets := make([]int, batch)
	for i := range targets {
		targets[i] = i % 10
	}
	CrossEntropy(h.Reshape(batch, -1).MatMul(head), targets).Backward()

	var bits uint64
	var abs float64
	for _, p := range []*Tensor{w1, b1, w2, b2, head} {
		if p.Grad() == nil {
			t.Fatal("a parameter has no gradient")
		}
		for _, v := range p.Grad().Float32s() {
			bits = bits*1099511628211 ^ uint64(math.Float32bits(v))
			abs += math.Abs(float64(v))
		}
	}
	return bits, abs
}

func mappedLimit() (int, bool) {
	mapPool.mu.Lock()
	defer mapPool.mu.Unlock()
	return mapPool.limit, mapPool.enabled
}

// The im2col matrix of the first convolution has no backward node of its
// own — its input does not require a gradient — so Conv2D's Release used
// to hand the buffer back while the product's backward still needed it.
// Off-heap storage makes that visible: retained, the buffer is reused and
// the gradients come out wrong; unretained, it is unmapped and the kernel
// faults. All three settings must now agree bit for bit.
func TestGradientsIndependentOfStorage(t *testing.T) {
	limit, enabled := mappedLimit()
	defer func() {
		if enabled {
			SetMappedLimit(limit)
		} else {
			SetMappedLimit(-1)
		}
	}()

	cases := []struct {
		name  string
		bytes int
	}{
		{"off-heap, buffers retained", 512 << 20},
		{"off-heap, every free unmapped", 0},
		{"on the Go heap", -1},
	}
	var want uint64
	var wantAbs float64
	for i, c := range cases {
		SetMappedLimit(c.bytes)
		got, abs := convStackGradients(t)
		if i == 0 {
			want, wantAbs = got, abs
			continue
		}
		if got != want {
			t.Errorf("%s: gradients differ from the retained run (fnv %016x against %016x, sum|g| %.9g against %.9g)",
				c.name, got, want, abs, wantAbs)
		}
	}
}

// Release must look at whether a recorded node captured the tensor, not
// only at whether the tensor itself carries one.
func TestReleaseRefusesWhenANodeCapturedIt(t *testing.T) {
	w := Randn(4, 6).SetRequiresGrad(true)
	col := Randn(6, 512) // no gradient of its own, so no node of its own
	if col.node != nil || col.requiresGrad {
		t.Fatal("the operand under test must have neither a node nor a gradient")
	}
	prod := w.MatMul(col) // records a node that saved col for its backward
	if col.consumers != 1 {
		t.Fatalf("the product recorded %d consumers of the operand, want 1", col.consumers)
	}

	col.Release()
	if col.released {
		t.Fatal("Release freed storage that a backward closure still reads")
	}
	prod.Sum().Backward()
	if w.Grad() == nil {
		t.Fatal("no gradient reached the filters")
	}
}

// The inference path keeps its recycling: without autograd there is no
// node and no consumer, and Release must still work.
func TestReleaseStillFreesUnusedStorage(t *testing.T) {
	var x *Tensor
	NoGrad(func() {
		a := Randn(64, 512)
		b := Randn(512, 8)
		_ = a.MatMul(b)
		x = a
	})
	if x.consumers != 0 {
		t.Fatalf("grad mode is off, so nothing should have recorded a consumer, got %d", x.consumers)
	}
	x.Release()
	if !x.released {
		t.Error("Release refused a tensor no node captured")
	}
}
