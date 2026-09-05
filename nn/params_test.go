package nn

import (
	"bytes"
	"testing"

	"github.com/fiber/ai/tensor"
)

func TestSaveLoadParamsRoundTrip(t *testing.T) {
	tensor.Seed(11)
	m := Sequential{NewLinear(4, 8), ReLU{}, NewLayerNorm(8), NewLinear(8, 3)}
	x := tensor.Randn(5, 4)
	var want *tensor.Tensor
	tensor.NoGrad(func() { want = m.Forward(x) })

	var buf bytes.Buffer
	if err := SaveParams(&buf, m); err != nil {
		t.Fatal(err)
	}
	fresh := Sequential{NewLinear(4, 8), ReLU{}, NewLayerNorm(8), NewLinear(8, 3)}
	if err := LoadParams(bytes.NewReader(buf.Bytes()), fresh); err != nil {
		t.Fatal(err)
	}
	var got *tensor.Tensor
	tensor.NoGrad(func() { got = fresh.Forward(x) })
	if !got.Equal(want) {
		t.Fatalf("loaded model differs:\n%v\n%v", got, want)
	}

	other := Sequential{NewLinear(4, 9), ReLU{}, NewLinear(9, 3)}
	if err := LoadParams(bytes.NewReader(buf.Bytes()), other); err == nil {
		t.Fatal("architecture mismatch not reported")
	}
	if err := LoadParams(bytes.NewReader([]byte("nope")), fresh); err == nil {
		t.Fatal("garbage accepted")
	}
}
