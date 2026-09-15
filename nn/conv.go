package nn

import (
	"math"

	"github.com/fiber/ai/tensor"
)

// Conv2D is a 2-D convolution layer over [batch, channels, height, width].
type Conv2D struct {
	W           *tensor.Tensor // [out, in, k, k]
	B           *tensor.Tensor // [out] or nil
	Stride, Pad int
}

// NewConv2D creates out filters of size k×k over in channels, He-initialised,
// with a bias; Stride 1 and Pad k/2 keep the spatial size for odd k.
func NewConv2D(in, out, k int) *Conv2D {
	s := float32(math.Sqrt(2 / float64(in*k*k)))
	return &Conv2D{W: tensor.Randn(out, in, k, k).MulScalar(s).SetRequiresGrad(true),
		B: tensor.Zeros(out).SetRequiresGrad(true), Stride: 1, Pad: k / 2}
}

func (c *Conv2D) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.Conv2D(x, c.W, c.B, c.Stride, c.Pad)
}

func (c *Conv2D) Params() []*tensor.Tensor {
	if c.B == nil {
		return []*tensor.Tensor{c.W}
	}
	return []*tensor.Tensor{c.W, c.B}
}

// Conv1D is a 1-D convolution layer over [batch, channels, length].
type Conv1D struct {
	W           *tensor.Tensor // [out, in, k]
	B           *tensor.Tensor // [out] or nil
	Stride, Pad int
}

// NewConv1D creates out filters of width k over in channels, He-initialised,
// with a bias; Stride 1 and Pad k/2 keep the length for odd k.
func NewConv1D(in, out, k int) *Conv1D {
	s := float32(math.Sqrt(2 / float64(in*k)))
	return &Conv1D{W: tensor.Randn(out, in, k).MulScalar(s).SetRequiresGrad(true),
		B: tensor.Zeros(out).SetRequiresGrad(true), Stride: 1, Pad: k / 2}
}

func (c *Conv1D) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.Conv1D(x, c.W, c.B, c.Stride, c.Pad)
}

func (c *Conv1D) Params() []*tensor.Tensor {
	if c.B == nil {
		return []*tensor.Tensor{c.W}
	}
	return []*tensor.Tensor{c.W, c.B}
}

// MaxPool2D halves (by default) the spatial size by taking window maxima.
type MaxPool2D struct{ K, Stride int }

// NewMaxPool2D pools k×k windows with stride k.
func NewMaxPool2D(k int) MaxPool2D { return MaxPool2D{K: k, Stride: k} }

func (p MaxPool2D) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.MaxPool2D(x, p.K, p.Stride)
}
func (MaxPool2D) Params() []*tensor.Tensor { return nil }

// GlobalAvgPool1D averages over the length of [batch, channels, length],
// giving [batch, channels]: the usual end of a 1-D convolutional encoder,
// where Flatten would tie the model to one input length.
type GlobalAvgPool1D struct{}

func (GlobalAvgPool1D) Forward(x *tensor.Tensor) *tensor.Tensor {
	return globalPool1D("GlobalAvgPool1D", x, (*tensor.Tensor).Mean)
}
func (GlobalAvgPool1D) Params() []*tensor.Tensor { return nil }

// GlobalMaxPool1D takes the maximum over the length of
// [batch, channels, length], giving [batch, channels]. Where the average
// asks how much of a pattern there is overall, the maximum asks whether
// it occurred at all.
type GlobalMaxPool1D struct{}

func (GlobalMaxPool1D) Forward(x *tensor.Tensor) *tensor.Tensor {
	return globalPool1D("GlobalMaxPool1D", x, (*tensor.Tensor).Max)
}
func (GlobalMaxPool1D) Params() []*tensor.Tensor { return nil }

func globalPool1D(op string, x *tensor.Tensor, reduce func(*tensor.Tensor, ...int) *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape()) != 3 {
		panic("nn: " + op + " expects [batch, channels, length] inputs")
	}
	return reduce(x, 2)
}
