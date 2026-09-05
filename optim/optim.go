// Package optim implements gradient-descent optimisers for tensor
// parameters: SGD with momentum and Adam/AdamW.
package optim

import (
	"math"

	"github.com/fiber/ai/internal/parallel"
	"github.com/fiber/ai/tensor"
)

// Optimizer updates parameters from their accumulated gradients.
type Optimizer interface {
	// Step applies one update using the parameters' current Grad.
	Step()
	// ZeroGrad clears the parameters' gradients.
	ZeroGrad()
}

func zeroGrad(params []*tensor.Tensor) {
	for _, p := range params {
		p.ZeroGrad()
	}
}

// SGD is stochastic gradient descent with optional momentum, Nesterov
// acceleration and L2 weight decay.
type SGD struct {
	Params      []*tensor.Tensor
	LR          float32
	Momentum    float32
	Nesterov    bool
	WeightDecay float32

	vel []*tensor.Tensor
}

// NewSGD returns plain SGD; set Momentum/WeightDecay on the result.
func NewSGD(params []*tensor.Tensor, lr float32) *SGD {
	return &SGD{Params: params, LR: lr}
}

func (s *SGD) ZeroGrad() { zeroGrad(s.Params) }

func (s *SGD) Step() {
	if s.Momentum != 0 && s.vel == nil {
		s.vel = make([]*tensor.Tensor, len(s.Params))
		for i, p := range s.Params {
			s.vel[i] = tensor.ZerosLike(p)
		}
	}
	tensor.NoGrad(func() {
		for i, p := range s.Params {
			g := p.Grad()
			if g == nil {
				continue
			}
			if s.WeightDecay != 0 {
				g = g.Add(p.MulScalar(s.WeightDecay))
			}
			if s.Momentum == 0 {
				p.AddScaledInPlace(g, -s.LR)
				continue
			}
			v := s.vel[i]
			v.MulScalarInPlace(s.Momentum).AddInPlace(g) // v = μ·v + g
			if s.Nesterov {
				p.AddScaledInPlace(g, -s.LR)
				p.AddScaledInPlace(v, -s.LR*s.Momentum)
			} else {
				p.AddScaledInPlace(v, -s.LR)
			}
		}
	})
}

// Adam implements Adam with bias correction. A non-zero WeightDecay is
// applied decoupled (AdamW).
type Adam struct {
	Params       []*tensor.Tensor
	LR           float32
	Beta1, Beta2 float32
	Eps          float32
	WeightDecay  float32

	m, v []*tensor.Tensor
	t    int
}

// NewAdam returns Adam with β1=0.9, β2=0.999, ε=1e-8.
func NewAdam(params []*tensor.Tensor, lr float32) *Adam {
	a := &Adam{Params: params, LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8}
	a.m = make([]*tensor.Tensor, len(params))
	a.v = make([]*tensor.Tensor, len(params))
	for i, p := range params {
		a.m[i] = tensor.ZerosLike(p)
		a.v[i] = tensor.ZerosLike(p)
	}
	return a
}

// NewAdamW returns Adam with decoupled weight decay.
func NewAdamW(params []*tensor.Tensor, lr, weightDecay float32) *Adam {
	a := NewAdam(params, lr)
	a.WeightDecay = weightDecay
	return a
}

func (a *Adam) ZeroGrad() { zeroGrad(a.Params) }

func (a *Adam) Step() {
	a.t++
	bc1 := 1 - float32(math.Pow(float64(a.Beta1), float64(a.t)))
	bc2 := 1 - float32(math.Pow(float64(a.Beta2), float64(a.t)))
	stepSize := a.LR / bc1
	b1, b2 := a.Beta1, a.Beta2
	tensor.NoGrad(func() {
		for i, p := range a.Params {
			if p.Grad() == nil {
				continue
			}
			g := p.Grad().Data()
			w := p.Data()
			m, v := a.m[i].Data(), a.v[i].Data()
			if a.WeightDecay != 0 {
				p.MulScalarInPlace(1 - a.LR*a.WeightDecay)
			}
			parallel.Range(len(w), 4096, func(lo, hi int) {
				for j := lo; j < hi; j++ {
					gj := g[j]
					m[j] = b1*m[j] + (1-b1)*gj
					v[j] = b2*v[j] + (1-b2)*gj*gj
					denom := float32(math.Sqrt(float64(v[j]/bc2))) + a.Eps
					w[j] -= stepSize * m[j] / denom
				}
			})
		}
	})
}
