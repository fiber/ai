package tensor

import (
	"math/rand/v2"
	"sync"
)

var (
	rngMu sync.Mutex
	rng   = rand.New(rand.NewPCG(0x5eed, 0xf1be))
)

// Seed re-seeds the package random number generator used by Rand, Randn
// and Uniform.
func Seed(seed uint64) {
	rngMu.Lock()
	defer rngMu.Unlock()
	rng = rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
}

func fillRandom(t *Tensor, r *rand.Rand, gen func(r *rand.Rand) float32) {
	if r == nil {
		rngMu.Lock()
		defer rngMu.Unlock()
		r = rng
	}
	for i := range t.data {
		t.data[i] = gen(r)
	}
}

// Rand returns a tensor of uniform samples in [0, 1).
func Rand(shape ...int) *Tensor { return RandFrom(nil, shape...) }

// RandFrom is Rand drawing from r (nil uses the package generator).
func RandFrom(r *rand.Rand, shape ...int) *Tensor {
	checkShape("Rand", shape)
	t := newTensorUninit(shape)
	fillRandom(t, r, func(r *rand.Rand) float32 { return r.Float32() })
	return t
}

// Randn returns a tensor of standard normal samples.
func Randn(shape ...int) *Tensor { return RandnFrom(nil, shape...) }

// RandnFrom is Randn drawing from r (nil uses the package generator).
func RandnFrom(r *rand.Rand, shape ...int) *Tensor {
	checkShape("Randn", shape)
	t := newTensorUninit(shape)
	fillRandom(t, r, func(r *rand.Rand) float32 { return float32(r.NormFloat64()) })
	return t
}

// Uniform returns a tensor of uniform samples in [lo, hi).
func Uniform(lo, hi float32, shape ...int) *Tensor { return UniformFrom(nil, lo, hi, shape...) }

// UniformFrom is Uniform drawing from r (nil uses the package generator).
func UniformFrom(r *rand.Rand, lo, hi float32, shape ...int) *Tensor {
	checkShape("Uniform", shape)
	t := newTensorUninit(shape)
	fillRandom(t, r, func(r *rand.Rand) float32 { return lo + (hi-lo)*r.Float32() })
	return t
}
