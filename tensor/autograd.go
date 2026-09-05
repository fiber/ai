package tensor

import (
	"sync/atomic"

	"github.com/fiber/ai/internal/kernel"
)

// node records how a tensor was produced so gradients can flow back.
type node struct {
	op       string
	inputs   []*Tensor
	backward func(gy *Tensor) // must call in.accumGrad(g) for each input needing grad
}

var noGradDepth atomic.Int32

// GradEnabled reports whether operations currently record the autograd
// graph.
func GradEnabled() bool { return noGradDepth.Load() == 0 }

// NoGrad runs fn with graph recording disabled. Use it for inference and
// for parameter updates. The switch is process-wide (not per goroutine),
// so it is meant to wrap whole phases, not individual calls.
func NoGrad(fn func()) {
	noGradDepth.Add(1)
	defer noGradDepth.Add(-1)
	fn()
}

// SetRequiresGrad marks t as a leaf whose gradient should be accumulated
// by Backward. It returns t for chaining.
func (t *Tensor) SetRequiresGrad(b bool) *Tensor {
	if b && t.node != nil {
		fail("SetRequiresGrad", "only leaf tensors can be marked; use Detach first")
	}
	t.requiresGrad = b
	if !b {
		t.grad = nil
	}
	return t
}

// RequiresGrad reports whether gradients flow to t.
func (t *Tensor) RequiresGrad() bool { return t.requiresGrad }

// IsLeaf reports whether t was created by the user rather than an
// operation on tensors requiring grad.
func (t *Tensor) IsLeaf() bool { return t.node == nil }

// Grad returns the accumulated gradient, or nil if none has been computed.
func (t *Tensor) Grad() *Tensor { return t.grad }

// ZeroGrad clears the accumulated gradient. The buffer is kept when it
// exists, so parameter tensors do not reallocate every step.
func (t *Tensor) ZeroGrad() {
	if t.grad != nil {
		clear(t.grad.data[:t.grad.size])
	}
}

// RetainGrad requests that Backward keeps the gradient of this non-leaf
// tensor in Grad (by default only leaves keep their gradient).
func (t *Tensor) RetainGrad() *Tensor {
	t.retainGrad = true
	return t
}

// Detach returns a tensor that shares t's storage but is not connected to
// the autograd graph and does not require grad.
func (t *Tensor) Detach() *Tensor {
	return view(t, t.data, t.shape, t.strides)
}

// record attaches a backward node to out if any input requires grad and
// grad mode is on. Backward closures must only touch detached tensors.
func record(out *Tensor, op string, inputs []*Tensor, backward func(gy *Tensor)) *Tensor {
	if !GradEnabled() {
		return out
	}
	for _, in := range inputs {
		if in.requiresGrad {
			out.requiresGrad = true
			out.node = &node{op: op, inputs: inputs, backward: backward}
			return out
		}
	}
	return out
}

// accumGrad adds g into t's gradient buffer. It is a no-op for tensors not
// requiring grad, which lets backward closures push unconditionally.
func (t *Tensor) accumGrad(g *Tensor) {
	if !t.requiresGrad {
		return
	}
	if !g.shape.Equal(t.shape) {
		fail("Backward", "internal: gradient shape %v does not match tensor shape %v", g.shape, t.shape)
	}
	if t.grad == nil {
		t.grad = newTensorUninit(t.shape)
		copyStrided(t.grad.data, g)
		return
	}
	kernel.Add(t.grad.data, g.values(), t.grad.data)
}

// Backward computes gradients of t (which must hold a single element) with
// respect to every leaf tensor that requires grad, accumulating into their
// Grad. Call ZeroGrad on parameters between training steps.
func (t *Tensor) Backward() {
	if t.size != 1 {
		fail("Backward", "tensor has %d elements; use BackwardWith to supply a gradient", t.size)
	}
	t.BackwardWith(Ones(t.shape...))
}

// BackwardWith is Backward seeded with an explicit gradient of t's shape.
func (t *Tensor) BackwardWith(grad *Tensor) {
	if !t.requiresGrad {
		fail("Backward", "tensor does not require grad")
	}
	if !grad.shape.Equal(t.shape) {
		fail("Backward", "gradient shape %v does not match tensor shape %v", grad.shape, t.shape)
	}
	order := topoOrder(t)
	t.accumGrad(grad.Detach())
	NoGrad(func() {
		for i := len(order) - 1; i >= 0; i-- {
			v := order[i]
			if v.node == nil || v.grad == nil {
				continue
			}
			v.node.backward(v.grad)
			if !v.retainGrad {
				v.grad = nil // intermediate gradient has been propagated
			}
		}
	})
}

// topoOrder returns the tensors reachable from root in an order where every
// tensor appears after all tensors that depend on... i.e. inputs before
// outputs; the caller walks it backwards.
func topoOrder(root *Tensor) []*Tensor {
	var order []*Tensor
	seen := map[*Tensor]bool{}
	type frame struct {
		t    *Tensor
		next int
	}
	stack := []frame{{t: root}}
	seen[root] = true
	for len(stack) > 0 {
		f := &stack[len(stack)-1]
		var inputs []*Tensor
		if f.t.node != nil {
			inputs = f.t.node.inputs
		}
		if f.next < len(inputs) {
			in := inputs[f.next]
			f.next++
			if !seen[in] && in.requiresGrad {
				seen[in] = true
				stack = append(stack, frame{t: in})
			}
			continue
		}
		order = append(order, f.t)
		stack = stack[:len(stack)-1]
	}
	return order
}

// checkInPlace panics if an in-place modification would corrupt the graph.
func (t *Tensor) checkInPlace(op string) {
	if t.requiresGrad && GradEnabled() {
		fail(op, "in-place modification of a tensor that requires grad; wrap the call in tensor.NoGrad")
	}
}
