// Package tensor implements an n-dimensional float32 tensor with
// broadcasting, strided views and reverse-mode automatic differentiation.
//
// # Design
//
//   - Data is float32, stored row-major. Views (Transpose, Permute, Narrow,
//     Expand, Reshape of contiguous tensors) share storage and only change
//     shape and strides; nothing is copied until an operation needs
//     contiguous memory. MatMul consumes strided operands directly, so
//     x.MatMul(w.T()) never materialises the transpose.
//   - Element-wise operations broadcast with NumPy semantics. The fast paths
//     (same layout, scalar, row/column broadcast) run on SIMD kernels; the
//     general case is a strided loop. All heavy work is split across
//     goroutines (see SetThreads).
//   - Every differentiable operation records one graph node holding its
//     inputs and a backward closure. A node is only recorded when an input
//     requires grad and grad mode is on (see NoGrad), so inference carries
//     no autograd overhead.
//   - Errors in shapes or arguments are programming errors and panic with a
//     *Error. Use Try to convert a panic into an error value.
//
// # Example
//
//	x := tensor.Randn(64, 32)
//	w := tensor.Randn(32, 8).SetRequiresGrad(true)
//	b := tensor.Zeros(8).SetRequiresGrad(true)
//	loss := tensor.MSELoss(x.MatMul(w).Add(b).Tanh(), target)
//	loss.Backward()
//	fmt.Println(w.Grad().Shape()) // [32 8]
package tensor
