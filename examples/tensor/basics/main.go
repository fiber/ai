// Example: a tour of the tensor API — construction, broadcasting, views,
// reductions and matrix products.
package main

import (
	"fmt"

	"github.com/fiber/ai/tensor"
)

func main() {
	fmt.Printf("backend: %s, threads: %d\n\n", tensor.Backend(), tensor.Threads())

	a := tensor.New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	fmt.Println("a =")
	fmt.Println(a)
	fmt.Println("shape", a.Shape(), "size", a.Size())

	// broadcasting: row vector, column vector and scalar
	row := tensor.New([]float32{10, 20, 30}, 3)
	col := tensor.New([]float32{100, 200}, 2, 1)
	fmt.Println("\na + row =")
	fmt.Println(a.Add(row))
	fmt.Println("\na * col =")
	fmt.Println(a.Mul(col))
	fmt.Println("\n(a - 2) / 2 =")
	fmt.Println(a.SubScalar(2).DivScalar(2))

	// views share storage: transposes and slices cost nothing
	at := a.T()
	fmt.Println("\naᵀ (view, contiguous =", at.IsContiguous(), ")")
	fmt.Println(at)
	fmt.Println("\na[:, 1:3] =")
	fmt.Println(a.Slice(1, 1, 3))

	// reductions along dimensions
	fmt.Println("\nsum over rows   :", a.Sum(0))
	fmt.Println("mean over cols  :", a.Mean(1))
	fmt.Println("max of all      :", a.Max().Item())
	fmt.Println("argmax per row  :", a.Argmax(1))

	// matrix products consume strided views directly
	fmt.Println("\na · aᵀ =")
	fmt.Println(a.MatMul(at))

	// batched matmul with broadcasting: [4, 2, 3] · [3, 5] -> [4, 2, 5]
	batch := tensor.Randn(4, 2, 3)
	w := tensor.Randn(3, 5)
	fmt.Println("\nbatched product shape:", batch.MatMul(w).Shape())

	// softmax rows sum to one
	logits := tensor.Randn(3, 4)
	fmt.Println("\nsoftmax(logits).Sum(1) =", logits.Softmax(-1).Sum(1))

	// shape errors are panics carrying *tensor.Error; Try converts them
	err := tensor.Try(func() { a.MatMul(a) })
	fmt.Println("\nerror handling:", err)
}
