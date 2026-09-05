// Tutorial chapter 1: a tensor is a slice with a shape.
package main

import (
	"fmt"

	"github.com/fiber/ai/tensor"
)

func main() {
	// Six numbers. As a Go slice they are just a row of values; the shape
	// tells the tensor how to read them: 2 rows of 3.
	temps := []float32{18.5, 21.0, 19.5, 22.0, 24.5, 23.0}
	t := tensor.New(temps, 2, 3)
	fmt.Println("t =")
	fmt.Println(t)
	fmt.Println("shape:", t.Shape(), " elements:", t.Size(), " dims:", t.Dims())

	// One element, one row, one column.
	fmt.Println("\nt[1,2]  =", t.At(1, 2))
	fmt.Println("row 0   =", t.Row(0))
	fmt.Println("column 1 =", t.Select(1, 1))

	// The same six numbers read as 3 rows of 2: no data is copied, only
	// the shape changes. Reshape, T and Narrow all return views.
	v := t.Reshape(3, 2)
	fmt.Println("\nt.Reshape(3, 2) =")
	fmt.Println(v)
	fmt.Println("t.T() (rows and columns swapped, still no copy) =")
	fmt.Println(t.T())

	// Because views share storage, writing through one is visible in all.
	v.Set(100, 0, 0)
	fmt.Println("\nafter v.Set(100, 0, 0): t[0,0] =", t.At(0, 0))

	// Data() hands out the backing slice itself; Float32s() gives a copy
	// you may keep and modify without touching the tensor.
	copyOf := t.Float32s()
	copyOf[0] = -1
	fmt.Println("Float32s() copy changed, t[0,0] still", t.At(0, 0))

	// Constructors for the shapes you need most.
	fmt.Println("\nZeros(2, 2) =")
	fmt.Println(tensor.Zeros(2, 2))
	fmt.Println("Arange(0, 5, 1) =", tensor.Arange(0, 5, 1))
	tensor.Seed(1)
	fmt.Println("Randn(2, 3) (normal noise, mean 0, spread 1) =")
	fmt.Println(tensor.Randn(2, 3))

	// A wrong shape is a panic with a readable message; Try turns it into
	// an error when you would rather handle it.
	err := tensor.Try(func() { tensor.New(temps, 4, 2) })
	fmt.Println("\nNew(temps, 4, 2):", err)
}
