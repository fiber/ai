// Tutorial chapter 2: broadcasting, or how one line replaces a loop.
package main

import (
	"fmt"

	"github.com/fiber/ai/tensor"
)

func main() {
	// Five days of readings from three sensors: rows are days, columns
	// are sensors. The sensors have different offsets and scales, which is
	// the normal state of any measured data.
	x := tensor.New([]float32{
		20.1, 1013, 41,
		21.4, 1009, 45,
		19.8, 1021, 38,
		23.0, 1004, 52,
		22.2, 1011, 47,
	}, 5, 3)
	fmt.Println("x (days × sensors) =")
	fmt.Println(x)

	// Subtracting a row of three numbers from a 5×3 matrix: the row is
	// applied to every day. That is broadcasting. No loop, no copy of the
	// row, one kernel call.
	offset := tensor.New([]float32{20, 1000, 40}, 3)
	fmt.Println("\nx - offset =")
	fmt.Println(x.Sub(offset))

	// A column of five numbers is applied to every sensor instead.
	weight := tensor.New([]float32{1, 1, 1, 0.5, 0.5}, 5, 1)
	fmt.Println("\nx * weight (a column) =")
	fmt.Println(x.Mul(weight))

	// The one thing every model input goes through: standardising each
	// column to mean 0 and spread 1, so that pressure (around 1000) does
	// not drown out temperature (around 20) just by being a bigger number.
	mean := x.Mean(0) // one number per column: shape [3]
	std := x.Std(0)
	fmt.Println("\nmean per sensor =", mean)
	fmt.Println("std  per sensor =", std)
	z := x.Sub(mean).Div(std)
	fmt.Println("standardised =")
	fmt.Println(z)
	fmt.Println("check: column means =", z.Mean(0), " column stds =", z.Std(0))

	// The rule: shapes are compared from the right; a dimension of size 1
	// stretches to match; a missing dimension counts as 1.
	//   [5 3] with [3]   -> ok, the row repeats down
	//   [5 3] with [5 1] -> ok, the column repeats across
	//   [5 3] with [5]   -> not ok: 3 and 5 do not match
	err := tensor.Try(func() { x.Add(tensor.Ones(5)) })
	fmt.Println("\nx + Ones(5):", err)
	fmt.Println("x + Ones(5, 1) works, shape", x.Add(tensor.Ones(5, 1)).Shape())
}
