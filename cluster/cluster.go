// Package cluster groups embedding rows: k-means with k-means++ seeding
// and nearest-centre search, both built on the tensor layer's matrix
// product so that fifty thousand rows against forty centres is one
// GEMM. Rows are expected to be unit length (see Normalize); the dot
// product is then the cosine similarity.
package cluster

import (
	"math"
	"math/rand/v2"

	"github.com/fiber/ai/tensor"
)

// Normalize returns x with every row scaled to unit length. Rows of
// length zero stay zero.
func Normalize(x *tensor.Tensor) *tensor.Tensor {
	norm := x.Square().Sum(1).Sqrt().Reshape(x.Dim(0), 1)
	nd := norm.Data()
	for i, v := range nd {
		if v == 0 {
			nd[i] = 1
		}
	}
	return x.Div(norm)
}

// Result of a k-means run.
type Result struct {
	Centres    *tensor.Tensor // [k×d], unit length
	Assignment []int          // centre index per row
	Similarity []float32      // cosine similarity of each row to its centre
	Sizes      []int          // rows per centre
	Iterations int
}

// KMeans clusters the unit-length rows of x into k groups by cosine
// similarity: k-means++ seeding, then up to iters rounds of assignment
// (one matrix product) and centre update, stopping early when no row
// changes centre.
func KMeans(x *tensor.Tensor, k, iters int, seed uint64) Result {
	n, d := x.Dim(0), x.Dim(1)
	if k <= 0 || k > n {
		panic("cluster: k must be between 1 and the number of rows")
	}
	r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	xd := x.Data()

	// k-means++: the first centre at random, each further one with a
	// probability proportional to its distance from the nearest centre.
	centres := make([]float32, 0, k*d)
	first := r.IntN(n)
	centres = append(centres, xd[first*d:(first+1)*d]...)
	best := make([]float32, n) // best similarity so far per row
	for i := range best {
		best[i] = -1
	}
	for len(centres) < k*d {
		c := tensor.New(centres[len(centres)-d:], 1, d)
		sim := x.MatMul(c.T()).Data() // [n×1] similarity to the newest centre
		var total float64
		weights := make([]float64, n)
		for i := range best {
			if sim[i] > best[i] {
				best[i] = sim[i]
			}
			w := float64(1 - best[i]) // cosine distance
			weights[i] = w * w
			total += weights[i]
		}
		pick := r.Float64() * total
		idx := n - 1
		for i, w := range weights {
			pick -= w
			if pick <= 0 {
				idx = i
				break
			}
		}
		centres = append(centres, xd[idx*d:(idx+1)*d]...)
	}
	cs := Normalize(tensor.New(centres, k, d))

	res := Result{Assignment: make([]int, n), Similarity: make([]float32, n), Sizes: make([]int, k)}
	for it := 1; it <= iters; it++ {
		res.Iterations = it
		sim := x.MatMul(cs.T()) // [n×k]
		newAssign := sim.Argmax(1)
		sd := sim.Data()
		changed := 0
		for i, a := range newAssign {
			res.Similarity[i] = sd[i*k+a]
			if a != res.Assignment[i] {
				changed++
			}
			res.Assignment[i] = a
		}
		// centre update: mean of the assigned rows, renormalised
		sums := make([]float32, k*d)
		for i := range res.Sizes {
			res.Sizes[i] = 0
		}
		for i, a := range res.Assignment {
			res.Sizes[a]++
			row := xd[i*d : (i+1)*d]
			acc := sums[a*d : (a+1)*d]
			for j, v := range row {
				acc[j] += v
			}
		}
		for c := 0; c < k; c++ {
			if res.Sizes[c] == 0 { // an empty centre restarts on the worst-fitting row
				worst := 0
				for i := range res.Similarity {
					if res.Similarity[i] < res.Similarity[worst] {
						worst = i
					}
				}
				copy(sums[c*d:(c+1)*d], xd[worst*d:(worst+1)*d])
				res.Similarity[worst] = 1
			}
		}
		cs = Normalize(tensor.New(sums, k, d))
		if changed == 0 && it > 1 {
			break
		}
	}
	res.Centres = cs
	return res
}

// Nearest returns, for every unit-length row of x, the index of the most
// similar centre and that similarity.
func Nearest(x, centres *tensor.Tensor) (index []int, similarity []float32) {
	sim := x.MatMul(centres.T())
	index = sim.Argmax(1)
	similarity = make([]float32, len(index))
	sd := sim.Data()
	k := centres.Dim(0)
	for i, a := range index {
		similarity[i] = sd[i*k+a]
	}
	return index, similarity
}

// Inertia is the mean cosine distance of rows to their centres, a single
// number to compare runs with different k or seeds (lower is tighter).
func (r Result) Inertia() float64 {
	var s float64
	for _, v := range r.Similarity {
		s += 1 - float64(v)
	}
	if len(r.Similarity) == 0 {
		return math.NaN()
	}
	return s / float64(len(r.Similarity))
}
