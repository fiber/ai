package cluster

import (
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/fiber/ai/tensor"
)

// planted makes n rows around three well-separated unit vectors in d
// dimensions, returning the rows and the true group of each.
func planted(r *rand.Rand, n, d int) (*tensor.Tensor, []int) {
	centres := tensor.RandnFrom(r, 3, d)
	cd := Normalize(centres).Data()
	data := make([]float32, n*d)
	truth := make([]int, n)
	for i := 0; i < n; i++ {
		g := i % 3
		truth[i] = g
		for j := 0; j < d; j++ {
			data[i*d+j] = cd[g*d+j] + float32(r.NormFloat64())*0.05
		}
	}
	return Normalize(tensor.New(data, n, d)), truth
}

func TestKMeansRecoversPlantedClusters(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	x, truth := planted(r, 600, 64)
	res := KMeans(x, 3, 20, 7)
	// every planted group must map to exactly one centre
	mapping := map[int]int{}
	for i, a := range res.Assignment {
		if prev, ok := mapping[truth[i]]; ok && prev != a {
			t.Fatalf("group %d split between centres %d and %d", truth[i], prev, a)
		}
		mapping[truth[i]] = a
	}
	if len(mapping) != 3 {
		t.Fatalf("groups merged: %v", mapping)
	}
	for i, s := range res.Similarity {
		if s < 0.85 { // noise of 0.05 per dimension over 64 dimensions
			t.Fatalf("row %d similarity %v to its centre", i, s)
		}
	}
	// Nearest: a member scores high, a random vector low
	idx, sim := Nearest(x.Narrow(0, 0, 3), res.Centres)
	for i := range idx {
		if sim[i] < 0.85 || idx[i] != res.Assignment[i] {
			t.Fatalf("Nearest disagrees: %v %v", idx, sim)
		}
	}
	_, sim = Nearest(Normalize(tensor.RandnFrom(r, 1, 64)), res.Centres)
	if sim[0] > 0.6 {
		t.Fatalf("random vector too similar: %v", sim[0])
	}
	if res.Inertia() > 0.1 { // ~0.07 expected for this noise level
		t.Fatalf("inertia %v", res.Inertia())
	}
}

func TestNearestSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}
	x := Normalize(tensor.Randn(50000, 768))
	c := Normalize(tensor.Randn(40, 768))
	Nearest(x, c) // warm up
	start := time.Now()
	Nearest(x, c)
	if el := time.Since(start); el > 200*time.Millisecond {
		t.Fatalf("50 000 × 768 against 40 centres took %v", el)
	} else {
		t.Logf("50 000 × 768 against 40 centres: %v", el)
	}
}

func TestCosine(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	if c := Cosine(a, a); math.Abs(float64(c)-1) > 1e-6 {
		t.Fatalf("same direction: %v", c)
	}
	if c := Cosine(a, []float32{-1, -2, -3, -4}); math.Abs(float64(c)+1) > 1e-6 {
		t.Fatalf("opposite: %v", c)
	}
	if c := Cosine([]float32{1, 0}, []float32{0, 1}); c != 0 {
		t.Fatalf("orthogonal: %v", c)
	}
	if c := Cosine(a, []float32{0, 0, 0, 0}); c != 0 {
		t.Fatalf("zero vector: %v", c)
	}
	rng := rand.New(rand.NewPCG(9, 9))
	for n := range []int{1, 7, 8, 9, 768, 1001} {
		x := tensor.RandnFrom(rng, n+1).Float32s()
		y := tensor.RandnFrom(rng, n+1).Float32s()
		var dot, nx, ny float64
		for i := range x {
			dot += float64(x[i]) * float64(y[i])
			nx += float64(x[i]) * float64(x[i])
			ny += float64(y[i]) * float64(y[i])
		}
		want := dot / math.Sqrt(nx*ny)
		if got := Cosine(x, y); math.Abs(float64(got)-want) > 1e-5 {
			t.Fatalf("n=%d: %v vs %v", n+1, got, want)
		}
	}
}

func TestSimilarities(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	q := Normalize(tensor.RandnFrom(rng, 5, 64))
	d := Normalize(tensor.RandnFrom(rng, 7, 64))
	s := Similarities(q, d)
	if s.Dim(0) != 5 || s.Dim(1) != 7 {
		t.Fatalf("shape %v", s.Shape())
	}
	qd, dd := q.Float32s(), d.Float32s()
	for i := 0; i < 5; i++ {
		for j := 0; j < 7; j++ {
			want := Cosine(qd[i*64:(i+1)*64], dd[j*64:(j+1)*64])
			if got := s.At(i, j); math.Abs(float64(got-want)) > 1e-5 {
				t.Fatalf("(%d,%d): %v vs %v", i, j, got, want)
			}
		}
	}
}

func BenchmarkCosine768(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 1))
	x := tensor.RandnFrom(rng, 768).Float32s()
	y := tensor.RandnFrom(rng, 768).Float32s()
	b.Run("vectorised", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			Cosine(x, y)
		}
	})
	b.Run("scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			var dot, nx, ny float64
			for k := range x {
				dot += float64(x[k]) * float64(y[k])
				nx += float64(x[k]) * float64(x[k])
				ny += float64(y[k]) * float64(y[k])
			}
			_ = dot / math.Sqrt(nx*ny)
		}
	})
}
