package cluster

import (
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
