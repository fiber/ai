package main

import "testing"

// The chapter's whole argument is that these three cases behave
// differently, so the test is the argument: two succeed, one cannot.
func TestPerceptronSeparatesAndAndOrButNotXor(t *testing.T) {
	for _, c := range []struct {
		name string
		f    func(a, b int) int
		want bool
	}{
		{"AND", and, true},
		{"OR", or, true},
		{"XOR", xor, false},
	} {
		p := &perceptron{}
		ok, at := p.train(targets(c.f), 0.1, 5000)
		if ok != c.want {
			if c.want {
				t.Errorf("%s: not separated after 5000 epochs, and it is separable", c.name)
			} else {
				t.Errorf("%s: separated after %d epochs, which is impossible for one unit", c.name, at)
			}
		}
	}
}

// One hidden unit is still one boundary and still fails; two are enough.
// If this ever passes with one unit, the nonlinearity or the training
// loop is not doing what the chapter says.
func TestHiddenLayerSolvesXorFromTwoUnits(t *testing.T) {
	y := targets(xor)
	one, _ := hidden(y, 1, 400)
	two, out := hidden(y, 2, 400)
	if one < 0.05 {
		t.Errorf("one hidden unit reached loss %.4f; it should not be able to", one)
	}
	if two > 0.01 {
		t.Errorf("two hidden units reached loss %.4f, want near zero", two)
	}
	for i, want := range y {
		if diff := out[i] - want; diff > 0.1 || diff < -0.1 {
			t.Errorf("output %d is %.2f, want %g", i, out[i], want)
		}
	}
}
