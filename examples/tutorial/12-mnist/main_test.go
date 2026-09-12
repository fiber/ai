package main

import (
	"math"
	"testing"

	"github.com/fiber/ai-data/mnist"
	"github.com/fiber/ai/tensor"
)

func splits(t *testing.T, trainLimit, testLimit int) (dataset, dataset) {
	t.Helper()
	trainSet, err := mnist.Train()
	if err != nil {
		t.Fatal(err)
	}
	testSet, err := mnist.Test()
	if err != nil {
		t.Fatal(err)
	}
	tr, mean, sd := load(trainSet, trainLimit, 0, 0)
	te, _, _ := load(testSet, testLimit, mean, sd)
	return tr, te
}

// The training split defines the scale and the test split borrows it, so
// the training pixels must come out with mean 0 and deviation 1 and the
// test pixels merely close to it.
func TestLoadScales(t *testing.T) {
	tr, te := splits(t, 5000, 2000)

	if got := tr.x.Shape(); got[0] != 5000 || got[1] != pixels {
		t.Fatalf("training shape is %v, want [5000 784]", got)
	}
	if len(te.labels) != 2000 {
		t.Fatalf("got %d test labels, want 2000", len(te.labels))
	}

	mean, sd := stats(tr.x)
	if math.Abs(mean) > 1e-4 || math.Abs(sd-1) > 1e-4 {
		t.Errorf("training pixels have mean %v and deviation %v, want 0 and 1", mean, sd)
	}
	mean, sd = stats(te.x)
	if math.Abs(mean) > 0.05 || math.Abs(sd-1) > 0.05 {
		t.Errorf("test pixels have mean %v and deviation %v, want them near 0 and 1", mean, sd)
	}

	for i, l := range tr.labels {
		if l < 0 || l > 9 {
			t.Fatalf("label %d of image %d is not a digit", l, i)
		}
	}
}

func stats(x *tensor.Tensor) (mean, sd float64) {
	var sum, sumSq float64
	values := x.Float32s()
	for _, v := range values {
		sum += float64(v)
		sumSq += float64(v) * float64(v)
	}
	mean = sum / float64(len(values))
	return mean, math.Sqrt(sumSq/float64(len(values)) - mean*mean)
}

// A short run on a small prefix, enough to catch the example rotting: if
// the loop, the loss or the optimiser breaks, this floor is missed by a
// wide margin rather than narrowly.
func TestMLPLearns(t *testing.T) {
	tr, te := splits(t, 4000, 2000)
	tensor.Seed(12)

	model := mlp()
	before, _ := evaluate(model, te, 256)
	if before > 25 {
		t.Fatalf("an untrained model already scores %.1f%%", before)
	}

	acc, confusion := train("MLP", model, tr, te, 2, 128, 1e-3, 12)
	if acc < 85 {
		t.Errorf("accuracy after two epochs is %.1f%%, want at least 85%%", acc)
	}

	total := 0
	for want := range digits {
		for got := range digits {
			total += confusion[want][got]
		}
	}
	if total != len(te.labels) {
		t.Errorf("confusion matrix holds %d images, want %d", total, len(te.labels))
	}
}

// The convolutional stack is the reason chapter 12 exists; check that the
// shapes line up all the way from flat rows to ten scores, so a change in
// Conv2D or MaxPool2D shows up here and not in a reader's terminal.
func TestCNNShapes(t *testing.T) {
	tr, _ := splits(t, 64, 0)
	tensor.Seed(1)

	var out *tensor.Tensor
	tensor.NoGrad(func() { out = cnn().Forward(tr.x) })
	if got := out.Shape(); got[0] != 64 || got[1] != digits {
		t.Fatalf("output shape is %v, want [64 10]", got)
	}
}
