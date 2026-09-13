// Tutorial chapter 12: handwritten digits — the first example in this
// tutorial that trains on data somebody else collected.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"time"

	"github.com/fiber/ai-data/mnist"
	"github.com/fiber/ai/data"
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

const (
	side   = 28
	pixels = side * side
	digits = 10
)

// toImages turns the flat [batch × 784] rows the loader produces into the
// [batch, channel, height, width] that Conv2D reads. It has no parameters.
type toImages struct{}

func (toImages) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Reshape(x.Dim(0), 1, side, side)
}

func (toImages) Params() []*tensor.Tensor { return nil }

// dataset is one split, ready for the model: pixels as a [n × 784] tensor
// and the digit of each row.
type dataset struct {
	x      *tensor.Tensor
	labels []int
}

// load reads a split and scales it. The scaling is one mean and one
// standard deviation for the whole image, not one per pixel: the pixels
// along the border are zero in every image of the set, so a per-pixel
// standard deviation would be zero there and the division would produce
// NaN. Ink is ink wherever it lands, so a single scale is also the more
// honest description of the data.
func load(set *mnist.Set, limit int, mean, sd float32) (dataset, float32, float32) {
	n := set.N
	if limit > 0 && limit < n {
		n = limit
	}
	values := make([]float32, n*pixels)
	buf := make([]float32, pixels)
	labels := make([]int, n)
	for i := range n {
		copy(values[i*pixels:], set.Float32(i, buf))
		labels[i] = int(set.Labels[i])
	}

	// The training split measures itself; every later split is scaled by
	// the numbers the training split produced, because at prediction time
	// the statistics of the incoming data are not yet known.
	if mean == 0 && sd == 0 {
		var sum, sumSq float64
		for _, v := range values {
			sum += float64(v)
			sumSq += float64(v) * float64(v)
		}
		m := sum / float64(len(values))
		mean = float32(m)
		sd = float32(math.Sqrt(sumSq/float64(len(values)) - m*m))
	}
	for i, v := range values {
		values[i] = (v - mean) / sd
	}
	return dataset{tensor.New(values, n, pixels), labels}, mean, sd
}

// evaluate runs the model over a split in batches — the whole test set at
// once would allocate a great deal for no gain — and returns the
// percentage correct together with the confusion matrix, rows the true
// digit and columns the predicted one.
func evaluate(m nn.Module, d dataset, batch int) (float64, [digits][digits]int) {
	var confusion [digits][digits]int
	correct := 0
	tensor.NoGrad(func() {
		for start := 0; start < len(d.labels); start += batch {
			end := min(start+batch, len(d.labels))
			rows := make([]int, end-start)
			for i := range rows {
				rows[i] = start + i
			}
			for i, p := range m.Forward(d.x.Rows(rows)).Argmax(1) {
				want := d.labels[start+i]
				confusion[want][p]++
				if p == want {
					correct++
				}
			}
		}
	})
	return 100 * float64(correct) / float64(len(d.labels)), confusion
}

// train runs the loop and prints one line per epoch, so the reader can see
// both what the model reaches and what it costs to get there.
func train(name string, model nn.Module, train, test dataset, epochs, batch int, lr float32, seed uint64) (float64, [digits][digits]int) {
	// Each model shuffles from its own stream, so training one model does
	// not depend on whether another ran first in the same process: with
	// one shared stream, the CNN of -model both starts from a different
	// batch order than the CNN of -model cnn, and the two are then not
	// comparable. The weights are seeded by the caller for the same
	// reason.
	r := rand.New(rand.NewPCG(seed, 0))
	opt := optim.NewAdam(model.Params(), lr)
	params := 0
	for _, p := range model.Params() {
		params += p.Size()
	}
	fmt.Printf("\n%s: %d parameters, %d images, %d epochs of batch %d\n", name, params, len(train.labels), epochs, batch)

	total := time.Duration(0)
	for epoch := 1; epoch <= epochs; epoch++ {
		start := time.Now()
		sum, steps := float32(0), 0
		for idx := range data.Batches(len(train.labels), batch, r) {
			labels := make([]int, len(idx))
			for i, row := range idx {
				labels[i] = train.labels[row]
			}
			loss := tensor.CrossEntropy(model.Forward(train.x.Rows(idx)), labels)
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
			sum += loss.Item()
			steps++
		}
		took := time.Since(start)
		total += took
		acc, _ := evaluate(model, test, batch)
		fmt.Printf("  epoch %2d  loss %.4f  test accuracy %.2f%%  %.1fs\n", epoch, sum/float32(steps), acc, took.Seconds())
	}

	acc, confusion := evaluate(model, test, batch)
	fmt.Printf("  %s: %.2f%% after %.1fs of training\n", name, acc, total.Seconds())
	return acc, confusion
}

// worst prints the digit pairs the model confuses most often. The point of
// the exercise is that the remaining error is not spread evenly: a handful
// of shapes account for most of it.
func worst(confusion [digits][digits]int, n int) {
	type pair struct {
		want, got, count int
	}
	var pairs []pair
	for want := range digits {
		for got := range digits {
			if want != got && confusion[want][got] > 0 {
				pairs = append(pairs, pair{want, got, confusion[want][got]})
			}
		}
	}
	for i := 1; i < len(pairs); i++ {
		for j := i; j > 0 && pairs[j].count > pairs[j-1].count; j-- {
			pairs[j], pairs[j-1] = pairs[j-1], pairs[j]
		}
	}
	fmt.Println("\n  most frequent confusions (true digit read as):")
	for _, p := range pairs[:min(n, len(pairs))] {
		fmt.Printf("    %d read as %d: %d times\n", p.want, p.got, p.count)
	}
}

func mlp() nn.Module {
	// The model of chapter 5, widened: the input is 784 pixels instead of
	// two coordinates and there are ten classes instead of three.
	return nn.Sequential{
		nn.NewLinear(pixels, 256), nn.ReLU{},
		nn.NewLinear(256, digits),
	}
}

func cnn() nn.Module {
	// Two convolution stages, each halved by pooling: 28 to 14 to 7. The
	// linear head then sees 32 channels of 7×7 instead of raw pixels.
	return nn.Sequential{
		toImages{},
		nn.NewConv2D(1, 16, 3), nn.ReLU{}, nn.NewMaxPool2D(2),
		nn.NewConv2D(16, 32, 3), nn.ReLU{}, nn.NewMaxPool2D(2),
		nn.Flatten{},
		nn.NewLinear(32*7*7, digits),
	}
}

func main() {
	which := flag.String("model", "both", "mlp, cnn or both")
	epochs := flag.Int("epochs", 5, "passes over the training set")
	batch := flag.Int("batch", 128, "images per step")
	limit := flag.Int("limit", 0, "train on the first n images only (0 = all 60000)")
	lr := flag.Float64("lr", 1e-3, "Adam learning rate")
	seed := flag.Uint64("seed", 12, "random seed")
	flag.Parse()

	trainSet, err := mnist.Train()
	if err != nil {
		log.Fatal(err)
	}
	testSet, err := mnist.Test()
	if err != nil {
		log.Fatal(err)
	}

	start := time.Now()
	trainData, mean, sd := load(trainSet, *limit, 0, 0)
	testData, _, _ := load(testSet, 0, mean, sd)
	fmt.Printf("MNIST: %d training and %d test images of %dx%d, loaded in %v\n",
		len(trainData.labels), len(testData.labels), trainSet.Rows, trainSet.Cols, time.Since(start).Round(time.Millisecond))
	fmt.Printf("scaled by mean %.4f and standard deviation %.4f of the training pixels\n", mean, sd)

	var confusion [digits][digits]int
	if *which == "mlp" || *which == "both" {
		tensor.Seed(*seed)
		_, confusion = train("MLP  784-256-10", mlp(), trainData, testData, *epochs, *batch, float32(*lr), *seed)
	}
	if *which == "cnn" || *which == "both" {
		tensor.Seed(*seed)
		_, confusion = train("CNN  16-32 filters", cnn(), trainData, testData, *epochs, *batch, float32(*lr), *seed)
	}
	worst(confusion, 5)
}
