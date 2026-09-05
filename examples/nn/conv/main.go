// Example: a small convolutional network classifies generated 16×16
// images by the shape drawn in them (horizontal bar, vertical bar,
// diagonal, square outline), at random positions and with noise.
package main

import (
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/fiber/ai/data"
	"github.com/fiber/ai/metrics"
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

const (
	size    = 16
	classes = 4
)

var names = []string{"horizontal", "vertical", "diagonal", "square"}

func draw(r *rand.Rand, img []float32, class int) {
	for i := range img {
		img[i] = float32(r.NormFloat64()) * 0.3 // background noise
	}
	set := func(y, x int) {
		if y >= 0 && y < size && x >= 0 && x < size {
			img[y*size+x] = 1
		}
	}
	oy, ox := 3+r.IntN(size-8), 3+r.IntN(size-8)
	switch class {
	case 0:
		for i := -3; i <= 3; i++ {
			set(oy, ox+i)
		}
	case 1:
		for i := -3; i <= 3; i++ {
			set(oy+i, ox)
		}
	case 2:
		for i := -3; i <= 3; i++ {
			set(oy+i, ox+i)
		}
	case 3:
		for i := -2; i <= 2; i++ {
			set(oy-2, ox+i)
			set(oy+2, ox+i)
			set(oy+i, ox-2)
			set(oy+i, ox+2)
		}
	}
}

func dataset(r *rand.Rand, n int) (*tensor.Tensor, []int) {
	imgs := make([]float32, n*size*size)
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		labels[i] = r.IntN(classes)
		draw(r, imgs[i*size*size:(i+1)*size*size], labels[i])
	}
	return tensor.New(imgs, n, 1, size, size), labels
}

func predict(m nn.Module, x *tensor.Tensor) []int {
	var p []int
	tensor.NoGrad(func() { p = m.Forward(x).Argmax(1) })
	return p
}

func main() {
	tensor.Seed(4)
	r := rand.New(rand.NewPCG(4, 0))
	xTrain, yTrain := dataset(r, 2000)
	xTest, yTest := dataset(r, 500)

	model := nn.Sequential{
		nn.NewConv2D(1, 8, 3), nn.ReLU{}, nn.NewMaxPool2D(2), // 8 × 8 × 8
		nn.NewConv2D(8, 16, 3), nn.ReLU{}, nn.NewMaxPool2D(2), // 16 × 4 × 4
		nn.Flatten{}, nn.NewLinear(16*4*4, 32), nn.ReLU{}, nn.NewLinear(32, classes),
	}
	opt := optim.NewAdam(model.Params(), 2e-3)
	start := time.Now()
	for epoch := 1; epoch <= 6; epoch++ {
		for idx := range data.Batches(xTrain.Dim(0), 64, r) {
			targets := make([]int, len(idx))
			for i, j := range idx {
				targets[i] = yTrain[j]
			}
			loss := tensor.CrossEntropy(model.Forward(xTrain.Rows(idx)), targets)
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
		}
		cm := metrics.Confusion(predict(model, xTest), yTest, classes)
		fmt.Printf("epoch %d  test accuracy %.1f%%\n", epoch, 100*cm.Accuracy())
	}
	fmt.Printf("trained in %.1fs\n\n", time.Since(start).Seconds())
	cm := metrics.Confusion(predict(model, xTest), yTest, classes)
	cm.Labels = names
	fmt.Print(cm)
}
