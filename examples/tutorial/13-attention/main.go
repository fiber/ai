// Tutorial chapter 11: attention — a layer that chooses what to look at,
// written by hand so that its choices can be read.
package main

import (
	"compress/gzip"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/fiber/ai/data"
	"github.com/fiber/ai/metrics"
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

const (
	minutesPerDay = 1440
	window        = 120
	horizon       = 30
	tokenDim      = 10 // eight airports and two position features
	keyDim        = 16
)

var airports = []string{"EIDW", "EGPH", "EGCC", "EGLL", "EHAM", "EBBR", "LFPG", "EDDF"}

func loadSeries(path string) (series map[string][]float32, days []string, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, err
	}
	rd := csv.NewReader(gz)
	header, err := rd.Read()
	if err != nil {
		return nil, nil, err
	}
	col := map[string]int{}
	for i, h := range header {
		col[h] = i
	}
	type key struct{ airport, day string }
	byKey := map[key][]float32{}
	seen := map[string]bool{}
	for {
		rec, err := rd.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		k := key{rec[col["airport"]], rec[col["date"]]}
		if byKey[k] == nil {
			byKey[k] = make([]float32, minutesPerDay)
		}
		seen[k.day] = true
		minute, _ := strconv.Atoi(rec[col["minute"]])
		v, _ := strconv.ParseFloat(rec[col["in_zone"]], 32)
		byKey[k][minute] = float32(v)
	}
	for d := range seen {
		days = append(days, d)
	}
	sort.Strings(days)
	series = map[string][]float32{}
	for _, a := range airports {
		for _, d := range days {
			series[a] = append(series[a], byKey[key{a, d}]...)
		}
	}
	return series, days, nil
}

// tokens builds, per example, 120 tokens of ten numbers: the eight
// airports' counts at that minute (scaled later) and two features that say
// where in the window the minute sits, because attention on its own has no
// notion of order.
func tokens(series map[string][]float32, target string, mu, sd float32) (x, y *tensor.Tensor, pos []int) {
	n := len(series[target])
	var xs, ys []float32
	for t := window; t+horizon-1 < n; t++ {
		for i := t - window; i < t; i++ {
			for _, a := range airports {
				xs = append(xs, (series[a][i]-mu)/sd)
			}
			p := float32(i-(t-window)) / float32(window-1)
			xs = append(xs, p, float32(math.Sin(math.Pi*float64(p))))
		}
		ys = append(ys, (series[target][t+horizon-1]-mu)/sd)
		pos = append(pos, t)
	}
	return tensor.New(xs, len(pos), window, tokenDim), tensor.New(ys, len(pos), 1), pos
}

// attentionPool is attention in its smallest form: keys and values are
// linear maps of the tokens, the query is one learned vector, the weights
// are a softmax over the 120 minutes, and the output is the weighted sum
// of the values. Everything else in the chapter is reading those weights.
type attentionPool struct {
	Wk, Wv *tensor.Tensor // [tokenDim × keyDim]
	Q      *tensor.Tensor // [keyDim × 1]
}

func newAttentionPool() *attentionPool {
	s := float32(1 / math.Sqrt(tokenDim))
	return &attentionPool{
		Wk: tensor.Randn(tokenDim, keyDim).MulScalar(s).SetRequiresGrad(true),
		Wv: tensor.Randn(tokenDim, keyDim).MulScalar(s).SetRequiresGrad(true),
		Q:  tensor.Randn(keyDim, 1).MulScalar(0.5).SetRequiresGrad(true),
	}
}

// weights returns the attention over the minutes, [batch × window].
func (a *attentionPool) weights(x *tensor.Tensor) *tensor.Tensor {
	b := x.Dim(0)
	k := x.Reshape(b*window, tokenDim).MatMul(a.Wk)                    // [b·120 × keyDim]
	scores := k.MatMul(a.Q).Reshape(b, window)                         // one number per minute
	return scores.MulScalar(float32(1 / math.Sqrt(keyDim))).Softmax(1) // sums to 1 over the window
}

func (a *attentionPool) Forward(x *tensor.Tensor) *tensor.Tensor {
	b := x.Dim(0)
	w := a.weights(x) // [b × 120]
	v := x.Reshape(b*window, tokenDim).MatMul(a.Wv).Reshape(b, window, keyDim)
	return w.Unsqueeze(2).Mul(v).Sum(1) // [b × keyDim]: the minutes the model chose
}

func (a *attentionPool) Params() []*tensor.Tensor { return []*tensor.Tensor{a.Wk, a.Wv, a.Q} }

func mae(m nn.Module, x, y *tensor.Tensor, sd float32) float64 {
	var v float64
	tensor.NoGrad(func() { v = metrics.MAE(m.Forward(x), y) * float64(sd) })
	return v
}

func main() {
	path := flag.String("data", "examples/airspace/testdata/counters.csv.gz", "airspace counters (csv.gz)")
	target := flag.String("airport", "EIDW", "ICAO code of the airport to forecast")
	flag.Parse()
	tensor.Seed(11)
	r := rand.New(rand.NewPCG(11, 0))

	series, days, err := loadSeries(*path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	// Scale with the training days of the target series, as in chapter 8.
	mu, sd := meanStd(series[*target][:4*minutesPerDay])
	x, y, pos := tokens(series, *target, mu, sd)
	var trainIdx, valIdx []int
	for i, t := range pos {
		switch day := (t + horizon - 1) / minutesPerDay; {
		case day < 4:
			trainIdx = append(trainIdx, i)
		case day == 4:
			valIdx = append(valIdx, i)
		}
	}
	xTrain, yTrain := x.Rows(trainIdx), y.Rows(trainIdx)
	xVal, yVal := x.Rows(valIdx), y.Rows(valIdx)
	fmt.Printf("%s, %d minutes ahead from %d tokens of %d numbers; %d examples train, %d validate (%s)\n",
		*target, horizon, window, tokenDim, len(trainIdx), len(valIdx), days[4])

	pool := newAttentionPool()
	model := nn.Sequential{pool, nn.NewLinear(keyDim, 32), nn.ReLU{}, nn.NewLinear(32, 1)}
	opt := optim.NewAdam(model.Params(), 1e-3)
	fmt.Printf("attention pooling + head, %d parameters\n\n", nn.NumParams(model))
	for epoch := 1; epoch <= 20; epoch++ {
		for idx := range data.Batches(xTrain.Dim(0), 64, r) {
			loss := tensor.MSELoss(model.Forward(xTrain.Rows(idx)), yTrain.Rows(idx))
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
		}
		if epoch%5 == 0 {
			fmt.Printf("epoch %2d  train %.2f  validation %.2f aircraft\n", epoch, mae(model, xTrain, yTrain, sd), mae(model, xVal, yVal, sd))
		}
	}
	fmt.Printf("\nchapter 8 on the same task: persistence 3.02, flat MLP 2.52, 1-D CNN 2.63 aircraft\n")

	// Where does it look? The weights over the window for one validation
	// example at 08:00, summed per ten minutes, oldest first.
	at := 8 * 60
	var ex int
	for i, t := range pos {
		if t == 4*minutesPerDay+at {
			ex = i
		}
	}
	var w []float32
	tensor.NoGrad(func() { w = pool.weights(x.Rows([]int{ex})).Float32s() })
	fmt.Printf("\nattention over the two hours before %s 08:00 (share of the total per ten minutes):\n", days[4])
	for b := 0; b < window; b += 10 {
		var s float32
		for _, v := range w[b : b+10] {
			s += v
		}
		fmt.Printf("  %3d to %3d minutes ago  %5.1f%%  %s\n", window-b, window-b-10, s*100, strings.Repeat("#", int(s*200)))
	}
}

func meanStd(s []float32) (mean, sd float32) {
	var sum, sq float64
	for _, v := range s {
		sum += float64(v)
	}
	mean = float32(sum / float64(len(s)))
	for _, v := range s {
		d := float64(v - mean)
		sq += d * d
	}
	return mean, float32(math.Sqrt(sq / float64(len(s))))
}
