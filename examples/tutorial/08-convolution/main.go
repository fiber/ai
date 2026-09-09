// Tutorial chapter 8: convolutions over time — a filter that slides
// along a counter series, and a forecaster built from a few of them.
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

	"github.com/fiber/ai/data"
	"github.com/fiber/ai/metrics"
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

const (
	minutesPerDay = 1440
	window        = 120 // minutes of history per example
	horizon       = 30  // minutes ahead
	filterWidth   = 7
)

// airports of the airspace example, west to east; the first is the target
// by default and the storm of 24 January 2025 hit it hardest.
var airports = []string{"EIDW", "EGPH", "EGCC", "EGLL", "EHAM", "EBBR", "LFPG", "EDDF"}

// loadSeries reads the airspace counters: for every airport one value per
// minute over the six days in order, the number of aircraft in its zone.
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

// windows builds one example per minute t: the last `window` minutes of
// every channel, channel after channel, and the target airport's count
// `horizon` minutes after the window ends. Returns x [n × channels·window],
// y [n × 1] and the position t of every row.
func windows(series map[string][]float32, channels []string, target string) (x, y *tensor.Tensor, pos []int) {
	n := len(series[target])
	var xs, ys []float32
	for t := window; t+horizon-1 < n; t++ {
		for _, a := range channels {
			xs = append(xs, series[a][t-window:t]...)
		}
		ys = append(ys, series[target][t+horizon-1])
		pos = append(pos, t)
	}
	return tensor.New(xs, len(pos), len(channels)*window), tensor.New(ys, len(pos), 1), pos
}

// toChannels is the whole of "write your own module": a flat row of
// channels·window numbers becomes [batch, channels, window], the shape
// Conv1D reads. It has no parameters.
type toChannels struct{ channels int }

func (m toChannels) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Reshape(x.Dim(0), m.channels, window)
}
func (toChannels) Params() []*tensor.Tensor { return nil }

// mae evaluates a model in aircraft, undoing the scaling.
func mae(m nn.Module, x, y *tensor.Tensor, sd float32) float64 {
	var v float64
	tensor.NoGrad(func() { v = metrics.MAE(m.Forward(x), y) * float64(sd) })
	return v
}

func train(name string, model nn.Sequential, xTrain, yTrain, xVal, yVal *tensor.Tensor, sd float32) {
	r := rand.New(rand.NewPCG(8, 0))
	opt := optim.NewAdam(model.Params(), 1e-3)
	fmt.Printf("\n%s (%d parameters)\n", name, nn.NumParams(model))
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
}

func main() {
	path := flag.String("data", "examples/airspace/testdata/counters.csv.gz", "airspace counters (csv.gz)")
	target := flag.String("airport", "EIDW", "ICAO code of the airport to forecast")
	flag.Parse()
	tensor.Seed(8)

	series, days, err := loadSeries(*path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if series[*target] == nil {
		fmt.Fprintf(os.Stderr, "no counters for %s; airports: %v\n", *target, airports)
		os.Exit(1)
	}
	fmt.Printf("%s: %d days of per-minute counts, %.0f aircraft in the zone at the busiest minute\n", *target, len(days), maxOf(series[*target]))
	fmt.Printf("each example: the last %d minutes, predicting the count %d minutes after the window\n", window, horizon)

	// The single-airport data set and the eight-airport one share the
	// same rows, so the split, the scaling and the targets are identical.
	x1, y, pos := windows(series, []string{*target}, *target)
	x8, _, _ := windows(series, airports, *target)

	// Split by the day the target falls on: days 1–4 train, day 5 validates,
	// day 6 (the storm) stays unseen until the end.
	var trainIdx, valIdx, stormIdx []int
	for i, t := range pos {
		switch day := (t + horizon - 1) / minutesPerDay; {
		case day < 4:
			trainIdx = append(trainIdx, i)
		case day == 4:
			valIdx = append(valIdx, i)
		default:
			stormIdx = append(stormIdx, i)
		}
	}
	fmt.Printf("%d examples train (%s to %s), %d validate (%s), %d are the storm day (%s)\n",
		len(trainIdx), days[0], days[3], len(valIdx), days[4], len(stormIdx), days[5])

	// Counts to a common scale, mean 0 and spread 1 over the training
	// targets; the same two numbers scale every split and every channel.
	mu, sd := meanStd(y.Rows(trainIdx).Float32s())
	scale := func(t *tensor.Tensor) *tensor.Tensor { return t.AddScalar(-mu).MulScalar(1 / sd) }
	split := func(x *tensor.Tensor) (tr, va, st *tensor.Tensor) {
		return scale(x.Rows(trainIdx)), scale(x.Rows(valIdx)), scale(x.Rows(stormIdx))
	}
	x1Train, x1Val, x1Storm := split(x1)
	x8Train, x8Val, x8Storm := split(x8)
	yTrain, yVal, yStorm := split(y)

	// Two baselines that cost nothing: "it stays as it is" and "as yesterday".
	persistence := func(xs, ys *tensor.Tensor) float64 {
		return metrics.MAE(xs.Narrow(1, window-1, 1), ys) * float64(sd)
	}
	yesterday := func(idx []int) float64 {
		var sum float64
		for _, i := range idx {
			t := pos[i] + horizon - 1
			sum += math.Abs(float64(series[*target][t] - series[*target][t-minutesPerDay]))
		}
		return sum / float64(len(idx))
	}
	fmt.Printf("\nbaselines on the validation day: persistence %.2f, same minute yesterday %.2f aircraft\n",
		persistence(x1Val, yVal), yesterday(valIdx))

	// Chapter 6's model on the airport alone: 120 unrelated columns.
	mlp1 := nn.Sequential{nn.NewLinear(window, 64), nn.ReLU{}, nn.NewLinear(64, 1)}
	train(fmt.Sprintf("flat MLP, %s alone", *target), mlp1, x1Train, yTrain, x1Val, yVal, sd)

	// The convolutional model on the airport alone: eight filters of width 7
	// slide along the two hours, eight more slide along their outputs
	// stepping four minutes at a time, a linear layer reads the result.
	cnn := func(channels int) nn.Sequential {
		second := nn.NewConv1D(8, 8, filterWidth)
		second.Stride = 4
		return nn.Sequential{toChannels{channels}, nn.NewConv1D(channels, 8, filterWidth), nn.ReLU{}, second, nn.ReLU{}, nn.Flatten{}, nn.NewLinear(8*window/4, 1)}
	}
	cnn1 := cnn(1)
	train(fmt.Sprintf("1-D CNN, %s alone", *target), cnn1, x1Train, yTrain, x1Val, yVal, sd)

	// The same two models reading all eight airports. For the MLP that is
	// 960 inputs; for the CNN it is eight channels under the same filters.
	mlp8 := nn.Sequential{nn.NewLinear(len(airports)*window, 64), nn.ReLU{}, nn.NewLinear(64, 1)}
	train("flat MLP, all eight airports", mlp8, x8Train, yTrain, x8Val, yVal, sd)
	cnn8 := cnn(len(airports))
	train("1-D CNN, all eight airports", cnn8, x8Train, yTrain, x8Val, yVal, sd)

	// What a first-layer filter learned: its seven weights on the target's
	// own channel, oldest minute first, and which other airport it weighs most.
	w := cnn8[1].(*nn.Conv1D).W.Float32s() // [8 filters × 8 channels × 7]
	ti := indexOf(airports, *target)
	fmt.Printf("\nfirst-layer filters of the eight-airport CNN, weights on %s's own minutes (oldest first):\n", *target)
	for f := 0; f < 3; f++ {
		fmt.Printf("  filter %d:", f)
		base := (f*len(airports) + ti) * filterWidth
		for _, v := range w[base : base+filterWidth] {
			fmt.Printf(" %6.2f", v)
		}
		best, bestNorm := "", 0.0
		for c, a := range airports {
			if c == ti {
				continue
			}
			var s float64
			for _, v := range w[(f*len(airports)+c)*filterWidth : (f*len(airports)+c+1)*filterWidth] {
				s += float64(v * v)
			}
			if s > bestNorm {
				best, bestNorm = a, s
			}
		}
		fmt.Printf("   listens most to %s among the others\n", best)
	}

	// The storm day: the models that read the neighbours expect a normal
	// day and are wrong by twice their usual error. That jump is the
	// signal the airspace example is built on.
	fmt.Println("\nmean absolute error in aircraft")
	fmt.Printf("%-26s %14s %10s\n", "", "validation day", "storm day")
	row := func(name string, val, storm float64) { fmt.Printf("%-26s %14.2f %10.2f\n", name, val, storm) }
	row("persistence", persistence(x1Val, yVal), persistence(x1Storm, yStorm))
	row("same minute yesterday", yesterday(valIdx), yesterday(stormIdx))
	row("flat MLP, alone", mae(mlp1, x1Val, yVal, sd), mae(mlp1, x1Storm, yStorm, sd))
	row("1-D CNN, alone", mae(cnn1, x1Val, yVal, sd), mae(cnn1, x1Storm, yStorm, sd))
	row("flat MLP, eight airports", mae(mlp8, x8Val, yVal, sd), mae(mlp8, x8Storm, yStorm, sd))
	row("1-D CNN, eight airports", mae(cnn8, x8Val, yVal, sd), mae(cnn8, x8Storm, yStorm, sd))
}

func indexOf(list []string, s string) int {
	for i, v := range list {
		if v == s {
			return i
		}
	}
	return -1
}

func maxOf(s []float32) float32 {
	var m float32
	for _, v := range s {
		m = max(m, v)
	}
	return m
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
