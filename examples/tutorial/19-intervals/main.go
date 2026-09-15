// Tutorial chapter 19: a forecast with an interval. Chapter 9 predicts
// one number thirty minutes ahead; this predicts three, the 10th, 50th
// and 90th percentile of what the count will be, using the pinball loss.
// Early stopping keeps the best epoch, and the trained model is written
// as safetensors so something other than this program can read it.
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
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/safetensors"
	"github.com/fiber/ai/tensor"
)

const (
	minutesPerDay = 1440
	window        = 120 // minutes of history per example
	horizon       = 30  // minutes ahead
	filterWidth   = 7
)

// quantiles: the middle one is the forecast, the outer two are the band.
var quantiles = []float32{0.1, 0.5, 0.9}

var airports = []string{"EIDW", "EGPH", "EGCC", "EGLL", "EHAM", "EBBR", "LFPG", "EDDF"}

// loadSeries reads the airspace counters: for every airport one value
// per minute over the six days in order, aircraft in its zone.
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

// windows builds one example per minute t: the last hist minutes of
// every channel, channel after channel, and the target's count horizon
// minutes after the window ends.
func windows(series map[string][]float32, target string, hist int) (x, y *tensor.Tensor, pos []int) {
	n := len(series[target])
	var xs, ys []float32
	for t := hist; t+horizon-1 < n; t++ {
		for _, a := range airports {
			xs = append(xs, series[a][t-hist:t]...)
		}
		ys = append(ys, series[target][t+horizon-1])
		pos = append(pos, t)
	}
	return tensor.New(xs, len(pos), len(airports)*hist), tensor.New(ys, len(pos), 1), pos
}

// toChannels reshapes a flat row into [batch, channels, length]. Unlike
// chapter 9's version it takes the length from the row, so the same
// model can be handed a longer history.
type toChannels struct{ channels int }

func (m toChannels) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Reshape(x.Dim(0), m.channels, x.Dim(1)/m.channels)
}
func (toChannels) Params() []*tensor.Tensor { return nil }

// forecaster: two convolutions over the history, then the length is
// averaged away and a linear layer produces one number per quantile.
// GlobalAvgPool1D rather than Flatten is what makes the model
// independent of how many minutes it is given.
func forecaster() nn.Sequential {
	second := nn.NewConv1D(16, 16, filterWidth)
	second.Stride = 4
	return nn.Sequential{
		toChannels{len(airports)},
		nn.NewConv1D(len(airports), 16, filterWidth), nn.ReLU{},
		second, nn.ReLU{},
		nn.GlobalAvgPool1D{},
		nn.NewLinear(16, len(quantiles)),
	}
}

// predict returns the three quantile outputs per row, in aircraft.
func predict(m nn.Module, x *tensor.Tensor, mu, sd float32) [][]float32 {
	var out [][]float32
	tensor.NoGrad(func() {
		p := m.Forward(x).MulScalar(sd).AddScalar(mu).Float32s()
		for i := 0; i < len(p); i += len(quantiles) {
			out = append(out, p[i:i+len(quantiles)])
		}
	})
	return out
}

// report measures what an interval is for: how often the truth falls
// inside it, how wide it is, and how good the middle line is on its own.
type report struct {
	coverage float64 // share of targets within [q10, q90]
	width    float64 // mean q90 − q10, in aircraft
	mae      float64 // mean absolute error of the median output
	crossed  int     // rows where the quantiles came out of order
}

func evaluate(m nn.Module, x, y *tensor.Tensor, mu, sd float32) report {
	preds := predict(m, x, mu, sd)
	truth := y.Float32s()
	var r report
	for i, p := range preds {
		t := truth[i]*sd + mu
		if t >= p[0] && t <= p[2] {
			r.coverage++
		}
		r.width += float64(p[2] - p[0])
		r.mae += math.Abs(float64(p[1] - t))
		if !(p[0] <= p[1] && p[1] <= p[2]) {
			r.crossed++
		}
	}
	n := float64(len(preds))
	r.coverage, r.width, r.mae = r.coverage/n*100, r.width/n, r.mae/n
	return r
}

func main() {
	path := flag.String("data", "examples/airspace/testdata/counters.csv.gz", "airspace counters (csv.gz)")
	target := flag.String("airport", "EIDW", "ICAO code of the airport to forecast")
	epochs := flag.Int("epochs", 60, "training epochs")
	out := flag.String("save", "", "write the trained model here as safetensors")
	flag.Parse()
	tensor.Seed(19)

	series, days, err := loadSeries(*path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	x, y, pos := windows(series, *target, window)

	// Split by the day the target falls on, as chapter 9 does: days 1-4
	// train, day 5 validates and chooses the epoch to keep, day 6 is the
	// storm and is not looked at until the end.
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
	fmt.Printf("%s, %d days of per-minute counts: %d examples train (%s to %s), %d validate (%s), %d the storm day (%s)\n",
		*target, len(days), len(trainIdx), days[0], days[3], len(valIdx), days[4], len(stormIdx), days[5])
	fmt.Printf("each example: %d minutes of all %d airports, predicting %s %d minutes after the window\n\n",
		window, len(airports), *target, horizon)

	// One scale for everything, from the training targets only. A
	// per-column Standardizer would be wrong here: the columns are the
	// same quantity at different minutes, so they must keep their
	// relative sizes.
	mu, sd := meanStd(y.Rows(trainIdx).Float32s())
	scale := func(t *tensor.Tensor) *tensor.Tensor { return t.AddScalar(-mu).MulScalar(1 / sd) }
	xTrain, xVal, xStorm := scale(x.Rows(trainIdx)), scale(x.Rows(valIdx)), scale(x.Rows(stormIdx))
	yTrain, yVal, yStorm := scale(y.Rows(trainIdx)), scale(y.Rows(valIdx)), scale(y.Rows(stormIdx))

	m := forecaster()
	opt := optim.NewAdamW(m.Params(), 1e-3, 1e-2)
	r := rand.New(rand.NewPCG(19, 0))
	fmt.Printf("%d parameters, three outputs per example: the %v quantiles\n", nn.NumParams(m), quantiles)

	// Early stopping: keep the parameters from the epoch with the best
	// validation loss, not the ones training happens to end on.
	best := nn.NewSnapshot(m)
	bestLoss, bestEpoch := float32(math.Inf(1)), 0
	valLoss := func() float32 {
		var v float32
		tensor.NoGrad(func() { v = tensor.PinballLoss(m.Forward(xVal), yVal.Reshape(yVal.Dim(0)), quantiles...).Item() })
		return v
	}
	for epoch := 1; epoch <= *epochs; epoch++ {
		for idx := range data.Batches(xTrain.Dim(0), 64, r) {
			yb := yTrain.Rows(idx)
			loss := tensor.PinballLoss(m.Forward(xTrain.Rows(idx)), yb.Reshape(yb.Dim(0)), quantiles...)
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
		}
		v := valLoss()
		if v < bestLoss {
			bestLoss, bestEpoch = v, epoch
			best.Capture(m)
		}
		if epoch <= 8 || epoch%10 == 0 {
			mark := ""
			if epoch == bestEpoch {
				mark = "  (best so far)"
			}
			fmt.Printf("  epoch %3d  validation pinball %.4f%s\n", epoch, v, mark)
		}
	}
	last := valLoss()
	if err := best.Restore(m); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("\nkept epoch %d (validation pinball %.4f); the last epoch was %.4f, %+.1f%%\n",
		bestEpoch, bestLoss, last, 100*float64(last-bestLoss)/float64(bestLoss))

	val, storm := evaluate(m, xVal, yVal, mu, sd), evaluate(m, xStorm, yStorm, mu, sd)
	fmt.Printf("\n%-16s %12s %14s %14s\n", "", "coverage", "band width", "median MAE")
	fmt.Printf("%-16s %11.1f%% %11.1f ac %11.2f ac\n", "validation day", val.coverage, val.width, val.mae)
	fmt.Printf("%-16s %11.1f%% %11.1f ac %11.2f ac\n", "storm day", storm.coverage, storm.width, storm.mae)
	fmt.Printf("\nthe band is meant to hold %.0f%% of the targets\n", 100*float64(quantiles[len(quantiles)-1]-quantiles[0]))
	if val.crossed+storm.crossed > 0 {
		fmt.Printf("quantiles came out of order on %d of %d rows\n", val.crossed+storm.crossed, len(valIdx)+len(stormIdx))
	} else {
		fmt.Println("the three outputs came out in order on every row")
	}

	// Widest and narrowest hours of the storm day: where the model knows
	// it does not know.
	preds := predict(m, xStorm, mu, sd)
	truth := yStorm.Float32s()
	type hour struct {
		h                int
		width, cover, at float64
	}
	byHour := map[int]*hour{}
	for i, p := range preds {
		h := ((pos[stormIdx[i]] + horizon - 1) % minutesPerDay) / 60
		if byHour[h] == nil {
			byHour[h] = &hour{h: h}
		}
		e := byHour[h]
		e.width += float64(p[2] - p[0])
		t := truth[i]*sd + mu
		if t >= p[0] && t <= p[2] {
			e.cover++
		}
		e.at++
	}
	var hours []*hour
	for _, e := range byHour {
		hours = append(hours, e)
	}
	sort.Slice(hours, func(i, j int) bool { return hours[i].width/hours[i].at > hours[j].width/hours[j].at })
	fmt.Println("\nstorm day by hour, widest bands first")
	fmt.Printf("  %-6s %12s %10s\n", "hour", "band width", "coverage")
	for _, e := range hours[:3] {
		fmt.Printf("  %02d:00  %9.1f ac %8.0f%%\n", e.h, e.width/e.at, 100*e.cover/e.at)
	}
	for _, e := range hours[len(hours)-2:] {
		fmt.Printf("  %02d:00  %9.1f ac %8.0f%%\n", e.h, e.width/e.at, 100*e.cover/e.at)
	}

	// The encoder averages over the length, so nothing in the model
	// depends on it being 120 minutes. Whether that is useful is a
	// different question from whether it runs.
	xLong, yLong, longPos := windows(series, *target, 240)
	var longIdx []int
	for i, t := range longPos {
		if (t+horizon-1)/minutesPerDay == 4 {
			longIdx = append(longIdx, i)
		}
	}
	long := evaluate(m, scale(xLong.Rows(longIdx)), scale(yLong.Rows(longIdx)), mu, sd)
	fmt.Printf("\nthe same trained model on 240 minutes of history instead of 120:\n")
	fmt.Printf("  validation day: coverage %.1f%%, band %.1f ac, median MAE %.2f ac\n", long.coverage, long.width, long.mae)

	if *out != "" {
		tensors := map[string]*tensor.Tensor{}
		names := []string{"conv1.w", "conv1.b", "conv2.w", "conv2.b", "head.w", "head.b"}
		for i, p := range m.Params() {
			tensors[names[i]] = p
		}
		meta := map[string]string{
			"quantiles": fmt.Sprint(quantiles),
			"target":    *target,
			"horizon":   fmt.Sprint(horizon),
			"scale":     fmt.Sprintf("mean %.4f sd %.4f", mu, sd),
		}
		if err := safetensors.Save(*out, tensors, meta); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		f, err := safetensors.Open(*out)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer f.Close()
		fmt.Printf("\nwrote %s:\n", *out)
		for _, n := range f.Names() {
			info, _ := f.Info(n)
			fmt.Printf("  %-10s %-5s %v\n", n, info.Dtype, info.Shape)
		}
		fmt.Printf("  metadata: %v\n", f.Metadata())
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
