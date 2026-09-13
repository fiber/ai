// Tutorial chapter 12: the autoencoder of chapter 11 produces a number
// per minute. This chapter turns that number into a decision, which is
// where most of the work in an operational system actually is.
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

const minutesPerDay = 1440

var (
	airports = []string{"EIDW", "EGPH", "EGCC", "EGLL", "EHAM", "EBBR", "LFPG", "EDDF"}
	counters = []string{"in_zone", "arrivals", "departures"}
)

func loadMinutes(path string) (rows [][]float32, days []string, err error) {
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
	type key struct {
		day    string
		minute int
	}
	byKey := map[key][]float32{}
	seen := map[string]bool{}
	width := len(airports) * len(counters)
	for {
		rec, err := rd.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		a := indexOf(airports, rec[col["airport"]])
		if a < 0 {
			continue
		}
		minute, _ := strconv.Atoi(rec[col["minute"]])
		k := key{rec[col["date"]], minute}
		seen[k.day] = true
		if byKey[k] == nil {
			byKey[k] = make([]float32, width)
		}
		for c, name := range counters {
			v, _ := strconv.ParseFloat(rec[col[name]], 32)
			byKey[k][a*len(counters)+c] = float32(v)
		}
	}
	for d := range seen {
		days = append(days, d)
	}
	sort.Strings(days)
	for _, d := range days {
		for m := 0; m < minutesPerDay; m++ {
			rows = append(rows, byKey[key{d, m}])
		}
	}
	return rows, days, nil
}

func indexOf(list []string, s string) int {
	for i, v := range list {
		if v == s {
			return i
		}
	}
	return -1
}

// flatten stacks the rows and appends the minute's place on a 24-hour
// clock, so that "empty at 09:00" and "empty at 03:00" are different.
func flatten(rows [][]float32) *tensor.Tensor {
	var xs []float32
	for i, r := range rows {
		xs = append(xs, r...)
		a := 2 * math.Pi * float64(i%minutesPerDay) / minutesPerDay
		xs = append(xs, float32(math.Sin(a)), float32(math.Cos(a)))
	}
	return tensor.New(xs, len(rows), len(rows[0])+2)
}

// hourly averages a per-minute score into 24 numbers a day. Chapter 11
// found minutes too noisy to act on; this is the same reduction, and it
// is also the first and cheapest way to cut a false-alarm rate.
func hourly(perMinute []float32) []float32 {
	out := make([]float32, len(perMinute)/60)
	for h := range out {
		var sum float32
		for i := range 60 {
			sum += perMinute[h*60+i]
		}
		out[h] = sum / 60
	}
	return out
}

func quantile(sorted []float32, q float64) float32 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(q * float64(len(sorted)-1))
	return sorted[i]
}

func main() {
	path := flag.String("data", "examples/airspace/testdata/counters.csv.gz", "airspace counters (csv.gz)")
	epochs := flag.Int("epochs", 40, "training epochs")
	flag.Parse()
	tensor.Seed(10)
	r := rand.New(rand.NewPCG(10, 0))

	rows, days, err := loadMinutes(*path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	all := flatten(rows)
	d := all.Dim(1)
	xTrain := all.Narrow(0, 0, 4*minutesPerDay)
	xVal := all.Narrow(0, 4*minutesPerDay, minutesPerDay)
	xStorm := all.Narrow(0, 5*minutesPerDay, minutesPerDay)
	fmt.Printf("%d days: %v — four to learn, one quiet to check, one with the storm\n\n", len(days), days)

	std := data.Fit(xTrain)
	for i, v := range std.Std.Float32s() {
		if v < 1 {
			std.Std.Data()[i] = 1
		}
	}
	enc := nn.Sequential{nn.NewLinear(d, 16), nn.GELU{}, nn.NewLinear(16, 3)}
	dec := nn.Sequential{nn.NewLinear(3, 16), nn.GELU{}, nn.NewLinear(16, d)}
	params := append(enc.Params(), dec.Params()...)
	opt := optim.NewAdam(params, 3e-3)
	zTrain := std.Transform(xTrain)
	for range *epochs {
		for idx := range data.Batches(zTrain.Dim(0), 64, r) {
			xb := zTrain.Rows(idx)
			loss := tensor.MSELoss(dec.Forward(enc.Forward(xb)), xb)
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
		}
	}

	score := func(x *tensor.Tensor) []float32 {
		var errs []float32
		tensor.NoGrad(func() {
			z := std.Transform(x)
			errs = dec.Forward(enc.Forward(z)).Sub(z).Square().Sum(1).Float32s()
		})
		return errs
	}
	trainH, valH, stormH := hourly(score(xTrain)), hourly(score(xVal)), hourly(score(xStorm))

	sorted := append([]float32(nil), trainH...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	fmt.Println("hourly reconstruction error")
	fmt.Printf("  four training days (%d hours): median %.2f, 90th %.2f, worst %.2f\n",
		len(trainH), quantile(sorted, 0.5), quantile(sorted, 0.9), quantile(sorted, 1))
	sv := append([]float32(nil), valH...)
	sort.Slice(sv, func(i, j int) bool { return sv[i] < sv[j] })
	fmt.Printf("  the quiet validation day:      median %.2f, worst %.2f\n", quantile(sv, 0.5), quantile(sv, 1))
	ss := append([]float32(nil), stormH...)
	sort.Slice(ss, func(i, j int) bool { return ss[i] < ss[j] })
	fmt.Printf("  the storm day:                 median %.2f, worst %.2f\n\n", quantile(ss, 0.5), quantile(ss, 1))

	// There are no labels. Nobody marked which minutes of the storm day
	// "should" have fired, and inventing them from the same statistic the
	// threshold uses would make any threshold look perfect. What we do
	// have is a day believed to be ordinary and a day known to be
	// disrupted, so the honest question is: how often does a threshold
	// fire on the quiet day, and how much of the storm day does it see?
	fire := func(hours []float32, t float32) int {
		n := 0
		for _, e := range hours {
			if e > t {
				n++
			}
		}
		return n
	}

	fmt.Println("threshold        quiet day     storm day    alarms/day    per operator")
	fmt.Println("                   fires         fires      one airport    (8 airports)")
	for _, q := range []float64{0.50, 0.75, 0.90, 0.99, 1.00} {
		t := quantile(sorted, q)
		fp, tp := fire(valH, t), fire(stormH, t)
		perDay := float64(fp)
		fmt.Printf("  q=%.2f %6.2f   %3d of 24     %3d of 24      %5.1f          %5.1f\n",
			q, t, fp, tp, perDay, 8*perDay)
	}

	// The one thing we can label with confidence: the quiet day should
	// produce no alarms at all. Scored as a classifier, at the threshold
	// most people reach for first.
	t90 := quantile(sorted, 0.90)
	pred := make([]int, 0, 48)
	truth := make([]int, 0, 48)
	for _, e := range valH {
		truth = append(truth, 0)
		pred = append(pred, btoi(e > t90))
	}
	for _, e := range stormH {
		truth = append(truth, 1) // the storm day, coarsely: it was disrupted
		pred = append(pred, btoi(e > t90))
	}
	m := metrics.Confusion(pred, truth, 2)
	fmt.Printf("\nAt q=0.90, scoring the storm day as one long event: precision %.0f%%, recall %.0f%%, accuracy %.0f%%\n",
		100*m.Precision(1), 100*m.Recall(1), 100*m.Accuracy())

	never := make([]int, len(truth))
	nm := metrics.Confusion(never, truth, 2)
	fmt.Printf("A detector that never fires at all:                accuracy %.0f%%\n", 100*nm.Accuracy())
	fmt.Printf("  — and if storms came one day in sixty instead of one in six, %.1f%%.\n", 100*(1-1.0/60))

	fmt.Printf("\nEight airports on this feed, so an operator sees the last column.\n")
	fmt.Printf("Four days of ordinary weather is 96 hours: not enough to place a\n")
	fmt.Printf("threshold, whatever the arithmetic above suggests.\n")
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}
