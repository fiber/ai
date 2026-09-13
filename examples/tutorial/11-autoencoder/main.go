// Tutorial chapter 10: the autoencoder — a model whose only output is
// how unusual its input is.
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
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

const minutesPerDay = 1440

var (
	airports = []string{"EIDW", "EGPH", "EGCC", "EGLL", "EHAM", "EBBR", "LFPG", "EDDF"}
	counters = []string{"in_zone", "arrivals", "departures"}
)

// loadMinutes returns one row per minute over the six days: for every
// airport its three counters, 24 numbers, plus the day labels.
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

// flatten stacks the rows; with clock set it appends two numbers per row
// that place the minute on the face of a 24-hour clock (sine and cosine of
// the time of day), so that "empty at 09:00" and "empty at 03:00" are
// different inputs.
func flatten(rows [][]float32, clock bool) *tensor.Tensor {
	var xs []float32
	width := len(rows[0])
	for i, r := range rows {
		xs = append(xs, r...)
		if clock {
			a := 2 * math.Pi * float64(i%minutesPerDay) / minutesPerDay
			xs = append(xs, float32(math.Sin(a)), float32(math.Cos(a)))
		}
	}
	if clock {
		width += 2
	}
	return tensor.New(xs, len(rows), width)
}

// score returns the reconstruction error of every row and, per row, the
// column that contributes most to it.
func score(enc, dec nn.Sequential, x *tensor.Tensor) (errs []float32, worst []int) {
	tensor.NoGrad(func() {
		sq := dec.Forward(enc.Forward(x)).Sub(x).Square()
		errs = sq.Sum(1).Float32s()
		worst = sq.Argmax(1)
	})
	return
}

func main() {
	path := flag.String("data", "examples/airspace/testdata/counters.csv.gz", "airspace counters (csv.gz)")
	flag.Parse()
	tensor.Seed(10)
	r := rand.New(rand.NewPCG(10, 0))

	rows, days, err := loadMinutes(*path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	names := make([]string, 0, 24)
	for _, a := range airports {
		for _, c := range counters {
			names = append(names, a+" "+c)
		}
	}
	fmt.Printf("%d minutes over %d days, %d numbers per minute (%d airports × %s)\n",
		len(rows), len(days), len(names), len(airports), strings.Join(counters, ", "))

	// Four ordinary days to learn what a minute looks like, one to check,
	// the storm day to score. Twice: without and with the clock.
	type fitted struct {
		enc, dec   nn.Sequential
		threshold  float32
		val, storm *tensor.Tensor
	}
	fit := func(clock bool) fitted {
		all := flatten(rows, clock)
		d := all.Dim(1)
		xTrain := all.Narrow(0, 0, 4*minutesPerDay)
		xVal := all.Narrow(0, 4*minutesPerDay, minutesPerDay)
		xStorm := all.Narrow(0, 5*minutesPerDay, minutesPerDay)

		// Every column to mean 0, spread 1, with the statistics of the
		// training days; a column that is almost always 0 keeps a spread of
		// at least 1, so a single arrival does not become a huge error.
		std := data.Fit(xTrain)
		for i, v := range std.Std.Float32s() {
			if v < 1 {
				std.Std.Data()[i] = 1
			}
		}
		xTrain, xVal, xStorm = std.Transform(xTrain), std.Transform(xVal), std.Transform(xStorm)

		// d numbers in, squeezed through 3, d numbers out. The model can only
		// reproduce what it has learned to compress: an ordinary minute.
		enc := nn.Sequential{nn.NewLinear(d, 16), nn.GELU{}, nn.NewLinear(16, 3)}
		dec := nn.Sequential{nn.NewLinear(3, 16), nn.GELU{}, nn.NewLinear(16, d)}
		opt := optim.NewAdam(append(enc.Params(), dec.Params()...), 2e-3)
		label := "without the clock"
		if clock {
			label = "with the clock"
		}
		fmt.Printf("\nautoencoder %d → 16 → 3 → 16 → %d, %d parameters, %s\n", d, d, nn.NumParams(enc)+nn.NumParams(dec), label)
		for epoch := 1; epoch <= 30; epoch++ {
			for idx := range data.Batches(xTrain.Dim(0), 64, r) {
				xb := xTrain.Rows(idx)
				loss := tensor.MSELoss(dec.Forward(enc.Forward(xb)), xb)
				opt.ZeroGrad()
				loss.Backward()
				opt.Step()
			}
			if epoch%10 == 0 {
				tr, _ := score(enc, dec, xTrain)
				va, _ := score(enc, dec, xVal)
				fmt.Printf("epoch %2d  mean error train %.2f  validation %.2f\n", epoch, mean(tr), mean(va))
			}
		}
		// The threshold: what the training days almost never exceed.
		trainErr, _ := score(enc, dec, xTrain)
		sorted := append([]float32(nil), trainErr...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		th := sorted[len(sorted)*999/1000]
		fmt.Printf("threshold %.1f (99.9th percentile of the training minutes)\n", th)
		return fitted{enc, dec, th, xVal, xStorm}
	}
	plain := fit(false)
	clock := fit(true)

	// Minutes over the threshold per hour, both models side by side.
	over := func(f fitted, x *tensor.Tensor) (perHour [24]int, total int, worst [24]string) {
		errs, w := score(f.enc, f.dec, x)
		var mx [24]float32
		for m, e := range errs {
			h := m / 60
			if e > f.threshold {
				perHour[h]++
				total++
			}
			if e > mx[h] {
				mx[h] = e
				worst[h] = names[w[m]]
			}
		}
		return
	}
	pv, pvTotal, _ := over(plain, plain.val)
	cv, cvTotal, _ := over(clock, clock.val)
	ps, psTotal, _ := over(plain, plain.storm)
	cs, csTotal, csWorst := over(clock, clock.storm)
	fmt.Printf("\nminutes over the threshold: validation day %s: %d without the clock, %d with; storm day %s: %d without, %d with\n",
		days[4], pvTotal, cvTotal, days[5], psTotal, csTotal)
	fmt.Println("\nstorm day by hour        without   with   largest contribution (with the clock)")
	for h := 0; h < 24; h++ {
		bar := strings.Repeat("#", cs[h])
		if cs[h] == 0 && ps[h] == 0 {
			fmt.Printf("%02d:00 %28s\n", h, "")
			continue
		}
		fmt.Printf("%02d:00 %19d %6d   %-22s %s\n", h, ps[h], cs[h], csWorst[h], bar)
	}
	_ = pv
	_ = cv

	// Minutes are noisy; hours are not. The same errors averaged per hour,
	// against the worst hour of the four training days.
	hourly := func(f fitted, x *tensor.Tensor) [24]float64 {
		errs, _ := score(f.enc, f.dec, x)
		var m [24]float64
		for i, e := range errs {
			m[i/60] += float64(e) / 60
		}
		return m
	}
	all := flatten(rows, true)
	std := data.Fit(all.Narrow(0, 0, 4*minutesPerDay))
	for i, v := range std.Std.Float32s() {
		if v < 1 {
			std.Std.Data()[i] = 1
		}
	}
	var worstTrainHour float64
	for d := 0; d < 4; d++ {
		for _, m := range hourly(clock, std.Transform(all.Narrow(0, d*minutesPerDay, minutesPerDay))) {
			worstTrainHour = max(worstTrainHour, m)
		}
	}
	hv, hs := hourly(clock, clock.val), hourly(clock, clock.storm)
	fmt.Printf("\nmean error per hour with the clock; the worst training hour was %.1f\n", worstTrainHour)
	fmt.Println("hour   validation   storm")
	for h := 0; h < 24; h++ {
		flag := ""
		if hs[h] > worstTrainHour {
			flag = "  <- above any training hour"
		}
		fmt.Printf("%02d:00 %10.1f %7.1f%s\n", h, hv[h], hs[h], flag)
	}
}

func mean(s []float32) float64 {
	var sum float64
	for _, v := range s {
		sum += float64(v)
	}
	return sum / float64(len(s))
}
