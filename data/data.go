// Package data prepares measured series for a model: counters to rates,
// sliding windows with time features, a split by time, standardisation
// with statistics that are kept, and shuffled mini-batches. Everything
// here runs once per data set; nothing is on a hot path.
package data

import (
	"iter"
	"math"
	"math/rand/v2"
	"time"

	"github.com/fiber/ai/tensor"
)

// Rates turns cumulative counter samples (bytes, packets, errors) taken
// every interval into per-second rates. A negative delta means the
// counter was reset, and the new value is taken as the delta; a rate
// above maxRate (the link speed, in the same unit per second) cannot be
// real and is treated the same way. A NaN sample marks a gap: the rate
// there and the one after it are NaN. The result has one entry fewer
// than the input.
func Rates(counter []float64, interval time.Duration, maxRate float64) []float32 {
	if len(counter) < 2 {
		return nil
	}
	sec := interval.Seconds()
	out := make([]float32, len(counter)-1)
	for i := 1; i < len(counter); i++ {
		prev, cur := counter[i-1], counter[i]
		if math.IsNaN(prev) || math.IsNaN(cur) {
			out[i-1] = float32(math.NaN())
			continue
		}
		d := cur - prev
		if d < 0 || (maxRate > 0 && d/sec > maxRate) {
			d = cur // reset: the counter started again from zero
			if maxRate > 0 && d/sec > maxRate {
				out[i-1] = float32(math.NaN())
				continue
			}
		}
		out[i-1] = float32(d / sec)
	}
	return out
}

// TimeFeatures encodes the time of day and the day of week as points on
// a circle: [sin(day) cos(day) sin(week) cos(week)], so that 23:59 is
// next to 00:00 and Sunday next to Monday.
func TimeFeatures(t time.Time) []float32 {
	day := float64(t.Hour()*3600+t.Minute()*60+t.Second()) / 86400
	week := (float64(int(t.Weekday())) + day) / 7
	return []float32{
		float32(math.Sin(2 * math.Pi * day)), float32(math.Cos(2 * math.Pi * day)),
		float32(math.Sin(2 * math.Pi * week)), float32(math.Cos(2 * math.Pi * week)),
	}
}

// Windows slides over a series and builds one example per position t:
// the input is series[t-window:t] followed by extra(t) (nil for none),
// the target is series[t+horizon-1], i.e. horizon steps after the last
// input. Positions whose window or target contains NaN are skipped.
// Returns inputs [n × (window+len(extra))], targets [n × 1] and the
// positions used.
func Windows(series []float32, window, horizon int, extra func(t int) []float32) (x, y *tensor.Tensor, positions []int) {
	if window <= 0 || horizon <= 0 {
		panic("data: window and horizon must be positive")
	}
	var xs, ys []float32
	width := -1
	for t := window; t+horizon-1 < len(series); t++ {
		target := series[t+horizon-1]
		if isNaN32(target) || hasNaN(series[t-window:t]) {
			continue
		}
		row := series[t-window : t]
		var ex []float32
		if extra != nil {
			ex = extra(t)
		}
		if width < 0 {
			width = window + len(ex)
		} else if window+len(ex) != width {
			panic("data: extra must return the same number of features for every position")
		}
		xs = append(xs, row...)
		xs = append(xs, ex...)
		ys = append(ys, target)
		positions = append(positions, t)
	}
	if len(positions) == 0 {
		return tensor.Zeros(0, window), tensor.Zeros(0, 1), nil
	}
	return tensor.New(xs, len(positions), width), tensor.New(ys, len(positions), 1), positions
}

func isNaN32(v float32) bool { return v != v }

func hasNaN(s []float32) bool {
	for _, v := range s {
		if v != v {
			return true
		}
	}
	return false
}

// SplitByTime returns the leading trainFraction of the rows for training
// and the rest for validation, as views: with time-ordered data this is
// the only split that does not leak the future into the past.
func SplitByTime(x, y *tensor.Tensor, trainFraction float64) (xTrain, yTrain, xVal, yVal *tensor.Tensor) {
	n := x.Dim(0)
	if y.Dim(0) != n {
		panic("data: x and y have different numbers of rows")
	}
	k := int(float64(n) * trainFraction)
	k = max(0, min(n, k))
	return x.Narrow(0, 0, k), y.Narrow(0, 0, k), x.Narrow(0, k, n-k), y.Narrow(0, k, n-k)
}

// Standardizer shifts and scales every column to mean 0 and spread 1
// with statistics from the training data. Keep Mean and Std with the
// model: whatever it sees later must be transformed with the same
// numbers.
type Standardizer struct {
	Mean, Std *tensor.Tensor // shape [columns]
}

// Fit computes the column statistics of x ([rows × columns]). Columns
// with zero spread get spread 1 so that they map to zero, not NaN.
func Fit(x *tensor.Tensor) *Standardizer {
	std := x.Std(0)
	sd := std.Data()
	for i, v := range sd {
		if v == 0 || v != v {
			sd[i] = 1
		}
	}
	return &Standardizer{Mean: x.Mean(0), Std: std}
}

// Transform returns (x − mean) / std.
func (s *Standardizer) Transform(x *tensor.Tensor) *tensor.Tensor { return x.Sub(s.Mean).Div(s.Std) }

// Inverse returns x · std + mean, back to the original units.
func (s *Standardizer) Inverse(x *tensor.Tensor) *tensor.Tensor { return x.Mul(s.Std).Add(s.Mean) }

// Batches yields index slices of the given size covering 0..n-1 exactly
// once in a fresh random order; the last slice may be shorter. Use it
// with Tensor.Rows:
//
//	for idx := range data.Batches(n, 32, r) {
//	    xb, yb := x.Rows(idx), y.Rows(idx)
//	}
func Batches(n, size int, r *rand.Rand) iter.Seq[[]int] {
	return func(yield func([]int) bool) {
		if n <= 0 || size <= 0 {
			return
		}
		perm := r.Perm(n)
		for lo := 0; lo < n; lo += size {
			if !yield(perm[lo:min(lo+size, n)]) {
				return
			}
		}
	}
}
