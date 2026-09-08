package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"strings"

	"github.com/fiber/ai/data"
	"github.com/fiber/ai/logtemplate"
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

const (
	window        = 24 // minutes of history the forecast sees
	horizon       = 15 // minutes ahead
	minutesPerDay = 1440
)

// forecaster is one model for all airports and one counter: the last 24
// minutes (standardised per airport), time of day and day of week on a
// circle, a one-hot of the airport; two outputs, expected value and log
// spread. Trained on movements per minute and, separately, on aircraft in
// the zone.
type forecaster struct {
	name     string
	model    nn.Sequential
	mean     []float32
	std      []float32
	n        int
	features int
	floor    float32 // smallest spread the band may have, in the counter's unit
}

func (f *forecaster) featureRow(hist []float32, ap, minute int) []float32 {
	row := make([]float32, 0, f.features)
	for _, v := range hist[len(hist)-window:] {
		row = append(row, (v-f.mean[ap])/f.std[ap])
	}
	row = append(row, timeFeatures(minute)...)
	for i := 0; i < f.n; i++ {
		if i == ap {
			row = append(row, 1)
		} else {
			row = append(row, 0)
		}
	}
	return row
}

// timeFeatures: minute counts from the first training day's midnight UTC.
func timeFeatures(minute int) []float32 {
	day := float64(minute%minutesPerDay) / minutesPerDay
	week := (float64((minute/minutesPerDay)%7) + day) / 7
	return []float32{float32(math.Sin(2 * math.Pi * day)), float32(math.Cos(2 * math.Pi * day)),
		float32(math.Sin(2 * math.Pi * week)), float32(math.Cos(2 * math.Pi * week))}
}

// trainForecaster builds examples from history[airport][minute].
func trainForecaster(name string, history [][]float32, floor float32, r *rand.Rand, epochs int, log func(string)) *forecaster {
	k := len(history)
	f := &forecaster{name: name, n: k, features: window + 4 + k, mean: make([]float32, k), std: make([]float32, k), floor: floor}
	for i, h := range history {
		var s, s2 float64
		for _, v := range h {
			s += float64(v)
			s2 += float64(v) * float64(v)
		}
		m := s / float64(len(h))
		f.mean[i] = float32(m)
		f.std[i] = float32(math.Max(float64(floor), math.Sqrt(math.Max(0, s2/float64(len(h))-m*m))))
	}
	var xs, ys []float32
	n := 0
	for i, h := range history {
		for t := window; t+horizon-1 < len(h); t++ {
			xs = append(xs, f.featureRow(h[:t], i, t)...)
			ys = append(ys, (h[t+horizon-1]-f.mean[i])/f.std[i])
			n++
		}
	}
	x := tensor.New(xs, n, f.features)
	y := tensor.New(ys, n, 1)
	f.model = nn.Sequential{nn.NewLinear(f.features, 64), nn.ReLU{}, nn.NewLinear(64, 64), nn.ReLU{}, nn.NewLinear(64, 2)}
	opt := optim.NewAdam(f.model.Params(), 1e-3)
	for e := 1; e <= epochs; e++ {
		var total float64
		batches := 0
		for idx := range data.Batches(n, 512, r) {
			out := f.model.Forward(x.Rows(idx))
			mu, ls := out.Narrow(1, 0, 1), out.Narrow(1, 1, 1)
			z := y.Rows(idx).Sub(mu).Div(ls.Exp())
			loss := z.Square().MulScalar(0.5).Add(ls).Mean()
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
			total += float64(loss.Item())
			batches++
		}
		if e == 1 || e%4 == 0 || e == epochs {
			log(fmt.Sprintf("forecast %s: epoch %d/%d loss %.3f (%d examples)", name, e, epochs, total/float64(batches), n))
		}
	}
	return f
}

// predict returns expected value and spread for airport ap at
// minute+horizon given its history up to minute.
func (f *forecaster) predict(hist []float32, ap, minute int) (mu, sigma float32) {
	tensor.NoGrad(func() {
		out := f.model.Forward(tensor.New(f.featureRow(hist, ap, minute), 1, f.features))
		mu = out.At(0, 0)*f.std[ap] + f.mean[ap]
		sigma = float32(math.Exp(float64(out.At(0, 1)))) * f.std[ap]
		out.Release()
	})
	if sigma < f.floor {
		sigma = f.floor
	}
	return mu, sigma
}

// autoencoder over the vector of all airports' counters for one minute.
type autoencoder struct {
	enc, dec  nn.Sequential
	mean, std []float32 // per input column; std floored so a rare count cannot dominate
	threshold float32
	names     []string
}

// scale standardises rows [n×d] with the autoencoder's statistics.
func (a *autoencoder) scale(x *tensor.Tensor) *tensor.Tensor {
	return x.Sub(tensor.New(a.mean, 1, len(a.mean))).Div(tensor.New(a.std, 1, len(a.std)))
}

// stdFloor keeps a column with almost no variance in training (a count
// that is usually 0) from turning a single 1 into a huge standardised error.
const stdFloor = 1.0

func trainAutoencoder(rows *tensor.Tensor, names []string, r *rand.Rand, epochs int, log func(string)) *autoencoder {
	d := rows.Dim(1)
	a := &autoencoder{names: names, mean: make([]float32, d), std: make([]float32, d)}
	raw := rows.Float32s()
	nrows := rows.Dim(0)
	for j := 0; j < d; j++ {
		var s1, s2 float64
		for i := 0; i < nrows; i++ {
			v := float64(raw[i*d+j])
			s1 += v
			s2 += v * v
		}
		m := s1 / float64(nrows)
		a.mean[j] = float32(m)
		a.std[j] = float32(math.Max(stdFloor, math.Sqrt(math.Max(0, s2/float64(nrows)-m*m))))
	}
	x := a.scale(rows)
	a.enc = nn.Sequential{nn.NewLinear(d, 32), nn.GELU{}, nn.NewLinear(32, 6)}
	a.dec = nn.Sequential{nn.NewLinear(6, 32), nn.GELU{}, nn.NewLinear(32, d)}
	opt := optim.NewAdam(append(a.enc.Params(), a.dec.Params()...), 2e-3)
	n := x.Dim(0)
	for e := 1; e <= epochs; e++ {
		var total float64
		b := 0
		for idx := range data.Batches(n, 256, r) {
			xb := x.Rows(idx)
			loss := tensor.MSELoss(a.dec.Forward(a.enc.Forward(xb)), xb)
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
			total += float64(loss.Item())
			b++
		}
		if e%10 == 0 || e == epochs {
			log(fmt.Sprintf("autoencoder: epoch %d/%d loss %.4f", e, epochs, total/float64(b)))
		}
	}
	scores := make([]float32, n)
	tensor.NoGrad(func() {
		rec := a.dec.Forward(a.enc.Forward(x))
		copy(scores, rec.Sub(x).Square().Sum(1).Float32s())
	})
	sort.Slice(scores, func(i, j int) bool { return scores[i] < scores[j] })
	a.threshold = scores[int(float64(n)*0.999)]
	log(fmt.Sprintf("autoencoder: threshold %.2f (99.9th percentile of %d training minutes)", a.threshold, n))
	return a
}

// score returns the reconstruction error of one minute and the counter
// that contributes most to it.
func (a *autoencoder) score(counters []float32) (score float32, worst string) {
	tensor.NoGrad(func() {
		x := a.scale(tensor.New(counters, 1, len(counters)))
		rec := a.dec.Forward(a.enc.Forward(x))
		sq := rec.Sub(x).Square()
		score = sq.Sum().Item()
		worst = a.names[sq.Argmax(1)[0]]
	})
	return
}

// metarWatch mines the reports into templates and keeps, per station, the
// training histogram of templates and per-template rates. It raises no
// first-seen alarms: a new template counts as a shift of the mix like any
// other, and a template far above its usual rate is a storm.
type metarWatch struct {
	miner    *logtemplate.Miner
	known    int
	hist     map[string][]float64 // station -> training histogram over templates (normalised)
	rate     map[string][]float64 // station -> reports per hour per template (training)
	recent   map[string][]int     // station -> template ids of the last hour
	recentAt map[string][]int     // matching minutes
	shiftRef map[string]float64   // station -> typical hourly shift during training (95th percentile)
}

func newMetarWatch() *metarWatch {
	return &metarWatch{miner: logtemplate.NewMiner(), hist: map[string][]float64{}, rate: map[string][]float64{},
		recent: map[string][]int{}, recentAt: map[string][]int{}, shiftRef: map[string]float64{}}
}

// train mines the training reports (station, minute, raw) and derives the
// per-station statistics.
func (w *metarWatch) train(reports []timedReport, trainMinutes int, log func(string)) {
	ids := make([]int, len(reports))
	for i, r := range reports {
		ids[i] = w.miner.Add(r.Raw)
	}
	w.known = w.miner.Len()
	perStation := map[string][]int{}
	for i, r := range reports {
		perStation[r.Station] = append(perStation[r.Station], i)
	}
	hours := math.Max(1, float64(trainMinutes)/60)
	for st, idx := range perStation {
		h := make([]float64, w.known)
		for _, i := range idx {
			h[ids[i]]++
		}
		rate := make([]float64, w.known)
		for t := range h {
			rate[t] = h[t] / hours
		}
		w.rate[st] = rate
		tot := float64(len(idx))
		for t := range h {
			h[t] /= tot
		}
		w.hist[st] = h
		// Typical shift: Hellinger distance of each training hour's mix to the whole.
		var shifts []float64
		byHour := map[int][]int{}
		for _, i := range idx {
			byHour[reports[i].Minute/60] = append(byHour[reports[i].Minute/60], ids[i])
		}
		for _, hourIDs := range byHour {
			shifts = append(shifts, hellinger(w.histogram(hourIDs), h))
		}
		sort.Float64s(shifts)
		if len(shifts) > 0 {
			w.shiftRef[st] = shifts[int(float64(len(shifts)-1)*0.95)]
		}
	}
	log(fmt.Sprintf("metar: %d reports → %d templates", len(reports), w.known))
}

func (w *metarWatch) histogram(ids []int) []float64 {
	h := make([]float64, max(w.known, w.miner.Len()))
	for _, id := range ids {
		if id < len(h) {
			h[id]++
		}
	}
	for i := range h {
		h[i] /= float64(max(1, len(ids)))
	}
	return h
}

// hellinger distance between two discrete distributions (0 identical, 1
// disjoint); b may be shorter than a (templates unseen in training).
func hellinger(a, b []float64) float64 {
	var s float64
	for i := range a {
		bi := 0.0
		if i < len(b) {
			bi = b[i]
		}
		d := math.Sqrt(a[i]) - math.Sqrt(bi)
		s += d * d
	}
	return math.Sqrt(s / 2)
}

// metarObservation is what one report contributes to the feeds.
type metarObservation struct {
	Template string
	Shift    float64 // Hellinger distance of the station's last hour to training
	ShiftRef float64
	Storm    string // set when this template's hourly rate is far above training
}

// observe feeds one report at a minute and returns its template and the
// station's current mix shift.
func (w *metarWatch) observe(station string, minute int, raw string) metarObservation {
	id := w.miner.Add(raw)
	w.recent[station] = append(w.recent[station], id)
	w.recentAt[station] = append(w.recentAt[station], minute)
	// drop reports older than an hour
	ids, at := w.recent[station], w.recentAt[station]
	cut := 0
	for cut < len(at) && at[cut] < minute-59 {
		cut++
	}
	w.recent[station], w.recentAt[station] = ids[cut:], at[cut:]
	obs := metarObservation{Template: strings.TrimSpace(w.miner.Template(id).String()), ShiftRef: w.shiftRef[station]}
	if ref, ok := w.hist[station]; ok {
		obs.Shift = hellinger(w.histogram(w.recent[station]), ref)
	}
	if id < w.known {
		n := 0
		for _, x := range w.recent[station] {
			if x == id {
				n++
			}
		}
		if r := w.rate[station]; r != nil && float64(n) > r[id]*4+3 {
			obs.Storm = fmt.Sprintf("%d reports in an hour, usually %.1f", n, r[id])
		}
	}
	return obs
}

// timedReport is a METAR with its minute index in the training/replay clock.
type timedReport struct {
	Station string
	Minute  int
	Raw     string
}
