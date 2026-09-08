package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/fiber/ai/tensor"
)

// alarm is a model alarm; event is a notable fact that needs no model.
type alarm struct {
	Minute  int    `json:"minute"`
	Airport string `json:"airport"`
	Kind    string `json:"kind"` // "band", "score", "metar"
	Text    string `json:"text"`
}

type event struct {
	Minute  int    `json:"minute"`
	Airport string `json:"airport"`
	Kind    string `json:"kind"`
	Text    string `json:"text"`
}

type point struct {
	Minute int     `json:"m"`
	Value  float32 `json:"v"`
	Mu     float32 `json:"mu"`
	Sigma  float32 `json:"sigma"`
}

type airportState struct {
	ICAO      string             `json:"icao"`
	Name      string             `json:"name"`
	Movements []point            `json:"movements"` // last 3 hours, with band
	InZone    []point            `json:"inZone"`
	Now       [nCounters]float32 `json:"now"`
	Metar     string             `json:"metar"`
	Template  string             `json:"template"`
	Shift     float64            `json:"shift"`
	ShiftRef  float64            `json:"shiftRef"`
	Outside   int                `json:"outside"`  // consecutive minutes outside the band
	Aircraft  [][4]float32       `json:"aircraft"` // dx, dy, alt, holding flag for the radar
	State     string             `json:"state"`    // "normal", "watch", "alarm"
	Reason    string             `json:"reason"`   // one sentence: what deviates from what
	Weather   weather            `json:"weather"`
	Expected  float32            `json:"expected"` // movements expected in the last 30 min
	Observed  float32            `json:"observed"`
}

// weather is the decoded current report of a station.
type weather struct {
	WindDir int    `json:"windDir"`
	WindKt  int    `json:"windKt"`
	GustKt  int    `json:"gustKt"`
	VisM    int    `json:"visM"`
	Wx      string `json:"wx"` // present weather in words
	Speci   bool   `json:"speci"`
	Time    string `json:"time"`
}

type scorePoint struct {
	Minute int     `json:"m"`
	Score  float32 `json:"score"`
	Worst  string  `json:"worst"`
}

type state struct {
	Mode      string         `json:"mode"`
	Date      string         `json:"date"`
	Minute    int            `json:"minute"`
	Clock     string         `json:"clock"`
	Airports  []airportState `json:"airports"`
	Scores    []scorePoint   `json:"scores"`
	Threshold float32        `json:"threshold"`
	Alarms    []alarm        `json:"alarms"`
	Events    []event        `json:"events"`
	Templates []templateRow  `json:"templates"`
	Training  []string       `json:"training"`
	Headline  string         `json:"headline"`
	DayAlarms []alarm        `json:"dayAlarms"` // the whole day, for the timeline
	DayEvents []event        `json:"dayEvents"`
}

type templateRow struct {
	Template string `json:"template"`
	Count    int    `json:"count"`
}

// watch holds the trained models and the running state.
type watch struct {
	mu sync.Mutex

	data    *dataset
	profile [][]float32 // per airport: expected movements per minute by minute of day, from the training days
	fcZone  *forecaster
	ae      *autoencoder
	mw      *metarWatch
	trainMs int // minutes of training history

	mode   string
	date   string
	minute int // minute of the replay day (or of the live day)

	rawMov    [][]float32    // per airport: movements per minute, training days then the running day
	rawZone   [][]float32    // aircraft in zone per minute
	hist      [][]float32    // movements smoothed over movSmooth minutes (the forecast quantity)
	histZone  [][]float32    // in-zone smoothed over zoneSmooth minutes
	goHour    [][]float32    // go-arounds per minute, for the hourly rate rule
	goRef     []float32      // training mean go-arounds per hour per airport
	outside   []int          // consecutive minutes the in-zone level is outside its band
	movRun    []int          // consecutive minutes the movement count is outside its Poisson range
	scoreRun  int            // consecutive minutes over the autoencoder threshold
	lastAlarm map[string]int // airport|hour -> minute, for bundling
	alarms    []alarm
	events    []event
	scores    []scorePoint
	preds     map[int][]pred // minute -> per airport predictions made horizon minutes ago
	metarNow  []string
	tmplNow   []string
	shiftNow  []float64
	tmplCount map[string]int
	seenEvent map[string]bool
	aircraft  [][][4]float32
	training  []string

	subs map[chan struct{}]bool
}

type pred struct{ zoneMu, zoneSigma float32 }

const (
	movSmooth   = 10 // minutes: the forecast quantity is the 10-minute mean rate
	zoneSmooth  = 10
	movWindow   = 30 // minutes over which observed and expected movements are compared
	movWindow2  = 60 // a second, longer window catches a closure at a quiet airport
	movTail     = 1e-4
	zoneMin     = 8 // consecutive minutes outside the in-zone band before an alarm
	movMin      = 3 // consecutive minutes with a Poisson tail below movTail
	scoreMin    = 3 // consecutive minutes over the autoencoder threshold
	minExpected = 3 // expected movements in the window below which the test says nothing
)

// poissonTail returns P(N ≤ k) and P(N ≥ k) for N ~ Poisson(λ).
func poissonTail(k int, lambda float64) (lower, upper float64) {
	if lambda <= 0 {
		if k == 0 {
			return 1, 1
		}
		return 1, 0
	}
	p := math.Exp(-lambda)
	cdf := p
	for i := 1; i <= k; i++ {
		p *= lambda / float64(i)
		cdf += p
	}
	// P(N ≥ k) = 1 - P(N ≤ k-1)
	return math.Min(1, cdf), math.Min(1, 1-(cdf-p))
}

// trailingMean of the last n values of s ending at i.
func trailingMean(s []float32, i, n int) float32 {
	lo := max(0, i-n+1)
	var sum float32
	for j := lo; j <= i; j++ {
		sum += s[j]
	}
	return sum / float32(i-lo+1)
}

// aeColumns are the counters the autoencoder sees: the state of the zone,
// not the rare event counts (those have their own rules).
var aeColumns = []int{cInZone, cArrivals, cDepartures, cHolding, cMeanGS, cFinal}

func aeVector(counters [][nCounters]float32) []float32 {
	out := make([]float32, 0, len(counters)*len(aeColumns))
	for _, v := range counters {
		for _, c := range aeColumns {
			out = append(out, v[c])
		}
	}
	return out
}

// newWatch trains on every day but the last one in the data set and
// prepares the replay of the last day.
func newWatch(d *dataset, replayDate string, logf func(string)) *watch {
	w := &watch{data: d, lastAlarm: map[string]int{}, preds: map[int][]pred{}, tmplCount: map[string]int{}, seenEvent: map[string]bool{}, subs: map[chan struct{}]bool{}}
	w.metarNow = make([]string, len(airports))
	w.tmplNow = make([]string, len(airports))
	w.shiftNow = make([]float64, len(airports))
	w.aircraft = make([][][4]float32, len(airports))
	w.outside = make([]int, len(airports))
	w.movRun = make([]int, len(airports))
	log := func(s string) {
		logf(s)
		w.mu.Lock()
		w.training = append(w.training, s)
		w.mu.Unlock()
	}
	if replayDate == "" {
		replayDate = d.Days[len(d.Days)-1]
	}
	var trainDays []string
	for _, day := range d.Days {
		if day < replayDate {
			trainDays = append(trainDays, day)
		}
	}
	if len(trainDays) == 0 || d.Counters[replayDate] == nil {
		trainDays = d.Days[:len(d.Days)-1]
		replayDate = d.Days[len(d.Days)-1]
	}
	w.date = replayDate
	w.train(trainDays, log)
	return w
}

func (w *watch) train(days []string, log func(string)) {
	r := rand.New(rand.NewPCG(1, 2))
	k := len(airports)
	w.rawMov = make([][]float32, k)
	w.rawZone = make([][]float32, k)
	w.hist = make([][]float32, k)
	w.histZone = make([][]float32, k)
	w.goHour = make([][]float32, k)
	w.goRef = make([]float32, k)
	var rows []float32
	var reports []timedReport
	for di, day := range days {
		c := w.data.Counters[day]
		for m := 0; m < minutesPerDay; m++ {
			minute := make([][nCounters]float32, k)
			for ap := 0; ap < k; ap++ {
				v := c[ap][m]
				minute[ap] = v
				w.rawMov[ap] = append(w.rawMov[ap], v[cArrivals]+v[cDepartures])
				w.rawZone[ap] = append(w.rawZone[ap], v[cInZone])
				i := len(w.rawMov[ap]) - 1
				w.hist[ap] = append(w.hist[ap], trailingMean(w.rawMov[ap], i, movSmooth))
				w.histZone[ap] = append(w.histZone[ap], trailingMean(w.rawZone[ap], i, zoneSmooth))
				w.goHour[ap] = append(w.goHour[ap], v[cGoArounds])
				w.goRef[ap] += v[cGoArounds]
			}
			rows = append(rows, aeVector(minute)...)
		}
		for _, rec := range w.data.Metar[day] {
			reports = append(reports, timedReport{Station: rec.Station, Minute: di*minutesPerDay + rec.Time.Hour()*60 + rec.Time.Minute(), Raw: rec.Raw})
		}
	}
	w.trainMs = len(days) * minutesPerDay
	for ap := range w.goRef {
		w.goRef[ap] /= float32(w.trainMs) / 60 // per hour
	}
	log(fmt.Sprintf("training on %s (%d days, %d minutes per airport)", strings.Join(days, ", "), len(days), w.trainMs))
	start := time.Now()
	// Movements: the expectation is the training days' profile by minute of
	// day, smoothed over ±15 minutes (the "same minute on an ordinary day"
	// rule); a Poisson test against it says whether a count is out of range.
	w.profile = make([][]float32, k)
	for ap := 0; ap < k; ap++ {
		raw := make([]float64, minutesPerDay)
		for i, v := range w.rawMov[ap] {
			raw[i%minutesPerDay] += float64(v)
		}
		prof := make([]float32, minutesPerDay)
		for m := 0; m < minutesPerDay; m++ {
			var sum float64
			for d := -15; d <= 15; d++ {
				sum += raw[(m+d+minutesPerDay)%minutesPerDay]
			}
			prof[m] = float32(sum / 31 / float64(len(days)))
		}
		w.profile[ap] = prof
	}
	log("movements: expectation from the training days' profile by minute of day")
	w.fcZone = trainForecaster("in zone", w.histZone, 1.0, r, 12, log)
	names := make([]string, 0, k*len(aeColumns))
	for _, ap := range airports {
		for _, c := range aeColumns {
			names = append(names, ap.ICAO+" "+counterNames[c])
		}
	}
	w.ae = trainAutoencoder(tensor.New(rows, w.trainMs, k*len(aeColumns)), names, r, 40, log)
	w.mw = newMetarWatch()
	w.mw.train(reports, w.trainMs, log)
	log(fmt.Sprintf("trained in %v", time.Since(start).Round(time.Second)))
}

// minuteIndex is the position in hist of the running day's minute m.
func (w *watch) minuteIndex(m int) int { return w.trainMs + m }

// clock is the minute of the running day as a UTC time.
func (w *watch) clock(m int) time.Time { return dayOf(w.date).Add(time.Duration(m) * time.Minute) }

// step feeds one minute of the running day: counters per airport, the
// METARs issued that minute, and the events the reduction found.
func (w *watch) step(m int, counters [][nCounters]float32, metars []metarRecord, evs []dayEvent, aircraft [][][4]float32) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.minute = m
	idx := w.minuteIndex(m)
	k := len(airports)
	flat := aeVector(counters)
	for ap := 0; ap < k; ap++ {
		v := counters[ap]
		mov := v[cArrivals] + v[cDepartures]
		for len(w.rawMov[ap]) <= idx { // fill gaps (live mode may skip minutes)
			w.rawMov[ap] = append(w.rawMov[ap], 0)
			w.rawZone[ap] = append(w.rawZone[ap], 0)
			w.hist[ap] = append(w.hist[ap], 0)
			w.histZone[ap] = append(w.histZone[ap], 0)
			w.goHour[ap] = append(w.goHour[ap], 0)
		}
		w.rawMov[ap][idx] = mov
		w.rawZone[ap][idx] = v[cInZone]
		w.hist[ap][idx] = trailingMean(w.rawMov[ap], idx, movSmooth)
		w.histZone[ap][idx] = trailingMean(w.rawZone[ap], idx, zoneSmooth)
		w.goHour[ap][idx] = v[cGoArounds]
	}
	if aircraft != nil {
		w.aircraft = aircraft
	}

	// Compare with the predictions made horizon minutes ago. Movements are
	// counts: over the last movWindow minutes the observed number is tested
	// against the Poisson distribution with the forecast rates as mean, so
	// a quiet airport (0.3 an hour) and a busy one (1.5 a minute) get the
	// same false-alarm rate. The in-zone level is a Gaussian band.
	if ps, ok := w.preds[m]; ok {
		for ap := 0; ap < k; ap++ {
			out := false
			var text string
			for _, win := range []int{movWindow, movWindow2} {
				var obs, exp float64
				for i := 0; i < win && m-i >= 0 && idx-i >= 0; i++ {
					obs += float64(w.rawMov[ap][idx-i])
					exp += float64(w.profile[ap][(m-i)%minutesPerDay])
				}
				lower, upper := poissonTail(int(obs+0.5), exp)
				if exp >= minExpected && (lower < movTail || upper < movTail) {
					out = true
					text = fmt.Sprintf("%.0f movements in %d min, expected %.0f", obs, win, exp)
					break
				}
			}
			if out {
				w.movRun[ap]++
			} else {
				w.movRun[ap] = 0
			}
			if w.movRun[ap] >= movMin {
				w.raise(m, ap, "band", text)
			}

			zone := w.histZone[ap][idx]
			p := ps[ap]
			zoneSigma := float32(math.Max(math.Max(float64(p.zoneSigma), 1), 0.35*float64(p.zoneMu)))
			if math.Abs(float64(zone-p.zoneMu)) > 3*float64(zoneSigma) {
				w.outside[ap]++
			} else {
				w.outside[ap] = 0
			}
			if w.outside[ap] >= zoneMin {
				w.raise(m, ap, "band", fmt.Sprintf("%.0f aircraft in zone, expected %.0f ± %.0f", zone, p.zoneMu, zoneSigma))
			}
		}
	}
	// Go-arounds: an hourly count far above the training rate.
	for ap := 0; ap < k; ap++ {
		var n float32
		for i := max(0, idx-59); i <= idx; i++ {
			n += w.goHour[ap][i]
		}
		if n >= 2 && n >= 4*w.goRef[ap]+2 {
			w.raise(m, ap, "go-arounds", fmt.Sprintf("%.0f aborted approaches in the last hour, usually %.1f", n, w.goRef[ap]))
		}
	}
	// Predictions for horizon minutes ahead.
	if idx+1 >= window {
		ps := make([]pred, k)
		for ap := 0; ap < k; ap++ {
			mu, sg := w.fcZone.predict(w.histZone[ap][:idx+1], ap, idx)
			ps[ap] = pred{mu, sg}
		}
		w.preds[m+horizon] = ps
	}
	// Autoencoder over the whole minute.
	score, worst := w.ae.score(flat)
	w.scores = append(w.scores, scorePoint{Minute: m, Score: score, Worst: worst})
	if len(w.scores) > 3*60 {
		w.scores = w.scores[len(w.scores)-3*60:]
	}
	if score > w.ae.threshold {
		w.scoreRun++
	} else {
		w.scoreRun = 0
	}
	if w.scoreRun >= scoreMin || score > 2*w.ae.threshold {
		ap := airportIndex(strings.Fields(worst)[0])
		w.raise(m, ap, "score", fmt.Sprintf("anomaly score %.1f (threshold %.1f), driven by %s", score, w.ae.threshold, worst))
	}
	// METARs.
	for _, rec := range metars {
		ap := airportIndex(rec.Station)
		if ap < 0 {
			continue
		}
		obs := w.mw.observe(rec.Station, w.minuteIndex(m), rec.Raw)
		w.metarNow[ap] = rec.Raw
		w.tmplNow[ap] = obs.Template
		w.shiftNow[ap] = obs.Shift
		w.tmplCount[obs.Template]++
		if pm, ok := parseMETAR(rec.Raw, w.clock(m)); ok {
			switch {
			case pm.GustKt >= 40:
				w.note(m, ap, "gust", fmt.Sprintf("wind %d kt gusting %d kt", pm.WindKt, pm.GustKt))
			case pm.VisM >= 0 && pm.VisM < 800:
				w.note(m, ap, "visibility", fmt.Sprintf("visibility %d m", pm.VisM))
			case pm.Speci:
				w.note(m, ap, "speci", "special report: "+shortMetar(rec.Raw))
			}
		}
		if obs.Storm != "" {
			w.raise(m, ap, "metar", obs.Template+": "+obs.Storm)
		}
		if obs.ShiftRef > 0 && obs.Shift > math.Max(0.35, 1.5*obs.ShiftRef) {
			w.raise(m, ap, "metar", fmt.Sprintf("report mix shifted (distance %.2f, usually under %.2f)", obs.Shift, obs.ShiftRef))
		}
	}
	// Events from the reduction.
	for _, e := range evs {
		ap := airportIndex(e.Airport)
		switch e.Kind {
		case "go-around":
			w.note(m, ap, "go-around", "aborted approach, aircraft "+e.Hex)
		case "holding":
			// too frequent to list one by one; the counter shows them
		}
		if counters[ap][cEmergency] > 0 && !w.seenEvent["emerg|"+e.Airport+"|"+e.Hex] {
			w.seenEvent["emerg|"+e.Airport+"|"+e.Hex] = true
		}
	}
	for ap := 0; ap < k; ap++ {
		if counters[ap][cEmergency] > 0 {
			key := fmt.Sprintf("emerg|%s|%d", airports[ap].ICAO, m/30)
			if !w.seenEvent[key] {
				w.seenEvent[key] = true
				w.note(m, ap, "emergency", fmt.Sprintf("%.0f aircraft squawking an emergency code", counters[ap][cEmergency]))
			}
		}
	}
	w.notify()
}

func shortMetar(raw string) string {
	f := strings.Fields(raw)
	if len(f) > 9 {
		f = f[:9]
	}
	return strings.Join(f, " ")
}

// raise records a model alarm, bundled per airport and hour: the first
// finding of an hour is reported, later ones of the same hour are folded
// into it.
func (w *watch) raise(m, ap int, kind, text string) {
	icao := "all"
	if ap >= 0 {
		icao = airports[ap].ICAO
	}
	key := fmt.Sprintf("%s|%d", icao, m/60)
	if last, ok := w.lastAlarm[key]; ok && m-last < 60 {
		return
	}
	w.lastAlarm[key] = m
	w.alarms = append(w.alarms, alarm{Minute: m, Airport: icao, Kind: kind, Text: text})
}

func (w *watch) note(m, ap int, kind, text string) {
	icao := "all"
	if ap >= 0 {
		icao = airports[ap].ICAO
	}
	key := fmt.Sprintf("%s|%s|%s", icao, kind, text)
	if w.seenEvent[key] {
		return
	}
	w.seenEvent[key] = true
	w.events = append(w.events, event{Minute: m, Airport: icao, Kind: kind, Text: text})
	if len(w.events) > 400 {
		w.events = w.events[len(w.events)-400:]
	}
}

// snapshot copies the state for the dashboard.
func (w *watch) snapshot() state {
	w.mu.Lock()
	defer w.mu.Unlock()
	s := state{Mode: w.mode, Date: w.date, Minute: w.minute, Clock: w.clock(w.minute).Format("15:04"), Threshold: w.ae.threshold}
	idx := w.minuteIndex(w.minute)
	for ap, a := range airports {
		as := airportState{ICAO: a.ICAO, Name: a.Name, Metar: w.metarNow[ap], Template: w.tmplNow[ap], Shift: w.shiftNow[ap], ShiftRef: w.mw.shiftRef[a.ICAO], Outside: w.outside[ap]}
		lo := max(0, w.minute-179)
		for m := lo; m <= w.minute; m++ {
			i := w.minuteIndex(m)
			if i >= len(w.hist[ap]) {
				break
			}
			p := point{Minute: m, Value: w.hist[ap][i]}
			var mu float32
			for d := 0; d < movSmooth; d++ {
				mu += w.profile[ap][(m-d+minutesPerDay)%minutesPerDay]
			}
			p.Mu = mu / movSmooth
			p.Sigma = float32(math.Max(0.05, math.Sqrt(float64(p.Mu)/movSmooth))) // Poisson spread of a 10-minute mean
			q := point{Minute: m, Value: w.histZone[ap][i]}
			if ps, ok := w.preds[m]; ok {
				q.Mu, q.Sigma = ps[ap].zoneMu, ps[ap].zoneSigma
			}
			as.Movements = append(as.Movements, p)
			as.InZone = append(as.InZone, q)
		}
		if day := w.data.Counters[w.date]; day != nil && w.mode == "replay" {
			as.Now = day[ap][w.minute]
		} else if idx < len(w.hist[ap]) {
			as.Now[cInZone] = w.histZone[ap][idx]
		}
		as.Aircraft = [][4]float32{}
		if ap < len(w.aircraft) && w.aircraft[ap] != nil {
			as.Aircraft = w.aircraft[ap]
		}
		// Observed against expected movements over the last 30 minutes.
		for i := 0; i < movWindow && w.minute-i >= 0 && idx-i >= 0 && idx-i < len(w.rawMov[ap]); i++ {
			as.Observed += w.rawMov[ap][idx-i]
			as.Expected += w.profile[ap][(w.minute-i)%minutesPerDay]
		}
		as.State, as.Reason = w.status(ap, as)
		if as.Metar != "" {
			if pm, ok := parseMETAR(as.Metar, w.clock(w.minute)); ok {
				as.Weather = weather{WindDir: pm.WindDir, WindKt: pm.WindKt, GustKt: pm.GustKt, VisM: pm.VisM, Wx: weatherWords(pm.Weather), Speci: pm.Speci, Time: pm.Time.Format("15:04")}
			}
		}
		s.Airports = append(s.Airports, as)
	}
	var inAlarm []string
	for _, as := range s.Airports {
		if as.State == "alarm" {
			inAlarm = append(inAlarm, fmt.Sprintf("%s: %s", as.Name, as.Reason))
		}
	}
	if len(inAlarm) == 0 {
		s.Headline = "all eight airports as expected"
	} else {
		s.Headline = strings.Join(inAlarm, " · ")
	}
	s.DayAlarms = append([]alarm{}, w.alarms...)
	s.DayEvents = append([]event{}, w.events...)
	// Empty lists are [] in the JSON, not null: the page indexes them.
	s.Scores = append([]scorePoint{}, w.scores...)
	n := len(w.alarms)
	s.Alarms = append([]alarm{}, w.alarms[max(0, n-60):]...)
	n = len(w.events)
	s.Events = append([]event{}, w.events[max(0, n-60):]...)
	s.Templates = []templateRow{}
	for t, c := range w.tmplCount {
		s.Templates = append(s.Templates, templateRow{Template: t, Count: c})
	}
	sort.Slice(s.Templates, func(i, j int) bool { return s.Templates[i].Count > s.Templates[j].Count })
	if len(s.Templates) > 25 {
		s.Templates = s.Templates[:25]
	}
	s.Training = append([]string{}, w.training...)
	return s
}

// status derives the tile's state and its one-sentence reason: the latest
// alarm of the last hour, else the deviation a run is building, else
// "as expected". Called with w.mu held.
func (w *watch) status(ap int, as airportState) (state, reason string) {
	icao := airports[ap].ICAO
	for i := len(w.alarms) - 1; i >= 0; i-- {
		a := w.alarms[i]
		if a.Airport != icao {
			continue
		}
		if w.minute-a.Minute < 60 {
			return "alarm", a.Text
		}
		break
	}
	if w.movRun[ap] > 0 {
		return "watch", fmt.Sprintf("%.0f movements in the last %d min, expected %.0f", as.Observed, movWindow, as.Expected)
	}
	if w.outside[ap] > 0 {
		return "watch", fmt.Sprintf("%.0f aircraft in the zone, outside the band for %d min", as.Now[cInZone], w.outside[ap])
	}
	if w.scoreRun > 0 && len(w.scores) > 0 && strings.HasPrefix(w.scores[len(w.scores)-1].Worst, icao) {
		return "watch", "anomaly score rising, driven by " + strings.TrimPrefix(w.scores[len(w.scores)-1].Worst, icao+" ")
	}
	return "normal", fmt.Sprintf("as expected: %.0f movements in the last %d min, expected %.0f", as.Observed, movWindow, as.Expected)
}

var wxWords = map[string]string{"DZ": "drizzle", "RA": "rain", "SN": "snow", "SG": "snow grains", "PL": "ice pellets", "GR": "hail", "GS": "small hail",
	"BR": "mist", "FG": "fog", "FU": "smoke", "HZ": "haze", "SQ": "squall", "FC": "funnel cloud", "SS": "sandstorm", "DS": "dust storm", "UP": "precipitation",
	"SH": "showers of", "TS": "thunderstorm with", "FZ": "freezing", "MI": "shallow", "BC": "patches of", "PR": "partial", "DR": "drifting", "BL": "blowing", "VC": "nearby"}

// weatherWords turns METAR weather groups ("-SHRA", "+TSGR") into words.
func weatherWords(groups []string) string {
	var out []string
	for _, g := range groups {
		var parts []string
		switch {
		case strings.HasPrefix(g, "+"):
			parts = append(parts, "heavy")
			g = g[1:]
		case strings.HasPrefix(g, "-"):
			parts = append(parts, "light")
			g = g[1:]
		}
		for len(g) >= 2 {
			code := g[:2]
			g = g[2:]
			if word, ok := wxWords[code]; ok {
				parts = append(parts, word)
			} else {
				parts = append(parts, code)
			}
		}
		out = append(out, strings.Join(parts, " "))
	}
	return strings.Join(out, ", ")
}

func (w *watch) subscribe() chan struct{} {
	ch := make(chan struct{}, 1)
	w.mu.Lock()
	w.subs[ch] = true
	w.mu.Unlock()
	return ch
}

func (w *watch) unsubscribe(ch chan struct{}) {
	w.mu.Lock()
	delete(w.subs, ch)
	w.mu.Unlock()
}

// notify wakes the dashboard subscribers; called with w.mu held.
func (w *watch) notify() {
	for ch := range w.subs {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}

// replayAircraft builds the radar view for minute m from the storm day's
// down-sampled tracks: the last position of every aircraft seen in the
// minute, with a holding flag from the events.
func replayAircraft(pts []trackPoint, m int) [][][4]float32 {
	out := make([][][4]float32, len(airports))
	lo, hi := m*60, m*60+59
	// pts sorted by second: binary search the window
	i := sort.Search(len(pts), func(i int) bool { return pts[i].Sec >= lo })
	last := map[[2]string]trackPoint{}
	for ; i < len(pts) && pts[i].Sec <= hi; i++ {
		last[[2]string{airports[pts[i].Airport].ICAO, pts[i].Hex}] = pts[i]
	}
	for _, p := range last {
		out[p.Airport] = append(out[p.Airport], [4]float32{p.DX, p.DY, p.AltFt, 0})
	}
	return out
}

// runReplay plays the last day of the data set.
func (w *watch) runReplay(speed int, headless bool, logf func(string)) {
	w.mode = "replay"
	day := w.data.Counters[w.date]
	metars := w.data.Metar[w.date]
	evs := w.data.Events[w.date]
	tracks := w.data.Tracks[w.date]
	mi, ei := 0, 0
	tick := time.Second / time.Duration(max(1, speed))
	for m := 0; m < minutesPerDay; m++ {
		var mm []metarRecord
		for mi < len(metars) && metars[mi].Time.Hour()*60+metars[mi].Time.Minute() <= m {
			mm = append(mm, metars[mi])
			mi++
		}
		var ee []dayEvent
		for ei < len(evs) && evs[ei].Minute <= m {
			ee = append(ee, evs[ei])
			ei++
		}
		counters := make([][nCounters]float32, len(airports))
		for ap := range airports {
			counters[ap] = day[ap][m]
		}
		var ac [][][4]float32
		if !headless && len(tracks) > 0 {
			ac = replayAircraft(tracks, m)
		}
		w.step(m, counters, mm, ee, ac)
		if headless {
			w.mu.Lock()
			for _, a := range w.alarms {
				if a.Minute == m {
					logf(fmt.Sprintf("%s ALARM %-5s %-6s %s", w.clock(m).Format("15:04"), a.Kind, a.Airport, a.Text))
				}
			}
			for _, e := range w.events {
				if e.Minute == m {
					logf(fmt.Sprintf("%s event %-10s %-6s %s", w.clock(m).Format("15:04"), e.Kind, e.Airport, e.Text))
				}
			}
			w.mu.Unlock()
			continue
		}
		time.Sleep(tick)
	}
}

// serve runs the dashboard.
func (w *watch) serve(addr string, logf func(string)) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(rw http.ResponseWriter, r *http.Request) {
		rw.Header().Set("Content-Type", "text/html; charset=utf-8")
		rw.Write(uiHTML)
	})
	mux.HandleFunc("/state", func(rw http.ResponseWriter, r *http.Request) {
		rw.Header().Set("Content-Type", "application/json")
		json.NewEncoder(rw).Encode(w.snapshot())
	})
	mux.HandleFunc("/events", func(rw http.ResponseWriter, r *http.Request) {
		fl, ok := rw.(http.Flusher)
		if !ok {
			http.Error(rw, "streaming unsupported", 500)
			return
		}
		rw.Header().Set("Content-Type", "text/event-stream")
		rw.Header().Set("Cache-Control", "no-cache")
		ch := w.subscribe()
		defer w.unsubscribe(ch)
		send := func() {
			b, _ := json.Marshal(w.snapshot())
			fmt.Fprintf(rw, "data: %s\n\n", b)
			fl.Flush()
		}
		send()
		for {
			select {
			case <-r.Context().Done():
				return
			case <-ch:
				send()
			}
		}
	})
	logf("dashboard on http://" + addr)
	return http.ListenAndServe(addr, mux)
}

// run wires the modes together.
func run(live, headless bool, speed int, addr, dataDir, replayDay string, logf func(string)) error {
	d, err := openData(dataDir)
	if err != nil {
		return err
	}
	logf("data:\n" + d.describe())
	w := newWatch(d, replayDay, logf)
	if live {
		go w.runLive(logf)
		return w.serve(addr, logf)
	}
	if headless {
		w.runReplay(speed, true, logf)
		w.mu.Lock()
		defer w.mu.Unlock()
		logf(fmt.Sprintf("replay of %s: %d alarms, %d events", w.date, len(w.alarms), len(w.events)))
		return nil
	}
	go w.runReplay(speed, false, logf)
	return w.serve(addr, logf)
}
