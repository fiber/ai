package main

import (
	"math"
	"sort"
)

// sample is one ADS-B position report of one aircraft.
type sample struct {
	T      float64 // unix seconds
	Lat    float64
	Lon    float64
	AltFt  float64 // barometric altitude; 0 on the ground
	Ground bool
	GS     float64 // knots, NaN when unknown
	Track  float64 // degrees, NaN when unknown
	VR     float64 // ft/min, NaN when unknown
	Squawk string
	Emerg  bool
}

// Counter indices per airport and minute. movements (arrivals plus
// departures) and inZone feed the forecast bands; the whole vector feeds
// the autoencoder.
const (
	cInZone     = iota // distinct aircraft in the zone during the minute
	cArrivals          // landings in the minute
	cDepartures        // take-offs in the minute
	cHolding           // distinct aircraft flying a holding pattern
	cGoArounds         // aborted approaches in the minute
	cEmergency         // distinct aircraft squawking 7500/7600/7700
	cMeanGS            // mean ground speed of aircraft below 5 000 ft AGL (kt)
	cFinal             // aircraft below 3 000 ft AGL within 10 NM
	nCounters
)

var counterNames = [nCounters]string{"in zone", "arrivals", "departures", "holding", "go-arounds", "emergency", "mean gs", "on final"}

// trackEvent is a detected movement of one aircraft.
type trackEvent struct {
	Kind   string // "arrival", "departure", "go-around", "emergency", "holding"
	Minute int    // minute of day
	Hex    string
}

// minuteContribution is what one aircraft adds to one airport's counters in
// one minute: presence, holding, low, emergency and speed samples. It is
// combined over aircraft by aggregate.
type minuteContribution struct {
	present, holding, emergency, final bool
	gsSum                              float64
	gsN                                int
}

// reduceTrack derives one aircraft's contributions and events for one
// airport from its time-ordered samples (any samples, in or out of the
// zone; the function filters). minuteOf maps a unix time to the minute of
// day the counters are kept in.
func reduceTrack(s []sample, ap airport, minuteOf func(t float64) int) (contrib map[int]*minuteContribution, events []trackEvent) {
	contrib = map[int]*minuteContribution{}
	get := func(m int) *minuteContribution {
		c := contrib[m]
		if c == nil {
			c = &minuteContribution{}
			contrib[m] = c
		}
		return c
	}
	// Per-sample geometry.
	type pt struct {
		sample
		dist, agl float64
		inZone    bool
	}
	pts := make([]pt, 0, len(s))
	for _, x := range s {
		d := distanceNM(x.Lat, x.Lon, ap.Lat, ap.Lon)
		if d > ap.radiusNM*1.5 {
			continue // keep some context outside the zone for approach geometry
		}
		agl := x.AltFt - ap.ElevFt
		if x.Ground {
			agl = 0
		}
		pts = append(pts, pt{sample: x, dist: d, agl: agl, inZone: d <= ap.radiusNM && agl < ap.ceilingFt})
	}
	if len(pts) == 0 {
		return contrib, nil
	}
	sort.Slice(pts, func(i, j int) bool { return pts[i].T < pts[j].T })

	// Presence, final, emergency, speed.
	for _, p := range pts {
		if !p.inZone {
			continue
		}
		c := get(minuteOf(p.T))
		c.present = true
		if p.agl < 3000 && p.dist <= 10 {
			c.final = true
		}
		if p.agl < 5000 && !math.IsNaN(p.GS) && p.GS > 30 {
			c.gsSum += p.GS
			c.gsN++
		}
		if p.Emerg || p.Squawk == "7500" || p.Squawk == "7600" || p.Squawk == "7700" {
			c.emergency = true
		}
	}
	// Ground contacts: runs of samples low and close to the field. What the
	// aircraft did before and after the run says whether it landed, took
	// off or went around.
	const lowAGL, nearNM, gapS = 1500.0, 5.0, 180.0
	type contact struct{ first, last int }
	var contacts []contact
	for i := 0; i < len(pts); i++ {
		if !(pts[i].agl < lowAGL && pts[i].dist <= nearNM) {
			continue
		}
		if n := len(contacts); n > 0 && pts[i].T-pts[contacts[n-1].last].T <= gapS {
			contacts[n-1].last = i
		} else {
			contacts = append(contacts, contact{i, i})
		}
	}
	maxAGL := func(lo, hi float64) (m float64, seen bool) {
		m = -1
		for _, p := range pts {
			if p.T >= lo && p.T <= hi {
				seen = true
				if p.agl > m {
					m = p.agl
				}
			}
		}
		return m, seen
	}
	const look = 360.0 // seconds of context on each side
	for _, c := range contacts {
		t0, t1 := pts[c.first].T, pts[c.last].T
		before, seenBefore := maxAGL(t0-look, t0-1)
		after, seenAfter := maxAGL(t1+1, t1+look)
		highBefore := seenBefore && before > 2500
		highAfter := seenAfter && after > 2500
		endsOnGround := false
		for i := c.first; i <= c.last; i++ {
			if pts[i].Ground || pts[i].agl < 150 {
				endsOnGround = true
			}
		}
		m := minuteOf(t0)
		switch {
		case highBefore && highAfter && !endsOnGround:
			events = append(events, trackEvent{Kind: "go-around", Minute: m})
		case highBefore && !highAfter:
			events = append(events, trackEvent{Kind: "arrival", Minute: m})
		case !highBefore && highAfter:
			events = append(events, trackEvent{Kind: "departure", Minute: minuteOf(t1)})
		case highBefore && highAfter && endsOnGround:
			// touched down and left again within the window: count both
			events = append(events, trackEvent{Kind: "arrival", Minute: m}, trackEvent{Kind: "departure", Minute: minuteOf(t1)})
		}
	}

	// Holding: within an 8-minute window (a racetrack circuit takes about
	// four), level flight whose heading turns
	// through more than 360° in one direction.
	const holdWindow, holdTurn, holdLevel = 480.0, 360.0, 1200.0
	for i := 0; i < len(pts); i++ {
		if !pts[i].inZone || math.IsNaN(pts[i].Track) {
			continue
		}
		turn, minAlt, maxAlt := 0.0, pts[i].agl, pts[i].agl
		last := pts[i].Track
		flagged := false
		for j := i + 1; j < len(pts) && pts[j].T-pts[i].T <= holdWindow; j++ {
			if math.IsNaN(pts[j].Track) {
				continue
			}
			turn += headingDelta(last, pts[j].Track)
			last = pts[j].Track
			minAlt = math.Min(minAlt, pts[j].agl)
			maxAlt = math.Max(maxAlt, pts[j].agl)
			if maxAlt-minAlt > holdLevel {
				break
			}
			if math.Abs(turn) >= holdTurn {
				for k := i; k <= j; k++ {
					if pts[k].inZone {
						get(minuteOf(pts[k].T)).holding = true
					}
				}
				events = append(events, trackEvent{Kind: "holding", Minute: minuteOf(pts[i].T)})
				i = j // continue after this pattern
				flagged = true
				break
			}
		}
		_ = flagged
	}
	return contrib, events
}

// aggregate turns per-aircraft contributions into the minute's counter
// vector for one airport.
type minuteAgg struct {
	present, holding, emergency, final int
	gsSum                              float64
	gsN                                int
	arrivals, departures, goArounds    int
}

func (m *minuteAgg) add(c *minuteContribution) {
	if c.present {
		m.present++
	}
	if c.holding {
		m.holding++
	}
	if c.emergency {
		m.emergency++
	}
	if c.final {
		m.final++
	}
	m.gsSum += c.gsSum
	m.gsN += c.gsN
}

func (m *minuteAgg) vector() [nCounters]float32 {
	var v [nCounters]float32
	v[cInZone] = float32(m.present)
	v[cArrivals] = float32(m.arrivals)
	v[cDepartures] = float32(m.departures)
	v[cHolding] = float32(m.holding)
	v[cGoArounds] = float32(m.goArounds)
	v[cEmergency] = float32(m.emergency)
	if m.gsN > 0 {
		v[cMeanGS] = float32(m.gsSum / float64(m.gsN))
	}
	v[cFinal] = float32(m.final)
	return v
}
