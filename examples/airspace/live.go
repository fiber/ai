package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"time"
)

// Live mode: once a minute, the aircraft around each airport from the
// adsb.lol API and the latest METARs from aviationweather.gov. Positions
// are kept per aircraft for 20 minutes and re-reduced every minute with
// the same rules as the archive, so live and replay counters mean the
// same thing.

type liveAircraft struct {
	Hex       string          `json:"hex"`
	Flight    string          `json:"flight"`
	AltBaro   json.RawMessage `json:"alt_baro"`
	GS        *float64        `json:"gs"`
	Track     *float64        `json:"track"`
	BaroRate  *float64        `json:"baro_rate"`
	Squawk    string          `json:"squawk"`
	Emergency string          `json:"emergency"`
	Lat       float64         `json:"lat"`
	Lon       float64         `json:"lon"`
	Seen      float64         `json:"seen_pos"`
}

// liveResponse is the readsb JSON both the aggregators and a local
// tar1090 serve; the aggregators call the list "ac", readsb "aircraft".
type liveResponse struct {
	AC       []liveAircraft `json:"ac"`
	Aircraft []liveAircraft `json:"aircraft"`
	Now      float64        `json:"now"` // milliseconds
}

// list is the aircraft under whichever key the server used.
func (r liveResponse) list() []liveAircraft {
	if len(r.AC) > 0 {
		return r.AC
	}
	return r.Aircraft
}

// liveOptions are the live-mode settings from the command line.
type liveOptions struct {
	source   string        // readsb JSON endpoint, per region or whole
	contact  string        // address sent in the User-Agent
	interval time.Duration // how often the feeds are polled
}

// runLive polls the feeds every interval and steps the watch every minute.
func (w *watch) runLive(opts liveOptions, logf func(string)) {
	w.mode = "live"
	client := &http.Client{Timeout: 30 * time.Second}
	ua := fmt.Sprintf(userAgentFmt, opts.contact)
	whole := !strings.Contains(opts.source, "%") // one feed, not two regions
	states := make([]backoff, len(liveRegions))
	lastSeen := make([]map[string]sample, len(airports))
	lastSeenAt := make([]time.Time, len(airports))
	var nextPoll time.Time
	windows := make([]map[string][]sample, len(airports)) // airport -> hex -> samples (20 min)
	for i := range windows {
		windows[i] = map[string][]sample{}
	}
	seenMetar := map[string]bool{}
	counted := map[string]int{} // airport|hex|kind -> minute, so an event counts once
	start := time.Now().UTC()
	w.date = start.Format("2006-01-02")
	for {
		now := time.Now().UTC()
		m := now.Hour()*60 + now.Minute()
		if now.Format("2006-01-02") != w.date {
			// a new day: keep the history running, reset the day index
			w.mu.Lock()
			w.trainMs += minutesPerDay
			w.date = now.Format("2006-01-02")
			w.mu.Unlock()
		}
		counters := make([][nCounters]float32, len(airports))
		aircraft := make([][][4]float32, len(airports))
		var evs []dayEvent
		// Two regional requests of 250 NM cover all eight fields (west: the
		// Irish Sea; east: the Benelux), and the samples are split by zone
		// locally. A whole feed from your own receiver is one request.
		fresh := make([]map[string]sample, len(airports))
		for i := range fresh {
			fresh[i] = map[string]sample{}
		}
		polled := false
		if !now.Before(nextPoll) {
			polled = true
			nextPoll = now.Add(opts.interval)
			regions := liveRegions
			if whole {
				regions = liveRegions[:1]
			}
			for ri, reg := range regions {
				if !states[ri].ready(now) {
					continue
				}
				if ri > 0 {
					time.Sleep(5 * time.Second)
				}
				hexes, err := fetchRegion(client, ua, opts.source, reg.lat, reg.lon, reg.radiusNM)
				if err != nil {
					if refusal(err) {
						// Turned away: hold off rather than keep knocking.
						wait := states[ri].refused(time.Now().UTC())
						logf(fmt.Sprintf("region %s: %v, not asking again for %s", reg.name, err, wait))
					} else {
						logf(fmt.Sprintf("region %s: %v", reg.name, err))
					}
					continue
				}
				states[ri].ok()
				for hex, smp := range hexes {
					for ai, ap := range airports {
						if distanceNM(smp.Lat, smp.Lon, ap.Lat, ap.Lon) <= ap.radiusNM*1.5 {
							fresh[ai][hex] = smp
						}
					}
				}
			}
		}
		// The minutes between two polls, and the airports of a region that
		// was refused, see the last known picture stamped with this minute,
		// so the counters stay on the one-minute grid instead of dropping
		// to an empty sky and raising alarms about it.
		for ai := range fresh {
			if len(fresh[ai]) > 0 {
				lastSeen[ai], lastSeenAt[ai] = fresh[ai], now
				continue
			}
			if !lastSeenAt[ai].IsZero() && now.Sub(lastSeenAt[ai]) <= carryMax {
				fresh[ai] = carryForward(lastSeen[ai], now)
			}
		}
		for ai, ap := range airports {
			hexes := fresh[ai]
			cut := float64(now.Unix()) - 20*60
			for hex, s := range hexes {
				windows[ai][hex] = append(windows[ai][hex], s)
			}
			var agg minuteAgg
			for hex, ss := range windows[ai] {
				// drop old samples, drop aircraft not seen for 20 minutes
				k := 0
				for _, s := range ss {
					if s.T >= cut {
						ss[k] = s
						k++
					}
				}
				ss = ss[:k]
				if len(ss) == 0 {
					delete(windows[ai], hex)
					continue
				}
				windows[ai][hex] = ss
				minuteOf := func(t float64) int {
					tt := time.Unix(int64(t), 0).UTC()
					return tt.Hour()*60 + tt.Minute()
				}
				contrib, events := reduceTrack(ss, ap, minuteOf)
				if c := contrib[m]; c != nil {
					agg.add(c)
					last := ss[len(ss)-1]
					dx, dy := ap.localNM(last.Lat, last.Lon)
					hold := float32(0)
					if c.holding {
						hold = 1
					}
					aircraft[ai] = append(aircraft[ai], [4]float32{float32(dx), float32(dy), float32(last.AltFt - ap.ElevFt), hold})
				}
				for _, e := range events {
					if e.Minute < m-2 || e.Minute > m {
						continue
					}
					key := fmt.Sprintf("%s|%s|%s", ap.ICAO, hex, e.Kind)
					if _, done := counted[key]; done {
						continue
					}
					counted[key] = m
					switch e.Kind {
					case "arrival":
						agg.arrivals++
					case "departure":
						agg.departures++
					case "go-around":
						agg.goArounds++
					}
					evs = append(evs, dayEvent{Airport: ap.ICAO, Minute: m, Kind: e.Kind, Hex: hex})
				}
			}
			counters[ai] = agg.vector()
		}
		for key, mm := range counted {
			if m-mm > 30 || mm > m {
				delete(counted, key)
			}
		}
		// The weather is asked for on the same cadence as the traffic.
		var metars []metarRecord
		if polled {
			stations := make([]string, len(airports))
			for i, ap := range airports {
				stations[i] = ap.ICAO
			}
			if recs, err := fetchLiveMETARs(client, ua, stations); err != nil {
				logf("metar: " + err.Error())
			} else {
				sort.Slice(recs, func(i, j int) bool { return recs[i].Time.Before(recs[j].Time) })
				for _, r := range recs {
					if !seenMetar[r.Raw] {
						seenMetar[r.Raw] = true
						metars = append(metars, r)
					}
				}
			}
		}
		w.step(m, counters, metars, evs, aircraft)
		time.Sleep(time.Until(now.Truncate(time.Minute).Add(time.Minute)))
	}
}

// liveRegions are the two 250 NM circles that cover the eight zones.
var liveRegions = []struct {
	name     string
	lat, lon float64
	radiusNM float64
}{
	{"west", 53.6, -3.4, 250}, // Dublin, Edinburgh, Manchester, Heathrow
	{"east", 50.8, 5.2, 250},  // Amsterdam, Brussels, Paris, Frankfurt
}

// adsb.lol is the default source. Its history archive is what the replay
// and the trained models were built from, so live and replay see the same
// picture, and one free service carries the load instead of two.
const adsbLol = "https://api.adsb.lol/v2/lat/%.4f/lon/%.4f/dist/%d"

// Every request says who is calling, so an operator seeing the traffic can
// tell what it is and reach whoever runs this copy. -contact fills in the
// address, and live mode does not start without one.
const userAgentFmt = "fiber-ai-airspace/1.0 (+https://github.com/fiber/ai; %s)"

const (
	// A region that refuses is left alone: ten minutes after the first
	// refusal, doubling while they continue, at most an hour.
	backoffMin = 10 * time.Minute
	backoffMax = time.Hour
	// How long the last poll may be carried forward for the minutes in
	// between. Beyond this the picture is too old to stand in for now.
	carryMax = 10 * time.Minute
)

// backoff is the per-region retreat after a 429 or a 403.
type backoff struct {
	until time.Time
	wait  time.Duration
}

// ready reports whether the region may be polled again.
func (b *backoff) ready(now time.Time) bool { return !now.Before(b.until) }

// refused records a refusal and returns how long the region is now left
// alone.
func (b *backoff) refused(now time.Time) time.Duration {
	switch {
	case b.wait == 0:
		b.wait = backoffMin
	case b.wait < backoffMax:
		b.wait *= 2
	}
	if b.wait > backoffMax {
		b.wait = backoffMax
	}
	b.until = now.Add(b.wait)
	return b.wait
}

// ok clears the retreat after a request that went through.
func (b *backoff) ok() { b.until, b.wait = time.Time{}, 0 }

// refusal reports whether the server turned us away rather than failed.
func refusal(err error) bool {
	s := err.Error()
	return strings.Contains(s, "429") || strings.Contains(s, "403")
}

// carryForward stamps the last known position of every aircraft with the
// given time, so the minutes between two polls see an unbroken track
// instead of an empty sky. Positions are stale by as much as the poll
// interval; the counters stay on the one-minute grid the models expect.
func carryForward(last map[string]sample, at time.Time) map[string]sample {
	out := make(map[string]sample, len(last))
	for hex, s := range last {
		s.T = float64(at.Unix())
		out[hex] = s
	}
	return out
}

// get asks for a URL with the client identified.
func get(client *http.Client, ua, url string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", ua)
	return client.Do(req)
}

// fetchRegion returns the current sample of every aircraft within radiusNM
// of a point, keyed by ICAO address.
func fetchRegion(client *http.Client, ua, source string, lat, lon, radiusNM float64) (map[string]sample, error) {
	url := source
	if strings.Contains(source, "%") {
		url = fmt.Sprintf(source, lat, lon, int(radiusNM))
	}
	resp, err := get(client, ua, url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("%s: %s", strings.SplitN(strings.TrimPrefix(url, "https://"), "/", 2)[0], resp.Status)
	}
	var lr liveResponse
	if err := json.NewDecoder(resp.Body).Decode(&lr); err != nil {
		return nil, err
	}
	now := lr.Now / 1000
	if now == 0 {
		now = float64(time.Now().Unix())
	}
	acs := lr.list()
	out := make(map[string]sample, len(acs))
	for _, a := range acs {
		if a.Lat == 0 && a.Lon == 0 || a.Hex == "" {
			continue
		}
		s := sample{T: now - a.Seen, Lat: a.Lat, Lon: a.Lon, GS: math.NaN(), Track: math.NaN(), VR: math.NaN(),
			Squawk: strings.TrimSpace(a.Squawk), Emerg: a.Emergency != "" && a.Emergency != "none"}
		var alt float64
		if json.Unmarshal(a.AltBaro, &alt) == nil {
			s.AltFt = alt
		} else {
			s.Ground = true
		}
		if a.GS != nil {
			s.GS = *a.GS
		}
		if a.Track != nil {
			s.Track = *a.Track
		}
		if a.BaroRate != nil {
			s.VR = *a.BaroRate
		}
		out[a.Hex] = s
	}
	return out, nil
}
