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

type liveResponse struct {
	AC  []liveAircraft `json:"ac"`
	Now float64        `json:"now"` // milliseconds
}

// runLive polls the feeds every minute and steps the watch.
func (w *watch) runLive(logf func(string)) {
	w.mode = "live"
	client := &http.Client{Timeout: 30 * time.Second}
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
		// adsb.lol rate-limits per-airport polling; two regional requests of
		// 250 NM cover all eight fields (west: the Irish Sea; east: the
		// Benelux), and the samples are split by zone locally.
		fresh := make([]map[string]sample, len(airports))
		for i := range fresh {
			fresh[i] = map[string]sample{}
		}
		for ri, reg := range liveRegions {
			if ri > 0 {
				time.Sleep(5 * time.Second)
			}
			hexes, err := fetchRegion(client, adsbLol, reg.lat, reg.lon, reg.radiusNM)
			if err != nil && strings.Contains(err.Error(), "429") {
				// Rate-limited: the other aggregator serves the same format.
				time.Sleep(3 * time.Second)
				hexes, err = fetchRegion(client, airplanesLive, reg.lat, reg.lon, reg.radiusNM)
			}
			if err != nil {
				logf(fmt.Sprintf("region %s: %v", reg.name, err))
				continue
			}
			for hex, smp := range hexes {
				for ai, ap := range airports {
					if distanceNM(smp.Lat, smp.Lon, ap.Lat, ap.Lon) <= ap.radiusNM*1.5 {
						fresh[ai][hex] = smp
					}
				}
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
		var metars []metarRecord
		stations := make([]string, len(airports))
		for i, ap := range airports {
			stations[i] = ap.ICAO
		}
		if recs, err := fetchLiveMETARs(client, stations); err != nil {
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

// The two public aggregators speak the same readsb JSON; adsb.lol is the
// first choice (its history archive is what the replay was built from),
// airplanes.live the fallback when it rate-limits.
const (
	adsbLol       = "https://api.adsb.lol/v2/lat/%.4f/lon/%.4f/dist/%d"
	airplanesLive = "https://api.airplanes.live/v2/point/%.4f/%.4f/%d"
)

// fetchRegion returns the current sample of every aircraft within radiusNM
// of a point, keyed by ICAO address.
func fetchRegion(client *http.Client, source string, lat, lon, radiusNM float64) (map[string]sample, error) {
	url := fmt.Sprintf(source, lat, lon, int(radiusNM))
	resp, err := client.Get(url)
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
	out := make(map[string]sample, len(lr.AC))
	for _, a := range lr.AC {
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
