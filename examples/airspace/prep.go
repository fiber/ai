package main

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/fiber/ai/internal/parallel"
)

// Prep reduces adsb.lol daily archives and a Mesonet METAR export to the
// files under testdata/. It is how the shipped data was made and is kept
// for reproducibility; running the example does not need it.
//
//	go run ./examples/airspace -prep -archives dir/with/v2025.01.NN.tar \
//	    -metar mesonet.csv -out examples/airspace/testdata -tracks 2025-01-24

// traceFile is the readsb trace_full JSON layout.
type traceFile struct {
	ICAO      string            `json:"icao"`
	Timestamp float64           `json:"timestamp"`
	Trace     []json.RawMessage `json:"trace"`
}

// parseTrace decodes one aircraft's day into samples. Each trace row is
// [dt, lat, lon, alt, gs, track, flags, vr, extra, source, ...]; alt is
// "ground" on the ground.
func parseTrace(tf *traceFile) (hex string, samples []sample) {
	samples = make([]sample, 0, len(tf.Trace))
	var squawk string
	emerg := false
	for _, row := range tf.Trace {
		var f []json.RawMessage
		if err := json.Unmarshal(row, &f); err != nil || len(f) < 8 {
			continue
		}
		var dt, lat, lon float64
		if json.Unmarshal(f[0], &dt) != nil || json.Unmarshal(f[1], &lat) != nil || json.Unmarshal(f[2], &lon) != nil {
			continue
		}
		s := sample{T: tf.Timestamp + dt, Lat: lat, Lon: lon, GS: math.NaN(), Track: math.NaN(), VR: math.NaN()}
		var alt float64
		if json.Unmarshal(f[3], &alt) == nil {
			s.AltFt = alt
		} else {
			var str string
			if json.Unmarshal(f[3], &str) == nil && str == "ground" {
				s.Ground = true
			}
		}
		var v float64
		if json.Unmarshal(f[4], &v) == nil {
			s.GS = v
		}
		if json.Unmarshal(f[5], &v) == nil {
			s.Track = v
		}
		if json.Unmarshal(f[7], &v) == nil {
			s.VR = v
		}
		if len(f) > 8 && len(f[8]) > 2 {
			var extra struct {
				Squawk    string `json:"squawk"`
				Emergency string `json:"emergency"`
			}
			if json.Unmarshal(f[8], &extra) == nil {
				if extra.Squawk != "" {
					squawk = extra.Squawk
				}
				if extra.Emergency != "" {
					emerg = extra.Emergency != "none"
				}
			}
		}
		s.Squawk, s.Emerg = squawk, emerg
		samples = append(samples, s)
	}
	return tf.ICAO, samples
}

// dayCounters holds one day's counter vectors: [airport][minute].
type dayCounters struct {
	Date   string
	Values [][]([nCounters]float32)
	Events []dayEvent
}

type dayEvent struct {
	Airport string
	Minute  int
	Kind    string
	Hex     string
}

// trackPoint is a down-sampled position for the storm-day map.
type trackPoint struct {
	Sec     int // second of day
	Airport int
	Hex     string
	DX, DY  float32 // NM east/north of the field
	AltFt   float32
}

// reduceArchive streams one daily tar and reduces every aircraft that came
// near a watched airport. Traces are gzip-JSON files under traces/xx/.
func reduceArchive(path string, keepTracks bool, log func(string)) (*dayCounters, []trackPoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	tr := tar.NewReader(bufio.NewReaderSize(f, 8<<20))

	date := ""
	day := &dayCounters{}
	aggs := make([][]minuteAgg, len(airports))
	for i := range aggs {
		aggs[i] = make([]minuteAgg, 1440)
	}
	var tracks []trackPoint
	var mu sync.Mutex
	var dayStart float64

	type job struct {
		hex     string
		samples []sample
	}
	jobs := make(chan job, 256)
	var wg sync.WaitGroup
	workers := parallel.Workers()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				local := make([]struct {
					ap      int
					contrib map[int]*minuteContribution
					events  []trackEvent
					pts     []trackPoint
				}, 0, 2)
				for ai, ap := range airports {
					near := false
					for _, s := range j.samples {
						if ap.inBox(s.Lat, s.Lon) {
							near = true
							break
						}
					}
					if !near {
						continue
					}
					minuteOf := func(t float64) int {
						m := int((t - dayStart) / 60)
						return max(0, min(1439, m))
					}
					c, ev := reduceTrack(j.samples, ap, minuteOf)
					var pts []trackPoint
					if keepTracks {
						lastSec := -100
						for _, s := range j.samples {
							sec := int(s.T - dayStart)
							if sec-lastSec < 20 || !ap.inBox(s.Lat, s.Lon) {
								continue
							}
							agl := s.AltFt - ap.ElevFt
							if s.Ground {
								agl = 0
							}
							if agl >= ap.ceilingFt {
								continue
							}
							dx, dy := ap.localNM(s.Lat, s.Lon)
							if math.Hypot(dx, dy) > ap.radiusNM {
								continue
							}
							pts = append(pts, trackPoint{Sec: sec, Airport: ai, Hex: j.hex, DX: float32(dx), DY: float32(dy), AltFt: float32(agl)})
							lastSec = sec
						}
					}
					local = append(local, struct {
						ap      int
						contrib map[int]*minuteContribution
						events  []trackEvent
						pts     []trackPoint
					}{ai, c, ev, pts})
				}
				if len(local) == 0 {
					continue
				}
				mu.Lock()
				for _, l := range local {
					for m, c := range l.contrib {
						aggs[l.ap][m].add(c)
					}
					for _, e := range l.events {
						switch e.Kind {
						case "arrival":
							aggs[l.ap][e.Minute].arrivals++
						case "departure":
							aggs[l.ap][e.Minute].departures++
						case "go-around":
							aggs[l.ap][e.Minute].goArounds++
						}
						day.Events = append(day.Events, dayEvent{Airport: airports[l.ap].ICAO, Minute: e.Minute, Kind: e.Kind, Hex: j.hex})
					}
					tracks = append(tracks, l.pts...)
				}
				mu.Unlock()
			}
		}()
	}

	n, kept := 0, 0
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			close(jobs)
			wg.Wait()
			return nil, nil, err
		}
		if h.Typeflag != tar.TypeReg || !strings.Contains(h.Name, "trace_full_") {
			continue
		}
		raw, err := io.ReadAll(tr)
		if err != nil {
			continue
		}
		zr, err := gzip.NewReader(strings.NewReader(string(raw)))
		var body []byte
		if err == nil {
			body, err = io.ReadAll(zr)
		}
		if err != nil {
			body = raw // some archives store plain JSON
		}
		var tf traceFile
		if json.Unmarshal(body, &tf) != nil {
			continue
		}
		n++
		if date == "" && tf.Timestamp > 0 {
			dayStart = math.Floor(tf.Timestamp/86400) * 86400
			date = time.Unix(int64(dayStart), 0).UTC().Format("2006-01-02")
			day.Date = date
		}
		// Cheap geographic pre-filter on the first, middle and last points
		// would miss transits; check every 20th point instead.
		hex, samples := parseTrace(&tf)
		near := false
		for i := 0; i < len(samples) && !near; i += 5 {
			for _, ap := range airports {
				if ap.inBox(samples[i].Lat, samples[i].Lon) {
					near = true
					break
				}
			}
		}
		if !near {
			continue
		}
		kept++
		jobs <- job{hex: hex, samples: samples}
		if n%5000 == 0 {
			log(fmt.Sprintf("%s: %d traces read, %d near an airport", filepath.Base(path), n, kept))
		}
	}
	close(jobs)
	wg.Wait()
	log(fmt.Sprintf("%s: %d traces, %d near an airport, %d events", filepath.Base(path), n, kept, len(day.Events)))

	day.Values = make([][]([nCounters]float32), len(airports))
	for ai := range airports {
		day.Values[ai] = make([]([nCounters]float32), 1440)
		for m := 0; m < 1440; m++ {
			day.Values[ai][m] = aggs[ai][m].vector()
		}
	}
	sort.Slice(day.Events, func(i, j int) bool { return day.Events[i].Minute < day.Events[j].Minute })
	sort.Slice(tracks, func(i, j int) bool { return tracks[i].Sec < tracks[j].Sec })
	return day, tracks, nil
}

// writeCounters appends one day's counters to a CSV writer:
// date,minute,airport,c0..c7.
func writeCounters(w *bufio.Writer, d *dayCounters) {
	for ai, ap := range airports {
		for m := 0; m < 1440; m++ {
			v := d.Values[ai][m]
			fmt.Fprintf(w, "%s,%d,%s", d.Date, m, ap.ICAO)
			for _, x := range v {
				fmt.Fprintf(w, ",%g", x)
			}
			w.WriteByte('\n')
		}
	}
}

func writeEvents(w *bufio.Writer, d *dayCounters) {
	for _, e := range d.Events {
		fmt.Fprintf(w, "%s,%d,%s,%s,%s\n", d.Date, e.Minute, e.Airport, e.Kind, e.Hex)
	}
}

func writeTracks(w *bufio.Writer, date string, pts []trackPoint) {
	for _, p := range pts {
		fmt.Fprintf(w, "%s,%d,%s,%s,%.2f,%.2f,%.0f\n", date, p.Sec, airports[p.Airport].ICAO, p.Hex, p.DX, p.DY, p.AltFt)
	}
}

// runPrep is the -prep mode.
func runPrep(archiveDir, metarCSV, outDir, trackDate string, log func(string)) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	tars, _ := filepath.Glob(filepath.Join(archiveDir, "*.tar"))
	sort.Strings(tars)
	if len(tars) == 0 {
		return fmt.Errorf("no *.tar in %s", archiveDir)
	}
	cf, err := os.Create(filepath.Join(outDir, "counters.csv.gz"))
	if err != nil {
		return err
	}
	defer cf.Close()
	cz := gzip.NewWriter(cf)
	cw := bufio.NewWriter(cz)
	fmt.Fprintln(cw, "date,minute,airport,in_zone,arrivals,departures,holding,go_arounds,emergency,mean_gs,on_final")
	ef, err := os.Create(filepath.Join(outDir, "events.csv.gz"))
	if err != nil {
		return err
	}
	defer ef.Close()
	ez := gzip.NewWriter(ef)
	ew := bufio.NewWriter(ez)
	fmt.Fprintln(ew, "date,minute,airport,kind,hex")
	var tw *bufio.Writer
	var tz *gzip.Writer
	for _, path := range tars {
		keep := trackDate != "" && strings.Contains(filepath.Base(path), strings.ReplaceAll(trackDate, "-", "."))
		day, tracks, err := reduceArchive(path, keep, log)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		writeCounters(cw, day)
		writeEvents(ew, day)
		if keep {
			tf, err := os.Create(filepath.Join(outDir, "tracks-"+day.Date+".csv.gz"))
			if err != nil {
				return err
			}
			tz = gzip.NewWriter(tf)
			tw = bufio.NewWriter(tz)
			fmt.Fprintln(tw, "date,second,airport,hex,dx_nm,dy_nm,alt_ft")
			writeTracks(tw, day.Date, tracks)
			tw.Flush()
			tz.Close()
			tf.Close()
			log(fmt.Sprintf("tracks for %s: %d points", day.Date, len(tracks)))
		}
	}
	cw.Flush()
	cz.Close()
	ew.Flush()
	ez.Close()

	// METAR: keep the stations we watch, sorted by time.
	if metarCSV != "" {
		in, err := os.Open(metarCSV)
		if err != nil {
			return err
		}
		recs, err := readMesonetCSV(in)
		in.Close()
		if err != nil {
			return err
		}
		sort.Slice(recs, func(i, j int) bool { return recs[i].Time.Before(recs[j].Time) })
		mf, err := os.Create(filepath.Join(outDir, "metar.csv.gz"))
		if err != nil {
			return err
		}
		mz := gzip.NewWriter(mf)
		mw := bufio.NewWriter(mz)
		fmt.Fprintln(mw, "station,valid,metar")
		n := 0
		for _, r := range recs {
			if airportIndex(r.Station) < 0 {
				continue
			}
			fmt.Fprintf(mw, "%s,%s,%s\n", r.Station, r.Time.Format("2006-01-02 15:04"), strings.ReplaceAll(r.Raw, ",", " "))
			n++
		}
		mw.Flush()
		mz.Close()
		mf.Close()
		log(fmt.Sprintf("metar: %d reports", n))
	}
	return nil
}
