package main

import (
	"bufio"
	"compress/gzip"
	"embed"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

//go:embed testdata/*.csv.gz testdata/ATTRIBUTION.md
var embedded embed.FS

// dataset is the prepared history: counters per day, airport and minute,
// the detected events, the METAR reports, and the storm day's positions.
type dataset struct {
	Days     []string                          // sorted dates, "2006-01-02"
	Counters map[string][][][nCounters]float32 // date -> [airport][minute]
	Events   map[string][]dayEvent             // date -> events
	Metar    map[string][]metarRecord          // date -> reports (sorted by time)
	Tracks   map[string][]trackPoint           // date -> down-sampled positions
}

// openData reads the prepared files from dir, or from the embedded
// testdata when dir is empty.
func openData(dir string) (*dataset, error) {
	var fsys fs.FS
	if dir == "" {
		sub, err := fs.Sub(embedded, "testdata")
		if err != nil {
			return nil, err
		}
		fsys = sub
	} else {
		fsys = os.DirFS(dir)
	}
	d := &dataset{Counters: map[string][][][nCounters]float32{}, Events: map[string][]dayEvent{}, Metar: map[string][]metarRecord{}, Tracks: map[string][]trackPoint{}}
	if err := d.readCounters(fsys, "counters.csv.gz"); err != nil {
		return nil, err
	}
	if err := d.readEvents(fsys, "events.csv.gz"); err != nil {
		return nil, err
	}
	if err := d.readMetar(fsys, "metar.csv.gz"); err != nil {
		return nil, err
	}
	matches, _ := fs.Glob(fsys, "tracks-*.csv.gz")
	for _, m := range matches {
		if err := d.readTracks(fsys, m); err != nil {
			return nil, err
		}
	}
	sort.Strings(d.Days)
	return d, nil
}

func openGz(fsys fs.FS, name string) (io.ReadCloser, error) {
	f, err := fsys.Open(name)
	if err != nil {
		return nil, err
	}
	z, err := gzip.NewReader(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	return struct {
		io.Reader
		io.Closer
	}{z, f}, nil
}

func (d *dataset) addDay(date string) {
	if _, ok := d.Counters[date]; ok {
		return
	}
	c := make([][][nCounters]float32, len(airports))
	for i := range c {
		c[i] = make([][nCounters]float32, minutesPerDay)
	}
	d.Counters[date] = c
	d.Days = append(d.Days, date)
}

func (d *dataset) readCounters(fsys fs.FS, name string) error {
	r, err := openGz(fsys, name)
	if err != nil {
		return err
	}
	defer r.Close()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	first := true
	for sc.Scan() {
		if first {
			first = false
			continue
		}
		f := strings.Split(sc.Text(), ",")
		if len(f) < 3+nCounters {
			continue
		}
		date, ap := f[0], airportIndex(f[2])
		m, _ := strconv.Atoi(f[1])
		if ap < 0 || m < 0 || m >= minutesPerDay {
			continue
		}
		d.addDay(date)
		var v [nCounters]float32
		for i := range v {
			x, _ := strconv.ParseFloat(f[3+i], 32)
			v[i] = float32(x)
		}
		d.Counters[date][ap][m] = v
	}
	return sc.Err()
}

func (d *dataset) readEvents(fsys fs.FS, name string) error {
	r, err := openGz(fsys, name)
	if err != nil {
		return err
	}
	defer r.Close()
	sc := bufio.NewScanner(r)
	first := true
	for sc.Scan() {
		if first {
			first = false
			continue
		}
		f := strings.Split(sc.Text(), ",")
		if len(f) < 5 {
			continue
		}
		m, _ := strconv.Atoi(f[1])
		d.Events[f[0]] = append(d.Events[f[0]], dayEvent{Airport: f[2], Minute: m, Kind: f[3], Hex: f[4]})
	}
	return sc.Err()
}

func (d *dataset) readMetar(fsys fs.FS, name string) error {
	r, err := openGz(fsys, name)
	if err != nil {
		return err
	}
	defer r.Close()
	recs, err := readMesonetCSV(r)
	if err != nil {
		return err
	}
	for _, rec := range recs {
		date := rec.Time.Format("2006-01-02")
		d.Metar[date] = append(d.Metar[date], rec)
	}
	return nil
}

func (d *dataset) readTracks(fsys fs.FS, name string) error {
	r, err := openGz(fsys, name)
	if err != nil {
		return err
	}
	defer r.Close()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	first := true
	for sc.Scan() {
		if first {
			first = false
			continue
		}
		f := strings.Split(sc.Text(), ",")
		if len(f) < 7 {
			continue
		}
		ap := airportIndex(f[2])
		if ap < 0 {
			continue
		}
		sec, _ := strconv.Atoi(f[1])
		dx, _ := strconv.ParseFloat(f[4], 32)
		dy, _ := strconv.ParseFloat(f[5], 32)
		alt, _ := strconv.ParseFloat(f[6], 32)
		d.Tracks[f[0]] = append(d.Tracks[f[0]], trackPoint{Sec: sec, Airport: ap, Hex: f[3], DX: float32(dx), DY: float32(dy), AltFt: float32(alt)})
	}
	return sc.Err()
}

// dayOf parses a date string as UTC midnight.
func dayOf(date string) time.Time {
	t, _ := time.Parse("2006-01-02", date)
	return t
}

// describe summarises the data set for the log.
func (d *dataset) describe() string {
	var b strings.Builder
	for _, day := range d.Days {
		mov := 0.0
		for ap := range airports {
			for m := 0; m < minutesPerDay; m++ {
				v := d.Counters[day][ap][m]
				mov += float64(v[cArrivals] + v[cDepartures])
			}
		}
		fmt.Fprintf(&b, "%s: %.0f movements, %d events, %d METARs, %d track points\n", day, mov, len(d.Events[day]), len(d.Metar[day]), len(d.Tracks[day]))
	}
	return strings.TrimRight(b.String(), "\n")
}

// attribution returns the data credits shipped with the example.
func attribution(dir string) string {
	if dir != "" {
		b, err := os.ReadFile(filepath.Join(dir, "ATTRIBUTION.md"))
		if err == nil {
			return string(b)
		}
	}
	b, _ := embedded.ReadFile("testdata/ATTRIBUTION.md")
	return string(b)
}
