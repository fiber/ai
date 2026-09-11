package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// metar is one parsed flight-weather report. Only the fields the watch
// uses are parsed; the raw text goes to the template miner as is.
type metar struct {
	Station  string
	Time     time.Time
	Raw      string
	Speci    bool
	WindDir  int // degrees, -1 variable/unknown
	WindKt   int
	GustKt   int
	VisM     int // metres, 9999 for 10 km or more, -1 unknown
	Weather  []string
	QNH      int // hPa, 0 unknown
	Auto     bool
	CAVOK    bool
	HasGusts bool
}

var (
	reWind = regexp.MustCompile(`^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?(KT|MPS)$`)
	reVis  = regexp.MustCompile(`^(\d{4})(NDV)?$`)
	reWx   = regexp.MustCompile(`^(\+|-|VC)?(MI|BC|PR|DR|BL|SH|TS|FZ)?(DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS)+$`)
	reQNH  = regexp.MustCompile(`^Q(\d{4})$`)
	reTime = regexp.MustCompile(`^(\d{2})(\d{2})(\d{2})Z$`)
)

// parseMETAR parses the body of a report. ref gives the month the
// day/time group belongs to.
func parseMETAR(raw string, ref time.Time) (metar, bool) {
	m := metar{Raw: strings.TrimSpace(raw), WindDir: -1, VisM: -1}
	f := strings.Fields(m.Raw)
	i := 0
	for i < len(f) && (f[i] == "METAR" || f[i] == "SPECI" || f[i] == "COR" || f[i] == "AUTO") {
		if f[i] == "SPECI" {
			m.Speci = true
		}
		if f[i] == "AUTO" {
			m.Auto = true
		}
		i++
	}
	if i >= len(f) || len(f[i]) != 4 {
		return m, false
	}
	m.Station = f[i]
	i++
	if i < len(f) {
		if t := reTime.FindStringSubmatch(f[i]); t != nil {
			d, _ := strconv.Atoi(t[1])
			h, _ := strconv.Atoi(t[2])
			mi, _ := strconv.Atoi(t[3])
			m.Time = time.Date(ref.Year(), ref.Month(), d, h, mi, 0, 0, time.UTC)
			i++
		}
	}
	inRemarks := false
	for ; i < len(f); i++ {
		tok := f[i]
		switch {
		case tok == "RMK" || tok == "TEMPO" || tok == "BECMG" || tok == "NOSIG":
			inRemarks = true // trend and remark groups: not the current state
		case inRemarks:
		case tok == "AUTO":
			m.Auto = true
		case tok == "CAVOK":
			m.CAVOK = true
			m.VisM = 9999
		case reWind.MatchString(tok):
			w := reWind.FindStringSubmatch(tok)
			if w[1] != "VRB" {
				m.WindDir, _ = strconv.Atoi(w[1])
			}
			m.WindKt, _ = strconv.Atoi(w[2])
			if w[3] != "" {
				m.GustKt, _ = strconv.Atoi(w[3])
				m.HasGusts = true
			}
			if w[4] == "MPS" {
				m.WindKt = int(float64(m.WindKt)*1.944 + 0.5)
				m.GustKt = int(float64(m.GustKt)*1.944 + 0.5)
			}
		case m.VisM < 0 && reVis.MatchString(tok):
			v := reVis.FindStringSubmatch(tok)
			m.VisM, _ = strconv.Atoi(v[1])
		case reWx.MatchString(tok) && len(tok) <= 9:
			m.Weather = append(m.Weather, tok)
		case reQNH.MatchString(tok):
			m.QNH, _ = strconv.Atoi(reQNH.FindStringSubmatch(tok)[1])
		}
	}
	return m, true
}

// metarRecord is one archived or live report with its issue time.
type metarRecord struct {
	Station string
	Time    time.Time
	Raw     string
}

// readMesonetCSV reads the Iowa State Mesonet export (station,valid,metar).
func readMesonetCSV(r io.Reader) ([]metarRecord, error) {
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = -1
	var out []metarRecord
	first := true
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if first {
			first = false
			if len(rec) > 0 && rec[0] == "station" {
				continue
			}
		}
		if len(rec) < 3 {
			continue
		}
		t, err := time.Parse("2006-01-02 15:04", rec[1])
		if err != nil {
			continue
		}
		out = append(out, metarRecord{Station: rec[0], Time: t.UTC(), Raw: rec[2]})
	}
	return out, nil
}

// fetchLiveMETARs asks aviationweather.gov for the latest reports of the
// given stations (raw text, one per line, newest first per station).
func fetchLiveMETARs(client *http.Client, ua string, stations []string) ([]metarRecord, error) {
	url := "https://aviationweather.gov/api/data/metar?format=raw&hours=1&ids=" + strings.Join(stations, ",")
	resp, err := get(client, ua, url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("aviationweather.gov: %s", resp.Status)
	}
	now := time.Now().UTC()
	var out []metarRecord
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		m, ok := parseMETAR(line, now)
		if !ok {
			continue
		}
		t := m.Time
		if t.After(now.Add(12 * time.Hour)) { // day/time group from last month
			t = t.AddDate(0, -1, 0)
		}
		out = append(out, metarRecord{Station: m.Station, Time: t, Raw: line})
	}
	return out, sc.Err()
}
