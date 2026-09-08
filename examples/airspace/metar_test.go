package main

import (
	"math"
	"strings"
	"testing"
	"time"
)

func TestParseMETAR(t *testing.T) {
	ref := time.Date(2025, 1, 24, 0, 0, 0, 0, time.UTC)
	cases := []struct {
		raw                         string
		wind, gust, vis             int
		speci, auto, cavok, hasGust bool
		wx                          string
	}{
		{"EIDW 240500Z 22032G54KT 200V260 9999 FEW015 SCT025 BKN045 08/00 Q0967 TEMPO 22038G63KT", 32, 54, 9999, false, false, false, true, ""},
		{"SPECI EGLL 240612Z AUTO 24018G31KT 0600 R27L/1200 FG BKN002 05/05 Q0982", 18, 31, 600, true, true, false, true, "FG"},
		{"EDDF 240850Z 20008KT 170V240 CAVOK 03/M02 Q1012 NOSIG", 8, 0, 9999, false, false, true, false, ""},
		{"LFPG 241000Z VRB03KT 4000 -RA BR OVC008 04/03 Q1001", 3, 0, 4000, false, false, false, false, "-RA BR"},
		{"METAR EHAM 241125Z 25025G40KT 220V280 9999 -SHRA FEW020CB SCT030 07/02 Q0990 TEMPO 26030G45KT", 25, 40, 9999, false, false, false, true, "-SHRA"},
	}
	for _, c := range cases {
		m, ok := parseMETAR(c.raw, ref)
		if !ok {
			t.Fatalf("parse failed: %s", c.raw)
		}
		if m.WindKt != c.wind || m.GustKt != c.gust || m.VisM != c.vis || m.Speci != c.speci || m.Auto != c.auto || m.CAVOK != c.cavok || m.HasGusts != c.hasGust {
			t.Errorf("%s\n got wind %d gust %d vis %d speci %v auto %v cavok %v gusts %v", c.raw, m.WindKt, m.GustKt, m.VisM, m.Speci, m.Auto, m.CAVOK, m.HasGusts)
		}
		if got := strings.Join(m.Weather, " "); got != c.wx {
			t.Errorf("%s\n weather %q, want %q", c.raw, got, c.wx)
		}
		if m.Time.Day() != 24 {
			t.Errorf("time %v", m.Time)
		}
	}
	// The trend group must not overwrite the current wind.
	m, _ := parseMETAR("EIDW 240500Z 22032G54KT 9999 FEW015 Q0967 TEMPO 22038G63KT", ref)
	if m.GustKt != 54 {
		t.Errorf("trend gust leaked: %d", m.GustKt)
	}
}

func TestMesonetCSV(t *testing.T) {
	in := "station,valid,metar\nEIDW,2025-01-24 00:00,EIDW 240000Z 14022G32KT 6000 -RA BKN009 06/06 Q0979\nEGLL,2025-01-24 00:20,COR EGLL 240020Z AUTO 19010KT 9999 FEW026 07/04 Q1000 NOSIG\n"
	recs, err := readMesonetCSV(strings.NewReader(in))
	if err != nil || len(recs) != 2 || recs[1].Station != "EGLL" || recs[1].Time.Minute() != 20 {
		t.Fatalf("%v %v", recs, err)
	}
}

func TestHellinger(t *testing.T) {
	a := []float64{0.5, 0.5, 0}
	if d := hellinger(a, a); d > 1e-12 {
		t.Fatalf("identical: %v", d)
	}
	if d := hellinger([]float64{1, 0}, []float64{0, 1}); math.Abs(d-1) > 1e-12 {
		t.Fatalf("disjoint: %v", d)
	}
	if d := hellinger([]float64{0.5, 0.5}, []float64{1}); d <= 0 || d >= 1 {
		t.Fatalf("shorter reference: %v", d)
	}
}

func TestMetarWatchShift(t *testing.T) {
	w := newMetarWatch()
	var reports []timedReport
	calm := []string{
		"EDDF %02d%02d50Z 20008KT 9999 FEW030 05/01 Q1015 NOSIG",
		"EDDF %02d%02d20Z 21009KT 9999 SCT035 06/01 Q1014 NOSIG",
	}
	for d := 0; d < 3; d++ {
		for h := 0; h < 24; h++ {
			for i, tmpl := range calm {
				reports = append(reports, timedReport{Station: "EDDF", Minute: d*1440 + h*60 + i*30, Raw: strings.ReplaceAll(strings.ReplaceAll(tmpl, "%02d%02d", "2410"), "%02d", "10")})
			}
		}
	}
	w.train(reports, 3*1440, func(string) {})
	// An hour of calm reports keeps the shift low.
	var last metarObservation
	for i := 0; i < 2; i++ {
		last = w.observe("EDDF", 5000+i*30, "EDDF 241050Z 20008KT 9999 FEW030 05/01 Q1015 NOSIG")
	}
	if last.Shift > 0.2 {
		t.Fatalf("calm shift %.2f", last.Shift)
	}
	// An hour of storm reports moves the mix.
	for i := 0; i < 4; i++ {
		last = w.observe("EDDF", 6000+i*15, "SPECI EDDF 241207Z 25035G58KT 2000 +SHRA BKN008 04/02 Q0975")
	}
	if last.Shift < 0.5 {
		t.Fatalf("storm shift %.2f", last.Shift)
	}
}
