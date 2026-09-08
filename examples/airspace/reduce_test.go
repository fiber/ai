package main

import (
	"math"
	"testing"
)

// synthetic tracks around EDDF: positions along a straight line through the
// field at a given track, with an altitude profile.
var eddf = airports[airportIndex("EDDF")]

func minuteOfTest(t float64) int { return int(t/60) % minutesPerDay }

// straightTrack builds samples every 10 s moving along heading hdg (degrees)
// at gs knots, starting startNM before the field and passing over it at
// time tPass; alt(dist) gives AGL altitude as a function of signed distance
// (negative before the field).
func straightTrack(hdg, gs, startNM, tPass float64, alt func(signedNM float64) float64) []sample {
	var out []sample
	nmPerSec := gs / 3600
	for d := -startNM; d <= startNM; d += nmPerSec * 10 {
		t := tPass + d/nmPerSec
		rad := hdg * deg
		dy := d * math.Cos(rad)
		dx := d * math.Sin(rad)
		lat := eddf.Lat + dy/60
		lon := eddf.Lon + dx/60/math.Cos(eddf.Lat*deg)
		a := alt(d)
		out = append(out, sample{T: t, Lat: lat, Lon: lon, AltFt: a + eddf.ElevFt, Ground: a <= 0, GS: gs, Track: hdg, VR: 0})
	}
	return out
}

func kinds(evs []trackEvent) map[string]int {
	m := map[string]int{}
	for _, e := range evs {
		m[e.Kind]++
	}
	return m
}

func TestArrival(t *testing.T) {
	// 3° glide from 30 NM out, on the ground after the field, then stays.
	s := straightTrack(250, 150, 30, 36000, func(d float64) float64 {
		if d >= 0 {
			return 0
		}
		return math.Min(9000, -d*318) // 318 ft per NM ≈ 3°
	})
	c, evs := reduceTrack(s, eddf, minuteOfTest)
	k := kinds(evs)
	if k["arrival"] != 1 || k["departure"] != 0 || k["go-around"] != 0 {
		t.Fatalf("arrival track gave %v", k)
	}
	present := 0
	for _, v := range c {
		if v.present {
			present++
		}
	}
	if present < 8 {
		t.Fatalf("expected presence in several minutes, got %d", present)
	}
}

func TestDeparture(t *testing.T) {
	s := straightTrack(70, 160, 30, 40000, func(d float64) float64 {
		if d <= 0 {
			return 0
		}
		return math.Min(9500, d*400)
	})
	_, evs := reduceTrack(s, eddf, minuteOfTest)
	if k := kinds(evs); k["departure"] != 1 || k["arrival"] != 0 {
		t.Fatalf("departure track gave %v", k)
	}
}

func TestGoAround(t *testing.T) {
	// Descends to 400 ft AGL over the field and climbs away again.
	s := straightTrack(250, 150, 30, 50000, func(d float64) float64 {
		return math.Min(9000, math.Abs(d)*318+400)
	})
	_, evs := reduceTrack(s, eddf, minuteOfTest)
	if k := kinds(evs); k["go-around"] != 1 || k["arrival"] != 0 || k["departure"] != 0 {
		t.Fatalf("go-around track gave %v", k)
	}
}

func TestHolding(t *testing.T) {
	// Two racetrack circuits at 15 NM, 7 000 ft, 4 minutes per circuit.
	var s []sample
	cx, cy := 10.0, 10.0 // NM east/north of the field
	r := 2.0
	for i := 0; i <= 96; i++ { // 8 minutes at 5 s
		tt := 60000 + float64(i)*5
		ang := float64(i) / 48 * 2 * math.Pi // one circuit per 48 samples
		dx := cx + r*math.Cos(ang)
		dy := cy + r*math.Sin(ang)
		hdg := math.Mod(90-(ang*180/math.Pi+90)+720, 360)
		s = append(s, sample{T: tt, Lat: eddf.Lat + dy/60, Lon: eddf.Lon + dx/60/math.Cos(eddf.Lat*deg), AltFt: 7000 + eddf.ElevFt, GS: 220, Track: hdg})
	}
	c, evs := reduceTrack(s, eddf, minuteOfTest)
	if kinds(evs)["holding"] < 1 {
		t.Fatalf("holding not detected: %v", kinds(evs))
	}
	hold := 0
	for _, v := range c {
		if v.holding {
			hold++
		}
	}
	if hold < 4 {
		t.Fatalf("expected several holding minutes, got %d", hold)
	}
	// A straight overflight at the same altitude must not be holding.
	_, evs = reduceTrack(straightTrack(90, 400, 40, 70000, func(float64) float64 { return 7000 }), eddf, minuteOfTest)
	if kinds(evs)["holding"] != 0 {
		t.Fatal("straight overflight flagged as holding")
	}
}

func TestEmergencyAndAggregate(t *testing.T) {
	s := straightTrack(90, 300, 40, 80000, func(float64) float64 { return 4000 })
	for i := range s {
		s[i].Squawk = "7700"
	}
	c, _ := reduceTrack(s, eddf, minuteOfTest)
	var agg minuteAgg
	for _, v := range c {
		if v.present {
			agg.add(v)
		}
	}
	v := agg.vector()
	if v[cEmergency] == 0 || v[cInZone] == 0 {
		t.Fatalf("emergency not counted: %v", v)
	}
	if v[cMeanGS] < 290 || v[cMeanGS] > 310 {
		t.Fatalf("mean gs %v", v[cMeanGS])
	}
}

func TestHeadingDelta(t *testing.T) {
	cases := [][3]float64{{350, 10, 20}, {10, 350, -20}, {90, 270, 180}, {0, 180, 180}, {45, 45, 0}}
	for _, c := range cases {
		if d := headingDelta(c[0], c[1]); math.Abs(d-c[2]) > 1e-9 {
			t.Errorf("headingDelta(%v,%v) = %v, want %v", c[0], c[1], d, c[2])
		}
	}
}
