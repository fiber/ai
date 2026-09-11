package main

import (
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func testMinuteOf(t float64) int {
	tt := time.Unix(int64(t), 0).UTC()
	return tt.Hour()*60 + tt.Minute()
}

// inZone returns the in-zone counter a single aircraft produces in one
// minute, the way the live loop computes it.
func inZone(t *testing.T, ss []sample, ap airport, minute int) float32 {
	t.Helper()
	contrib, _ := reduceTrack(ss, ap, testMinuteOf)
	var agg minuteAgg
	if c := contrib[minute]; c != nil {
		agg.add(c)
	}
	return agg.vector()[cInZone]
}

// A minute with no poll of its own sees the last known picture, so the
// counters do not drop to an empty sky between two polls.
func TestCarryForward(t *testing.T) {
	ap := airports[0]
	at := time.Date(2026, 9, 11, 10, 30, 0, 0, time.UTC)
	s := sample{
		T: float64(at.Unix()), Lat: ap.Lat, Lon: ap.Lon,
		AltFt: ap.ElevFt + 2000, GS: 180, Track: 90, VR: math.NaN(),
	}
	fetched := testMinuteOf(s.T)
	if got := inZone(t, []sample{s}, ap, fetched); got != 1 {
		t.Fatalf("in-zone counter of the polled minute = %v, want 1", got)
	}
	next := at.Add(time.Minute)
	carried := carryForward(map[string]sample{"abc123": s}, next)
	c, ok := carried["abc123"]
	if !ok {
		t.Fatal("carryForward dropped the aircraft")
	}
	if c.Lat != s.Lat || c.Lon != s.Lon || c.AltFt != s.AltFt {
		t.Fatalf("carried sample moved: %+v", c)
	}
	if got, want := c.T, float64(next.Unix()); got != want {
		t.Fatalf("carried timestamp = %v, want %v", got, want)
	}
	if got := inZone(t, []sample{s, c}, ap, testMinuteOf(c.T)); got != 1 {
		t.Fatalf("in-zone counter of the carried minute = %v, want 1", got)
	}
}

// A refused region is left alone for ten minutes, then twenty, up to an
// hour, and the wait is forgotten after a request that goes through.
func TestBackoff(t *testing.T) {
	var b backoff
	now := time.Date(2026, 9, 11, 10, 0, 0, 0, time.UTC)
	if !b.ready(now) {
		t.Fatal("a fresh region should be polled")
	}
	want := []time.Duration{10 * time.Minute, 20 * time.Minute, 40 * time.Minute, time.Hour, time.Hour}
	for i, w := range want {
		if got := b.refused(now); got != w {
			t.Fatalf("refusal %d waits %s, want %s", i+1, got, w)
		}
		if b.ready(now.Add(w - time.Minute)) {
			t.Fatalf("refusal %d: polled again before the wait was over", i+1)
		}
		if !b.ready(now.Add(w)) {
			t.Fatalf("refusal %d: still held off after the wait", i+1)
		}
	}
	b.ok()
	if !b.ready(now) || b.wait != 0 {
		t.Fatalf("a successful poll should clear the retreat, got until=%v wait=%s", b.until, b.wait)
	}
}

// The aggregators call the aircraft list "ac", a local readsb "aircraft".
// Both decode the same way, and every request says who is calling.
func TestDecodeReadsb(t *testing.T) {
	const body = `{"now":1.7e12,%q:[{"hex":"abc123","flight":"TEST","alt_baro":3000,"gs":180,"track":90,"baro_rate":0,"squawk":"1000","lat":53.6,"lon":-3.4,"seen_pos":2}]}`
	var seenUA string
	for _, key := range []string{"ac", "aircraft"} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			seenUA = r.Header.Get("User-Agent")
			fmt.Fprintf(w, body, key)
		}))
		ua := fmt.Sprintf(userAgentFmt, "you@example.com")
		got, err := fetchRegion(srv.Client(), ua, srv.URL, 53.6, -3.4, 250)
		srv.Close()
		if err != nil {
			t.Fatalf("%s: %v", key, err)
		}
		s, ok := got["abc123"]
		if !ok || len(got) != 1 {
			t.Fatalf("%s: decoded %d aircraft, want the one", key, len(got))
		}
		if s.Lat != 53.6 || s.Lon != -3.4 || s.AltFt != 3000 {
			t.Fatalf("%s: decoded %+v", key, s)
		}
		if !strings.Contains(seenUA, "github.com/fiber/ai") || !strings.Contains(seenUA, "you@example.com") {
			t.Fatalf("%s: User-Agent was %q", key, seenUA)
		}
	}
}
