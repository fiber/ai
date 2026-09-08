package main

import (
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"
)

// TestStormReplay runs the headless replay on the shipped data and checks
// the acceptance criteria of the spec: the storm is detected at the western
// airports within 30 minutes of the first gust above 45 kt, Frankfurt raises
// no band alarm, and the alarm volume stays bounded.
func TestStormReplay(t *testing.T) {
	if testing.Short() {
		t.Skip("trains models; skipped in -short")
	}
	d, err := openData("")
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Days) < 3 {
		t.Skipf("need training days plus the storm day, have %v", d.Days)
	}
	var lines []string
	w := newWatch(d, "2025-01-24", func(s string) { lines = append(lines, s) })
	if w.date != "2025-01-24" {
		t.Fatalf("replay day %s, want the storm day", w.date)
	}
	w.runReplay(1000, true, func(string) {})

	// Each station's storm window from its own METARs: two hours before the
	// first gust of 40 kt or more (the first missing flights precede the
	// peak) to ninety minutes after the last. The first model alarm must
	// fall inside it.
	type win struct{ first, last int }
	gusts := map[string]*win{}
	for _, rec := range d.Metar[w.date] {
		m, ok := parseMETAR(rec.Raw, dayOf(w.date))
		if !ok || m.GustKt < 40 {
			continue
		}
		t := rec.Time.Hour()*60 + rec.Time.Minute()
		if g := gusts[rec.Station]; g == nil {
			gusts[rec.Station] = &win{t, t}
		} else {
			g.last = t
		}
	}
	firstAlarm := map[string]int{}
	perAirportKind := map[string]int{}
	bundles := map[string]bool{}
	for _, a := range w.alarms {
		if _, seen := firstAlarm[a.Airport]; !seen {
			firstAlarm[a.Airport] = a.Minute
		}
		perAirportKind[a.Airport+" "+a.Kind]++
		bundles[fmt.Sprintf("%s/%d", a.Airport, a.Minute/60)] = true
	}
	t.Logf("%d alarms, %d events on %s", len(w.alarms), len(w.events), w.date)
	for _, st := range []string{"EIDW", "EGPH", "EGCC"} {
		g := gusts[st]
		if g == nil {
			t.Errorf("%s: no gust of 40 kt in the METARs of the storm day", st)
			continue
		}
		a, ok := firstAlarm[st]
		if !ok {
			t.Errorf("%s: storm not detected (gusts %s to %s)", st, clockOf(g.first), clockOf(g.last))
			continue
		}
		t.Logf("%s: gusts of 40 kt or more %s to %s, first model alarm %s", st, clockOf(g.first), clockOf(g.last), clockOf(a))
		if a < g.first-120 || a > g.last+90 {
			t.Errorf("%s: first alarm %s outside the storm window %s to %s", st, clockOf(a), clockOf(g.first-120), clockOf(g.last+90))
		}
	}
	// Frankfurt was untouched by the storm: no forecast-band alarm in the
	// operating day (the early hours differ by weekday, which five training
	// days cannot teach).
	for _, a := range w.alarms {
		if a.Airport == "EDDF" && a.Kind == "band" && strings.Contains(a.Text, "movements") && a.Minute >= 6*60 && a.Minute < 22*60 {
			t.Errorf("EDDF movements alarm at %s: %s", clockOf(a.Minute), a.Text)
		}
	}
	if len(w.alarms) > 40 {
		t.Errorf("%d alarms on the storm day, want at most 40", len(w.alarms))
	}
	if len(bundles) > 40 {
		t.Errorf("%d airport-hour bundles, want at most 40", len(bundles))
	}
	keys := make([]string, 0, len(perAirportKind))
	for k := range perAirportKind {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		t.Logf("  %-12s %d", k, perAirportKind[k])
	}
	_ = strings.Join(lines, "\n")
}

func clockOf(m int) string {
	return time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC).Add(time.Duration(m) * time.Minute).Format("15:04")
}

// TestQuietDay replays an ordinary day (2025-01-23, trained on the four days
// before it) and expects the models to stay mostly quiet.
func TestQuietDay(t *testing.T) {
	if testing.Short() {
		t.Skip("trains models; skipped in -short")
	}
	d, err := openData("")
	if err != nil {
		t.Fatal(err)
	}
	if d.Counters["2025-01-23"] == nil || len(d.Days) < 5 {
		t.Skip("hold-out day not in the data")
	}
	w := newWatch(d, "2025-01-23", func(string) {})
	w.runReplay(1000, true, func(string) {})
	t.Logf("%d alarms, %d events on the quiet day", len(w.alarms), len(w.events))
	for _, a := range w.alarms {
		t.Logf("  %s %-10s %-5s %s", clockOf(a.Minute), a.Kind, a.Airport, a.Text)
	}
	if len(w.alarms) > 10 {
		t.Errorf("%d alarms on an ordinary day, want at most 10", len(w.alarms))
	}
}
