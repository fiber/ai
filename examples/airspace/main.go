package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"
)

var (
	addr      = flag.String("addr", "127.0.0.1:8080", "dashboard address")
	speed     = flag.Int("speed", 10, "replay: simulated minutes per second")
	live      = flag.Bool("live", false, "watch the live feeds instead of replaying the storm day")
	headless  = flag.Bool("headless", false, "no dashboard: print alarms and events for the replay and exit")
	dataDir   = flag.String("data", "", "directory with the prepared data (default: the embedded testdata)")
	day       = flag.String("day", "", "replay: the day to replay, trained on the days before it (default: the last day)")
	prep      = flag.Bool("prep", false, "build the data files from adsb.lol archives and a Mesonet METAR export")
	archives  = flag.String("archives", "", "prep: directory holding the daily *.tar archives")
	metarCSV  = flag.String("metar", "", "prep: Mesonet CSV export (station,valid,metar)")
	outDir    = flag.String("out", "examples/airspace/testdata", "prep: output directory")
	trackDate = flag.String("tracks", "2025-01-24", "prep: day whose positions are kept for the map")
	interval  = flag.Duration("interval", 2*time.Minute, "live: how often the feeds are polled")
	contact   = flag.String("contact", "", "live: contact address sent in the User-Agent (required with -live)")
	source    = flag.String("source", adsbLol, "live: readsb JSON endpoint; with lat/lon/dist verbs it is filled in per region, otherwise fetched as it stands")
)

// liveOpts carries the live-mode flags to the poller.
var liveOpts liveOptions

// liveHelp is what -live without -contact prints. The example is in a
// public repository: a free aggregator can carry one identified client
// polling every other minute, not a thousand anonymous ones.
const liveHelp = `-live needs -contact: every request has to say who is calling.

  go run ./examples/airspace -live -contact you@example.com

Read the terms of the source you point at first (the default is
api.adsb.lol). Better still, feed from your own receiver and leave the
shared services alone:

  go run ./examples/airspace -live -contact you@example.com \
      -source http://your-pi/data/aircraft.json

Without -live the example replays the bundled storm day and touches no
network at all.`

func main() {
	flag.Parse()
	logf := func(s string) { fmt.Fprintf(os.Stderr, "%s %s\n", time.Now().Format("15:04:05"), s) }
	if *prep {
		if *archives == "" {
			log.Fatal("-prep needs -archives")
		}
		if err := runPrep(*archives, *metarCSV, *outDir, *trackDate, logf); err != nil {
			log.Fatal(err)
		}
		return
	}
	if *live {
		if *contact == "" {
			log.Fatal(liveHelp)
		}
		if *interval < time.Minute {
			logf("interval raised to one minute: the counters are per minute and the feeds are free")
			*interval = time.Minute
		}
		liveOpts = liveOptions{source: *source, contact: *contact, interval: *interval}
	}
	if err := run(*live, *headless, *speed, *addr, *dataDir, *day, logf); err != nil {
		log.Fatal(err)
	}
}
