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
)

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
	if err := run(*live, *headless, *speed, *addr, *dataDir, *day, logf); err != nil {
		log.Fatal(err)
	}
}
