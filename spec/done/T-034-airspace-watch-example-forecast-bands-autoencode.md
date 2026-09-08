---
id: T-034
title: Airspace watch example: forecast bands, autoencoder and METAR templates over live and historical ADS-B and flight weather data
status: done
scope:
  - examples/airspace/
manual:
  - docs/manual/applications.md
done: 2026-09-08
created: 2026-09-08
---

## Goal

A publishable counterpart to the private network-watch application: one
program that watches eight European airports the way an operator
watches a network, on public data with open licences. Airports are the
entities, aircraft movements the counters, flight-weather reports the
event stream. It trains on ordinary days, replays Storm Éowyn
(2025-01-24) with the models running, and can also run live. It is the
worked example for the `applications.md` page and shows the `nn`,
`logtemplate`, `data` and `metrics` packages together on data anyone
can fetch.

## Design

### Data

- **Airports:** EIDW Dublin, EGPH Edinburgh (Belfast was tried first;
  the archive's receiver coverage there is too thin to count movements),
  EGCC Manchester, EGLL
  London Heathrow, EHAM Amsterdam, EBBR Brussels, LFPG Paris CDG, EDDF
  Frankfurt, west to east so a front is seen crossing them. Each has a
  zone: 40 NM radius, below 10 000 ft.
- **ADS-B history:** adsb.lol daily archives (GitHub releases of
  `adsblol/globe_history_2025`, ODbL 1.0 / CC0), one gzip-JSON trace
  per aircraft per day. A `-prep` mode reads the tar, keeps positions
  inside the zones and reduces them to per-airport, per-minute counters:
  aircraft in zone, arrivals, departures, aircraft in holding, go-arounds,
  emergency squawks (7500/7600/7700), mean ground speed and mean vertical
  rate of arriving traffic. Arrival: a trace that descends through
  1 500 ft within 5 NM of the field and does not climb again;
  departure: the mirror image; holding: a level segment whose heading
  accumulates more than 360° within 12 minutes inside the zone;
  go-around: descent below 1 500 ft within 5 NM followed by a climb of
  more than 500 ft without a gap in the trace.
- **METAR history:** the Iowa State Mesonet ASOS archive
  (`asos.py`, raw METAR text, all eight stations, no key); live METARs
  from the aviationweather.gov cache file (`metars.cache.csv.gz`,
  updated every minute, US National Weather Service, public domain).
- **Live ADS-B:** adsb.lol `v2/lat/{lat}/lon/{lon}/dist/{nm}` per
  airport once a minute (JSON with position, altitude, ground speed,
  track, vertical rate, squawk, emergency flag); same reduction as the
  archive, from a sliding window of positions per aircraft.
- **Shipped data** under `examples/airspace/testdata/`: the per-minute
  counters for 2025-01-20 to 2025-01-24 (training days and the storm),
  the METAR lines per station and minute for the same days, and for the
  storm day a down-sampled position track set for the map (one point per
  aircraft per 20 s inside the zones), a few MB gzipped. Attribution
  files name both sources and their licences; the raw archives are not
  redistributed.

### Models (as in the network workbook)

- **Forecaster:** one MLP for all airports, features the last 24 minutes
  of the counter (standardised per airport), time of day and day of week
  as sin/cos, a one-hot of the airport; two outputs, expected value and
  log spread, trained on movements per minute (arrivals + departures)
  and on aircraft in zone. Alarm when the measured value is outside
  three spreads for three consecutive minutes.
- **Autoencoder:** over the vector of all airports' counters for the
  minute (8 airports × 8 counters), threshold at the 99.5th percentile
  of training error; the top contributing counter names the airport.
- **METAR watch:** `logtemplate.Miner` over the reports, rates per
  template per station against the training band, and a template-mix
  shift per station (Hellinger distance of the last hour's template
  histogram to the training histogram). No first-seen alarms.
- **Events** (not alarms): emergency squawks, go-arounds, SPECI reports,
  wind gusts above 40 kt, visibility below 800 m, de-duplicated per
  aircraft and per report.

### Program

- `go run ./examples/airspace` replays the storm day at 10 simulated
  minutes per second after training on the four days before, serving
  the dashboard on 127.0.0.1:8080; `-live` switches to the live feeds
  (training then comes from the shipped history); `-headless` prints
  alarms and events for the replay and exits; `-prep` builds the
  testdata from a downloaded archive and Mesonet CSV (documented for
  reproducibility, not needed to run the example).
- Dashboard (embedded HTML, no external assets): a panel per airport
  with the movement sparkline and its band, a small radar view of the
  zone with aircraft dots and holding rings (local Cartesian projection,
  canvas), the autoencoder score timeline, the alarm feed, the event
  feed, and the METAR template table; state pushed by server-sent
  events, as in the network application.
- Alarm rules are the network ones: bands and sustained deviation, no
  fixed thresholds; alarms bundle per airport and hour in the feed.

### Alternatives considered

- Wikipedia edit stream (rejected as dull), the free German GTFS-RT
  stream (47 MB per 10 s, alert texts are boilerplate, opaque stop ids,
  no Berlin), weather stations alone (no event stream).
- A tiled map: needs external assets or a tile server; the local radar
  views show what matters (positions relative to the field) without.

## Acceptance

- Replay, headless: the storm is detected at EIDW, EGPH and EGCC by a
  model alarm in the storm morning (05:30 to 09:00 UTC: the observable
  effect begins when the morning wave should start and does not, or when
  approaches start failing; the first gusts over 45 kt came at 01:00 to
  03:00 when there is no traffic to miss), EDDF raises no forecast-band
  alarm that day, and the total number of model alarms on the storm day
  is under 40 across all airports (bundled per airport and hour, under
  15).
- Training days: fewer than 5 model alarms per day in a hold-out day.
- Unit tests for the trace reduction (arrival, departure, holding,
  go-around, emergency) on synthetic tracks, and for the METAR parser
  and the template-mix distance.
- `-live` runs against the real feeds for at least an hour without
  error on the M2 (manual check; the test suite never touches the
  network).
- Manual: `applications.md` gains the example with a screenshot-free
  walkthrough (what each panel shows, how the alarms arise, licences).

## Notes

Built 2026-09-08. What changed against the design while building:

- **Movements are counts, not levels.** The forecaster band on movements
  per minute failed twice: with an absolute spread floor it was too wide
  for Edinburgh's 0.3 movements a minute, and the autoregressive model
  followed a closed airport's zeros into predicting zeros, so Dublin was
  found at 08:00 instead of 06:00. The rule is now the workbook's own
  baseline, the training days' profile by minute of day, with a Poisson
  test over 30- and 60-minute windows (tail under 1e-4 for three minutes
  running). The forecaster with its band stays for aircraft in the zone,
  a level. Result: Dublin at 06:06 ("0 movements in 60 min, expected
  10"), Edinburgh at 06:53, both within the first hour of their missing
  morning wave.
- **Rare counts out of the autoencoder.** A single go-around in a column
  whose training variance was near zero gave standardised errors in the
  thousands. Go-arounds and emergencies now have their own rules (two
  aborted approaches in an hour where training had none; emergency
  squawks as events); the autoencoder sees the six zone-state counters
  with a standard-deviation floor of one, threshold at the 99.9th
  percentile with three minutes' persistence.
- **Belfast replaced by Edinburgh:** the archive's coverage at Belfast
  counted 40 to 60 movements a day and none on the storm day; Edinburgh
  counts 280 on an ordinary day and 49 on the storm day.
- **Acceptance windows from the METARs:** each station's storm window is
  two hours before its first gust of 40 kt to ninety minutes after the
  last, instead of a fixed offset from the first 45-kt gust (Dublin's
  came at 03:00 when there is no traffic to miss).

Results, storm day 2025-01-24, trained on 01-19 to 01-23: 27 alarms in
27 airport-hours (Dublin 06:06 to 11:00 hourly, Edinburgh 06:53 to
19:00, Manchester go-arounds 11:33 to 13:00, Dublin go-arounds 12:29 and
13:00, Paris anomaly score 14:51 and 15:00 on approach speeds); 68
events; no movements alarm at Frankfurt. Quiet day 2025-01-23 trained on
01-19 to 01-22: 7 alarms, all autoencoder scores on holding or final
counts, 6 events. The METAR mix-shift rule did not fire on the storm
day: the reports change one field at a time (wind group, then gusts,
then SPECI), so the template mix moves gradually; the templates table
shows the shift, the alarm rule would need a longer window to catch it.
Live mode ran against api.adsb.lol and aviationweather.gov for the
smoke test; a longer live run is the user's check. Prep takes about
5.5 minutes per archived day on the M2 Pro (JSON parsing of 50 000
traces); the shipped testdata is 4 MB.

Live mode, smoke-tested three times: per-airport polling of adsb.lol
hit its rate limit (429) after three requests, so the loop now makes
two regional requests of 250 NM (Irish Sea, Benelux) that cover all
eight zones and splits the aircraft locally; on a 429 it falls back to
airplanes.live, which serves the same readsb JSON. Four live minutes
stepped cleanly with 20 to 60 aircraft per eastern zone and the current
METARs of all stations.

