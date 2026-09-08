---
id: T-035
title: Airspace dashboard: anomalies legible at a glance (state and reason per airport, marked excursions, incident timeline, decoded weather)
status: done
scope:
  - examples/airspace/
manual:
  - docs/manual/applications.md
done: 2026-09-08
created: 2026-09-08
---

## Goal

The first dashboard showed curves, dots and lists; a viewer could not
tell where an anomaly was or why. Make the page answer, at a glance:
which airports are abnormal right now, what exactly deviates from what
was expected, and how the day unfolded across all airports.

## Design

- **State and reason first.** Each airport tile leads with a state pill
  (normal / watch / alarm) and one sentence: the latest alarm text of
  the last hour, or the current deviation while a run is building
  ("movements 3 of expected 12 in the last 30 min"), or "as expected".
  The page header carries the situation in one line: the airports in
  alarm with their reasons, or "all eight airports as expected".
- **Excursions marked in the chart.** The movement chart draws the
  expected band and the smoothed actual; where the actual leaves the
  band the gap is filled red, so the anomaly is the coloured area, not
  a subtle crossing of lines. The in-zone level gets the same treatment
  in a second, thinner trace.
- **Decoded weather instead of templates.** The METAR watch keeps its
  rule, but the tile shows what the report says: wind and gusts in knots
  with a direction arrow, visibility, present weather in words, SPECI
  flagged. The template table is dropped from the page (the rule and the
  miner stay in the code).
- **Incident timeline.** One lane per airport over the day, west to
  east; alarm hours as bars coloured by kind, events as ticks (gusts,
  visibility, aborted approaches, emergencies), the current time as a
  cursor. This is the picture of the front crossing the stations.
- **Feeds stay,** alarms and events, newest first, with the airport
  name; the training log collapses behind a details element.
- `snapshot` gains per-airport `state`, `reason` and decoded `weather`,
  and the whole day's alarms and events (a few hundred small records)
  for the timeline.

## Acceptance

- On the storm replay at 08:00, the header names Dublin and Edinburgh
  with their reasons, both tiles are in the alarm state with the missing
  movements shaded red, Frankfurt is "as expected"; the timeline shows
  the two western lanes red from 06:00 and the others clear.
- Live mode shows all eight tiles "as expected" on an ordinary day with
  decoded weather present.
- `go test ./examples/airspace` passes unchanged; the manual section is
  updated to describe the page as it is.

## Notes

Built 2026-09-08. Verified on the replay at 08:31: header "Dublin: 0
movements in 30 min, expected 14 · Edinburgh: 1 movements in 60 min,
expected 17", both tiles in alarm, the six others normal with their
"as expected: n of m" sentences, decoded weather on all tiles (Edinburgh
32 kt gusting 48, light rain, 4 700 m), timeline and feeds populated.

