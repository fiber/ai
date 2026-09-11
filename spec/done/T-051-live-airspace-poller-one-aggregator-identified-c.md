---
id: T-051
title: Live airspace poller: one aggregator, identified client, gentler cadence
status: done
scope:
  - examples/airspace/
manual:
  - docs/manual/applications.md
done: 2026-09-11
created: 2026-09-11
---

## Goal

The live mode of the airspace example was an impolite network client, and
one of the two public aggregators it used has blocked it. Running it for
three days produced roughly 9400 requests to api.adsb.lol, two every
minute, sent with Go's default `User-Agent` and no way for either operator
to find out who was calling. When adsb.lol answered 429 the code
immediately re-asked api.airplanes.live for the same data; that service
now answers 403 with a request to contact its operators.

Anyone who runs `go run ./examples/airspace -live` from the repository
inherits that behaviour, so the example has to become a client the
operators of a free service would not mind having. It keeps working
against adsb.lol alone, identifies itself, asks less often, and retreats
when it is told to.

The second half of the problem is multiplication. One anonymous client a
minute drew a block; a thousand readers of the tutorial running the same
example would be an outage. So live mode stops being something that can
be run anonymously at all: it refuses to start without a contact address,
and it can be pointed at the reader's own receiver instead of a shared
aggregator. The default path through the example stays the bundled
replay, which needs no network.

## Design

`examples/airspace/live.go` only.

**One aggregator.** The `airplanesLive` constant and the fallback branch
go away. adsb.lol is the single source; its history archive is what the
replay and the trained models were built from, so this is also the source
that matches them.

**An identified client.** `fetchRegion` and `fetchLiveMETARs` send a
`User-Agent` naming the project, its repository and a contact address, so
an operator seeing the traffic can tell what it is and reach whoever runs
it. `userAgentFmt` holds the shape and `-contact` fills in the address.

**No anonymous live mode.** `-live` without `-contact` exits with a
message: name yourself, read the source's terms, or feed from your own
receiver. There is no default address, so the repository cannot be cloned
and pointed at a free service by a thousand people who never thought
about it.

**Your own receiver.** `-source` takes the URL of any endpoint serving
readsb JSON. With `%.4f/%.4f/%d` verbs it is filled in per region as
adsb.lol is; without them it is fetched as it stands, which is what a
local tar1090 or readsb does (`http://pi/data/aircraft.json`). The
decoder accepts both the `ac` key the aggregators use and the `aircraft`
key readsb writes, and a plain URL is fetched once per poll rather than
once per region.

**A gentler cadence.** A `-interval` flag (default 2 minutes) sets how
often the feeds are polled, halving the request rate. The watch keeps
stepping once a minute, because the counters, the profile and the Poisson
alarms are all defined on the one-minute grid the archive was reduced on;
stepping every second minute would leave zero-filled minutes in the
trailing windows and raise false "quieter than expected" alarms.

For the minutes in between, the last fetch is carried forward: each
aircraft's newest sample is appended again with its timestamp advanced by
one minute. `reduceTrack` then sees an unbroken one-minute track and
presence, in-zone level and ground contacts stay continuous. Positions
are up to `-interval` minutes stale on carried minutes, which is
documented in the example's README and in the manual. Movement events are
unaffected: they are deduplicated per aircraft and kind by the existing
`counted` map.

**Backing off.** A 429 or 403 from a region no longer triggers a second
request anywhere. The region is skipped and not polled again for
`backoffMinutes` (10), doubling up to an hour while the refusals
continue and resetting on the first success. The message logged names the
wait, so the operator of the copy can see the client is holding off.

Rejected: keeping a second aggregator behind a flag (the point is to stop
spreading the same load over two free services); polling every minute with
only the fallback removed (does not reduce the load that drew the 429s);
stepping the watch at the poll interval (breaks the one-minute grid, as
above).

## Acceptance

- `go vet ./...`, `GOARCH=amd64 go vet ./...` and `go test ./...` pass.
- No occurrence of `airplanes.live` remains under `examples/`.
- `go run ./examples/airspace -live -interval 2 -contact you@example.com`
  polls api.adsb.lol twice per two minutes (once per region), not four
  times, and every request carries a `User-Agent` naming the repository
  and that address.
- `go run ./examples/airspace -live` without `-contact` exits with a
  message naming the three ways forward and does not open a connection.
- `-source http://host/data/aircraft.json` is fetched once per poll with
  no formatting applied, and its `aircraft` key is decoded like `ac`:
  `TestDecodeReadsb` asserts both keys yield the same samples.
- With the feed carried forward, a minute with no fetch still produces
  non-zero in-zone counters for airports that had aircraft in the
  previous minute: `TestCarryForward` builds a window from one fetch,
  carries it, and asserts the in-zone counter of the carried minute
  equals that of the fetched minute.
- A region that answers 429 is not re-requested in the following minute;
  `TestBackoff` drives the backoff state machine and asserts the wait
  doubles from 10 minutes to at most 60 and resets after a success.
- `docs/manual/applications.md` and `examples/airspace/README.md` state
  the cadence, the carry-forward, the single source, the mandatory
  contact address and the local-receiver option; the manual line
  promising "one poll per minute" is corrected.

## Notes

Opened after api.airplanes.live began answering 403 with "Please contact
us at contact@airplanes.live", on 2026-09-11, while the example had been
running live for three days.
