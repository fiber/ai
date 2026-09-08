# airspace

Eight European airports watched the way an operator watches a network:
aircraft movements are the counters, METAR reports the event stream,
fiber/ai's `nn`, `logtemplate`, `data` and `metrics` packages do the
watching. Trains on five ordinary days, replays Storm Éowyn (24 January
2025) with the models running, or follows the live feeds.

```
go run ./examples/airspace              # replay the storm day, dashboard on 127.0.0.1:8080
go run ./examples/airspace -headless    # print alarms and events instead
go run ./examples/airspace -live        # adsb.lol and aviationweather.gov, one poll a minute
```

Flags: `-speed` (replay minutes per second, default 10), `-addr`, `-data`
(a directory produced by `-prep` instead of the embedded files).

Files: `reduce.go` turns position reports into per-minute counters and
movement events (shared by archive and live), `metar.go` parses reports,
`models.go` holds the zone forecaster, the autoencoder and the METAR
template watch, `watch.go` the movement profile with its Poisson test,
the replay/live loop, alarms, events and the dashboard,
`prep.go` the reduction of the raw archives, `ui.html` the page.

The data under `testdata/` is derived from adsb.lol (ODbL / CC0) and
U.S. National Weather Service METARs via the Iowa Environmental Mesonet;
see `testdata/ATTRIBUTION.md`. The manual page
[applications.md](../../docs/manual/applications.md) walks through what
the panels show and how the alarms arise.
