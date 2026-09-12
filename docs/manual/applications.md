# Application packages

Building blocks above the tensor layer for the projects in the network
workbook. They are ordinary Go packages with no assembly; the heavy
lifting is a matrix product here and there.

## logtemplate: fifty million syslog lines a day

`logtemplate` turns a stream of log lines into templates and counts, so
that only templates, a few thousand a day instead of millions of lines,
need embedding, clustering or a person's attention.

```go
m := logtemplate.NewMiner()
for line := range lines {
    id := m.Add(line)          // template ID, dense and stable
}
for _, t := range m.Templates() {  // most frequent first
    fmt.Println(t.Count, t)        // "142999 <*> systemd[<NUM>]: Started session <NUM> of user <*>"
}
```

Two stages. `Mask` replaces the variable parts of a line by placeholders
in one pass over the bytes: IPv4/IPv6 addresses (`<IP>`), MAC addresses
(`<MAC>`), quoted strings within one token (`<STR>`), hexadecimal
identifiers of four or more characters mixing digits and letters or
starting with `0x` (`<HEX>`), and numbers including digit runs inside
words (`<NUM>`: `eth0` becomes `eth<NUM>`, `ge-0/0/3` becomes
`ge-<NUM>/<NUM>/<NUM>`). Words made only of hex letters (`deadbeef`)
stay, they cannot be told from English. Then the `Miner` runs the Drain
algorithm: a fixed-depth tree keyed by token count and leading tokens,
leaves holding templates; a line joins the template that agrees with it
in at least `Threshold` (0.5) of the positions, and disagreeing positions
become `<*>`.

Throughput on one M2 Pro core: about 1.2 million lines a second, so a
day of fifty million lines takes under a minute. A `Miner` is meant for
one goroutine; shard by source when you want more. `ResetCounts` rolls
the counters over without forgetting templates; `Template(id)` looks one
up.

## cluster: k-means and nearest centre over embeddings

`cluster` groups rows of an embedding matrix, such as the 768-dimensional
vectors EmbeddingGemma returns for each template. Rows are compared by
cosine similarity, which is a plain matrix product once rows have unit
length:

```go
x := cluster.Normalize(embeddings)          // [n×768], rows of length 1
res := cluster.KMeans(x, 40, 30, seed)      // 40 centres, at most 30 rounds
res.Assignment[i]                           // centre of row i
res.Similarity[i]                           // how well it fits
res.Sizes, res.Inertia()                    // rows per centre, mean distance

idx, sim := cluster.Nearest(newRows, res.Centres)   // for new templates
if sim[0] < 0.5 { /* fits no known cluster: show it to a person */ }
```

`KMeans` seeds with k-means++ and stops when no row changes centre; an
emptied centre restarts on the worst-fitting row. Each round is one
`MatMul` of the rows against the centres: 50 000 rows of 768 against 40
centres take about 10 ms on the M2 Pro.

`examples/syslog` puts both together on a generated day of logs and
prints the dashboard: clusters with their line counts and most frequent
template, and new lines judged against the day's clusters.

### Cosine similarity

Two forms, both on top of the SIMD kernels:

```go
c := cluster.Cosine(a, b)                 // one pair of []float32, no allocation
s := cluster.Similarities(queries, docs)  // [q×d] matrix through one GEMM
```

`Cosine` is for the odd comparison, a query against a handful of
candidates: one fused kernel pass over both vectors, 128 ns for 768
dimensions on the M2 Pro against 1 968 ns for NumPy's dot-over-norms.
`Similarities` is for search and expects unit-length rows (`Normalize`
once; EmbeddingGemma's vectors already are): one matrix product, then
`Argmax` or a sort per row. A thousand queries against ten thousand
documents take 7.4 ms, level with NumPy on Accelerate. `Nearest` is the
same product with the argmax done for you.

## data: from measurements to examples

The steps every project in the workbook takes before a model sees a
number, once per data set:

```go
rates := data.Rates(counter, time.Minute, 125e6)  // bytes/s from cumulative bytes, resets handled, gaps as NaN
x, y, pos := data.Windows(rates, 24, 15, func(t int) []float32 {
    return data.TimeFeatures(start.Add(time.Duration(t) * time.Minute))
})                                                // [n×28] inputs, [n×1] targets 15 minutes ahead
xTrain, yTrain, xVal, yVal := data.SplitByTime(x, y, 0.8)
s := data.Fit(xTrain)                             // keep s.Mean and s.Std with the model
xTrain, xVal = s.Transform(xTrain), s.Transform(xVal)
for idx := range data.Batches(xTrain.Dim(0), 256, r) {
    xb, yb := xTrain.Rows(idx), yTrain.Rows(idx)
    // train on the batch
}
```

`Rates` treats a negative delta and a rate above the link speed as a
counter reset (the new value counts from zero) and a NaN sample as a
gap; `Windows` skips positions whose window or target touches a gap and
returns the positions it used so results can be lined up with time
stamps again. `SplitByTime` returns views, `Batches` yields shuffled
index slices for `Tensor.Rows`.

## metrics: judging a classifier or a forecast

```go
m := metrics.Confusion(pred, truth, 6)
m.Labels = []string{"printer", "camera", "workstation", "server", "phone", "unknown"}
fmt.Println(m)               // the table with per-class recall and precision
m.Recall(1), m.Precision(1), m.F1(1)
metrics.MAE(pred, y), metrics.RMSE(pred, y)
```

The per-class numbers are the ones that matter when classes are uneven:
a model that always says "workstation" scores 85 % accuracy on a network
with 40 cameras and no recall on cameras at all. Pair the table with
`tensor.CrossEntropyWeighted(logits, targets, weights)` during training,
which multiplies each example's loss by its class's weight (for example
the inverse class frequency) and normalises by the batch's total weight
as PyTorch does.

## The airspace watch: all of it on public data

`examples/airspace` puts the packages above and the `nn` models
together on data anyone can fetch: aircraft positions and flight-weather
reports around eight European airports, Dublin to Frankfurt. It is the
public counterpart of a network watch: airports are the entities,
movements per minute the counters, METAR reports the event stream.

```
go run ./examples/airspace              # train on four ordinary days, replay Storm Éowyn
go run ./examples/airspace -headless    # the same, alarms and events printed, no browser
go run ./examples/airspace -live -contact you@example.com
```

Live mode needs `-contact`, and it is not decoration: the address goes
into the `User-Agent` of every request, so the operator of a free service
can see what the traffic is and reach whoever is causing it. Running it
anonymously is what got an earlier version of this example blocked. The
other flags of live mode are `-interval` (how often the feeds are polled,
two minutes by default) and `-source`, the endpoint to read. If you have
a receiver of your own, point `-source` at its `aircraft.json` and leave
the shared services out of it entirely:

```
go run ./examples/airspace -live -contact you@example.com \
    -source http://your-pi/data/aircraft.json
```

The counters stay on the one-minute grid the models were trained on
whatever the poll interval is: for the minutes between two polls the last
known picture is carried forward, so positions can be a poll interval
stale but the series has no holes. A source that answers 429 or 403 is
left alone for ten minutes, then twenty, up to an hour, and those minutes
are carried forward too.

**What it watches.** For every airport and minute: aircraft in a 40 NM
zone below 10 000 ft, landings, take-offs, aircraft in holding patterns,
aborted approaches, emergency squawks, the mean ground speed of low
traffic and the number of aircraft on final. These come from a reduction
of raw position reports (`reduce.go`) that the archive and the live feed
share, so replay and live counters mean the same thing. The METARs of
the eight stations are the text stream.

**What the models and rules do.** Movements per minute are counts, so
their expectation is the training days' profile by minute of day (the
"same minute on an ordinary day" rule the workbook insists on beating)
and the test is Poisson: over the last 30 and 60 minutes the observed
number of movements against the expected one, an alarm when the tail
probability is under one in ten thousand for three minutes running. That
gives a quiet airport with twenty movements an hour and a busy one with
ninety the same false-alarm rate, which no fixed band does. Aircraft in
the zone is a level, not a count: one forecaster with an expectation
band per airport (the recipe of tutorial chapter 6) predicts it fifteen
minutes ahead from the last 24 minutes, time of day and day of week; an
alarm needs eight minutes outside three spreads. An autoencoder over the
48-value vector of all airports' zone state (in zone, arrivals,
departures, holding, mean speed, on final) scores the minute as a whole,
threshold at the 99.9th percentile of training error, alarm after three
minutes over it or one at double. Aborted approaches have a rate rule:
two in an hour where the training days had none. And a
`logtemplate.Miner` over the METARs keeps a per-station rate per
template and a mix-shift measure (Hellinger distance of the last hour's
template histogram to the training histogram). There is no first-seen
alarm: a new report kind simply moves the mix like any other.

**Alarms and events are kept apart.** Alarms come from the models and
rules only: movements out of Poisson range, the zone level outside its
band, an autoencoder score over the threshold, a burst of aborted
approaches, a template far above its usual rate or a shifted mix. They
bundle per airport and hour: the first finding of an hour is reported,
later ones fold into it. Events are facts that need no model and are shown for
context: gusts over 40 kt, visibility under 800 m, special reports,
aborted approaches, emergency codes, de-duplicated per aircraft and
report.

**Reading the page.** The header says in one line what is abnormal
right now ("Dublin: 0 movements in 30 min, expected 14 · Edinburgh: 1
movements in 60 min, expected 17") or that all eight airports are as
expected. Each airport tile leads with a state, normal, watch or alarm,
and the sentence behind it, then the last three hours of movements: the
expected band in blue, the ten-minute mean of the actual in white, and
the excursions filled red, so the anomaly is the coloured area and not a
line crossing another. Beside it a 40 NM radar view of the zone, below
it the current weather decoded from the METAR (wind and gusts,
visibility, present weather, SPECI flagged). The incident timeline
draws one lane per airport, west to east, with alarm hours as bars and
events as ticks; on the storm day the two western lanes turn red from
six in the morning while Frankfurt's stays clear. The anomaly score
timeline, the alarm feed and the event feed complete the page; the
METAR template table of the first version is gone, because a METAR's
"template" is only the shape of the code with the numbers removed, and
the numbers are the news.

**The replay.** The shipped data covers 2025-01-19 to 2025-01-24. The
first five days train, the sixth is Storm Éowyn: record gusts over
Ireland and Scotland from the early morning, Dublin without a movement
until nine and Edinburgh with 49 for the day against 280, aborted
approaches at Manchester, the front moving east across the stations
through the day, Frankfurt untouched with 1 106 movements. (Belfast was
the first choice for the north, but the archive's receiver coverage
there is too thin to count movements; Edinburgh's is dense.)

**Data and licences.** Positions are derived from the adsb.lol history
archives (ODbL / CC0) and reduced to per-minute counters plus a
down-sampled track set for the storm day; METARs are U.S. National
Weather Service data via the Iowa Environmental Mesonet. Live mode reads
api.adsb.lol and aviationweather.gov, one request per region every two
minutes, identified by the address you pass in `-contact`. `testdata/ATTRIBUTION.md` has the
details; `-prep` rebuilds the files from the raw archives.

## MNIST: the benchmark everybody knows

`examples/tutorial/12-mnist` trains a classifier on handwritten digits, the one
dataset in this field whose numbers a reader can check against their own
experience. It is the program of [tutorial chapter 12](../tutorial/12-mnist.md)
and the only example that trains on data collected by somebody else.

```
go run ./examples/tutorial/12-mnist                  # both models, five epochs
go run ./examples/tutorial/12-mnist -model cnn -epochs 10
go run ./examples/tutorial/12-mnist -limit 1000      # how each model copes with scarce data
```

The data comes from `github.com/fiber/ai-data`, a separate module that
embeds the four original idx files and verifies them by checksum. Nothing
is downloaded at run time, and because the module is imported only by
this example, anyone who imports `tensor` never fetches those 11 MB.

Two models on the same input, both from `nn`: a 784-256-10 perceptron,
and a convolutional network of two 3×3 stages with max pooling. On an
Apple M2 Pro, five epochs of batch 128 with Adam at 1e-3:

| | parameters | test accuracy | training |
|---|---:|---:|---:|
| MLP 784-256-10 | 203 530 | 97.82% | 1.0 s |
| CNN 16-32 filters | 20 490 | 98.85% | 35.6 s |

The convolutional model is ten times smaller and better, because the
filters are shared across every position instead of learned once per
pixel — and it is thirty-five times slower to train, because it does far
more arithmetic per parameter. The example also prints the worst pairs of
the confusion matrix; after B-009 the remaining 115 errors are spread
almost evenly, no pair reaching eight.

`benchmarks/python/mnist.py` is the same two models in PyTorch on the
same data and optimiser, for anyone who wants to check the figures. On
the six performance cores of an M2 Pro: 2.0 s and 51.6 s there against
1.0 s and 35.6 s here, with test accuracy 97.88% and a mean 98.70% over
seeds 12 to 15 against 97.82% and 98.66%. The accuracies are the same
number — the ranges overlap — and with `--init he`, which gives the
PyTorch model our initialisation, its mean becomes 98.67%. The remaining
difference between the two libraries on this task is the default
initialisation, not the convolution.

Pixels are scaled by one mean and one standard deviation for the whole
set, not per pixel. Border pixels are zero in every image, so a per-pixel
deviation is zero there and the division produces NaN — the kind of thing
real data does and generated data never does.
