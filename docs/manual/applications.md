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
