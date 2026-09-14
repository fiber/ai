# 18. A text classifier on frozen embeddings

Chapter 10 ended with a search: a question in plain words, the nearest
templates by cosine. A service usually needs something blunter than a
ranking — *which team gets this line?* — and the way that is built in
practice is neither a language model nor a fine-tuned encoder. The
encoder stays frozen and turns every line into a vector once. A layer
of a few thousand numbers, trained in a fraction of a second, turns the
vector into a decision. This chapter builds that and measures it
against the obvious alternative on lines whose wording neither has seen.

```
go run ./examples/tutorial/18-text-classifier
```

It needs the EmbeddingGemma weights like chapter 10, and takes about
fifteen seconds once they are in place. Without them it says what is
missing and stops.

## Lines, in shapes

The data is generated, and the chapter is only as honest as the way it
is generated. Five categories a syslog line can be routed to — access,
network, storage, jobs, hardware — each written in eight *shapes*, with
the pids, addresses, users and counts filled in at random:

```
5 categories, 8 shapes each, 2 held out of training
training 1200 lines, test on seen wordings 300 lines, on held-out wordings 400 lines, foreign 160 lines

  access    seen      sshd[34629]: Accepted publickey for root from 10.68.26.243 port 44214 ssh2
            held out  vsftpd[21253]: CONNECT: Client 10.191.182.239, anonymous login refused
  network   seen      kernel: eth94: link is up, 1000 Mbps full duplex
            held out  firewall: DROP IN=eth6 SRC=10.111.104.66 DST=10.127.172.212 PROTO=TCP SPT=10408 DPT=39170
  storage   seen      kernel: EXT4-fs (sdc74): remounting filesystem read-only after error
            held out  rsyslogd: no space left on device writing /var/log/ledger.log
  jobs      seen      CRON[31338]: (root) CMD (/usr/local/bin/backup --job nightly)
            held out  puppet-agent[33100]: Applied catalog in 14.46 seconds (search run)
  hardware  seen      ipmi: fan 70 speed 27 RPM below threshold on chassis 75
            held out  kernel: thermal thermal_zone47: critical temperature 95 C reached
```

The split is by shape, not by line. Two shapes of every category are
held out entirely: not one line of `rsyslogd: no space left on device`
is in the training set. Fresh lines of the six seen shapes are the
easy test — same wording, different numbers — and the held-out shapes
are the real one, because a real feed brings wordings nobody labelled.
Splitting by line instead of by shape would have let both classifiers
score 100 % and told you nothing, which is chapter 7's warning about
time series in a different costume.

## Two classifiers, one training loop

The first is the alternative anyone would try first. A *bag of words*:
one column per distinct word in the training lines, 1 if the line
contains it, and a `nn.NewLinear(200, 5)` on top. Words containing
digits are dropped, so pids and addresses do not become one-off
features. The vocabulary is built from the training lines and nothing
else, so a word that only occurs in a held-out shape has no column at
all.

The second is the chapter's subject: `gemma.Embed` on every line, and
`nn.NewLinear(768, 5)` on the vectors. Both heads train in the same
function — cross-entropy, AdamW, 50 epochs in batches of 64, the loop
of chapter 7 — and it takes a tenth of a second either way:

```go
head := nn.NewLinear(x.Dim(1), len(categories))
opt := optim.NewAdamW(head.Params(), 1e-2, 0.01)
for range epochs {
    for idx := range data.Batches(len(labels), 64, r) {
        loss := tensor.CrossEntropy(head.Forward(x.Rows(idx)), labelsOf(idx))
        opt.ZeroGrad(); loss.Backward(); opt.Step()
    }
}
```

```
bag of words: 200 distinct words in the training lines
  head trained in 0.1s
  accuracy on seen wordings 100.0%, on held-out wordings 52.0%

EmbeddingGemma loaded in 1.046s
  1200 lines embedded in 6.3s: 190 lines/s, 768 numbers each
  head of 3845 numbers trained in 0.1s
  accuracy on seen wordings 100.0%, on held-out wordings 94.5%

nearest centroid, no training at all: seen 99.7%, held out 92.5%
```

On the seen wordings both are perfect, and that number is worthless:
it says the heads can tell six shapes apart when given the shape's own
words. On the held-out wordings the bag of words falls to 52 % and the
embedding head holds at 94.5 %. The confusion matrices say how each
fails:

```
held-out wordings, bag of words:
truth \ pred   access  network  storage     jobs hardware   recall
   access       40       40        0        0        0    50.0%
  network       40       40        0        0        0    50.0%
  storage        0       40        0       40        0     0.0%
     jobs        0       32        0       48        0    60.0%
 hardware        0        0        0        0       80   100.0%
precision    50.0%    26.3%     NaN%    54.5%   100.0%    52.0%

held-out wordings, embedding head:
truth \ pred   access  network  storage     jobs hardware   recall
   access       80        0        0        0        0   100.0%
  network        0       80        0        0        0   100.0%
  storage        0        0       72        8        0    90.0%
     jobs        0        0        0       80        0   100.0%
 hardware        0       14        0        0       66    82.5%
precision   100.0%    85.1%   100.0%    90.9%   100.0%    94.5%
  wrong, one line per kind of mistake:
    jobs      for storage   0.58  rsyslogd: no space left on device writing /var/log/billing.log
    network   for hardware  0.43  snmpd[38326]: chassis 34 intrusion sensor asserted on edge-7
```

The bag of words is not guessing at random; it is doing exactly what
it can. `account locked after bad attempts from` has `from`, which
three sshd shapes use and one network shape, and goes to access. `vsftpd: CONNECT: Client
... anonymous login refused` has `client`, which the DNS shape owns,
and goes to network. `no space left on device writing /var/log/x.log`
has `device` from a storage shape and `var`, `log` from logrotate, and
jobs wins. `thin pool is 72% full` goes to network on the strength of
`full duplex`. Hardware scores 100 % because its held-out shapes happen
to reuse `chassis` and `temperature`, words the seen shapes also have.
A bag of words classifies vocabulary; when the vocabulary changes, it
has whatever the overlap happens to be.

The embedding head's two mistakes are the kind a person could make.
`snmpd` is a network daemon reporting a chassis sensor, and it is put
under network at 0.43 confidence — the head is telling you it is
unsure. `no space left on device writing /var/log/billing.log` goes to
jobs at 0.58, which is wrong and not absurd: the line is about a log
file. Both are on the boundary between categories, and the head's
confidence says so.

*Nearest centroid* — the mean vector of each category, nearest by
cosine, no training at all — is two points behind the trained head on
the held-out shapes. That is worth knowing: for five well-separated
categories, the encoder has done nearly all the work, and the head is
polishing. Thirty categories that overlap is where the head should
earn its keep, and that is a measurement to make before believing it.

## Lines from nowhere

A classifier with five outputs answers every question with one of the
five. What happens when the line is about none of them? A sixth kind,
application errors, that neither classifier ever saw a line of:

```
foreign lines (application errors), embedding head:
  storage   0.34  orders[21002]: NullPointerException in OrderService.submit at line 1
  jobs      0.67  api[37899]: HTTP 500 on POST /checkout/nightly after 11 ms
  jobs      0.42  db-pool[14766]: connection pool exhausted, 29 waiters, timeout 94 ms
  storage   0.49  postgres[22934]: deadlock detected, process 10656 waits for ShareLock on transaction 37471
  jobs      0.90  cache[9009]: miss rate 66% over the last 6 s on weekly-full
  storage   0.33  ingest[47882]: JSON parse error at offset 67 in message from web-3
  jobs      0.38  grpc[13432]: rpc error: code = DeadlineExceeded desc = search.Get after 97 ms
  access    0.45  web[34967]: unhandled promise rejection in auth handler: TypeError
```

Every one gets a team, because that is all the head can do. The only
signal it has is how sure it is, the softmax maximum, and that is
worth looking at as a distribution rather than a line at a time:

```
confidence               median    10th pct   worst
  seen wordings          0.998     0.995     0.988
  held-out wordings      0.878     0.510     0.399
  foreign lines          0.472     0.330     0.290

threshold   sent to a human: seen   held out    foreign lines caught
  0.90                         0.0%    57.0%     99.4%
  0.95                         0.0%    76.8%    100.0%
  0.98                         0.0%    88.8%    100.0%
  0.99                         0.7%    99.8%    100.0%
```

This is chapter 12 again, with a confidence instead of a reconstruction
error. Against the *seen* wordings the answer looks free: a threshold
of 0.90 catches 99 % of the foreign lines and sends nothing to a human.
Against the held-out wordings — the ones a real feed brings — the same
threshold sends 57 % of correctly classifiable lines to a person to
catch the foreign ones, and there is no setting in the table that
catches the foreign lines without doing that. The `cache miss rate`
line is routed to jobs at 0.90 confidence and would pass any threshold
below it.

So "none of the above" is not something a classifier has; it is
something you build, and it costs. The cheap version is the threshold,
priced as above. The better version is a sixth category with labelled
examples of the things that are none of the five — which is to say, the
fix is again labelled data rather than a better model.

## What this costs to run

Embedding is the whole cost: 190 lines a second on a laptop, one
forward pass of a 300 M-parameter encoder per line, and chapter 10's
manual page says how batching and threads move that number. The head is
3 845 float32s and retrains from scratch in a tenth of a second, which
changes how you work: every time somebody corrects a label, retrain.
The vectors are kept; only the head moves. Chapter 8 is the service
this goes into — the encoder loaded once, the head reloaded on a
signal.

## What to remember

- Frozen encoder, small trained head: embed once, decide cheaply,
  retrain in seconds. This is the pattern for text in a Go service.
- Test on wordings the head never saw. Splitting by line hides
  everything.
- A bag of words classifies vocabulary. It is a fair baseline and it
  is what you are replacing.
- A classifier has no "none of the above" unless you build it, and the
  threshold version costs on exactly the lines you wanted it for.

## What to try

Move a shape from held out to seen and watch the bag of words recover
on that category — that is what labelling one more wording buys it, and
what the embedding head already had. Then write two shapes of your own
in a wording deliberately unlike the others, and see which classifier
notices. And add `application` as a sixth category with the foreign
shapes: the confidence table becomes uninteresting, which is the point.
