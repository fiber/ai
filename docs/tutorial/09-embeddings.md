# 9. Embeddings: text as geometry

Everything so far took numbers in. Logs, tickets, alerts and
documentation are text, and a model that only eats numbers cannot read
them. An embedding model is the bridge: it takes a sentence and returns
a vector, in a way that puts sentences with similar meaning at nearby
points. This chapter trains nothing. It loads a pretrained encoder,
EmbeddingGemma, and looks at what its vectors do.

```
go run ./examples/tutorial/09-embeddings
```

The program needs the model's weights once, about 1.2 GB; the models
page of the manual says where to get them and how to point
`FIBERAI_MODELS` at the directory. Without them the program says so and
stops.

```
EmbeddingGemma loaded in 941ms: 24 layers, vectors of 768 numbers
16 templates embedded in 107ms
```

The sixteen inputs are syslog templates as an operator would store them:
an interface going down, a BGP neighbour coming up, a failed login, a
full disk, a finished backup.

## A vector is 768 numbers

```
"interface GigabitEthernet0/1 changed state to down"
  starts -0.127 0.030 0.083 0.023 0.024 ... (768 numbers, length 1)
```

None of the 768 means anything on its own. The model was trained so
that the *direction* of the vector carries the meaning, and every
vector has length 1, so the only thing left to compare is the angle
between two of them. The ruler for that is the cosine of the angle:
1 for the same direction, 0 for unrelated, computed by
`cluster.Cosine` in a hundred nanoseconds.

## What the ruler says

Every pair of the sixteen templates, sorted:

```
closest pairs:
  0.929  %BGP-5-ADJCHANGE: neighbor 10.0.0.1 Down
         %BGP-5-ADJCHANGE: neighbor 10.0.0.1 Up
  0.896  interface GigabitEthernet0/1 changed state to down
         interface GigabitEthernet0/1 changed state to up
  0.716  %BGP-5-ADJCHANGE: neighbor 10.0.0.1 Down
         OSPF neighbor 10.0.0.3 on Vlan10 went from FULL to DOWN
farthest pairs:
  0.239  accepted password for user backup from 10.0.0.9
         fan speed sensor reading abnormal on chassis 1
  0.221  fan speed sensor reading abnormal on chassis 1
         backup job nightly-full completed successfully
```

The two closest pairs are the surprise worth remembering: an interface
going *down* and the same interface coming *up* are nearer to each
other than to anything else in the list. Embeddings encode topic, not
polarity. "Something about that interface's state" is what the vector
says; whether the news is good or bad is a small component of it.

```
interface down vs interface up: 0.896
interface down vs BGP neighbor down: 0.556
interface down vs backup completed: 0.271
```

Two consequences for anyone building on this. Clustering log
templates by embedding groups them by subject, which is usually what
you want for a dashboard and never what you want for an alarm. And a
search for "things going down" will return things coming up as well;
if the difference matters, it has to be modelled on top, with a
classifier as in chapter 5, trained on the vectors.

## Search in plain words

A question is embedded with the "query" prompt (the model was trained
with distinct prompts for questions and documents) and compared with
every template:

```
"a network link went down"
  0.523  interface GigabitEthernet0/1 changed state to down
  0.432  OSPF neighbor 10.0.0.3 on Vlan10 went from FULL to DOWN
  0.400  %BGP-5-ADJCHANGE: neighbor 10.0.0.1 Down

"someone failed to log in"
  0.477  authentication failure for user admin from 203.0.113.7
  0.382  session opened for user root by cron
  0.374  power supply 2 failed on switch access-7

"the device is running out of resources"
  0.380  CPU utilization exceeded 90 percent on router core-1
  0.366  memory usage at 95 percent on firewall fw-2
  0.356  power supply 2 failed on switch access-7
```

No word of the first query appears in its top answer. That is the
whole reason to use an embedding rather than a text index: the query
says *link*, the template says *interface*, and the vectors know they
are the same thing. Note the absolute numbers, though: 0.52 for a good
match, 0.37 for a poor one. Similarities between a question and a
document are lower than between two documents, and the threshold that
separates a hit from a miss has to be found on your own data, not
assumed.

## Scale

Sixteen templates take a tenth of a second. A day of logs is millions
of lines, and embedding each line would take hours; embedding the few
thousand templates that `logtemplate` extracts from them takes seconds,
once. That is the shape of every real system built on this: reduce
first, embed what is left, store the vectors, and compare with the
ruler. The `cluster` package does the comparing for thousands of
vectors at once (`Similarities`), and chapter 10 is about a model that
does not need labels either.
