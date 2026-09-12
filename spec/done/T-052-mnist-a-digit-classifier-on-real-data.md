---
id: T-052
title: MNIST: a digit classifier on real data
status: done
scope:
  - examples/mnist/
  - benchmarks/python/
  - docs/tutorial/
  - go.mod
  - go.sum
manual:
  - docs/manual/applications.md
done: 2026-09-12
created: 2026-09-12
---

## Goal

Eleven tutorial chapters and not one of them touches real data. Every
dataset is generated in the example that uses it: spirals, sine waves,
synthetic counters. That is right for teaching a gradient, and wrong as
the last impression the tutorial leaves, because the first thing a reader
does afterwards is load a file from disk and discover the tutorial never
showed them how.

It is also the missing piece of evidence. Chapter 5 opens with "which of
ten digits" as its motivating example and then classifies spiral arms. A
Go developer weighing this library against PyTorch wants one number they
already have a feel for, and MNIST is the only such number in this field:
everybody knows roughly what an MLP and a small CNN should reach, and
roughly how long it should take. Without it the framework's public
evidence is GEMM throughput, which is meaningful to a few hundred people
in the world.

So: a twelfth chapter that trains on the real MNIST files, reports
accuracy and wall time for an MLP and a small convolutional network, and
shows the reader what handling real data looks like.

## Design

**Where the data lives.** In a separate module, github.com/fiber/ai-data,
already published. The four original idx files are embedded there and
verified by checksum in that module's tests, so the example works offline
on the first run and nothing is downloaded from a third party when many
people run it at once — the lesson of T-051, applied before it costs us
anything. Because ai-data is only imported by this example, someone who
imports github.com/fiber/ai/tensor never fetches those 11 MB: the Go
tool downloads module zips for what it needs to build.

Alternatives rejected: committing the data here (2.6x the repository, paid
by every consumer for ever); downloading on first run (a public example
pointed at somebody else's bandwidth, which is exactly how T-051 started);
a committed subset (still bytes in the module zip, and a subset cannot
produce the headline number the site needs).

**The example.** `examples/mnist`, one `main.go`:

- `-model mlp|cnn|both` (default both), `-epochs`, `-batch`, `-limit` to
  train on a prefix of the training set, `-seed`.
- MLP: 784-256-10 with ReLU, the same shape as chapter 5 scaled up, so the
  reader recognises it.
- CNN: two 3x3 convolution layers with max pooling between them, then a
  linear head; `nn.Conv2D`, `nn.MaxPool2D` and `nn.Flatten` already exist
  and are otherwise unexercised by any example.
- Both report accuracy on the 10000 test images and wall time per epoch,
  so the chapter's table is the program's own output rather than prose.
- Input is `Float32` from ai-data, standardised by the training mean and
  standard deviation; the CNN reshapes to [batch, 1, 28, 28].

**The chapter.** `docs/tutorial/12-mnist.md`, appended rather than
inserted after chapter 8. Inserting means renumbering three chapters,
their example directories and every reference to them, which buys the
reader nothing. Chapter 8 gets one cross-reference forward. The chapter
covers: what the idx format is and why a data module rather than a
download; the MLP as the baseline; what the convolution buys and what it
costs in time; and where the remaining error is, with a look at the
confusion between 4 and 9.

## Acceptance

- `go test ./examples/mnist/` trains an MLP on a small prefix within a few
  seconds and asserts an accuracy floor, so the example cannot rot
  silently.
- `go run ./examples/mnist` on the full set reaches at least 97.5% test
  accuracy with the MLP and at least 98.5% with the CNN.
- A PyTorch counterpart of the example, `benchmarks/python/mnist.py`,
  trains the same two architectures on the same data with the same
  optimiser, batch size and epoch count, so the chapter's comparison is
  reproducible rather than asserted. Both sides are run with the same
  number of threads and over several seeds before any claim is made.
- `go vet ./...` and `GOARCH=amd64 go vet ./...` clean, `go test ./...`
  passes.
- The chapter's numbers are the ones the program prints on this machine,
  with the machine named.
- docs/manual/applications.md gains the example.

## Notes
The example found B-009 before it found anything about digits. Comparing
against PyTorch produced a gap of 0.3 points that survived every
explanation offered for it — initialisation, seed variance, max-pool tie
breaking — and turned out to be `Conv2D` releasing the im2col buffer its
own backward still read. Every figure written here is from after that
fix; the ones measured before it were produced with corrupted gradients.

That is the argument for the chapter existing, beyond the tutorial. Eleven
chapters of synthetic data never exercised the library hard enough to
find a use-after-free in the convolution backward, because nothing there
is deep enough for the allocator to recycle a live buffer, and no
synthetic benchmark has an external reference to disagree with. A number
everybody knows is a test, not decoration.

Measured on an Apple M2 Pro, six performance cores, five epochs of batch
128 with Adam at 1e-3:

| | parameters | accuracy | training |
|---|---:|---:|---:|
| MLP 784-256-10 | 203530 | 97.82% | 1.0 s |
| CNN 16-32 filters | 20490 | 98.85% | 35.6 s |
| PyTorch 2.8, same MLP | 203530 | 97.88% | 2.0 s |
| PyTorch 2.8, same CNN | 20490 | 98.72% | 51.6 s |

Over seeds 12, 13 and 14 the convolutional models give 98.85 / 98.64 /
98.44 (mean 98.64) against PyTorch's 98.72 / 98.75 / 98.72 (mean 98.73).
PyTorch is steadier across seeds and a tenth of a point ahead on the
mean; we are 1.45× faster on the CNN and twice as fast on the MLP.

Two claims in the chapter's closing section were checked rather than
assumed, and one of them was wrong when written: at `-limit 1000
-epochs 5` the CNN scores *below* the MLP (86.5 against 87.7) because
forty optimiser steps do not start it. The order reverses at 40 epochs,
89.2 against 93.4. The chapter now says so, which makes a better point
than the original claim did.
