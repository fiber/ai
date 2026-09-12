# Benchmarks

The M2 Pro figures were re-measured on 2026-09-12 against the current
code, both sides in one session; the sections for other machines keep the
dates in their headings. Machine and versions:

| | |
|---|---|
| Machine | Apple M2 Pro (6 performance + 4 efficiency cores), macOS 26 (Darwin 25.6) |
| fiber/ai | Go 1.26.2, `darwin/arm64`; GEMM on the AMX back-end, everything else NEON |
| fiber/ai (generic) | same code with `FIBERAI_KERNEL=generic` — pure Go, no assembly |
| NumPy | 2.0.2 (Accelerate BLAS), Python 3.9.6 |
| PyTorch | 2.8.0 CPU (Accelerate BLAS, SLEEF vector math, 6 threads by default) |
| Worker count | 6, the performance cores — the default on Apple Silicon since T-047, and what PyTorch also chooses here |

Method: every case is warmed up once, then repeated for ≥ 0.7 s and the mean
time per call is reported. Each call allocates a fresh output, as it would
in user code. GB/s counts the bytes an ideal implementation must move
(inputs read + output written). GFLOPS = 2·m·n·k / time.

Reproduce:

```sh
go run ./cmd/bench > benchmarks/results/go.md
FIBERAI_KERNEL=generic go run ./cmd/bench -quick > benchmarks/results/go-generic.md
cd benchmarks/python && python3 -m venv .venv && .venv/bin/pip install numpy torch
.venv/bin/python bench.py > ../results/python.md
```

Raw output: [results/go.md](benchmarks/results/go.md),
[results/go-generic.md](benchmarks/results/go-generic.md),
[results/python.md](benchmarks/results/python.md).

## Headline

Two machines, both sides measured the same day with the same shapes.
Apple M2 Pro: `go run ./cmd/bench` (GEMM on AMX, the rest NEON, results
released) against NumPy 2.0.2 and PyTorch 2.8.0 on Accelerate,
12 September 2026, both runs in `benchmarks/results/m2pro-2026-09-12/`.
Xeon Gold 6130, one socket pinned with `numactl`: fiber/ai AVX-512 of
8 September, PyTorch 2.14+cpu with MKL and oneDNN (16 threads) from the
same session for attention, convolution and EmbeddingGemma and from the
x86 round for the rest. Bold marks the faster side; "level" is within
5 %.

| Workload | M2 Pro fiber/ai | M2 Pro PyTorch | Xeon fiber/ai | Xeon PyTorch |
|---|---:|---:|---:|---:|
| SGEMM 2048², all cores (GFLOPS) | 2 225 (level) | 2 192 | **1 268** | 780 |
| SGEMM 1024² | 2 118 (2 465 with B packed once) | **2 688** | 1 234 (1 327 with B packed once) | **1 434** |
| SGEMM 512² | 1 505 | **2 157** | **1 002** | 991 (level) |
| SGEMM 256² | 975 | **1 121** | 540 | **627** |
| SGEMM 128² | 553 | **782** | 99 | **236** |
| [256×768]·[768×3072], B packed per call | 1 705 | **2 328** | **1 111** | 980 |
| [256×768]·[768×3072], B packed once (weights) | **2 507** | 2 328 | **1 331** | 980 |
| [1×4096]·[4096×4096] | **14.2** | 11.2 | **13.0** | 11.0 |
| x + y, 1M (released) | **37 µs** | 68 µs | **17 µs** | 24 µs |
| x + y, 16M (released) | 1.37 ms (level) | 1.39 ms | **13.9 ms** | 17.5 ms |
| exp, 16M | **1.60 ms** | 3.10 ms | **10.7 ms** | 13.8 ms |
| tanh, 1M (released) | **90 µs** | 713 µs | 54 µs (level; 484 unreleased, cold buffer) | 54 µs |
| sum(), 4096² | 584 µs (level) | 593 µs | 2.40 ms (level) | 2.42 ms |
| sum(dim=0), 4096² | **638 µs** | 2.23 ms | **2.68 ms** | 6.29 ms |
| max(dim=1), 4096² | **537 µs** | 1.22 ms | **2.40 ms** | 2.46 ms |
| softmax(dim=1), 4096² | **2.32 ms** | 4.60 ms | **11.0 ms** | 14.4 ms |
| layernorm, 4096² | **1.81 ms** | 2.37 ms | **11.0 ms** | 14.1 ms |
| transpose + copy, 4096² | **12.6 ms** | 23.2 ms | **25.5 ms** | 68.5 ms |
| attention [8×8×512×64] (GFLOPS) | **1 005** | 590 | 763 | **1 141** |
| conv2d [32×64×56×56]·64×3×3 (GFLOPS) | 348 (level) | 287 | 73 | **438** |
| MLP forward, batch 256 (samples/s) | **827 K** | 520 K | **473 K** | 361 K |
| MLP forward + backward | 205 K (level) | 212 K | 74 K | **124 K** |
| MLP train step (Adam) | **178 K** | 117 K | 71 K (level) | 71 K |
| EmbeddingGemma, 32 × 65 tokens (sentences/s) | **105** | 88 † | **59** | 37 |

† The PyTorch EmbeddingGemma figure is from 8 September: the benchmark
venv has no `sentence-transformers`, so that one row could not be
re-measured in this session. Every other M2 Pro number is from
12 September.

The convolution case is measured with `-only conv`. In a full suite run
it follows the element-wise section, which leaves 404 MiB retained in the
mapped pool and forces some 3 900 collections, and it then reports 184
GFLOPS instead of 348. Attention does not shift that way (991 in the
suite, 1 005 alone). See
[results/m2pro-2026-09-12/go-conv-isolated.md](benchmarks/results/m2pro-2026-09-12/go-conv-isolated.md).

**Where fiber/ai is ahead:** everything memory-bound that we wrote
kernels for (element-wise, exp, tanh on Apple, reductions, softmax,
layernorm, transposes), MLP inference on both machines, the full
EmbeddingGemma encoder on the Xeon, attention on Apple, and the largest
GEMM on the Xeon.

**Where it is behind, honestly:**

- **GEMM below 2048² on both machines, when B is packed per call.**
  Accelerate and MKL win by 20–50 % at 1024 and 512 and by 2× at 128 and
  256. Our blocked driver pays packing and dispatch that only amortise at
  the largest sizes. With the packed-operand cache (T-037), which keeps
  a reused right operand packed, the transformer projection
  [256×768]·[768×3072] goes from 25 % behind to 10 % ahead on the M2, and
  the inference rows (MLP forward, EmbeddingGemma) gain 9–14 %, and the
  fused epilogues (T-040: bias, activation, gated product and folded
  pre-norm applied on the finished output block) give EmbeddingGemma
  another 13 % on the M2; the square, fresh-operand case is unchanged
  and still behind.
- **The backward pass.** forward+backward loses 20 % on the M2 and 40 %
  on the Xeon: PyTorch's autograd fuses more and allocates less; our
  backward builds each gradient as its own tensor. The Adam step wins
  it back on the M2 (fused kernel) and is level on the Xeon.
- **Attention on x86** (2.3× behind at the last Xeon run) and
  **convolution on x86** (6×): each fused attention task was a K=64
  product through the general GEMM driver with its own packing; T-041
  replaced that with a micro-kernel driver that packs K and V once per
  head (M2: 947 → 1 110 GFLOPS; Xeon pinned 479 → 533, both sockets
  437 → 747); the profile then showed a third of the Xeon's time in Go
  packing loops, and the AVX2 register-transpose packing (T-042) took
  it to 618 pinned; the AVX-512 exponential (T-043) 655, one thread per
  core (T-045) 693, packing K and V inside each head's task instead of
  in a phase of its own (T-046) 763. An online-softmax block structure
  like oneDNN's measured slower at these sizes (the scores fit L1
  anyway) and stays behind a switch. Still 1.5× behind oneDNN's fused
  attention, and what is left is arithmetic under all-core load, not
  data movement. Limited to AVX2 (`ONEDNN_MAX_CPU_ISA=AVX2
  ATEN_CPU_CAPABILITY=avx2`) PyTorch still reaches 1 023 GFLOPS while
  our AVX2 back-end runs at 543: their kernel barely depends on the
  vector width, ours does; on AVX2, the deployment case, attention is
  1.9× behind while EmbeddingGemma stays ahead (49 against 36).
  The im2col convolution is memory-bound on the Xeon; oneDNN has a
  dedicated primitive, the implicit-GEMM candidate is on the list.
- **tanh on the Xeon** (9× behind in the unreleased 1M row): not the
  kernel. With the AVX-512 exp and tanh (T-043) the released rows are
  tanh 1M 54 µs (MKL 54, level) and exp 50 µs; every unreleased 1M row
  costs 480 µs, `relu` and `x * 2.5` included, because the result buffer
  comes back cold after a rotation through 256 MiB of unreleased
  results and this socket moves 17 GB/s (see the allocator note below
  the Xeon table). Python's reference counting frees a dropped result
  at once; in Go that is `Release()`.

Earlier headline tables quoted single best runs (M2 GEMM 2 301, train
164 K). The M2 varies between runs with the same code — two runs of one
binary an hour apart gave 2 089 and 2 157 GFLOPS on the same shape, a
3 % spread, and more than that on the inference rows — so this table uses
one run of each side taken minutes apart, and differences under about
5 % should be read as "level" whether or not the word is there.

## Matrix multiply

n×n · n×n, float32, GFLOPS (higher is better), M2 Pro:

| n | AMX 1 thread | AMX all workers | NEON 1 thread | NEON all workers | pure Go all | NumPy | PyTorch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 542 | 553 | 86 | 86 | 7.7 | 792 | **782** |
| 256 | 747 | 975 | 95 | 408 | 40.6 | 1 127 | **1 121** |
| 512 | 1 094 | 1 505 | 101 | 518 | 43.7 | 2 144 | **2 157** |
| 1024 | 1 182 | 2 118 | 101 | 542 | 43.9 | 2 636 | **2 688** |
| 2048 | 1 036 | **2 225** | 102 | 546 | – | 2 225 | 2 192 |

All workers is six here, the performance cores. The n=128 row is where
T-050's small-product path shows: 376 on 5 September, 553 now. The other
sizes are unchanged within the run-to-run spread. The pure Go column is
the same source with `FIBERAI_KERNEL=generic` and is what the assembly is
worth: a factor of 48 at n=1024.

Other shapes, all threads:

| shape | AMX, B packed per call | AMX, B packed once | NumPy | PyTorch |
|---|---:|---:|---:|---:|
| [1×4096]·[4096×4096] (matrix–vector, memory-bound) | **14.2** | 13.9 | 11.3 | 11.2 |
| [8×4096]·[4096×4096] | 71.7 | **272.0** | 70.6 | 69.3 |
| [64×1024]·[1024×1024] | 950 | **1 621** | 1 288 | 1 292 |
| [256×768]·[768×3072] (transformer FFN) | 1 705 | **2 507** | 2 150 | 2 328 |
| [512×512]·[512×512] | 1 541 | 1 732 | – | – |
| [1024×1024]·[1024×1024] | 2 157 | **2 465** | – | – |
| [1024×1024]·[1024×1024]ᵀ (transposed view, no copy) | – | 2 469 | 2 420 | 2 375 |

Two columns, because the answer depends on whether the right-hand operand
is reused. Packed per call is the honest number for a one-off product;
packed once is what a layer with fixed weights actually gets, and it is
the packed-operand cache of T-037 doing the work. Reading only one column
would flatter or wrong us depending on which.

Notes:

- The AMX tile alone reaches 1.55 TFLOPS on one thread and 3.3 TFLOPS on
  the six performance cores; the driver (packing, one barrier per K
  block) is what separates 2.3 from that at n=2048 and more at small n.
  The M4, with a single performance cluster, reaches 1.4 TFLOPS on one
  thread and 1.7 on all (PyTorch there uses SME: 1.9).
- NEON scaling from 1 to 6 workers reaches 5.4× at n=2048 (102 to 546).
  Adding the four efficiency cores on top was worth about 1.5 performance
  cores when they were still in the pool; T-047 took them out because
  every parallel round waited for the slowest worker.
- The matrix–vector case is pure memory bandwidth and the `axpy` path
  streams B once with all cores; both back-ends beat Accelerate there.
- Small M (8 rows) with NEON still packs the whole B matrix; reading B
  in place for M ≤ MR is on the roadmap (T-005). AMX is ahead of
  Accelerate on that shape anyway.

## Element-wise operations

GB/s (higher is better); pure Go is fiber/ai with `FIBERAI_KERNEL=generic`.

| op | n | fiber/ai | released | pure Go | NumPy | PyTorch |
|---|---:|---:|---:|---:|---:|---:|
| x + y | 16M | 123.9 | **147.2** | 95.3 | 88.9 | 144.5 |
| x * y | 16M | 125.8 | – | 96.2 | 89.7 | 137.2 |
| x + row (broadcast) | 16M | 88.4 | – | 63.0 | 45.9 | **157.4** |
| x * 2.5 | 16M | 127.7 | – | 82.2 | 91.9 | **156.0** |
| exp(x) | 16M | 83.8 | **94.6** | 20.6 | 5.1 | 43.3 |
| tanh(x) | 16M | 80.5 | **89.6** | 5.0 | 8.5 | 11.9 |
| relu(x) | 16M | 133.0 | – | 13.4 | 23.4 | **157.5** |
| x + y | 1M | 244.6 | **337.8** | 59.5 | 108.3 | 184.3 |
| exp(x) | 1M | 85.7 | **97.3** | 14.7 | 5.1 | 40.2 |
| tanh(x) | 1M | 78.3 | **93.2** | 5.0 | 8.5 | 11.8 |
| relu(x) | 1M | **174.6** | – | 9.7 | 23.9 | 123.1 |

"Released" is the same call with `Release()` on the previous result, so
the allocator hands back a warm buffer instead of a freshly mapped one.
It is what a loop that reuses its intermediates gets, and the gap between
the two columns is the cost of a cold 64 MB allocation.

Time per call for small arrays (64K elements), where fixed overhead
dominates:

| op | fiber/ai | pure Go | NumPy | PyTorch |
|---|---:|---:|---:|---:|
| x + y | 6.2 µs | 9.3 µs | **6.7 µs** | 30.9 µs |
| exp(x) | **9.6 µs** | 36.6 µs | 102.9 µs | 45.6 µs |
| tanh(x) | **9.8 µs** | 75.1 µs | 62.6 µs | 78.8 µs |

Notes:

- At 16M elements every implementation is bound by memory bandwidth, and
  the remaining difference is allocation, not arithmetic. PyTorch's edge
  on the simple ops comes from its caching allocator reusing the output
  buffer; releasing ours closes it (147 against 144 GB/s on `x + y`).
- `exp` and `tanh` are hand-written NEON kernels (Cody–Waite reduction
  and a polynomial, ≈ 1 ulp). Both are an order of magnitude ahead of
  NumPy and several times ahead of PyTorch's SLEEF. The earlier tables in
  this file recorded `tanh` at 5.0 GB/s, going through `math.Tanh` per
  element; that is the change between them and now.
- The 64K rows show fixed overhead rather than bandwidth. NumPy's
  single-threaded recycling loop is still marginally ahead on `x + y`,
  and the transcendentals are not close.

## Reductions on a 4096×4096 matrix

Time per call (lower is better):

| op | fiber/ai | pure Go | NumPy | PyTorch |
|---|---:|---:|---:|---:|
| sum() | **584 µs** | 742 µs | 2.71 ms | 593 µs |
| sum(dim=0) | **638 µs** | 1.18 ms | 1.33 ms | 2.23 ms |
| sum(dim=1) | 543 µs | 733 µs | 2.78 ms | **576 µs** |
| max(dim=1) | **537 µs** | 3.65 ms | 1.19 ms | 1.22 ms |
| softmax(dim=1) | **2.32 ms** | 15.5 ms | – | 4.60 ms |
| layernorm (last dim) | **1.81 ms** | 5.90 ms | – | 2.37 ms |
| transpose + contiguous copy | **12.6 ms** | 12.8 ms | 62.0 ms | 23.2 ms |

Row reductions run at 115 to 125 GB/s, which is memory speed on this
machine. `sum(dim=0)` folds rows into per-goroutine partial sums so it
streams the matrix once, where NumPy and PyTorch both do noticeably
worse. LayerNorm now computes mean and variance in one pass over each row
into an L1-resident scratch buffer and keeps only the per-row statistics
for backward; the earlier five-pass version measured 3.37 ms and was
behind PyTorch, which is the change between that table and this one.
`transpose + contiguous` is a pure permutation and the one row where the
assembly buys nothing — pure Go matches it.

## Training step: MLP 784 → 512 → 512 → 10, batch 256, Adam

| phase | fiber/ai | pure Go | PyTorch |
|---|---:|---:|---:|
| forward (no grad) | **310 µs · 827 K samples/s** | 9.18 ms · 28 K | 493 µs · 520 K |
| forward + backward | 1.25 ms · 205 K | 22.1 ms · 12 K | 1.21 ms · 212 K (level) |
| forward + backward + Adam step | **1.44 ms · 178 K** | 22.8 ms · 11 K | 2.19 ms · 117 K |

The step is ~90 % matrix products (340 MFLOP forward, ~2× that backward),
so this table is the GEMM table again, and it moved when GEMM moved to
AMX: the earlier NEON figures here were 1.51 ms forward and 4.50 ms for
the full step.

Two things are worth noting. The autograd machinery is cheap —
forward+backward costs 4.0× the forward, against the 3× the GEMMs alone
would predict, so roughly a quarter of the backward is bookkeeping. And
the optimiser step is where Python overhead shows: PyTorch spends 0.98 ms
on Adam for 668 K parameters, fiber/ai 0.19 ms.

## Apple M4 (4 performance + 6 efficiency cores)

Same workloads, run by hand on an M4 with Go 1.27.0, NumPy 2.5.3 and
PyTorch 2.14.0 (raw: [results/go-m4.md](benchmarks/results/go-m4.md),
[results/python-m4.md](benchmarks/results/python-m4.md)).

| Workload | fiber/ai | NumPy | PyTorch | ratio |
|---|---:|---:|---:|---:|
| SGEMM 128², all cores (GFLOPS) | 93.8 | 913 | 953 | 10× behind |
| SGEMM 2048², 1 core (GFLOPS) | **120.7** | – | 1 878 (SME) | 15× behind |
| SGEMM 2048², all cores (GFLOPS) | 614 | 1 884 | 1 881 | 3.1× behind |
| [1×4096]·[4096×4096] (GFLOPS) | 30.4 | 34.8 | 33.2 | par |
| [8×4096]·[4096×4096] (GFLOPS) | 72.9 | 93.0 | 93.1 | 1.3× behind |
| exp, 16M | **2.09 ms** | 24.9 ms | 3.57 ms | 1.7× ahead |
| softmax(dim=1), 4096² | **2.74 ms** | – | 6.02 ms | 2.2× ahead |
| sum(dim=0), 4096² | **0.62 ms** | 1.24 ms | 2.79 ms | 4.5× ahead |
| max(dim=1), 4096² | **0.57 ms** | 1.03 ms | 2.38 ms | 4.2× ahead |
| layernorm, 4096² | 4.66 ms | – | **1.94 ms** | 2.4× behind |
| transpose + copy, 4096² | 10.8 ms | 51.2 ms | 11.2 ms | par |
| x + y, 16M | 2.84 ms | 2.23 ms | **2.14 ms** | 1.3× behind |
| MLP forward, batch 256 (samples/s) | 189 K | – | **835 K** | 4.4× behind |
| MLP train step, batch 256 (samples/s) | 64.7 K | – | **117 K** | 1.8× behind |

Observations:

- The NEON single-core figure is 86 % of the M4 performance core's FMA
  peak, the same efficiency as on the M2 Pro. The M4 (4P+6E) scales to 5×
  one core.
- Accelerate on the M4 goes through **SME** and delivers 1.9 TFLOPS — less
  than the M2 Pro's 2.7 TFLOPS on AMX, because the base M4 has one matrix
  unit for its single performance cluster. SME is a documented ISA on the
  M4, so an SME kernel in Go assembly (TODO T-006) can target the same
  unit; the M4 numbers above are its acceptance baseline.
- Everything that is not a GEMM is where it was on the M2 Pro: ahead on
  exp, softmax, column reductions and max; behind on layernorm (five
  passes per row instead of one fused pass) and on tanh.

## Apple AMX back-end (M2 Pro): what it is worth against NEON

Accelerate's SGEMM numbers on the Macs come from the AMX coprocessor;
spec T-010 puts fiber/ai's GEMM tile on the same unit (Go assembly with
the instruction words documented by github.com/corsix/amx). AMX is the
default back-end on Apple Silicon now; the NEON column here is the same
binary with `FIBERAI_KERNEL=neon`, measured in the same session (raw:
[results/m2pro-2026-09-12/go-neon.md](benchmarks/results/m2pro-2026-09-12/go-neon.md)).

| Workload | NEON (6 workers) | AMX | NumPy | PyTorch |
|---|---:|---:|---:|---:|
| SGEMM 512², all workers (GFLOPS) | 518 | 1 505 | 2 144 | **2 157** |
| SGEMM 1024² | 542 | 2 118 | 2 636 | **2 688** |
| SGEMM 2048² | 546 | **2 225** | 2 225 | 2 192 |
| SGEMM 1024², 1 thread | 101 | 1 182 | – | 2 652 (Accelerate ignores the thread limit) |
| [64×1024]·[1024²], packed per call | 411 | 950 | 1 288 | **1 292** |
| [256×768]·[768×3072], packed per call | 498 | 1 705 | 2 150 | **2 328** |
| [1024²]·[1024²]ᵀ (view) | 553 | **2 469** | 2 420 | 2 375 |
| MLP inference (samples/s) | 349 K | **827 K** | – | 520 K |
| MLP forward + backward | 102 K | 205 K | – | 212 K (level) |
| MLP training step | 98 K | **178 K** | – | 117 K |

AMX is worth 3 to 4× over NEON on every GEMM shape and 2.4× on MLP
inference. It is also the reason the training-step row reads the way it
does: on NEON that row was 98 K, behind PyTorch.

The raw tile reaches 1.55 TFLOPS on one thread and 3.3 TFLOPS on six
(the units sit with the six performance cores; the efficiency cores'
unit is slow), so the driver — packing, one barrier per K block, the C
tile in and out of Z once per K block — is what separates 2.25 from
Accelerate's 2.7 at n=1024 and below. Two findings on the way: the
coprocessor executes in order, so the loads of step k+4 must be issued
before the outer products of step k (four X/Y register slots; 3× faster
than the naive loop), and `set`/`clr` per tile costs 0.7 µs, hence once
per task with the goroutine locked to its thread.

## Apple M4 MacBook Air (9 September 2026)

Four performance and six efficiency cores, fanless; the AMX back-end is
on by default here as on the M2. PyTorch 2.14 on Accelerate, which uses
the M4's SME unit, four threads; same day, commit 0ceb7eb; raw runs in
`results/m4air-2026-09-09/`. Bold marks the faster side.

| Workload | fiber/ai | NumPy | PyTorch |
|---|---:|---:|---:|
| SGEMM 2048², all cores (GFLOPS) | **1 716** | 1 648 | 1 631 |
| SGEMM 1024² | **1 685** | 1 628 | 1 543 |
| SGEMM 512² | **1 621** | 1 480 | 1 479 |
| SGEMM 128² | 559 | 941 | **954** |
| [256×768]·[768×3072], B packed once | **1 576** (1 350 per call) | 1 542 | 1 507 |
| [1×4096]·[4096×4096] | 24.7 | **35.1** | 35.1 |
| x + y, 1M (released) | **51 µs** | 109 µs | 59 µs |
| x + y, 16M (released) | 2.42 ms | 2.20 ms | **2.14 ms** |
| exp, 16M | **1.86 ms** | 25.5 ms | 3.90 ms |
| tanh, 1M (released) | **94 µs** | 787 µs | 721 µs |
| sum(), 4096² | **595 µs** | 2.25 ms | 906 µs |
| softmax(dim=1), 4096² | **2.73 ms** | – | 5.71 ms |
| layernorm, 4096² | 2.64 ms | – | **2.39 ms** |
| transpose + copy, 4096² | **11.5 ms** | 50.4 ms | 14.4 ms |
| attention [8×8×512×64] (GFLOPS) | **920** → 988 | – | 729 |
| conv2d [32×64×56×56]·64×3×3 (GFLOPS) | 357 | – | 332 (level) |
| MLP forward, batch 256 (samples/s) | 666 K → **1.07 M** (performance cores, T-047) | – | 815 K |
| MLP forward + backward | 202 K → 254 K | – | **348 K** |
| MLP train step (Adam) | 163 K → **225 K** | – | 204 K |
| EmbeddingGemma (sentences/s) | 83 → **91** (performance cores) | – | 80 |

The M4 Air is where PyTorch's Accelerate stack is strongest, and the
picture differed from the M2 Pro in one place: with all ten cores the
MLP rows were behind, all three phases, where the M2 Pro had them ahead.
The Air has four performance and six efficiency cores against the M2
Pro's six and four, and every parallel round waited for an efficiency
core. With four workers (T-047 makes the performance cores the Apple
Silicon default) the forward is 1.3× ahead and the training step ahead;
the backward pass stays behind, the fusion gap. The M2 Pro gained a
quarter on the training step the same way (147 K → 183 K when T-047
landed; it measures 178 K now) and lost about a tenth on attention, whose
whole-head tasks the efficiency cores could grind through alone;
`FIBERAI_WORKERS=10` gets that back.

## Cloud VM, 6 vCPU AVX2 (12 September 2026)

The machine most like production: a KVM guest with six AVX2 vCPUs, no
AVX-512, a hypervisor between us and the page tables. fiber/ai at
da3c3b9 built with Go 1.27.1 on the guest itself, against NumPy 2.5.2 /
OpenBLAS and PyTorch 2.14.0+cpu / MKL with six threads, same session;
raw runs in `results/kvm-avx2-2026-09-12/`.

**Read the ranges, not the medians.** This guest is shared and its
run-to-run spread dwarfs the Mac's: three runs of one binary gave 215,
233 and 262 GFLOPS on SGEMM 2048² and 44, 51 and 58 at 128². Every
figure below is the median of three, and differences under about 10 %
mean nothing here. The 9 September table this replaces was single runs,
so it was never better than that either.

| Workload | fiber/ai AVX2 | NumPy | PyTorch |
|---|---:|---:|---:|
| SGEMM 2048², all cores (GFLOPS) | **255** | 167 | 188 |
| SGEMM 1024² | **248** | 184 | 173 |
| SGEMM 512² | **235** | 213 | 185 |
| SGEMM 256² | 175 (level) | 154 | 173 |
| SGEMM 128² | 44 | 74 | **104** |
| x + y, 1M (released) | **136 µs** | 724 µs | 141 µs |
| x + y, 16M (released) | **6.12 ms** | 40.7 ms | 30.9 ms |
| exp, 16M | **6.34 ms** | 76.7 ms | 32.7 ms |
| tanh, 1M (released) | **304 µs** | 6.1 ms | 1.22 ms |
| sum(), 4096² | **1.69 ms** | 9.95 ms | 2.01 ms |
| sum(dim=0), 4096² | **2.38 ms** | 8.98 ms | 7.30 ms |
| softmax(dim=1), 4096² | **9.51 ms** | – | 34.2 ms |
| layernorm, 4096² | **7.59 ms** | – | 25.6 ms |
| transpose + copy, 4096² | **21.6 ms** | 216 ms | 128 ms |
| attention [8×8×512×64] (GFLOPS) | **163** | – | 128 |
| conv2d [32×64×56×56]·64×3×3 (GFLOPS) | 74 | – | **130** |
| MLP forward, batch 256 (samples/s) | **145 K** | – | 101 K |
| MLP forward + backward | 25 K | – | **44 K** |
| MLP train step (Adam) | 31 K (level) | – | 32 K |
| **tiny autoencoder, forward (samples/s)** | **927 K** | – | 231 K |
| **tiny autoencoder, training step** | **225 K** | – | 31 K |

### What changed since 9 September

Same machine, same workloads, three months of kernel and allocator work
in between — and this is the first time any of it has been measured on
x86 rather than assumed from the Mac.

| Workload | 9 Sep | 12 Sep | |
|---|---:|---:|---|
| transpose + copy, 4096² | 74 ms | **21.6 ms** | 3.4× faster |
| layernorm, 4096² | 11.0 ms | **7.59 ms** | 1.4× |
| tanh, 1M (released) | 397 µs | **304 µs** | 1.3× |
| SGEMM 512² | 180 | **235** | 1.3× |
| x + y, 1M (released) | 163 µs | **136 µs** | 1.2× |
| sum(), 4096² | 1.92 ms | 1.69 ms | 1.1× |
| exp 16M, x+y 16M, softmax | – | – | unchanged |
| sum(dim=0), 4096² | 2.10 ms | 2.38 ms | 13 % slower |
| SGEMM 128² | 52 | 44 | 15 % slower |

The transpose row is the one to notice. It allocates and writes a 64 MB
result, page faults cost 2–3× more under a hypervisor than on the Mac,
and off-heap mapped storage with a free list removes most of them. The
allocation-cost section below predicted this was "the single most
valuable change for x86 deployments"; it was, and by more than it gained
on the Mac.

Two rows went the other way and neither is inside the noise. `sum(dim=0)`
and SGEMM 128² both sit below the whole three-run range of the September
figure. The 128² case is the interesting one: T-050's small-product path
took the M2 Pro from 376 to 553 GFLOPS at that size and does nothing
here — the spec recorded "Xeon and VM: pending" and this is the answer.
The path is tuned for the AMX tile, and on AVX2 the driver it replaces
was not the bottleneck.

### Small products: where this machine actually loses and wins

The shapes below 256² are where fiber/ai is furthest behind on x86, and
the shapes an anomaly-detection model is made of are where it is
furthest ahead. Both are in the same table, which is the point.

| shape | fiber/ai | PyTorch |
|---|---:|---:|
| 32² (GFLOPS) | **12.6** | 10.5 |
| 64² | 31.3 | **37.3** |
| 96² | 38.2 | **86.0** |
| 128² | 51.4 | **126.3** |
| 160² | 47.8 | **130.3** |
| 192² | 129.0 | **171.1** |
| 256² | **184.1** | 147.4 |
| [64×24]·[24×16] | **10.8** | 8.4 |
| [64×16]·[16×3] | **1.5** | 1.1 |
| [256×24]·[24×16] | 16.0 | **19.1** |

Between 96² and 192² PyTorch is two to two-and-a-half times ahead, and
that band is unfixed work. Below and above it we are ahead. A
24→16→3→16→24 autoencoder at batch 64 is made entirely of the narrow
shapes at the bottom of that table, and it runs 4× faster on the forward
pass and 7× faster on a full training step than the same model in
PyTorch — because at that size a step is per-operation overhead, and
that is what Python spends.

### Notes

Attention and convolution are measured with `-only`, like the M2 Pro
rows, and their spread on this guest is wide enough to be worth stating:
attention ranged 124 to 185 GFLOPS over three runs, convolution 48 to 77.
Treat both as order-of-magnitude.

The matrix–vector row is gone from this table: at 6 vCPU it is entirely
memory bandwidth and the guest's is not stable enough between runs to
report a number anyone could reproduce.

Everything that allocates still suffers under the hypervisor. The
unreleased 64K row remains the extreme: a fresh 256 KB mapping pays
about 0.7 ms of page faults here, so small results that are not released
rotate through a thousand fresh mappings before the first forced
collection. A lower collection budget for small size classes on such
hosts, or the heap path for them, is still the candidate (TODO).

## Small products and the tiny autoencoder (M2 Pro)

Below about 160² a product takes one call with micro-kernels that read
their operands in place (T-050) instead of the blocked driver. Fresh
operands, result released; PyTorch 2.8 on Accelerate the same session
(`bench.py --only small`). "Before" is the blocked driver, measured
before T-050.

| shape | before | fiber/ai | PyTorch |
|---|---:|---:|---:|
| 32² (GFLOPS) | 42 | **63** | 52 |
| 64² | 156 | 216 | **305** |
| 96² | 264 | 371 | **566** |
| 128² | 379 | 558 | **785** |
| 160² | 526 | 671 | **890** |
| 192² (driver) | 732 | 743 | **992** |
| 256² (driver) | 1 082 | 1 105 | 1 143 (level) |
| [64×24]·[24×16] | – | 25 | **38** |
| [64×16]·[16×3] | – | **4.0** | 3.1 |
| [256×24]·[24×16] | – | 36 | **100** |
| tiny autoencoder 24→16→3→16→24, batch 64, forward (samples/s) | – | **3.08 M** | 1.84 M |
| same, forward + backward + Adam step | – | **1.16 M** | 253 K |

The training step of a model this small is 4.6× ahead: at this size a
step is per-operation overhead, and that is where Python pays. The
squares are still behind Accelerate by 1.3–1.5× below 160² (they were
1.8–2.1× behind): what remains on the M2 is the AMX tile store per
k-step block and the result recycling, not packing. Narrow products
([256×24]·[24×16]) stay behind because the 32-wide AMX tile half idles at
n = 16. Xeon and VM: pending.

## Attention, convolution and EmbeddingGemma (M2 Pro, AMX)

Rows added with the transformer work; PyTorch 2.8 (Accelerate BLAS, 6
threads by default) via `benchmarks/python/bench.py --only
attention,conv,embed`, fiber/ai via `go run ./cmd/bench -only
attention,conv,embed`. Attention and convolution run under `NoGrad` with
the result released; EmbeddingGemma is float32 on both sides, 32 copies
of a 65-token sentence as one batch.

| row | PyTorch | fiber/ai |
|---|---:|---:|
| attention [8×8×512×64], GFLOPS over the two products | 590 | 1 005 |
| same with a causal mask | 592 | 832 |
| attention [1×8×2048×64], long sequence | 711 | 958 |
| conv2d [32×64×56×56] · 64 filters 3×3, GFLOPS | 287 | 348 |
| EmbeddingGemma, sentences/s | 88 (8 September) | 105 |

Each row here is measured with `-only`, not inside a full suite run: the
convolution case in particular reports 184 GFLOPS rather than 348 when it
follows the element-wise section, because the mapped pool is holding
404 MiB by then and the collector is running constantly. That is a
property of the harness, not of the convolution, and it is on the list to
fix.

The EmbeddingGemma row could not be re-measured on the PyTorch side in
this session — the benchmark venv has no `sentence-transformers` — so
that one figure is from 8 September while ours is current. A full
24-layer encoder, tokenizer to unit vector, a fifth ahead of the Python
stack on the same machine. Its parity with `sentence-transformers` is
cosine 1.000000 on 64 short and 3 long (up to 1753-token) inputs; see
[docs/manual/models.md](docs/manual/models.md).

## Allocation cost (why the element-wise numbers are what they are)

Every element-wise operation allocates its result; Go zero-fills it and,
once the GC has released earlier results to the OS, the pages fault back
in on the next write. The `GOGC` experiment below is from 5 September and
is kept because it is what motivated the fix rather than because it
describes the code now: results of 64 KiB and more are mapped outside the
Go heap and recycled through a pool, and `Release()` returns one for
immediate reuse. The "released" column of the element-wise table is that
same effect measured on the current code — 337 against 245 GB/s on
`x + y` at 1M elements.

Raising `GOGC` from 100 to 800 — which keeps freed memory resident
instead of returning it — changed nothing in the kernels and this much in
the timings:

| op | n | GOGC=100 | GOGC=800 |
|---|---:|---:|---:|
| x + y | 64K | 26.3 µs | **11.5 µs** |
| relu(x) | 64K | 24.4 µs | **9.1 µs** |
| x + y | 1M | 153 µs | **113 µs** |
| exp(x) | 1M | 221 µs | **180 µs** |
| x + y | 16M | 2.05 ms | 2.69 ms |

So below a few MB the cost is the allocator, not the arithmetic; above
that the 64 MB results are fresh pages either way. On the x86 KVM guest
the same effect is much larger, because page faults are 2–3× more
expensive under virtualisation: `x + y` on 16M elements ran at 8 GB/s
while `sum()` over the same data ran at 32 GB/s. Writing results into
caller-provided or pooled buffers was the fix, and it landed: off-heap
mapped storage with a free list, `Release()`, and an opt-in heap pool
(`SetPoolLimit`). It remains the single most valuable change for x86
deployments, where the x86 tables here still predate it.

## x86: Intel Xeon Gold 6130 (Skylake-SP), one socket

The production-class comparison: bare metal, 16 cores per socket, AVX-512
with two FMA units per core. fiber/ai pinned to socket 0 with
`GOMAXPROCS=16 numactl --cpunodebind=0 --membind=0`; NumPy 2.5.3 links
OpenBLAS 0.3.34 (Haswell kernels), PyTorch 2.14.0+cpu links MKL 2024.2
(32 threads, hyperthreads included). Go 1.27.1. Raw:
[results/skylake-sp-6130-1socket/](benchmarks/results/skylake-sp-6130-1socket/).

| Workload | fiber/ai AVX-512 | fiber/ai AVX2 | NumPy / OpenBLAS | PyTorch / MKL |
|---|---:|---:|---:|---:|
| SGEMM 1024², 1 thread (GFLOPS, blas bench) | **162** | – | – | 169 |
| SGEMM 1024², 1 thread (GFLOPS, incl. output allocation) | 122 → 156 (T-015) | 74 | – | – |
| SGEMM 1024², all cores, tensor level (result released) | 730 → 1 208 (T-015/T-016) | – | – | – |
| SGEMM 2048², all cores, tensor level (result released) | 1 307 (T-015/T-016) | – | – | – |
| tanh, 1M | 2.73 ms → 484 µs (T-018) | | 421 µs | **54 µs** |
| SGEMM 512², all cores (GFLOPS) | 245 | 241 | 717 | **991** |
| SGEMM 1024², all cores (GFLOPS) | 530 → 1 242 after T-014/T-016 (blas bench) | 439 | 1 303 | **1 434** |
| SGEMM 2048², all cores (GFLOPS) | 732 → **1 285** after T-014/T-016 (blas bench) | – | 881 | 780 |
| [1×4096]·[4096×4096] (GFLOPS) | **13.0** | 13.0 | 10.9 | 11.0 |
| [256×768]·[768×3072] (GFLOPS) | 323 → **1 162** (T-015/T-016) | 344 | 1 088 | 980 |
| x + y, 64K | 194 µs → 49 µs, **8.8 µs** released (T-015/T-017) | | 16.3 µs | 17.8 µs |
| x + y, 1M | 1.49 ms → 650 µs, **17.4 µs** released (T-015/T-017/T-019) | | 506 µs | 24 µs |
| x + y, 16M | 22.8 ms → **13.9 ms** released (T-015) | | 26.7 ms | 17.5 ms |
| exp, 16M | 20.0 ms → **10.7 ms** (T-015) | | 24.6 ms | 13.8 ms |
| sum(), 4096² | 2.40 ms | | 4.79 ms | **2.42 ms** |
| sum(dim=0), 4096² | **2.68 ms** | | 4.80 ms | 6.29 ms |
| max(dim=1), 4096² | **2.40 ms** | | 5.03 ms | 2.46 ms |
| softmax(dim=1), 4096² | 20.9 ms → **11.0 ms** (T-015) | | – | 14.4 ms |
| layernorm, 4096² | 36.1 ms → **11.0 ms** (T-015/T-020) | | – | 14.1 ms |
| transpose + copy, 4096² | **25.5 ms** | | 819 ms | 68.5 ms |
| MLP forward, batch 256 (samples/s) | 59 K → 288 K (T-015/T-019) | | – | **361 K** |
| MLP forward + backward (samples/s) | 16 K → 65 K (T-014..T-019) | | – | **124 K** |
| MLP train step, batch 256 (samples/s) | 14 K → 64 K (T-014..T-019) | | – | **71 K** |

What this says:

- **The AVX-512 micro-kernel is right.** It passed the start-up
  self-verification and every test on the first run on real hardware, it
  is 1.65× faster than the AVX2 kernel on the same core, and at 162
  GFLOPS single-threaded it sits at ~90 % of the core's AVX-512 turbo
  peak — level with MKL's single thread. This is the like-for-like
  comparison the Macs could not give: kernel against kernel, we are
  there.
- **Multi-core scaling was the problem — and it was the driver.** The
  first run reached 759 GFLOPS at n=1024 with 16 workers (32 % of the
  socket's ~2 000 GFLOPS AVX-512 peak at its 1.95 GHz all-core clock).
  Spec T-014 took it to **1 147** (57 %) in four measured steps: pack A
  once per K block instead of per task, replace channel polling in the
  worker pool with an atomic spin (a profile showed 20 % of CPU in
  runtime locks), spin through the rounds instead of parking (perf showed
  the cores only 70 % busy), and a finer task grid (the last round of a
  K block left most cores waiting). Blocking, hyperthreading and NUMA
  placement were measured and ruled out along the way. Spec T-016 then
  widened the AVX-512 tile from 12×32 to 14×32 (28 accumulators, MC=112)
  for **1 242** at n=1024 and **1 285** at n=2048 (62–64 % of peak);
  software prefetch in the kernel changed nothing. At n=2048 fiber/ai is
  ahead of both MKL (780) and OpenBLAS (881); at n=1024 MKL's 1 434 (72 %
  of peak) is still 15 % ahead.
- **Two sockets are slower than one** (n=1024: 162 vs 530 GFLOPS with 64
  vs 16 threads). NUMA and hyperthreads are invisible to Go's scheduler;
  a topology-aware default thread count is TODO T-013.
- **Element-wise operations were dominated by allocation.** A fresh
  1M-float result cost 647 µs on this machine: Go zero-fills it on the
  allocating thread and its pages fault in one at a time. Spec T-015 moved
  results of 128 KiB and more off the Go heap (mmap, huge pages, first
  touch inside the parallel kernels, cleanup-based reuse), which took
  `x + y` 1M from 1.49 ms to ~500 µs and 16M from 22.8 to 15.3 ms. The
  remaining gap to PyTorch's 24 µs at 1M is cache residency: its
  reference counting hands the same buffer out again immediately, while a
  collector-driven library rotates through cold buffers between
  collections. `Release()` closes that (28.8 µs at 1M, 8.8 µs at 64K,
  13.9 ms at 16M: two of three rows ahead of PyTorch), together with
  owner-first chunk assignment in `parallel.Range` (T-017) so the same
  core touches the same slice on every call. `nn` releases its
  intermediates itself, which is what lifted MLP inference 59 K → 176 K
  samples/s. This machine's memory sustains ~25 GB/s (`sum()` says so for
  both libraries), so anything that leaves the caches costs the same for
  everyone.
- Where no allocation dominates we already win on this machine:
  matrix-vector, column sums, max, transposes.


Transformer round (T-031/T-032/B-004, 2026-09-08), same socket and
pinning, Go 1.27.1. The MLP rows moved with the storage and RMSNorm work
of that round; the attention rows are after the B-004 fix (before it the
masked row ran at 39 GFLOPS). The Python columns are open: the Xeon's
venv has NumPy and PyTorch but no `sentence-transformers`, and the
machine cannot reach PyPI.

| Workload | fiber/ai AVX-512 | PyTorch / MKL |
|---|---:|---:|
| SGEMM 1024², all cores, tensor level | 1 145 | – |
| SGEMM 2048², all cores, tensor level | 1 156 → **1 268** (workers pinned to cores, T-045) | – |
| [256×768]·[768×3072] | 1 016 | – |
| MLP forward, batch 256 (samples/s) | 288 K → 384 K → 415 K (T-040) → **473 K** (T-042) | 361 K |
| MLP forward + backward (samples/s) | 65 K → 72 K | 124 K |
| MLP train step (samples/s) | 64 K → 67 K | 71 K |
| attention [8×8×512×64] (GFLOPS) | 479 → 533 (T-041) → 618 (T-042) → 655 (T-043) → 693 (T-045) → 763 (T-046) | **1 141** |
| same with causal mask | 470 (was 39) → 515 → 595 → 662 → 713 | **1 129** |
| attention [8×8×512×64], AVX2 back-end | 543 | **1 023** (oneDNN and ATen limited to AVX2; MKL limit pending) |
| EmbeddingGemma, AVX2 back-end (sentences/s) | **49** | 36 (same limits) |
| attention [1×8×2048×64], long sequence | 539 → 698 → 748 | **1 141** |
| conv2d [32×64×56×56]·64×3×3 (GFLOPS) | 73 | **438** |
| EmbeddingGemma, 32 × 65 tokens (sentences/s) | **49 → 58** (T-040, pre-norms folded, gated FFN fused) | 37 |
| [64×1024]·[1024×1024], B packed once (T-037) | 551 → **699** (541 per call, T-042) | 699 (level) |
| [8×4096]·[4096×4096], B packed once | **89** (58 per call) | 58 |
| [256×768]·[768×3072], B packed once | 1 240 → **1 331** (1 111 per call, T-042) | 980 |
| EmbeddingGemma with the cache warm (all hits) | 43–47, no gain on this machine | 37 |
| cluster.Cosine, 768-d pair | **103 ns** | NumPy 4 332 ns |
| exp / tanh, one core, ns per element (kernel benchmark) | AVX2 0.51 / 0.57 → **AVX-512 0.27 / 0.32** (T-043) | – |

The unreleased element-wise 1M rows on this machine (about 480 µs
whatever the operation, `relu` and `x * 2.5` included) do not measure
kernels but DRAM: an unreleased 4 MB result is reclaimed by a forced
collection once 256 MiB are outstanding, so each one comes back cold,
and 12 MB of traffic at this socket's 17 GB/s is 480 µs. The forced
collection itself is about 1 ms per 64 results (gctrace) and the
buffers are reused (567 K hits against 1 K misses over the section); two
allocator changes on the way to that finding (a bounded wait, then
synchronous reclamation through weak pointers, T-044) made the
recycling deterministic but not faster, because it never was the cost.
The released rows are the kernel comparison: exp 50 µs, tanh 54 (MKL
54), gelu 119.
| cluster.Similarities 1 000×10 000 | **15.5 ms** | NumPy/OpenBLAS 85 ms |

PyTorch 2.14.0+cpu with MKL and oneDNN, pinned to the same socket with
`OMP_NUM_THREADS=16`; unpinned over both sockets it reaches 2 189 /
2 118 / 611 on the three GFLOPS rows. `sentence-transformers` 6.0.1 in
float32 for the embedding row (installed from offline wheels, the
machine has no PyPI access).

The embedding row is the one this round was about: the full encoder,
tokenizer to unit vector, runs 1.2–1.3× faster than the Python stack on
the production machine and 1.1× on the M2 Pro (with the packed-operand
cache). The cache lifts the M2 by 14 % and the Xeon not at all: with the
cache warm and every product a hit the Xeon still needs 740 ms per
batch, of which the products account for about 360 ms at its GEMM rate;
the rest is the element-wise work between them (146 RMSNorms, GELU,
residual adds, RoPE, layout copies, some 2 GB of traffic per batch),
which the Xeon's memory system serves at a tenth of the M2's bandwidth.
Fusing those passes into the GEMM epilogues is the next lever there. The two rows where PyTorch
is clearly ahead, attention (2.3×) and convolution (6×), are the next
kernels to write (see TODO).

Two placement results from the same session. Without `numactl`, the
Linux default (all 32 physical cores of both sockets) reaches 887 GFLOPS
at 2048² and 46 K training samples/s; `FIBERAI_WORKERS=16` unpinned
gives the same 887 and 46 K; the pinned socket gives 1 156 and 67 K.
Worker count is not the lever, memory placement is: pages first touched
by workers on both sockets are spread over both nodes. Pinning worker
threads to one node's cores is the candidate fix (TODO). The 1M-element
parallel threshold is right when pinned: n=128 runs at 99 GFLOPS on all
cores against 65 on one; the earlier unpinned reading of 58 was the
placement problem, not the threshold. Convolution at 73 GFLOPS against
1 150 for GEMM says the im2col path is memory-bound on this machine and
is the strongest case yet for the implicit-GEMM candidate. Attention
runs at about 42 % of the GEMM rate on both machines (479 of 1 156 on the
Xeon, 981 of 2 300 on the M2): each fused task is a K=64 product with
its own packing, where the blocked GEMM is at its weakest. oneDNN's
fused attention primitive gets twice that; an attention micro-kernel
that packs K and V once per head and skips the general GEMM was the
candidate and is now in (T-041): 947 → 1 110 GFLOPS on the M2, where
what remains is the AMX tile store at depth 64 and the exponential. On
the Xeon the same change gives 479 → 533 pinned (PyTorch 1 141 the same
day) and 437 → 747 on both sockets: the new path carries little memory
traffic, so it scales across the sockets where the old one did not, but
per core it reaches 33 GFLOPS against 72 for the plain GEMM. The AVX2
exponential (about 1.2 ns per element on this machine, three times the
M2's) is the first suspect; a profile of the pinned run decides.

Where the socket's remaining 37 % go (2048²: 1 268 of a 2 000 GFLOPS
all-core peak): the kernel alone runs at 95 % of the single-core peak
(174 of 179 GFLOPS at 2.8 GHz), the scaling curve loses from the second
core on (93 % efficiency at 2, 74 % at 4, 56 % at 12, nothing from the
last four), K blocking and loop order change nothing (KC 512/1024/2048
within 3 %, the rows strategy 30 % worse), finer tasks 2 %, one thread
per physical core +10 % (now the default on one package). What is left
is the AVX-512 clock under all-core load and shared resources between
cores; not cheap to buy back in software.

The two-socket run (all 64 hardware threads, no pinning) is in
[results/skylake-sp-6130-2socket/](benchmarks/results/skylake-sp-6130-2socket/)
as a record of the problem, not as a result.

## x86: KVM guest at Broadwell feature level (AVX2 only)

A 6-vCPU cloud VM without AVX-512. AVX2 kernel selected by detection, all
tests pass. NumPy 2.5.2 (OpenBLAS), PyTorch 2.14.0+cpu (MKL, 6 threads).
Raw: [results/x86_64-avx2-kvm/](benchmarks/results/x86_64-avx2-kvm/).

| Workload | fiber/ai AVX2 | NumPy / OpenBLAS | PyTorch / MKL |
|---|---:|---:|---:|
| SGEMM 1024², 1 thread (GFLOPS) | 37 | – | 54 |
| SGEMM 512², all vCPUs (GFLOPS) | 105 | 179 | **183** |
| SGEMM 1024², all vCPUs (GFLOPS) | 141 | 146 | **170** |
| SGEMM 2048², all vCPUs (GFLOPS) | **191** | **191** | 145 |
| [1×4096]·[4096×4096] (GFLOPS) | 8.7 | 14.2 | **16.0** |
| x + y, 64K | 159 µs | **40 µs** | 41 µs |
| x + y, 1M | 1.64 ms | 852 µs | **270 µs** |
| x + y, 16M | **24.2 ms** | 42.3 ms | 35.0 ms |
| exp, 16M | **19.3 ms** | 68.6 ms | 35.6 ms |
| sum(), 4096² | **2.09 ms** | 9.16 ms | 2.43 ms |
| sum(dim=0), 4096² | **2.42 ms** | 9.71 ms | 6.51 ms |
| max(dim=1), 4096² | **2.16 ms** | 9.50 ms | 4.41 ms |
| softmax(dim=1), 4096² | **23.5 ms** | – | 45.9 ms |
| layernorm, 4096² | 38.7 ms | – | **30.1 ms** |
| transpose + copy, 4096² | **86 ms** | 188 ms | 134 ms |
| MLP forward, batch 256 (samples/s) | 42 K | – | **116 K** |
| MLP train step, batch 256 (samples/s) | 13 K | – | **31 K** |

On the VM the picture is friendlier than on the bare-metal Xeon: page
faults on fresh allocations cost everyone, so on the large memory-bound
workloads fiber/ai is ahead of both, and the largest GEMM is level with
OpenBLAS and ahead of MKL. Single-threaded, MKL's AVX2 kernel is 1.45×
faster than ours on this (virtual) Broadwell — the AVX2 micro-kernel
deserves the same tuning attention as the AVX-512 one once real AVX2
hardware with a visible clock is available. Small arrays (64K) and the
MLP step stay 2–4× behind for the allocation reasons described above.

### Thread placement on the Xeon (follow-up)

Pinning the 16 workers to 16 physical cores (`numactl --physcpubind`)
gives the same result as letting them float over the socket's 32
hardware threads (793 vs 758 GFLOPS at n=1024, 1 007 vs 1 018 at
n=2048), and 32 physical cores across both sockets with interleaved
memory reach 1 036 — no more than 16. Hyperthreading and NUMA are not
what limits scaling; the blocked driver is. The MC/KC sweep on this
machine is the next measurement.

## What the numbers say

- The kernels are close to the hardware limits: the NEON GEMM runs at
  about 87 % of one core's FMA peak, and the streaming operations run at
  memory bandwidth.
- On GEMM the picture depends on the size. Accelerate is ahead below
  1024² — by 1.3× at 512², more at 128² — and we are level at 2048²
  (2 225 against 2 225 and 2 192). Where the right-hand operand is reused,
  which is what a layer with fixed weights does, the packed-operand cache
  puts us ahead on the transformer FFN shape as well.
- Away from GEMM the wins are ours and they come from the kernels:
  exp and tanh by an order of magnitude over NumPy and several times over
  PyTorch, softmax by 2×, column sums by 3.5×, max by 2.3×, transposes by
  1.8×, LayerNorm by 1.3×. Inference and the training step on the MLP are
  ahead; forward+backward is level.
- Two of the losses in earlier editions of this file are gone: `tanh` was
  5 GB/s through `math.Tanh` and is now 90, and LayerNorm was 3.37 ms and
  is now 1.81. What remains is GEMM below 1024² and the largest
  element-wise operations, where the gap is allocation rather than
  arithmetic and closes when the result is released.
- Highest-value next steps, in order: unpacked-B GEMM for small M,
  resetting the allocator between benchmark sections so the convolution
  figure is not an artifact of what ran before it, and re-measuring the
  x86 back-ends, which predate off-heap results entirely.
