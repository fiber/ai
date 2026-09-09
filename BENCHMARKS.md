# Benchmarks

All numbers measured on the same machine, one after another, on 2026-09-05:

| | |
|---|---|
| Machine | Apple M2 Pro (6 performance + 4 efficiency cores), macOS 26 (Darwin 25.6) |
| fiber/ai | Go 1.26.2, NEON back-end, 10 goroutines (`darwin/arm64`) |
| fiber/ai (generic) | same code with `FIBERAI_KERNEL=generic` — pure Go, no assembly |
| NumPy | 2.0.2 (Accelerate BLAS), Python 3.9.6 |
| PyTorch | 2.8.0 CPU (Accelerate BLAS, SLEEF vector math, 6 threads by default) |

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
released) against NumPy 2.5.3 and PyTorch 2.14 on Accelerate, 8 September
2026, both runs in `benchmarks/results/m2pro-2026-09-08/`. Xeon Gold 6130,
one socket pinned with `numactl`: fiber/ai AVX-512 from the same day,
PyTorch 2.14+cpu with MKL and oneDNN (16 threads) from the same session
for attention, convolution and EmbeddingGemma and from the x86 round
for the rest. Bold marks the faster side; "level" is within 5 %.

| Workload | M2 Pro fiber/ai | M2 Pro PyTorch | Xeon fiber/ai | Xeon PyTorch |
|---|---:|---:|---:|---:|
| SGEMM 2048², all cores (GFLOPS) | 2 269 (level) | 2 227 | **1 156** | 780 |
| SGEMM 1024² | 2 135 (2 471 with B packed once) | **2 682** | 1 217 (1 327 with B packed once) | **1 434** |
| SGEMM 512² | 1 509 | **2 154** | **1 002** | 991 (level) |
| SGEMM 256² | 916 | **1 116** | 540 | **627** |
| SGEMM 128² | 369 | **755** | 99 | **236** |
| [256×768]·[768×3072], B packed per call | 1 795 | **2 358** | **1 111** | 980 |
| [256×768]·[768×3072], B packed once (weights) | **2 594** | 2 358 | **1 331** | 980 |
| [1×4096]·[4096×4096] | **13.4** | 11.3 | **13.0** | 11.0 |
| x + y, 1M (released) | **43 µs** | 68 µs | **17 µs** | 24 µs |
| x + y, 16M (released) | 1.51 ms | **1.37 ms** | **13.9 ms** | 17.5 ms |
| exp, 16M | **1.60 ms** | 3.09 ms | **10.7 ms** | 13.8 ms |
| tanh, 1M | **123 µs** | 719 µs | 484 µs | **54 µs** |
| sum(), 4096² | 616 µs (level) | 596 µs | 2.40 ms (level) | 2.42 ms |
| sum(dim=0), 4096² | **744 µs** | 2.18 ms | **2.68 ms** | 6.29 ms |
| max(dim=1), 4096² | **575 µs** | 1.18 ms | **2.40 ms** | 2.46 ms |
| softmax(dim=1), 4096² | **2.22 ms** | 6.27 ms | **11.0 ms** | 14.4 ms |
| layernorm, 4096² | **1.85 ms** | 2.21 ms | **11.0 ms** | 14.1 ms |
| transpose + copy, 4096² | **10.9 ms** | 22.4 ms | **25.5 ms** | 68.5 ms |
| attention [8×8×512×64] (GFLOPS) | **1 110** | 553 | 655 | **1 141** |
| conv2d [32×64×56×56]·64×3×3 (GFLOPS) | 337 (level) | 311 | 73 | **438** |
| MLP forward, batch 256 (samples/s) | **875 K** | 526 K | **473 K** | 361 K |
| MLP forward + backward | 179 K | **221 K** | 74 K | **124 K** |
| MLP train step (Adam) | **148 K** | 121 K | 71 K (level) | 71 K |
| EmbeddingGemma, 32 × 65 tokens (sentences/s) | **106** | 88 | **59** | 37 |

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
  it to 618 pinned. Still 1.8× behind oneDNN's fused attention; what
  is left there is the AVX2 exponential (an AVX-512 version is the
  candidate) and the micro-kernel's own 60 % of peak.
  The im2col convolution is memory-bound on the Xeon; oneDNN has a
  dedicated primitive, the implicit-GEMM candidate is on the list.
- **tanh on the Xeon** (9× behind in the 1M row): not the kernel. With
  the AVX-512 exp and tanh (T-043) one Xeon core does 0.27 and 0.32 ns
  per element (AVX2: 0.51 and 0.57), yet every 1M element-wise row
  costs the same 480 µs, `relu` and `x * 2.5` included, while the
  released variant of `x + y` takes 23 µs. The row measures the
  recycling of unreleased 4 MB results, a forced GC every 64 of them,
  which the Xeon pays far more dearly than the M2 (see the allocator
  note below the Xeon table).

The M2 inference rows and the "B packed once" entries were taken after
T-037 (packed-operand cache); the rest of the table is from the run of
the same morning. Earlier headline tables quoted single best runs (M2 GEMM 2 301, train
164 K); the M2 varies by about 10 % between runs with the same code, so
this table uses one run of each side taken minutes apart.

## Matrix multiply

n×n · n×n, float32, GFLOPS (higher is better), M2 Pro:

| n | AMX 1 thread | AMX all cores | NEON 1 thread | NEON 10 threads | pure Go 10 threads | NumPy | PyTorch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 357 | 376 | 70 | 73 | 7.5 | 754 | 783 |
| 256 | 770 | 1 003 | 90 | 280 | 37.7 | 1 134 | 1 133 |
| 512 | 1 091 | 1 550 | 99 | 487 | 47.0 | 2 142 | 2 131 |
| 1024 | 1 195 | 2 183 | 100 | 592 | 48.6 | 2 693 | 2 665 |
| 2048 | 1 038 | **2 301** | 99 | 622 | – | 2 241 | 2 243 |

Other shapes, all threads:

| shape | AMX | NEON | NumPy | PyTorch |
|---|---:|---:|---:|---:|
| [1×4096]·[4096×4096] (matrix–vector, memory-bound) | **13.8** | 13.8 | 11.1 | 11.3 |
| [8×4096]·[4096×4096] | **75.6** | 55 | 69.5 | 68.0 |
| [64×1024]·[1024×1024] | 1 004 | 332 | **1 292** | 1 275 |
| [256×768]·[768×3072] (transformer FFN) | 1 835 | 519 | **2 389** | 2 357 |
| [1024×1024]·[1024×1024]ᵀ (transposed view, no copy) | 2 158 | 574 | **2 431** | 2 398 |

Notes:

- The AMX tile alone reaches 1.55 TFLOPS on one thread and 3.3 TFLOPS on
  the six performance cores; the driver (packing, one barrier per K
  block) is what separates 2.3 from that at n=2048 and more at small n.
  The M4, with a single performance cluster, reaches 1.4 TFLOPS on one
  thread and 1.7 on all (PyTorch there uses SME: 1.9).
- NEON scaling from 1 to 10 threads reaches 6× at n=2048; the four
  efficiency cores are worth about 1.5 performance cores.
- The matrix–vector case is pure memory bandwidth and the `axpy` path
  streams B once with all cores; both back-ends beat Accelerate there.
- Small M (8 rows) with NEON still packs the whole B matrix; reading B
  in place for M ≤ MR is on the roadmap (T-005). AMX is ahead of
  Accelerate on that shape anyway.

## Element-wise operations

GB/s (higher is better); pure Go is fiber/ai with `FIBERAI_KERNEL=generic`.

| op | n | fiber/ai | pure Go | NumPy | PyTorch |
|---|---:|---:|---:|---:|---:|
| x + y | 16M | 99.6 | 95.3 | 89.4 | 146.7 |
| x * y | 16M | 100.7 | 96.2 | 90.0 | 147.1 |
| x + row (broadcast) | 16M | 71.5 | 63.0 | 46.9 | 159.0 |
| x * 2.5 | 16M | 88.5 | 82.2 | 92.6 | 157.0 |
| exp(x) | 16M | **65.4** | 20.6 | 5.1 | 44.6 |
| tanh(x) | 16M | 5.0 | 5.0 | 8.6 | 11.6 |
| relu(x) | 16M | 89.3 | 13.4 | 24.5 | 157.5 |
| x + y | 1M | 73.0 | 59.5 | 125.4 | 175.0 |
| exp(x) | 1M | 37.2 | 14.7 | 5.1 | 38.8 |
| relu(x) | 1M | 49.2 | 9.7 | 24.1 | 122.6 |

Time per call for small arrays (64K elements), where fixed overhead
dominates:

| op | fiber/ai | pure Go | NumPy | PyTorch |
|---|---:|---:|---:|---:|
| x + y | 27.7 µs | 41.2 µs | **7.1 µs** | 30.9 µs |
| exp(x) | 50.4 µs | 83.3 µs | 103.2 µs | **46.4 µs** |

Notes:

- At 16M elements every implementation is bound by memory bandwidth.
  PyTorch's edge on the simple ops (150 vs 100 GB/s) comes from its caching
  allocator: it reuses the output buffer, whereas each Go call gets a fresh,
  zero-filled 64 MB allocation from the runtime — an extra write pass plus
  page faults. Streaming the same op into an existing tensor with
  `AddInPlace` avoids this.
- `exp` is a hand-written NEON kernel (Cody–Waite reduction + degree-6
  polynomial, ≈ 1 ulp); it is 13× faster than NumPy and ahead of PyTorch's
  SLEEF. `tanh` (and GELU, which uses it) still go through `math.Tanh` per
  element and are the slowest thing in the table — next on the list.
- The 64K case shows the cost of Go's allocator for a 256 KB result (page
  faults on freshly mapped memory) relative to NumPy's single-threaded,
  buffer-recycling loop. It is still on par with PyTorch, whose dispatcher
  overhead is similar in size.

## Reductions on a 4096×4096 matrix

Time per call (lower is better):

| op | fiber/ai | pure Go | NumPy | PyTorch |
|---|---:|---:|---:|---:|
| sum() | 614 µs | 751 µs | 2.76 ms | **593 µs** |
| sum(dim=0) | **653 µs** | 1.22 ms | 1.32 ms | 2.14 ms |
| sum(dim=1) | 603 µs | 752 µs | 2.73 ms | **576 µs** |
| max(dim=1) | **588 µs** | 3.18 ms | 1.17 ms | 1.14 ms |
| softmax(dim=1) | **2.91 ms** | 14.4 ms | – | 6.47 ms |
| layernorm (last dim) | 3.37 ms | 7.29 ms | – | **2.35 ms** |
| transpose + contiguous copy | **11.0 ms** | 10.9 ms | 60.2 ms | 22.7 ms |

Row reductions run at ~110 GB/s, i.e. at memory speed. `sum(dim=0)` folds
rows into per-goroutine partial sums so it streams the matrix once (NumPy
and PyTorch both do noticeably worse here). LayerNorm makes five passes over
each 16 KB row; fusing mean/variance into one pass would close the gap to
PyTorch.

## Training step: MLP 784 → 512 → 512 → 10, batch 256, Adam

| phase | fiber/ai | pure Go | PyTorch |
|---|---:|---:|---:|
| forward (no grad) | 1.51 ms · 170 K samples/s | 9.53 ms · 27 K | **0.48 ms · 532 K** |
| forward + backward | 4.24 ms · 60 K | 23.6 ms · 11 K | **1.17 ms · 219 K** |
| forward + backward + Adam step | 4.50 ms · 57 K | 22.4 ms · 11 K | **2.21 ms · 116 K** |

The step is ~90 % matrix products (340 MFLOP forward, ~2× that backward),
so this table is the GEMM table again: PyTorch's forward runs at ~700 GFLOPS
on AMX, ours at ~225 GFLOPS on NEON. Two things are worth noting. The
autograd machinery itself is cheap: forward+backward costs 2.8× the forward
in fiber/ai, the theoretical 3× for the GEMMs involved. And the optimiser
step is where Python overhead shows — PyTorch spends 1.04 ms on Adam for
668 K parameters, fiber/ai 0.26 ms.

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

## Apple AMX back-end (M2 Pro, opt-in `FIBERAI_AMX=1`)

Accelerate's SGEMM numbers on the Macs come from the AMX coprocessor;
spec T-010 puts fiber/ai's GEMM tile on the same unit (Go assembly with
the instruction words documented by github.com/corsix/amx). Same
`cmd/bench`, results released as in the other tables:

| Workload | NEON (10 cores) | AMX | NumPy | PyTorch |
|---|---:|---:|---:|---:|
| SGEMM 512², all cores (GFLOPS) | 514 | 1 409 | **2 142** | 2 131 |
| SGEMM 1024² | 581 | 1 881 | **2 693** | 2 665 |
| SGEMM 2048² | 636 | **2 250** | 2 241 | 2 243 |
| SGEMM 1024², 1 thread | 96 | 1 049 | – | 2 690 (Accelerate ignores the thread limit) |
| [64×1024]·[1024²] | 332 | 928 | **1 292** | 1 275 |
| [256×768]·[768×3072] | 500 | 1 562 | **2 389** | 2 357 |
| [1024²]·[1024²]ᵀ (view) | 562 | 1 893 | **2 431** | 2 398 |
| MLP inference (samples/s) | 221 K | **569 K** | – | 532 K |
| MLP forward + backward | 96 K | 162 K | – | **219 K** |
| MLP training step | 85 K | **146 K** | – | 116 K |

The raw tile reaches 1.55 TFLOPS on one thread and 3.3 TFLOPS on six
(the units sit with the six performance cores; the efficiency cores'
unit is slow), so the driver — packing, one barrier per K block, the C
tile in and out of Z once per K block — is what separates 2.25 from
Accelerate's 2.7 at n=1024 and below. Two findings on the way: the
coprocessor executes in order, so the loads of step k+4 must be issued
before the outer products of step k (four X/Y register slots; 3× faster
than the naive loop), and `set`/`clr` per tile costs 0.7 µs, hence once
per task with the goroutine locked to its thread.

## Attention, convolution and EmbeddingGemma (M2 Pro, AMX)

Rows added with the transformer work; PyTorch 2.14 (Accelerate BLAS, 6
threads by default) via `benchmarks/python/bench.py --only
attention,conv,embed`, fiber/ai via `go run ./cmd/bench -only
attention,conv,embed`. Attention and convolution run under `NoGrad` with
the result released; EmbeddingGemma is float32 on both sides, 32 copies
of a 65-token sentence as one batch.

| row | PyTorch | fiber/ai |
|---|---:|---:|
| attention [8×8×512×64], GFLOPS over the two products | 561 | 1 110 |
| same with a causal mask | 566 | 1 070 |
| attention [1×8×2048×64], long sequence | 652.5 | 1 120 |
| conv2d [32×64×56×56] · 64 filters 3×3, GFLOPS | 314 | 367 |
| EmbeddingGemma, sentences/s | 91 | 106 |

The embedding row is the one that matters for the syslog work: a full
24-layer encoder, tokenizer to unit vector, a fifth ahead of the Python
stack on the same machine. Its parity with `sentence-transformers` is
cosine 1.000000 on 64 short and 3 long (up to 1753-token) inputs; see
[docs/manual/models.md](docs/manual/models.md).

## Allocation cost (why the element-wise numbers are what they are)

Every element-wise operation allocates its result; Go zero-fills it and,
once the GC has released earlier results to the OS, the pages fault back
in on the next write. On the M2 Pro, raising `GOGC` from 100 to 800 —
which keeps freed memory resident instead of returning it — changes
nothing in the kernels and this much in the timings:

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
caller-provided or pooled buffers (TODO T-003) is the fix; it is the
single most valuable change for x86 deployments.

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
| SGEMM 2048², all cores, tensor level | 1 156 | – |
| [256×768]·[768×3072] | 1 016 | – |
| MLP forward, batch 256 (samples/s) | 288 K → 384 K → 415 K (T-040) → **473 K** (T-042) | 361 K |
| MLP forward + backward (samples/s) | 65 K → 72 K | 124 K |
| MLP train step (samples/s) | 64 K → 67 K | 71 K |
| attention [8×8×512×64] (GFLOPS) | 479 → 533 (T-041) → 618 (T-042) → 655 (T-043) | **1 141** |
| same with causal mask | 470 (was 39) → 515 → 595 → 662 | **1 129** |
| attention [1×8×2048×64], long sequence | 539 → 698 → 748 | **1 141** |
| conv2d [32×64×56×56]·64×3×3 (GFLOPS) | 73 | **438** |
| EmbeddingGemma, 32 × 65 tokens (sentences/s) | **49 → 58** (T-040, pre-norms folded, gated FFN fused) | 37 |
| [64×1024]·[1024×1024], B packed once (T-037) | 551 → **699** (541 per call, T-042) | 699 (level) |
| [8×4096]·[4096×4096], B packed once | **89** (58 per call) | 58 |
| [256×768]·[768×3072], B packed once | 1 240 → **1 331** (1 111 per call, T-042) | 980 |
| EmbeddingGemma with the cache warm (all hits) | 43–47, no gain on this machine | 37 |
| cluster.Cosine, 768-d pair | **103 ns** | NumPy 4 332 ns |
| exp / tanh, one core, ns per element (kernel benchmark) | AVX2 0.51 / 0.57 → **AVX-512 0.27 / 0.32** (T-043) | – |

The element-wise 1M rows on this machine (about 480 µs whatever the
operation, `relu` and `x * 2.5` included; `x + y` released 23 µs) do not
measure kernels: an unreleased 4 MB result is reclaimed by a forced GC
once 256 MiB are outstanding, one GC per 64 results, and its cost on 32
hardware threads is what the row shows. `Release()` removes it; the
released rows are the kernel comparison. Whether the forced collection
can be made cheaper or rarer on Linux is an open item (TODO).
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

- The NEON kernels are close to the hardware limits: 87 % of FMA peak for
  GEMM on one core, memory bandwidth for the streaming ops.
- Against NumPy, fiber/ai is faster on everything except a large GEMM on
  Apple Silicon, where NumPy is really Apple's AMX coprocessor.
- Against PyTorch, fiber/ai wins on exp, softmax, column sums, max and
  transposes, ties on row reductions, and loses on large GEMM (AMX), large
  simple element-wise ops (allocator) and tanh/GELU (not yet vectorised).
- Highest-value next steps, in order: vectorised tanh/log, fused
  LayerNorm statistics, unpacked-B GEMM for small M, and measuring the
  AVX2/AVX-512 back-ends on x86 against OpenBLAS/MKL for a like-for-like
  GEMM comparison.
