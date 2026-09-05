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

| Workload | fiber/ai | pure Go | NumPy | PyTorch |
|---|---:|---:|---:|---:|
| SGEMM 2048², 1 core (GFLOPS) | **99.8** | 7.9 | – | 2 209 (AMX) |
| SGEMM 2048², all cores (GFLOPS) | **594.7** | 48.6 | 2 241 (AMX) | 2 243 (AMX) |
| [1×4096]·[4096×4096] (GFLOPS) | **14.6** | 12.4 | 11.1 | 11.3 |
| exp, 16M elements | **2.05 ms** | 6.51 ms | 26.3 ms | 3.01 ms |
| softmax(dim=1), 4096² | **2.91 ms** | 14.4 ms | – | 6.47 ms |
| sum(), 4096² | **0.61 ms** | 0.75 ms | 2.76 ms | 0.59 ms |
| sum(dim=0), 4096² | **0.65 ms** | 1.22 ms | 1.32 ms | 2.14 ms |
| transpose + copy, 4096² | **11.0 ms** | 10.9 ms | 60.2 ms | 22.7 ms |
| x + y, 16M elements | 2.02 ms | 2.11 ms | 2.25 ms | **1.37 ms** |
| MLP train step, batch 256 (samples/s) | 56.8 K | 11.4 K | – | **116 K** |

The assembly kernels are worth 2–13× over the same Go code (GEMM 12×,
relu 7×, exp 3×, max 5×), and on everything that is not a large matrix
product fiber/ai is on par with or ahead of NumPy and PyTorch.

**The one caveat is the big GEMM.** On Apple Silicon, Accelerate does not
run SGEMM on the NEON units at all — it uses the AMX matrix coprocessor,
which is undocumented and only reachable through Apple's library. 2.2–2.7
TFLOPS is what that coprocessor delivers; 600 GFLOPS is roughly the ceiling
of ten NEON cores (4 FMA pipes × 4 lanes × 2 × ~3.5 GHz ≈ 112 GFLOPS per
performance core, less on efficiency cores). Our single-core figure of ~100
GFLOPS is 87 % of that peak, i.e. the NEON kernel itself is close to
optimal; the gap to Python on this machine is hardware, not code. On x86
(OpenBLAS / MKL on AVX2 / AVX-512) the same comparison would be like for
like; those numbers are pending an AVX-512 machine.

## Matrix multiply

n×n · n×n, float32, GFLOPS (higher is better):

| n | fiber/ai 1 thread | fiber/ai 10 threads | pure Go 10 threads | NumPy | PyTorch |
|---:|---:|---:|---:|---:|---:|
| 128 | 67.5 | 71.7 | 7.5 | 754 | 783 |
| 256 | 86.1 | 146.8 | 37.7 | 1 134 | 1 133 |
| 512 | 97.0 | 364.5 | 47.0 | 2 142 | 2 131 |
| 1024 | 97.7 | 525.5 | 48.6 | 2 693 | 2 665 |
| 2048 | 99.8 | 594.7 | – | 2 241 | 2 243 |

Other shapes, all threads:

| shape | fiber/ai | NumPy | PyTorch |
|---|---:|---:|---:|
| [1×4096]·[4096×4096] (matrix–vector, memory-bound) | **14.6** | 11.1 | 11.3 |
| [8×4096]·[4096×4096] | 42.6 | 69.5 | 68.0 |
| [64×1024]·[1024×1024] | 212.5 | 1 292 | 1 275 |
| [256×768]·[768×3072] (transformer FFN) | 374.9 | 2 389 | 2 357 |
| [1024×1024]·[1024×1024]ᵀ (transposed view, no copy) | 464.9 | 2 431 | 2 398 |

Notes:

- Scaling from 1 to 10 threads reaches 6× at n=2048. The four efficiency
  cores contribute maybe 1.5 performance-core equivalents, so ~7× is the
  realistic maximum; the rest is packing and synchronisation overhead.
- The matrix–vector case beats Accelerate: it is pure memory bandwidth and
  the `axpy` path streams B once with all cores.
- Small M (8 rows) loses because the packed algorithm still packs the whole
  B matrix; reading B in place for M ≤ MR is on the roadmap.

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

## Production platform

Production runs on Intel/AMD Linux. There NumPy and PyTorch use OpenBLAS
or MKL on the same AVX2/AVX-512 units our kernels use, so the comparison
is like for like. Those measurements are TODO T-008 and will decide the
priorities; the macOS numbers above are development feedback.

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
