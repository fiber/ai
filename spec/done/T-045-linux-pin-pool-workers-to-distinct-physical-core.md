---
id: T-045
title: Linux: pin pool workers to distinct physical cores of the affinity mask
status: done
scope:
  - internal/parallel/
manual:
  - docs/manual/performance.md
done: 2026-09-09
created: 2026-09-09
---

## Goal

On the pinned Xeon socket (16 physical cores, 32 hardware threads) the
pool runs 16 workers on 32 logical CPUs and the Go scheduler knows
nothing about hyperthread siblings. Two workers landing on the two
threads of one core share its FMA ports while another core idles.
Measured with 2048² SGEMM: `numactl --cpunodebind=0` 1 102–1 177
GFLOPS, the same run with `--physcpubind` to one CPU per physical core
1 309 GFLOPS (+19 %), nothing else changed. The scaling curve says the
same: 93 % efficiency with 2 workers, 74 % with 4, 56 % with 12, and the
last four cores add nothing. Pin the pool's worker threads to distinct
physical cores of the process's affinity mask, so that placement is
right without numactl incantations.

## Design

- **`coreCPUs(root, cpus)`** (topology.go, testable with the fake
  sysfs): one CPU per (package, core) among the affinity mask's CPUs,
  the lowest CPU number of each core, ordered by package then core id,
  so with fewer workers than cores one package fills first.
- **Linux** (`topology_linux.go`): `pinCPUs()` computes that list once
  from `affinityCPUs()`; `pinThread(cpu)` calls `sched_setaffinity(0,
  …)` for the calling thread. Each pool helper locks itself to its OS
  thread at start and pins to list[id], helper ids being 1-based so
  list[0] stays for the caller's goroutine, which also works on every
  job and is not pinned (user code). Helpers beyond the list run
  unpinned. A failed syscall leaves the helper unpinned. `FIBERAI_PIN=0`
  disables pinning; other platforms have stubs (macOS has no affinity
  API; the AMX back-end already locks its threads per task).
- Memory placement across NUMA nodes is not part of this: with the
  default unpinned mask (both sockets) the workers spread over both
  packages' cores and first-touch decides where pages live, as today.

## Acceptance

- `coreCPUs` returns one CPU per core in package order for the fake
  16-CPU topology (two packages × 4 cores × 2 threads) and subsets of
  it; `go test ./internal/parallel` on both machines; `go test ./...`
  on the Xeon; `GOOS=linux go vet`.
- Baselines (Xeon Gold 6130 one socket via `numactl --cpunodebind=0`,
  2026-09-09; PyTorch 2.14/MKL same machine): SGEMM 2048² 1 147–1 177
  GFLOPS (MKL 780; 1 309 with physcpubind), 1024² 1 105–1 147 (MKL
  1 434), MLP train step 71 K samples/s (PyTorch 71 K), attention 655
  (PyTorch 1 141); unpinned default (both sockets) 887 GFLOPS at 2048²
  and 46 K training. Targets with pinning on and plain
  `--cpunodebind=0`: 2048² ≥ 1 280, 1024² ≥ 1 220, train ≥ 78 K,
  attention ≥ 700; the unpinned default not worse than today; M2 rows
  unchanged (no pinning there).
- Manual: performance.md's back-end/worker table gets `FIBERAI_PIN` and
  a sentence on what pinning is worth.

## Notes

- First Xeon run (343738d, pinning always on): one socket 2048² 1 265
  (target 1 280 missed by 1 %; physcpubind 1 309), 1 024² 1 237 (met),
  attention 693 (target 700 missed by 1 %), training 67 K (71 before,
  target 78 K missed: the step is small parallel rounds, FMA contention
  was never its cost); two-socket default with pinning 2048² 795–803
  and training 39.0 K. A run of the previous bundle was first mistaken
  for the unpinned comparison (and the pinning briefly called neutral);
  the real one, same day and code with pinning off across packages
  (bb71d1e): 921 and 43.8 K. Pinning across two sockets is 11–14 %
  worse. Consequence: pin by default only within one package, where it
  is measured to help; FIBERAI_PIN=1 forces it across packages.
  NUMA-local memory placement remains the open item for the two-socket
  case. (The 887 / 46 K of the day before: another day on a shared
  machine; two-socket figures vary about 10 % between days.)
- Control run after the change (bb71d1e): one socket 1024² 1 240, 2048²
  1 269; two sockets 921 and 43.8 K. Closed with 2048² and attention 1 %
  under their targets and training unchanged.
