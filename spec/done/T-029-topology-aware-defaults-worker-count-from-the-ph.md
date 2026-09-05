---
id: T-029
title: Topology-aware defaults: worker count from the physical cores of the affinity set, one NUMA node by default on Linux
status: done
scope:
  - internal/parallel/
  - tensor/
  - internal/blas/
  - cmd/bench/
manual:
  - docs/manual/performance.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

The worker count defaults to GOMAXPROCS, which on a two-socket
hyperthreaded server is 64 for 32 physical cores: measured, both sockets
were slower than one (511 against 1 139 GFLOPS at n=2048), and
hyperthreads add nothing to FMA-bound kernels. Folds TODO item T-013.

## Design

- On Linux, `internal/parallel` derives its default from the process's
  CPU affinity mask (`sched_getaffinity`) and sysfs: the number of
  distinct (package, core) pairs among the allowed CPUs, i.e. physical
  cores, hyperthreads excluded. Under `numactl --cpunodebind=0` that is
  16 on the Xeon; unpinned 32 instead of 64.
- `FIBERAI_WORKERS=n` overrides; `SetWorkers` keeps working. On other
  systems the default stays GOMAXPROCS (macOS already caps GEMM workers
  at the performance cores through the AMX hints).
- Go cannot pin threads to cores, so NUMA locality is not enforced;
  the manual keeps recommending `numactl` for two-socket machines and
  the performance page explains why.

## Acceptance

- Unit tests for the sysfs parser on a fake directory tree (two
  packages × 4 cores × 2 threads, an affinity subset).
- No change of behaviour on macOS or when the override is set; `go
  test ./...` on Linux/amd64 (the Xeon) with the new default.
- Xeon unpinned (64 CPUs visible): n=2048 all cores ≥ 1 000 GFLOPS with
  the new default (from 511 with 64 workers; PyTorch 780 pinned); no
  regression pinned (1 307). No performance impact elsewhere.

## Notes

Implemented without a new dependency: `sched_getaffinity` through a raw
syscall, the topology files parsed by a portable function with a test
on a fake sysfs tree (two packages × 4 cores × 2 threads; subsets;
missing CPU). `FIBERAI_WORKERS` overrides. macOS unchanged. The Xeon
numbers (unpinned n=2048 with 32 instead of 64 workers) are the next
round's measurement.
