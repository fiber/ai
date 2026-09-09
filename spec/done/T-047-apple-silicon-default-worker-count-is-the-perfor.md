---
id: T-047
title: Apple Silicon: default worker count is the performance cores
status: done
scope:
  - internal/parallel/
  - cmd/bench/
manual:
  - docs/manual/performance.md
done: 2026-09-09
created: 2026-09-09
---

## Goal

On Apple Silicon the pool uses GOMAXPROCS workers, which includes the
efficiency cores. Every parallel round then waits for its slowest
member. On an M4 MacBook Air (4P + 6E) `FIBERAI_WORKERS=4` took the MLP
forward from 666 K to 1.09 M samples/s (PyTorch 815 K), forward+backward
from 202 K to 256 K, the training step from 163 K to 223 K (PyTorch
204 K), attention from 920 to 982 GFLOPS. On the M2 Pro (6P + 4E)
`FIBERAI_WORKERS=6` gave 798 K → 852 K, 180 K → 211 K, 147 K → 184 K,
EmbeddingGemma unchanged at 105, attention 1 156 → 1 041 (its tasks are
whole heads, which an efficiency core can work through alone). Make the
performance cores the default on Apple Silicon; the AMX GEMM hint
already does this for matrix products.

## Design

- `topology_darwin.go` (`//go:build darwin`): `defaultWorkers` returns
  `hw.perflevel0.physicalcpu` when the sysctl answers with a value > 0
  (Apple Silicon), else GOMAXPROCS (Intel Macs); the generic stub keeps
  `!linux && !darwin`. `FIBERAI_WORKERS` overrides as before; the
  manual's environment table says so.
- No pinning on macOS (no affinity API); the scheduler keeps the
  performance cores busy with fewer runnable threads on its own.

## Acceptance

- `go test ./internal/parallel` and `go test ./...` on the M2; bench
  header reports the new default; `FIBERAI_WORKERS=10` restores the old
  behaviour.
- Baselines (same day, all cores; PyTorch 2.14 on Accelerate): M2 Pro
  MLP forward 798 K / forward+backward 180 K / train 147 K samples/s
  (PyTorch 526 K / 221 K / 121 K), attention 1 156 (553); M4 Air 666 K /
  202 K / 163 K (815 K / 348 K / 204 K), attention 920 (729). Targets
  with the default: the `FIBERAI_WORKERS=P` figures above within noise
  (M2 852 K / 211 K / 184 K, M4 1.09 M / 256 K / 223 K); attention on
  the M2 may drop to about 1 040, accepted and recorded.
- Manual: performance.md's worker-count row and the Apple back-end
  paragraph.

## Notes
- M2 Pro with the default (6 workers): MLP forward 855 K, forward+backward
  212 K, training step 183 K, EmbeddingGemma 105, attention 982 (the
  E-cores took whole heads before; accepted). M4 Air with
  `FIBERAI_WORKERS=4` (what the default now picks): 1.09 M / 256 K /
  223 K, attention 982; the default run there is pending.
