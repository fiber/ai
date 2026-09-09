---
id: B-006
title: "smeprobe does not build outside darwin/arm64: missing build constraint breaks go test ./... on Linux"
status: done
scope:
  - internal/kernel/smeprobe/
manual: none
done: 2026-09-09
created: 2026-09-09
---

## Goal

`go test ./...` on the Xeon fails to build `internal/kernel/smeprobe`
(`syscall.SysctlUint32`, `syscall.SYS_SIGPROCMASK` are Darwin-only). The
probe is an Apple M4 experiment and must not break the build anywhere
else.

## Design

`main.go` gets `//go:build darwin && arm64`; a `main_other.go` with the
inverse constraint keeps the package buildable everywhere and prints
that the probe runs on Apple Silicon only. The assembly file already
carries the arm64 suffix.

## Acceptance

- `GOOS=linux GOARCH=amd64 go vet ./...` and `go build ./...` succeed;
  the probe still runs on the M2/M4.
- No performance impact.

## Notes

Fixed: `main.go` constrained to darwin/arm64, `main_other.go` prints a
notice elsewhere; `GOOS=linux GOARCH=amd64 go vet ./...` passes.

