---
id: B-001
title: Hook exit code 2 is lost when the gate runs through go run
status: done
scope:
  - cmd/gate/
  - .claude/
  - .gitignore
manual:
  - docs/manual/process.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

The Claude Code hook must actually block. `go run ./cmd/gate hook` exits
with status 1 whenever the program exits non-zero ("exit status 2" is
printed, the wrapper's own code is 1), and Claude Code treats exit code 1
as a non-blocking error: the message is shown, the edit still happens.
Found by pipe-testing the hook right after installing it.

## Design

The hook command builds the gate into an ignored binary and executes that
directly, so the process exit code reaches Claude Code unchanged:

    go build -o .claude/gate-bin ./cmd/gate || exit 2; exec .claude/gate-bin hook

A failed build exits 2 as well (fail closed). `.claude/gate-bin` is
gitignored. The git pre-commit hook keeps using `go run`: any non-zero
status aborts a commit, so it was never affected.

While here, the "write a spec first" message gains the `-manual` flag it
was missing.

## Acceptance

- `echo '{"tool_name":"Edit","tool_input":{"file_path":"<repo>/tensor/x.go"}}' | sh -c '<hook command>'`
  exits with status 2.
- The suggestion printed on a block includes `-manual`.
- `go test ./cmd/gate` passes.

## Notes

The settings watcher only picks up `.claude/settings.json` files that
existed when the session started; a new file needs `/hooks` or a restart
before the hook fires in the live session.
