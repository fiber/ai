---
id: B-003
title: Gate hook depends on the shell working directory and locks the session out after cd
status: done
scope:
  - cmd/gate/
  - .claude/
manual:
  - docs/manual/process.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

The hook command built the gate relative to the current directory
(`go build ./cmd/gate`). After a `cd tensor` in the session shell every
hooked tool — Bash, Edit, Write, even a subagent, which inherits the same
directory — failed with "stat .../tensor/cmd/gate: directory not found"
and exit code 2, so nothing could be run or edited, including the fix.
The session was unblocked by hand from a terminal on the machine.

## Design

Two changes so this cannot recur:

- `.claude/settings.json`: the hook command first changes to
  `$(git rev-parse --show-toplevel)`, so building and running the gate
  never depends on where the shell stands. (Applied by hand to unblock;
  recorded here.)
- `cmd/gate`: the hook input JSON carries `cwd`, the directory the tool
  call runs in. `gate hook` resolves relative file paths and Bash write
  targets against that directory instead of `os.Getwd()`, which after the
  settings change is always the repository root. Without a `cwd` field
  the current directory is used as before.

## Acceptance

- With the shell in a subdirectory, a hooked Bash command that writes a
  relative path is checked against the right repository path (unit test:
  hook JSON with `cwd` set to a subdirectory of the temp repo).
- A `cd` in the session shell no longer disables the hook (verified by
  hand once).
- `docs/manual/process.md` mentions that the hook is directory-independent.

## Notes
