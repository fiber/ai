# Development process

The authoritative rules are in [PROCESS.md](../../PROCESS.md) at the
repository root; this page is the practical guide.

## The three lists

- `TODO.md` — open work items, `- [ ] T-nnn — title (spec/...)`.
- `BUGS.md` — open bugs, `- [ ] B-nnn — title (spec/...)`. Record a bug
  here as soon as it is known, even without a spec.
- `DONE.md` — completed items and fixed bugs, newest first, each pointing
  to its spec in `spec/done/`.

## Specs

A spec is a Markdown file in `spec/` named `<ID>-<slug>.md` with front
matter:

```
---
id: T-012
title: Vectorised tanh kernel
status: open            # open | done
scope:                  # what this spec allows to change
  - internal/kernel/
  - tensor/ops_unary.go
manual:                 # manual pages that must change with it, or none
  - docs/manual/performance.md
created: 2026-09-06
---
```

followed by `## Goal`, `## Design`, `## Acceptance` and optionally
`## Notes`. Write it before touching code; keep it honest while coding; it
is the record of what was decided and why.

Scope patterns: `dir/` covers everything below the directory; other
patterns are matched as globs against the repository path (`*.s` matches
any assembly file).

## The gate

`go run ./cmd/gate` enforces the rules:

| Command | Does |
|---|---|
| `gate check [-staged]` | validates every spec, the three lists, the manual index and the coverage of all changed (or staged) files |
| `gate new [-bug] -scope a/,b.go -manual page\|none title...` | allocates an id, writes the spec from `spec/TEMPLATE.md`, adds the list entry |
| `gate done T-012` | sets `status: done`, moves the spec to `spec/done/`, updates TODO/BUGS and DONE |
| `gate install` | activates the git pre-commit hook for this clone |
| `gate hook` | the Claude Code PreToolUse hook (reads JSON on stdin) |

It runs automatically in two places:

- **Before every commit** (`.githooks/pre-commit`): the staged change set
  must pass `gate check -staged`. `git commit --no-verify` is not used.
- **Before every edit and shell command in Claude Code**
  (`.claude/settings.json`): an Edit/Write to a code file with no covering
  open spec is refused with the missing spec named; a Bash command that
  redirects into, moves, copies or deletes a code file is checked the same
  way, and `--no-verify` commits are refused. The hook builds the gate into
  the ignored binary `.claude/gate-bin` and runs that, because a `go run`
  wrapper would turn the blocking exit code 2 into a 1. A settings file
  created during a session becomes active after `/hooks` or a restart.

What counts as code: everything except `*.md`, `spec/`, `.claude/`,
`.githooks/`, `benchmarks/results/`, `.gitignore` and `orig/`.

## The manual

`docs/manual/` is part of the deliverable. Every spec names the pages it
changes; `gate check` fails when a spec is completed without those pages
being part of the same change, and when a manual page is not linked from
`docs/manual/README.md`. A spec that changes nothing user-visible declares
`manual: none`.

## Typical session

```sh
go run ./cmd/gate new -scope internal/kernel/,tensor/ops_unary.go \
    -manual docs/manual/performance.md "Vectorised tanh kernel"
$EDITOR spec/T-012-vectorised-tanh-kernel.md
# implement, test, update docs/manual/performance.md
go run ./cmd/gate done T-012
go run ./cmd/gate check
git add -A && git commit -m "Add vectorised tanh kernel"
```
