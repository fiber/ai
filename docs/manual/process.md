# Development process

The authoritative rules are in [PROCESS.md](../../PROCESS.md) at the
repository root; this page is the practical guide.

## The four lists

- `TODO.md` — open work items, `- [ ] T-nnn — title (spec/...)`.
- `BUGS.md` — open bugs, `- [ ] B-nnn — title (spec/...)`. Record a bug
  here as soon as it is known, even without a spec.
- `DONE.md` — completed work items, newest first, each pointing to its
  spec in `spec/done/`.
- `BUGS-FIXED.md` — fixed bugs, newest first, same format. `gate done`
  sorts an entry into the right file by its id.

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

**Beat Python (rule 7).** When the scope touches `internal/`, `tensor/`,
`nn/` or `optim/`, the Acceptance section must name the NumPy/PyTorch
figure for the same workload and the target relative to it — or state
"no performance impact". The gate rejects the spec otherwise, and `gate
new` reminds you when the scope is performance-relevant.

Scope patterns: `dir/` covers everything below the directory; other
patterns are matched as globs against the repository path (`*.s` matches
any assembly file).

## The gate

`go run ./cmd/gate` enforces the rules:

| Command | Does |
|---|---|
| `gate check [-staged]` | validates every spec, the three lists, the manual index and the coverage of all changed (or staged) files |
| `gate new [-bug] -scope a/,b.go -manual page\|none title...` | allocates an id, writes the spec from `spec/TEMPLATE.md`, adds the list entry |
| `gate done T-012` | sets `status: done`, moves the spec to `spec/done/`, removes the TODO/BUGS item and adds it to DONE.md (T) or BUGS-FIXED.md (B) |
| `gate install` | activates the git pre-commit hook for this clone |
| `gate hook` | the Claude Code PreToolUse hook (reads JSON on stdin) |

It runs automatically in two places:

- **Before every commit** (`.githooks/pre-commit`): the staged change set
  must pass `gate check -staged`. `git commit --no-verify` is not used.
- **Before every edit and shell command in Claude Code**
  (`.claude/settings.json`): an Edit/Write to a code file with no covering
  open spec is refused with the missing spec named. A Bash command is
  checked by its **write targets**: the file after `>`/`>>`, the arguments
  of `tee`, `sed -i`, `mv`, `cp`, `rm`, `gofmt -w`, `git mv`/`git rm`, and
  Python file writes — `open` in write mode, `Path.write_*`,
  `shutil.copy/move`, `os.rename/remove` (a variable is resolved through
  one literal assignment in the same command). Text inside heredoc bodies
  is not a shell target, so specs and manual pages may talk about code
  freely; Python write calls are recognised inside heredocs, because that
  is where scripts live, so documentation should not spell one out
  verbatim. A target the gate cannot resolve (a shell or Python variable)
  is refused with "use a literal path". `--no-verify` commits are refused. The hook builds the gate into
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
