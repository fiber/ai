---
id: T-001
title: Process gate: lists, specs and enforcement tool
status: done
scope:
  - cmd/gate/
  - .githooks/
  - .claude/
manual:
  - docs/manual/README.md
  - docs/manual/process.md
done: 2026-09-05
created: 2026-09-05
---

## Goal

Every change to the code base is preceded by a written spec and tracked in
TODO.md / BUGS.md / DONE.md, and this is enforced mechanically rather than
by discipline: neither an editor session driven by Claude Code nor a git
commit can modify code that no open spec covers.

## Design

`cmd/gate` is a dependency-free Go program with these commands:

- `check [-staged]` — validates spec files (front matter, required
  sections, id/filename/directory consistency), the three lists (every open
  spec listed, every done spec in DONE.md and absent from TODO/BUGS) and
  that every changed non-documentation file is covered by the scope of an
  open spec or of a done spec that is itself part of the change set.
  Changed = staged files with `-staged`, otherwise all working-tree changes
  including untracked files.
- `hook` — reads the Claude Code PreToolUse JSON from stdin. For
  Edit/Write/MultiEdit/NotebookEdit it checks the target file; for Bash it
  extracts paths that the command writes to (redirections, `tee`, `sed -i`,
  `mv`, `cp`, `rm`, `gofmt -w`, Python `open(..., 'w')`) and checks each.
  It also refuses `git commit --no-verify`. Exit 2 with an explanatory
  message blocks the tool call.
- `new [-bug] -scope a/,b.go title...` — allocates the next id, writes a
  spec from `spec/TEMPLATE.md` and adds the list entry.
- `done <id>` — sets `status: done` and `done: <date>`, moves the file to
  `spec/done/`, removes the TODO/BUGS line and prepends the DONE.md line.
- `install` — sets `core.hooksPath` to `.githooks`.

The manual under `docs/manual/` is enforced too: every spec declares the
manual pages it changes (`manual:` list, or `none`); when a spec is
completed the gate requires those pages to be in the same change set, and
every page must be linked from `docs/manual/README.md`.

Scope patterns: a trailing `/` matches a directory prefix, otherwise the
pattern is matched with `path.Match` against the repo-relative path (and
against the base name when it contains no `/`).

Exempt from coverage: `*.md`, `spec/`, `.claude/`, `.githooks/`,
`benchmarks/results/`, `.gitignore`, `orig/`.

Rejected alternatives: a prompt/LLM hook (non-deterministic, slow); a
Makefile-only check (does not stop edits, only commits); putting the rules
in CLAUDE.md alone (not enforced).

## Acceptance

- `go run ./cmd/gate check` passes on the repository after this change.
- Editing a code file with no covering spec is refused by the Claude Code
  hook with exit code 2 and a message naming the file and how to create a
  spec; the same edit passes once a covering spec exists.
- A Bash command redirecting into an uncovered `.go` file is refused; one
  writing to a `.md` file is not.
- `git commit` of an uncovered code change fails in the pre-commit hook.
- `gate new` / `gate done` round-trip leaves `gate check` passing.
- Completing a spec whose declared manual page is unchanged fails
  `gate check`; an unlinked manual page fails it too.
- Unit tests cover front-matter parsing, scope matching, list validation
  and Bash path extraction.

## Notes
