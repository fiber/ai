# Development process

Four lists, one spec per change, and a gate that refuses work outside the
rules. `go run ./cmd/gate` is the tool; a git pre-commit hook and a
Claude Code hook run it automatically.

## The lists

| File | Holds | Line format |
|---|---|---|
| `TODO.md` | open work items | `- [ ] T-042 — title (spec/T-042-slug.md)` — the spec reference may be missing while the item is only an idea |
| `BUGS.md` | open bugs | `- [ ] B-007 — title (spec/B-007-slug.md)` |
| `DONE.md` | finished work items, newest first | `- 2026-09-05 T-042 — title (spec/done/T-042-slug.md)` |
| `BUGS-FIXED.md` | fixed bugs, newest first | `- 2026-09-05 B-007 — title (spec/done/B-007-slug.md)` |

IDs are `T-nnn` for work items and `B-nnn` for bugs and are never reused.

## The rules

1. **No code without a spec.** Before a file outside the documentation set
   is created or modified there must be an open spec in `spec/` whose
   `scope` covers the file. Documentation (`*.md`, `spec/`, `.claude/`,
   `.githooks/`, `benchmarks/results/`, `.gitignore`) is exempt.
2. **A spec is a Markdown file `spec/<ID>-<slug>.md`** with the front matter
   and sections of `spec/TEMPLATE.md`: id, title, status, scope, created;
   sections Goal, Design, Acceptance. Write it before coding; refine it
   while coding; it is the record of what was decided and why.
3. **Every open spec is listed** in `TODO.md` (T) or `BUGS.md` (B) as an
   unchecked item.
4. **Finishing moves the spec** to `spec/done/` with `status: done` and a
   `done:` date, removes the item from `TODO.md`/`BUGS.md` and adds a line
   to `DONE.md` (work items) or `BUGS-FIXED.md` (bugs). The commit that completes a spec may still touch files in
   its scope (the gate treats a done spec that is part of the change set as
   covering).
5. **Bugs are recorded first** in `BUGS.md`; fixing one needs a `B-` spec
   like any other change.
6. **Commits go through the gate.** `git commit --no-verify` is not used.
7. **The manual is maintained.** Every spec declares the `docs/manual/`
   pages it changes (`manual:` list, or `none` when nothing user-visible
   changes). Completing a spec without touching those pages fails the
   gate, as does a manual page that is not linked from
   `docs/manual/README.md`.

## Workflow

```sh
go run ./cmd/gate new -scope tensor/,internal/kernel/ -manual docs/manual/performance.md "Vectorised tanh kernel"
#   -> spec/T-002-vectorised-tanh-kernel.md, TODO.md entry
$EDITOR spec/T-002-vectorised-tanh-kernel.md        # fill in Goal / Design / Acceptance
# ... implement, test ...
go run ./cmd/gate done T-002                        # moves the spec, updates TODO.md and DONE.md
git add -A && git commit                            # pre-commit runs `gate check -staged`
```

`go run ./cmd/gate new -bug -scope ... -manual none "title"` creates a bug
spec and the `BUGS.md` entry. `go run ./cmd/gate check` validates the whole tree at any
time.

## Enforcement

- `.githooks/pre-commit` runs `gate check -staged` (activate once per clone
  with `go run ./cmd/gate install`).
- `.claude/settings.json` runs `gate hook` before every Edit/Write and
  every Bash command; edits to uncovered code files are refused with the
  spec that is missing, and shell commands are checked by their write
  targets: redirections, `tee`, `sed -i`, `mv`/`cp`/`rm`, `gofmt -w`,
  `git mv`/`rm` arguments and Python file writes (`open` in write mode,
  `Path.write_*`, `shutil.copy/move`, `os.rename/remove`). Words that
  merely appear in heredoc text are not targets. A write whose target is
  a variable is refused ("use a literal path").
