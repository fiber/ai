We are building the github.com/fiber/ai package, an AI framework for the Go
programming language.

Python still dominates AI; we want to change that in favour of Go. Existing
Go frameworks are slow and awkward. We build one that is modern, fast and
fun to use. Implement extensive testing and example applications.

## Process (enforced by `go run ./cmd/gate`, see PROCESS.md)

- No code change without an open spec in `spec/` whose `scope` covers the
  file. Create one first: `go run ./cmd/gate new -scope <dirs> -manual <pages|none> "title"`,
  then fill in Goal / Design / Acceptance before implementing.
- Track work in `TODO.md`, bugs in `BUGS.md`, finished items in `DONE.md`.
  Record a bug in `BUGS.md` as soon as it is found.
- When a spec is implemented and tested: `go run ./cmd/gate done <id>`
  (moves it to `spec/done/`, updates the lists), update the manual pages
  the spec declared, run `go run ./cmd/gate check`, then commit.
- The manual in `docs/manual/` is part of every user-visible change.
- The gate runs as a Claude Code hook before every edit and shell command
  and as the git pre-commit hook. Do not work around it (`--no-verify`,
  writing code files from outside the repo tools); widen or add a spec
  instead.

## Conventions

- Code comments, commit messages and documentation in English.
- Commit messages describe the change plainly; no tool or model attribution
  trailers.
- `orig/` holds the earlier `vektor` prototype: read-only reference, not
  compiled, not modified.
- Run `go vet ./...` and `GOARCH=amd64 go vet ./...` (assembles the amd64
  kernels) and `go test ./...` before completing a spec; the AVX2 path can
  be exercised under Rosetta with
  `FIBERAI_KERNEL=avx2 FIBERAI_KERNEL_FORCE=1 GOARCH=amd64 go test ./...`.
