package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const validSpec = `---
id: T-001
title: Something
status: open
scope:
  - tensor/
  - cmd/bench/main.go
manual:
  - none
created: 2026-09-05
---

## Goal
g
## Design
d
## Acceptance
a, no performance impact
`

func TestParseSpec(t *testing.T) {
	s, err := parseSpec("spec/T-001-something.md", []byte(validSpec))
	if err != nil {
		t.Fatal(err)
	}
	if s.ID != "T-001" || s.Title != "Something" || s.Status != "open" || len(s.Scope) != 2 || s.Scope[1] != "cmd/bench/main.go" {
		t.Fatalf("parsed %+v", s)
	}
	bad := []struct{ name, content string }{
		{"spec/X-1.md", validSpec},                                            // filename/id mismatch
		{"spec/done/T-001-x.md", validSpec},                                   // open spec in done dir
		{"spec/T-001-x.md", strings.Replace(validSpec, "## Design\n", "", 1)}, // missing section
		{"spec/T-001-x.md", strings.Replace(validSpec, "scope:\n  - tensor/\n  - cmd/bench/main.go\n", "", 1)},
		{"spec/T-001-x.md", strings.Replace(validSpec, "status: open", "status: done", 1)}, // done without date, wrong dir
		{"spec/T-001-x.md", strings.Replace(validSpec, "manual:\n  - none\n", "", 1)},      // manual missing
		{"spec/T-001-x.md", strings.Replace(validSpec, "  - none", "  - README.md", 1)},    // manual page outside docs/manual/
		{"spec/T-001-x.md", "no front matter"},
	}
	for _, b := range bad {
		if _, err := parseSpec(b.name, []byte(b.content)); err == nil {
			t.Errorf("%s: expected error", b.name)
		}
	}
}

func TestCovers(t *testing.T) {
	cases := []struct {
		pattern, rel string
		want         bool
	}{
		{"tensor/", "tensor/ops.go", true},
		{"tensor/", "tensor/sub/x.go", true},
		{"tensor/", "tensorflow/x.go", false},
		{"cmd/bench/main.go", "cmd/bench/main.go", true},
		{"cmd/bench/main.go", "cmd/bench/other.go", false},
		{"internal/kernel/*.s", "internal/kernel/kernel_neon_arm64.s", true},
		{"*.go", "deep/dir/file.go", true},
		{"go.mod", "go.mod", true},
		{"./nn/", "nn/nn.go", true},
	}
	for _, c := range cases {
		if got := covers(c.pattern, c.rel); got != c.want {
			t.Errorf("covers(%q, %q) = %v, want %v", c.pattern, c.rel, got, c.want)
		}
	}
}

func TestExempt(t *testing.T) {
	for _, p := range []string{"README.md", "spec/T-001-x.md", ".claude/settings.json", ".githooks/pre-commit", "benchmarks/results/go.md", "orig/foo.go", ".gitignore", "docs/notes.md"} {
		if !exempt(p) {
			t.Errorf("%s should be exempt", p)
		}
	}
	for _, p := range []string{"tensor/ops.go", "go.mod", "cmd/gate/main.go", "benchmarks/python/bench.py", "internal/kernel/x.s"} {
		if exempt(p) {
			t.Errorf("%s should not be exempt", p)
		}
	}
}

func TestCoveringSpec(t *testing.T) {
	specs := []Spec{
		{ID: "T-001", Status: "open", Scope: []string{"tensor/"}, Path: "spec/T-001-a.md"},
		{ID: "T-002", Status: "done", Scope: []string{"nn/"}, Path: "spec/done/T-002-b.md"},
	}
	if _, err := coveringSpec("tensor/x.go", specs, nil); err != nil {
		t.Error(err)
	}
	if _, err := coveringSpec("optim/x.go", specs, nil); err == nil || !strings.Contains(err.Error(), "gate new -scope optim/") {
		t.Errorf("expected helpful error, got %v", err)
	}
	if _, err := coveringSpec("nn/x.go", specs, nil); err == nil {
		t.Error("done spec outside the change set must not cover")
	}
	if _, err := coveringSpec("nn/x.go", specs, map[string]bool{"spec/done/T-002-b.md": true}); err != nil {
		t.Error("done spec inside the change set must cover:", err)
	}
}

func TestBashWriteTargets(t *testing.T) {
	root := t.TempDir()
	os.MkdirAll(filepath.Join(root, "tensor"), 0o755)
	os.WriteFile(filepath.Join(root, "tensor", "ops.go"), []byte("package tensor"), 0o644)
	os.WriteFile(filepath.Join(root, "README.md"), nil, 0o644)
	wd, _ := os.Getwd()
	os.Chdir(root)
	defer os.Chdir(wd)

	has := func(targets []string, name string) bool {
		for _, t := range targets {
			if strings.HasSuffix(t, name) {
				return true
			}
		}
		return false
	}
	detected := []string{
		"cat > tensor/ops.go <<'EOF'\npackage tensor\nEOF",
		"cat > tensor/new.go <<'EOF'\nEOF",
		"echo x >> tensor/ops.go",
		"echo x 2>tensor/ops.go",
		"sed -i '' 's/a/b/' tensor/ops.go",
		"mv other.go tensor/ops.go",
		"cp x tensor/ops.go && go test",
		"rm -f tensor/ops.go",
		"gofmt -w tensor/ops.go",
		"go test ./... | tee tensor/ops.go",
		"git rm tensor/ops.go",
		"python3 - <<'PY'\np='tensor/ops.go'; s=open(p).read()\nopen(p,'w').write(s)\nPY",
		"python3 -c \"open('tensor/ops.go', 'w').write('x')\"",
		"python3 - <<'PY'\nfrom pathlib import Path\nPath('tensor/ops.go').write_text('x')\nPY",
	}
	for _, cmd := range detected {
		tg, err := bashWriteTargets(root, root, cmd)
		if err != nil || !has(tg, "tensor/ops.go") && !has(tg, "tensor/new.go") {
			t.Errorf("not detected: %q -> %v, %v", cmd, tg, err)
		}
	}
	notDetected := []string{
		"go test ./tensor/ && cat tensor/ops.go",
		"grep -n foo tensor/ops.go | head",
		"cat > NOTES.md <<'EOF'\nrun bench.py and see tensor/ops.go\nEOF",                 // heredoc text is not a target
		"python3 - <<'PY'\np='NOTES.md'\nopen(p,'w').write('mentions tensor/ops.go')\nPY", // markdown target only
		"echo hi > /tmp/elsewhere/x.go",
		"cat > nonexistent/dir/x.go",
		"git commit -m 'touch tensor/ops.go'",
	}
	for _, cmd := range notDetected {
		tg, err := bashWriteTargets(root, root, cmd)
		if err != nil || len(tg) != 0 {
			t.Errorf("false positive: %q -> %v, %v", cmd, tg, err)
		}
	}
	unknown := []string{
		"for f in tensor/*.go; do sed -i '' s/a/b/ $f; done",
		"python3 - <<'PY'\nimport sys\nopen(sys.argv[1], 'w')\nPY",
	}
	for _, cmd := range unknown {
		if _, err := bashWriteTargets(root, root, cmd); err != errUnknownTarget {
			t.Errorf("variable target must be refused: %q -> %v", cmd, err)
		}
	}
	if !noVerifyRe.MatchString("git commit --no-verify -m x") || !noVerifyRe.MatchString("git commit -n -m x") || noVerifyRe.MatchString("git commit -m 'no-verify text'") {
		t.Error("no-verify detection")
	}
	// relative targets resolve against the tool's working directory
	if tg, err := bashWriteTargets(root, filepath.Join(root, "tensor"), "cat > ops.go <<'EOF'\nEOF"); err != nil || !has(tg, "tensor/ops.go") {
		t.Errorf("cwd-relative target: %v, %v", tg, err)
	}
	// the no-verify check ignores heredoc bodies (documentation may quote it)
	if shell, _ := splitHeredocs("cat > NOTES.md <<'EOF'\ngit commit --no-verify is forbidden\nEOF"); noVerifyRe.MatchString(shell) {
		t.Error("heredoc body must not trigger the no-verify rule")
	}
}

func TestPythonBaselineRule(t *testing.T) {
	perf := strings.Replace(validSpec, "## Acceptance\na, no performance impact\n", "## Acceptance\nfast enough\n", 1)
	if _, err := parseSpec("spec/T-001-x.md", []byte(perf)); err == nil || !strings.Contains(err.Error(), "Python baseline") {
		t.Errorf("performance scope without baseline must fail, got %v", err)
	}
	ok := strings.Replace(validSpec, "## Acceptance\na, no performance impact\n", "## Acceptance\nmatches PyTorch at n=1024\n", 1)
	if _, err := parseSpec("spec/T-001-x.md", []byte(ok)); err != nil {
		t.Errorf("baseline given: %v", err)
	}
	nonPerf := strings.Replace(perf, "  - tensor/\n  - cmd/bench/main.go\n", "  - cmd/gate/\n", 1)
	if _, err := parseSpec("spec/T-001-x.md", []byte(nonPerf)); err != nil {
		t.Errorf("non-performance scope needs no baseline: %v", err)
	}
}

// TestRepoRoundTrip builds a throw-away git repository and exercises
// check, new, done and the hook end to end.
func TestRepoRoundTrip(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	root := t.TempDir()
	root, _ = filepath.EvalSymlinks(root)
	run := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	run("init", "-q")
	run("config", "user.email", "t@example.com")
	run("config", "user.name", "t")
	write := func(rel, content string) {
		t.Helper()
		os.MkdirAll(filepath.Dir(filepath.Join(root, rel)), 0o755)
		if err := os.WriteFile(filepath.Join(root, rel), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	tmpl, _ := os.ReadFile(filepath.Join("..", "..", "spec", "TEMPLATE.md"))
	write("spec/TEMPLATE.md", string(tmpl))
	write("TODO.md", "# TODO\n\n")
	write("BUGS.md", "# BUGS\n\n(none open)\n")
	write("DONE.md", "# DONE\n\nnewest first\n\n")
	write("BUGS-FIXED.md", "# BUGS-FIXED\n\nnewest first\n\n")
	write("README.md", "docs\n")
	write("docs/manual/README.md", "# Manual\n\n- [Usage](usage.md)\n")
	write("docs/manual/usage.md", "# Usage\n")
	run("add", "-A")
	run("commit", "-q", "-m", "init")

	if err := runCheck(root, false); err != nil {
		t.Fatalf("clean repo must pass: %v", err)
	}
	// uncovered code change fails
	write("pkg/a.go", "package pkg\n")
	if err := runCheck(root, false); err == nil || !strings.Contains(err.Error(), "pkg/a.go") {
		t.Fatalf("expected coverage error, got %v", err)
	}
	// hook blocks the same file, allows docs
	hook := func(tool, file, command string) error {
		in := `{"tool_name":"` + tool + `","cwd":` + jsonString(root) + `,"tool_input":{"file_path":"` + file + `","command":` + jsonString(command) + `}}`
		return runHook(root, strings.NewReader(in))
	}
	wd, _ := os.Getwd()
	os.Chdir(root)
	defer os.Chdir(wd)
	if err := hook("Edit", filepath.Join(root, "pkg", "a.go"), ""); err == nil {
		t.Fatal("hook must block uncovered edit")
	}
	if err := hook("Write", filepath.Join(root, "NOTES.md"), ""); err != nil {
		t.Fatalf("hook must allow docs: %v", err)
	}
	if err := hook("Bash", "", "cat > pkg/a.go <<'EOF'\nx\nEOF"); err == nil {
		t.Fatal("hook must block bash redirect into uncovered code")
	}
	if err := hook("Bash", "", "git commit --no-verify -m x"); err == nil {
		t.Fatal("hook must block --no-verify")
	}
	// a spec fixes it
	if err := runNew(root, false, "pkg/", "docs/manual/usage.md", "Package pkg"); err != nil {
		t.Fatal(err)
	}
	if err := runCheck(root, false); err != nil {
		t.Fatalf("covered change must pass: %v", err)
	}
	if err := hook("Edit", filepath.Join(root, "pkg", "a.go"), ""); err != nil {
		t.Fatalf("hook must allow covered edit: %v", err)
	}
	// a bug spec
	if err := runNew(root, true, "pkg/", "none", "Crash on empty input"); err != nil {
		t.Fatal(err)
	}
	bugs, _ := os.ReadFile(filepath.Join(root, "BUGS.md"))
	if !strings.Contains(string(bugs), "- [ ] B-001 — Crash on empty input (spec/B-001-crash-on-empty-input.md)") || strings.Contains(string(bugs), "(none open)") {
		t.Fatalf("BUGS.md:\n%s", bugs)
	}
	// done moves the spec and keeps check passing (the done spec is in the change set)
	if err := runDone(root, "T-001"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "spec", "done", "T-001-package-pkg.md")); err != nil {
		t.Fatal("spec not moved")
	}
	todo, _ := os.ReadFile(filepath.Join(root, "TODO.md"))
	if strings.Contains(string(todo), "T-001") {
		t.Fatal("TODO.md still lists T-001")
	}
	done, _ := os.ReadFile(filepath.Join(root, "DONE.md"))
	if !strings.Contains(string(done), "T-001 — Package pkg (spec/done/T-001-package-pkg.md)") {
		t.Fatalf("DONE.md:\n%s", done)
	}
	if err := runCheck(root, false); err == nil || !strings.Contains(err.Error(), "docs/manual/usage.md") {
		t.Fatalf("completing a spec without its manual page must fail, got %v", err)
	}
	write("docs/manual/usage.md", "# Usage\n\nDocumented.\n")
	if err := runCheck(root, false); err != nil {
		t.Fatalf("after done with manual: %v", err)
	}
	write("docs/manual/orphan.md", "# Orphan\n")
	if err := runCheck(root, false); err == nil || !strings.Contains(err.Error(), "orphan.md") {
		t.Fatalf("unlinked manual page must fail, got %v", err)
	}
	os.Remove(filepath.Join(root, "docs/manual/orphan.md"))
	if err := runDone(root, "B-001"); err != nil {
		t.Fatal(err)
	}
	bugs, _ = os.ReadFile(filepath.Join(root, "BUGS.md"))
	if !strings.Contains(string(bugs), "(none open)") {
		t.Fatalf("placeholder not restored:\n%s", bugs)
	}
	fixedList, _ := os.ReadFile(filepath.Join(root, "BUGS-FIXED.md"))
	if !strings.Contains(string(fixedList), "B-001 — Crash on empty input (spec/done/B-001-crash-on-empty-input.md)") {
		t.Fatalf("BUGS-FIXED.md:\n%s", fixedList)
	}
	if done, _ = os.ReadFile(filepath.Join(root, "DONE.md")); strings.Contains(string(done), "B-001") {
		t.Fatal("fixed bug must not be listed in DONE.md")
	}
	// once committed, a done spec no longer covers new edits
	run("add", "-A")
	run("commit", "-q", "-m", "work")
	write("pkg/b.go", "package pkg\n")
	if err := runCheck(root, false); err == nil {
		t.Fatal("done spec must not cover changes after its commit")
	}
	// staged mode only looks at the index
	if err := runCheck(root, true); err != nil {
		t.Fatalf("nothing staged must pass: %v", err)
	}
}

func jsonString(s string) string {
	b := strings.Builder{}
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}
