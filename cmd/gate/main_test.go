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
a
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
	if tg := bashWriteTargets(root, "cat > tensor/ops.go <<'EOF'\npackage tensor\nEOF"); !has(tg, "tensor/ops.go") {
		t.Errorf("heredoc redirect not detected: %v", tg)
	}
	if tg := bashWriteTargets(root, "cat > tensor/new.go <<'EOF'\nEOF"); !has(tg, "tensor/new.go") {
		t.Errorf("new file in existing dir not detected: %v", tg)
	}
	if tg := bashWriteTargets(root, "sed -i '' 's/a/b/' tensor/ops.go"); !has(tg, "tensor/ops.go") {
		t.Errorf("sed -i not detected: %v", tg)
	}
	if tg := bashWriteTargets(root, "python3 - <<'PY'\np='tensor/ops.go'; open(p,'w').write('x')\nPY"); !has(tg, "tensor/ops.go") {
		t.Errorf("python write not detected: %v", tg)
	}
	if tg := bashWriteTargets(root, "go test ./tensor/ && cat tensor/ops.go"); len(tg) != 0 {
		t.Errorf("read-only command flagged: %v", tg)
	}
	if tg := bashWriteTargets(root, "echo hi > /tmp/elsewhere/x.go"); len(tg) != 0 {
		t.Errorf("path outside repo flagged: %v", tg)
	}
	if tg := bashWriteTargets(root, "cat > nonexistent/dir/x.go"); len(tg) != 0 {
		t.Errorf("path in missing dir flagged: %v", tg)
	}
	if !noVerifyRe.MatchString("git commit --no-verify -m x") || !noVerifyRe.MatchString("git commit -n -m x") || noVerifyRe.MatchString("git commit -m 'no-verify text'") {
		t.Error("no-verify detection")
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
		in := `{"tool_name":"` + tool + `","tool_input":{"file_path":"` + file + `","command":` + jsonString(command) + `}}`
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
