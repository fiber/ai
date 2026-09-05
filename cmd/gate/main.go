// gate enforces the development process described in PROCESS.md: every
// code change is covered by a spec in spec/, open specs are listed in
// TODO.md or BUGS.md, finished specs live in spec/done/ and DONE.md.
//
//	gate check [-staged]          validate specs, lists and changed files
//	gate hook                     Claude Code PreToolUse hook (JSON on stdin)
//	gate new [-bug] -scope a/,b.go -manual page.md|none title...
//	gate done <id>                move a spec to spec/done/ and update the lists
//	gate install                  point git at .githooks
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	root, err := repoRoot()
	if err != nil {
		fatal(err)
	}
	var cmdErr error
	switch os.Args[1] {
	case "check":
		fs := flag.NewFlagSet("check", flag.ExitOnError)
		staged := fs.Bool("staged", false, "check staged files instead of the working tree")
		fs.Parse(os.Args[2:])
		cmdErr = runCheck(root, *staged)
	case "hook":
		cmdErr = runHook(root, os.Stdin)
		if cmdErr != nil {
			fmt.Fprintln(os.Stderr, cmdErr)
			os.Exit(2) // exit code 2 blocks the tool call in Claude Code
		}
	case "new":
		fs := flag.NewFlagSet("new", flag.ExitOnError)
		bug := fs.Bool("bug", false, "create a bug spec (B-nnn) listed in BUGS.md")
		scope := fs.String("scope", "", "comma-separated scope patterns")
		manual := fs.String("manual", "", "comma-separated manual pages to update, or none")
		fs.Parse(os.Args[2:])
		cmdErr = runNew(root, *bug, *scope, *manual, strings.Join(fs.Args(), " "))
	case "done":
		if len(os.Args) != 3 {
			usage()
			os.Exit(2)
		}
		cmdErr = runDone(root, os.Args[2])
	case "install":
		cmdErr = runInstall(root)
	default:
		usage()
		os.Exit(2)
	}
	if cmdErr != nil {
		fatal(cmdErr)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: gate check [-staged] | hook | new [-bug] -scope a/,b.go -manual page|none title... | done <id> | install")
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "gate:", err)
	os.Exit(1)
}

func repoRoot() (string, error) {
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return "", errors.New("not inside a git repository")
	}
	return strings.TrimSpace(string(out)), nil
}

// ---------------------------------------------------------------------------
// Specs
// ---------------------------------------------------------------------------

// Spec is one spec file.
type Spec struct {
	ID      string
	Title   string
	Status  string // "open" or "done"
	Scope   []string
	Created string
	Done    string
	Manual  []string // manual pages this spec changes; "none" if not user-visible
	Path    string   // repo-relative
}

var idRe = regexp.MustCompile(`^[TB]-\d{3,}$`)

// parseSpec reads front matter and checks the required sections.
func parseSpec(rel string, content []byte) (Spec, error) {
	s := Spec{Path: rel}
	text := string(content)
	if !strings.HasPrefix(text, "---\n") {
		return s, fmt.Errorf("%s: missing front matter", rel)
	}
	end := strings.Index(text[4:], "\n---")
	if end < 0 {
		return s, fmt.Errorf("%s: unterminated front matter", rel)
	}
	front := text[4 : 4+end]
	body := text[4+end+4:]
	var key string
	for _, line := range strings.Split(front, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		if strings.HasPrefix(line, "  - ") || strings.HasPrefix(line, "- ") {
			item := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "- "))
			switch key {
			case "scope":
				s.Scope = append(s.Scope, item)
			case "manual":
				s.Manual = append(s.Manual, item)
			default:
				return s, fmt.Errorf("%s: list item under %q", rel, key)
			}
			continue
		}
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			return s, fmt.Errorf("%s: bad front matter line %q", rel, line)
		}
		key = strings.TrimSpace(k)
		v = strings.TrimSpace(v)
		switch key {
		case "id":
			s.ID = v
		case "title":
			s.Title = v
		case "status":
			s.Status = v
		case "created":
			s.Created = v
		case "done":
			s.Done = v
		case "scope":
			for _, p := range strings.Split(v, ",") {
				if p = strings.TrimSpace(p); p != "" {
					s.Scope = append(s.Scope, p)
				}
			}
		case "manual":
			for _, p := range strings.Split(v, ",") {
				if p = strings.TrimSpace(p); p != "" {
					s.Manual = append(s.Manual, p)
				}
			}
		default:
			return s, fmt.Errorf("%s: unknown front matter key %q", rel, key)
		}
	}
	var errs []string
	if !idRe.MatchString(s.ID) {
		errs = append(errs, "id must look like T-001 or B-001")
	}
	if s.Title == "" {
		errs = append(errs, "title missing")
	}
	if s.Status != "open" && s.Status != "done" {
		errs = append(errs, "status must be open or done")
	}
	if len(s.Scope) == 0 {
		errs = append(errs, "scope missing")
	}
	if s.Created == "" {
		errs = append(errs, "created missing")
	}
	if len(s.Manual) == 0 {
		errs = append(errs, "manual missing (list the docs/manual/ pages this changes, or 'none')")
	}
	for _, m := range s.Manual {
		if m != "none" && (!strings.HasPrefix(m, manualDir) || !strings.HasSuffix(m, ".md")) {
			errs = append(errs, fmt.Sprintf("manual entry %q must be a page under %s or 'none'", m, manualDir))
		}
	}
	if s.Status == "done" && s.Done == "" {
		errs = append(errs, "done date missing")
	}
	if base := filepath.Base(rel); !strings.HasPrefix(base, s.ID+"-") && base != s.ID+".md" {
		errs = append(errs, fmt.Sprintf("file name must start with %s-", s.ID))
	}
	inDone := strings.HasPrefix(filepath.ToSlash(rel), "spec/done/")
	if inDone && s.Status != "done" {
		errs = append(errs, "specs in spec/done/ must have status: done")
	}
	if !inDone && s.Status == "done" {
		errs = append(errs, "done specs belong in spec/done/")
	}
	for _, section := range []string{"## Goal", "## Design", "## Acceptance"} {
		if !strings.Contains(body, "\n"+section) && !strings.HasPrefix(body, section) {
			errs = append(errs, "section "+section+" missing")
		}
	}
	if performanceScope(s.Scope) && !hasPythonBaseline(body) {
		errs = append(errs, "performance spec without a Python baseline: the Acceptance section must name the NumPy/PyTorch figure and target, or state 'no performance impact' (PROCESS.md rule 7)")
	}
	if len(errs) > 0 {
		return s, fmt.Errorf("%s: %s", rel, strings.Join(errs, "; "))
	}
	return s, nil
}

// performanceDirs are the parts of the tree where changes can move
// benchmark numbers; specs touching them must state a Python baseline.
var performanceDirs = []string{"internal/", "tensor/", "nn/", "optim/"}

func performanceScope(scope []string) bool {
	for _, p := range scope {
		p = strings.TrimPrefix(strings.TrimSpace(p), "./")
		for _, d := range performanceDirs {
			if strings.HasPrefix(p, d) || p+"/" == d {
				return true
			}
		}
	}
	return false
}

// hasPythonBaseline checks the Acceptance section for a NumPy/PyTorch
// reference or an explicit no-impact statement.
func hasPythonBaseline(body string) bool {
	i := strings.Index(body, "## Acceptance")
	if i < 0 {
		return false
	}
	sec := body[i+len("## Acceptance"):]
	if j := strings.Index(sec, "\n## "); j >= 0 {
		sec = sec[:j]
	}
	lower := strings.ToLower(sec)
	return strings.Contains(lower, "pytorch") || strings.Contains(lower, "numpy") || strings.Contains(lower, "no performance impact")
}

// loadSpecs reads spec/*.md and spec/done/*.md.
func loadSpecs(root string) ([]Spec, []error) {
	var specs []Spec
	var errs []error
	seen := map[string]string{}
	for _, dir := range []string{"spec", "spec/done"} {
		entries, err := os.ReadDir(filepath.Join(root, dir))
		if err != nil {
			if dir == "spec" {
				errs = append(errs, errors.New("spec/ directory missing"))
			}
			continue
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") || e.Name() == "TEMPLATE.md" {
				continue
			}
			rel := path.Join(dir, e.Name())
			data, err := os.ReadFile(filepath.Join(root, rel))
			if err != nil {
				errs = append(errs, err)
				continue
			}
			s, err := parseSpec(rel, data)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			if other, dup := seen[s.ID]; dup {
				errs = append(errs, fmt.Errorf("%s: id %s already used by %s", rel, s.ID, other))
				continue
			}
			seen[s.ID] = rel
			specs = append(specs, s)
		}
	}
	return specs, errs
}

// covers reports whether a scope pattern matches a repo-relative path.
func covers(pattern, rel string) bool {
	rel = filepath.ToSlash(rel)
	pattern = strings.TrimPrefix(pattern, "./")
	if strings.HasSuffix(pattern, "/") {
		return strings.HasPrefix(rel, pattern) || rel+"/" == pattern
	}
	if pattern == rel {
		return true
	}
	if ok, _ := path.Match(pattern, rel); ok {
		return true
	}
	if !strings.Contains(pattern, "/") {
		if ok, _ := path.Match(pattern, path.Base(rel)); ok {
			return true
		}
	}
	return false
}

// exempt lists files that need no spec: documentation, the process files
// themselves, benchmark results and the retired prototype.
func exempt(rel string) bool {
	rel = filepath.ToSlash(rel)
	if strings.HasSuffix(rel, ".md") || rel == ".gitignore" {
		return true
	}
	for _, prefix := range []string{"spec/", ".claude/", ".githooks/", "benchmarks/results/", "orig/"} {
		if strings.HasPrefix(rel, prefix) {
			return true
		}
	}
	return false
}

// coveringSpec returns the spec that permits changing rel, or an error
// explaining what is missing. changed is the set of paths in the current
// change set; a done spec in that set still covers its scope so that the
// commit completing a spec can carry the final code.
func coveringSpec(rel string, specs []Spec, changed map[string]bool) (Spec, error) {
	for _, s := range specs {
		if s.Status != "open" && !changed[s.Path] {
			continue
		}
		for _, p := range s.Scope {
			if covers(p, rel) {
				return s, nil
			}
		}
	}
	var open []string
	for _, s := range specs {
		if s.Status == "open" {
			open = append(open, fmt.Sprintf("  %s %s (scope %s)", s.ID, s.Title, strings.Join(s.Scope, ", ")))
		}
	}
	msg := fmt.Sprintf("%s is not covered by any open spec.\nWrite a spec first: go run ./cmd/gate new -scope %s -manual docs/manual/<page>.md|none \"title\"  (see PROCESS.md)", rel, scopeSuggestion(rel))
	if len(open) > 0 {
		msg += "\nOpen specs:\n" + strings.Join(open, "\n")
	}
	return Spec{}, errors.New(msg)
}

func scopeSuggestion(rel string) string {
	dir := path.Dir(filepath.ToSlash(rel))
	if dir == "." {
		return rel
	}
	return dir + "/"
}

// ---------------------------------------------------------------------------
// Lists
// ---------------------------------------------------------------------------

func readLines(root, name string) ([]string, error) {
	data, err := os.ReadFile(filepath.Join(root, name))
	if err != nil {
		return nil, fmt.Errorf("%s missing", name)
	}
	return strings.Split(strings.TrimRight(string(data), "\n"), "\n"), nil
}

func hasOpenItem(lines []string, id string) bool {
	for _, l := range lines {
		if strings.HasPrefix(strings.TrimSpace(l), "- [ ] "+id+" ") || strings.HasPrefix(strings.TrimSpace(l), "- [ ] "+id+"\t") {
			return true
		}
	}
	return false
}

func mentions(lines []string, id string) bool {
	for _, l := range lines {
		if strings.HasPrefix(strings.TrimSpace(l), "- ") && strings.Contains(l, " "+id+" ") {
			return true
		}
	}
	return false
}

// checkLists validates TODO.md, BUGS.md and DONE.md against the specs.
func checkLists(root string, specs []Spec) []error {
	var errs []error
	todo, err := readLines(root, "TODO.md")
	if err != nil {
		errs = append(errs, err)
	}
	bugs, err := readLines(root, "BUGS.md")
	if err != nil {
		errs = append(errs, err)
	}
	done, err := readLines(root, "DONE.md")
	if err != nil {
		errs = append(errs, err)
	}
	fixed, err := readLines(root, "BUGS-FIXED.md")
	if err != nil {
		errs = append(errs, err)
	}
	if len(errs) > 0 {
		return errs
	}
	for _, s := range specs {
		list, name := todo, "TODO.md"
		doneList, doneName := done, "DONE.md"
		if strings.HasPrefix(s.ID, "B-") {
			list, name = bugs, "BUGS.md"
			doneList, doneName = fixed, "BUGS-FIXED.md"
		}
		switch s.Status {
		case "open":
			if !hasOpenItem(list, s.ID) {
				errs = append(errs, fmt.Errorf("%s: open spec %s has no '- [ ] %s — ...' entry in %s", s.Path, s.ID, s.ID, name))
			}
		case "done":
			if !mentions(doneList, s.ID) {
				errs = append(errs, fmt.Errorf("%s: done spec %s is not listed in %s", s.Path, s.ID, doneName))
			}
			if hasOpenItem(todo, s.ID) || hasOpenItem(bugs, s.ID) {
				errs = append(errs, fmt.Errorf("%s: done spec %s is still an open item in TODO.md/BUGS.md", s.Path, s.ID))
			}
		}
	}
	return errs
}

// ---------------------------------------------------------------------------
// Manual
// ---------------------------------------------------------------------------

const manualDir = "docs/manual/"

// checkManualIndex requires every manual page to be linked from the
// manual's table of contents so nothing is written and then forgotten.
func checkManualIndex(root string) []error {
	entries, err := os.ReadDir(filepath.Join(root, manualDir))
	if err != nil {
		return []error{errors.New(manualDir + " missing")}
	}
	index, err := os.ReadFile(filepath.Join(root, manualDir, "README.md"))
	if err != nil {
		return []error{errors.New(manualDir + "README.md (table of contents) missing")}
	}
	var errs []error
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".md") || name == "README.md" {
			continue
		}
		if !strings.Contains(string(index), "("+name+")") {
			errs = append(errs, fmt.Errorf("%s%s is not linked from %sREADME.md", manualDir, name, manualDir))
		}
	}
	return errs
}

// checkManualUpdated requires that a spec being completed in this change
// set (status done and part of the change) has touched every manual page
// it declared, and that declared pages exist.
func checkManualUpdated(specs []Spec, changed map[string]bool) []error {
	var errs []error
	for _, s := range specs {
		if s.Status != "done" || !changed[s.Path] {
			continue
		}
		for _, m := range s.Manual {
			if m == "none" {
				continue
			}
			if !changed[m] {
				errs = append(errs, fmt.Errorf("%s: declares manual page %s but the page is not part of this change", s.Path, m))
			}
		}
	}
	return errs
}

// ---------------------------------------------------------------------------
// check
// ---------------------------------------------------------------------------

func gitLines(root string, args ...string) ([]string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git %s: %v", strings.Join(args, " "), err)
	}
	var lines []string
	for _, l := range strings.Split(string(out), "\n") {
		if l != "" {
			lines = append(lines, l)
		}
	}
	return lines, nil
}

// changedFiles returns repo-relative paths of the change set.
func changedFiles(root string, staged bool) ([]string, error) {
	if staged {
		return gitLines(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR")
	}
	lines, err := gitLines(root, "status", "--porcelain", "-uall")
	if err != nil {
		return nil, err
	}
	var files []string
	for _, l := range lines {
		if len(l) < 4 {
			continue
		}
		name := l[3:]
		if i := strings.LastIndex(name, " -> "); i >= 0 {
			name = name[i+4:]
		}
		if !strings.HasPrefix(l, " D") && !strings.HasPrefix(l, "D ") {
			files = append(files, name)
		}
	}
	return files, nil
}

func runCheck(root string, staged bool) error {
	specs, errs := loadSpecs(root)
	errs = append(errs, checkLists(root, specs)...)
	errs = append(errs, checkManualIndex(root)...)
	files, err := changedFiles(root, staged)
	if err != nil {
		return err
	}
	changed := map[string]bool{}
	for _, f := range files {
		changed[f] = true
	}
	errs = append(errs, checkManualUpdated(specs, changed)...)
	for _, f := range files {
		if exempt(f) {
			continue
		}
		if _, err := coveringSpec(f, specs, changed); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		var b strings.Builder
		for _, e := range errs {
			b.WriteString("- " + e.Error() + "\n")
		}
		return fmt.Errorf("%d problem(s):\n%s", len(errs), b.String())
	}
	mode := "working tree"
	if staged {
		mode = "staged changes"
	}
	fmt.Printf("gate: ok (%d specs, %d changed files, %s)\n", len(specs), len(files), mode)
	return nil
}

// ---------------------------------------------------------------------------
// hook (Claude Code PreToolUse)
// ---------------------------------------------------------------------------

type hookInput struct {
	ToolName  string `json:"tool_name"`
	Cwd       string `json:"cwd"` // directory the tool call runs in
	ToolInput struct {
		FilePath     string `json:"file_path"`
		NotebookPath string `json:"notebook_path"`
		Command      string `json:"command"`
	} `json:"tool_input"`
}

func runHook(root string, stdin io.Reader) error {
	data, err := io.ReadAll(stdin)
	if err != nil {
		return err
	}
	var in hookInput
	if err := json.Unmarshal(data, &in); err != nil {
		return nil // not a tool payload we understand; never block on that
	}
	base := in.Cwd
	if base == "" {
		base, _ = os.Getwd()
	}
	var targets []string
	switch in.ToolName {
	case "Edit", "Write", "MultiEdit", "NotebookEdit":
		p := in.ToolInput.FilePath
		if p == "" {
			p = in.ToolInput.NotebookPath
		}
		if p != "" {
			targets = append(targets, p)
		}
	case "Bash":
		cmd := in.ToolInput.Command
		shell, _ := splitHeredocs(cmd)
		if noVerifyRe.MatchString(shell) {
			return errors.New("git commit --no-verify bypasses the process gate and is not allowed (PROCESS.md rule 6)")
		}
		var err error
		if targets, err = bashWriteTargets(root, base, cmd); err != nil {
			return fmt.Errorf("blocked by the process gate: %v", err)
		}
	default:
		return nil
	}
	if len(targets) == 0 {
		return nil
	}
	specs, errs := loadSpecs(root)
	if len(errs) > 0 {
		var b strings.Builder
		for _, e := range errs {
			b.WriteString("- " + e.Error() + "\n")
		}
		return fmt.Errorf("spec files are invalid; fix them before changing code:\n%s", b.String())
	}
	files, _ := changedFiles(root, false)
	changed := map[string]bool{}
	for _, f := range files {
		changed[f] = true
	}
	for _, t := range targets {
		rel, ok := relPath(root, base, t)
		if !ok || exempt(rel) {
			continue
		}
		if _, err := coveringSpec(rel, specs, changed); err != nil {
			return fmt.Errorf("blocked by the process gate: %v", err)
		}
	}
	return nil
}

// relPath converts an absolute path, or one relative to base, into a
// repo-relative one; ok is false for paths outside the repository.
func relPath(root, base, p string) (string, bool) {
	if !filepath.IsAbs(p) {
		p = filepath.Join(base, p)
	}
	rel, err := filepath.Rel(canonical(root), canonical(p))
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", false
	}
	return filepath.ToSlash(rel), true
}

// canonical resolves symlinks (macOS puts temporary directories behind
// /var -> /private/var); for a path that does not exist yet its directory
// is resolved instead.
func canonical(p string) string {
	if r, err := filepath.EvalSymlinks(p); err == nil {
		return r
	}
	if d, err := filepath.EvalSymlinks(filepath.Dir(p)); err == nil {
		return filepath.Join(d, filepath.Base(p))
	}
	return p
}

var (
	noVerifyRe = regexp.MustCompile(`git\s+commit[^\n|;&]*(--no-verify|\s-n\b)`)
	heredocRe  = regexp.MustCompile(`<<-?\s*['"]?(\w+)['"]?`)
	// files the gate cares about when they are written
	codeFileRe = regexp.MustCompile(`\.(go|s|S|c|h|py|sh|mod|sum|json|ya?ml|toml|txt|csv)$`)
	// shell commands whose arguments are written, moved or deleted
	writeCmds  = map[string]bool{"tee": true, "mv": true, "cp": true, "rm": true, "truncate": true, "install": true, "patch": true}
	pyAssignRe = regexp.MustCompile(`(?m)^\s*(\w+)\s*=\s*['"]([^'"\n]+)['"]`)
	pyWriteRes = []*regexp.Regexp{
		regexp.MustCompile(`open\(\s*([^,()]+?)\s*,\s*['"][wax]`),
		regexp.MustCompile(`Path\(\s*([^()]+?)\s*\)\.write_`),
		regexp.MustCompile(`shutil\.(?:copy|copy2|copyfile|move)\([^,()]+,\s*([^,()]+?)\s*\)`),
		regexp.MustCompile(`os\.(?:rename|replace)\([^,()]+,\s*([^,()]+?)\s*\)`),
		regexp.MustCompile(`os\.remove\(\s*([^()]+?)\s*\)`),
	}
)

// errUnknownTarget is returned when a write goes to a path the gate
// cannot see (a shell or Python variable).
var errUnknownTarget = errors.New("cannot determine the write target; use a literal path")

// splitHeredocs separates heredoc bodies from the shell text so that
// words inside documentation or code being written are not mistaken for
// shell arguments.
func splitHeredocs(cmd string) (shell string, bodies string) {
	var sh, bd []string
	var marker string
	for _, line := range strings.Split(cmd, "\n") {
		if marker != "" {
			if strings.TrimSpace(line) == marker {
				marker = ""
			} else {
				bd = append(bd, line)
			}
			continue
		}
		sh = append(sh, line)
		if m := heredocRe.FindStringSubmatch(line); m != nil {
			marker = m[1]
		}
	}
	return strings.Join(sh, "\n"), strings.Join(bd, "\n")
}

// shellTokens splits shell text into words, keeping operators separate
// and dropping quotes.
func shellTokens(text string) []string {
	var toks []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			toks = append(toks, cur.String())
			cur.Reset()
		}
	}
	var quote rune
	rs := []rune(text)
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case quote != 0:
			if r == quote {
				quote = 0
			} else {
				cur.WriteRune(r)
			}
		case r == '\'' || r == '"':
			quote = r
		case r == ' ' || r == '\t' || r == '\n':
			flush()
		case r == '|' || r == ';' || r == '&':
			flush()
			if i+1 < len(rs) && rs[i+1] == r {
				toks = append(toks, string(r)+string(r))
				i++
			} else {
				toks = append(toks, string(r))
			}
		case r == '>':
			// redirection, possibly prefixed by a file descriptor or &
			if prev := cur.String(); prev == "1" || prev == "2" || prev == "&" {
				cur.Reset()
			} else {
				flush()
			}
			if i+1 < len(rs) && rs[i+1] == '>' {
				toks = append(toks, ">>")
				i++
			} else {
				toks = append(toks, ">")
			}
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return toks
}

func isOperator(t string) bool {
	return t == "|" || t == "||" || t == ";" || t == "&" || t == "&&" || t == ">" || t == ">>"
}

// shellTargets returns the tokens a shell command writes to. An error is
// returned when a target is a variable.
func shellTargets(shell string) ([]string, error) {
	toks := shellTokens(shell)
	var out []string
	add := func(t string) error {
		if strings.HasPrefix(t, "$") || strings.Contains(t, "${") {
			return errUnknownTarget
		}
		out = append(out, t)
		return nil
	}
	for i := 0; i < len(toks); i++ {
		t := toks[i]
		switch {
		case t == ">" || t == ">>":
			if i+1 < len(toks) && !isOperator(toks[i+1]) {
				if err := add(toks[i+1]); err != nil {
					return nil, err
				}
				i++
			}
		case writeCmds[t],
			t == "sed" && i+1 < len(toks) && strings.HasPrefix(toks[i+1], "-i"),
			(t == "gofmt" || t == "goimports") && i+1 < len(toks) && toks[i+1] == "-w",
			t == "git" && i+1 < len(toks) && (toks[i+1] == "mv" || toks[i+1] == "rm"):
			if t == "git" {
				i++
			}
			for j := i + 1; j < len(toks) && !isOperator(toks[j]); j++ {
				a := toks[j]
				if strings.HasPrefix(a, "-") {
					continue
				}
				if err := add(a); err != nil {
					return nil, err
				}
				i = j
			}
		}
	}
	return out, nil
}

// pythonTargets returns paths written by Python snippets in text.
func pythonTargets(text string) ([]string, error) {
	vars := map[string]string{}
	for _, m := range pyAssignRe.FindAllStringSubmatch(text, -1) {
		vars[m[1]] = m[2]
	}
	var out []string
	for _, re := range pyWriteRes {
		for _, m := range re.FindAllStringSubmatch(text, -1) {
			arg := strings.TrimSpace(m[1])
			if (strings.HasPrefix(arg, "'") || strings.HasPrefix(arg, `"`)) && len(arg) >= 2 {
				out = append(out, arg[1:len(arg)-1])
				continue
			}
			if v, ok := vars[arg]; ok {
				out = append(out, v)
				continue
			}
			return nil, errUnknownTarget
		}
	}
	return out, nil
}

// bashWriteTargets returns the repository files a shell command writes,
// moves or deletes: redirection and write-command arguments in the shell
// text plus Python write calls in the command and its heredoc bodies.
// Words that merely appear in heredoc text are not targets. Paths outside
// the repository or in directories that do not exist are ignored.
func bashWriteTargets(root, base, cmd string) ([]string, error) {
	shell, bodies := splitHeredocs(cmd)
	targets, err := shellTargets(shell)
	if err != nil {
		return nil, err
	}
	py, err := pythonTargets(shell + "\n" + bodies)
	if err != nil {
		return nil, err
	}
	targets = append(targets, py...)
	seen := map[string]bool{}
	var out []string
	for _, t := range targets {
		t = strings.TrimSpace(t)
		if t == "" || t == "/dev/null" || strings.HasPrefix(t, "http") || seen[t] {
			continue
		}
		seen[t] = true
		if !codeFileRe.MatchString(t) {
			continue // documentation and other exempt files need no check
		}
		rel, ok := relPath(root, base, t)
		if !ok {
			continue
		}
		abs := filepath.Join(root, rel)
		if _, err := os.Stat(abs); err != nil {
			if _, err := os.Stat(filepath.Dir(abs)); err != nil {
				continue
			}
		}
		out = append(out, abs)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// new / done / install
// ---------------------------------------------------------------------------

func today() string { return time.Now().Format("2006-01-02") }

func nextID(root, prefix string) (string, error) {
	max := 0
	consider := func(id string) {
		if !strings.HasPrefix(id, prefix+"-") {
			return
		}
		if n, err := strconv.Atoi(strings.TrimPrefix(id, prefix+"-")); err == nil && n > max {
			max = n
		}
	}
	specs, _ := loadSpecs(root)
	for _, s := range specs {
		consider(s.ID)
	}
	re := regexp.MustCompile(`\b` + prefix + `-\d+\b`)
	for _, name := range []string{"TODO.md", "BUGS.md", "DONE.md"} {
		lines, err := readLines(root, name)
		if err != nil {
			continue
		}
		for _, l := range lines {
			for _, id := range re.FindAllString(l, -1) {
				consider(id)
			}
		}
	}
	return fmt.Sprintf("%s-%03d", prefix, max+1), nil
}

func slugify(title string) string {
	var b strings.Builder
	lastDash := true
	for _, r := range strings.ToLower(title) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			lastDash = false
		case !lastDash:
			b.WriteByte('-')
			lastDash = true
		}
	}
	s := strings.Trim(b.String(), "-")
	if len(s) > 48 {
		s = strings.Trim(s[:48], "-")
	}
	return s
}

func runNew(root string, bug bool, scope, manual, title string) error {
	title = strings.TrimSpace(title)
	if title == "" || scope == "" || manual == "" {
		return errors.New("usage: gate new [-bug] -scope a/,b.go -manual docs/manual/page.md|none title words...")
	}
	prefix, list := "T", "TODO.md"
	if bug {
		prefix, list = "B", "BUGS.md"
	}
	id, err := nextID(root, prefix)
	if err != nil {
		return err
	}
	rel := path.Join("spec", id+"-"+slugify(title)+".md")
	tmpl, err := os.ReadFile(filepath.Join(root, "spec", "TEMPLATE.md"))
	if err != nil {
		return errors.New("spec/TEMPLATE.md missing")
	}
	var scopeLines strings.Builder
	for _, p := range strings.Split(scope, ",") {
		if p = strings.TrimSpace(p); p != "" {
			scopeLines.WriteString("  - " + p + "\n")
		}
	}
	text := string(tmpl)
	text = strings.Replace(text, "id: T-000", "id: "+id, 1)
	text = strings.Replace(text, "title: One-line title", "title: "+title, 1)
	text = strings.Replace(text, "created: 2026-01-01", "created: "+today(), 1)
	text = regexp.MustCompile(`(?m)^scope:\n(  - .*\n)+`).ReplaceAllString(text, "scope:\n"+scopeLines.String())
	var manualLines strings.Builder
	for _, p := range strings.Split(manual, ",") {
		if p = strings.TrimSpace(p); p != "" {
			manualLines.WriteString("  - " + p + "\n")
		}
	}
	text = regexp.MustCompile(`(?m)^manual:\n(  - .*\n)+`).ReplaceAllString(text, "manual:\n"+manualLines.String())
	if err := os.WriteFile(filepath.Join(root, rel), []byte(text), 0o644); err != nil {
		return err
	}
	entry := fmt.Sprintf("- [ ] %s — %s (%s)", id, title, rel)
	if err := appendListItem(root, list, entry); err != nil {
		return err
	}
	fmt.Printf("created %s and added to %s\n", rel, list)
	if performanceScope(strings.Split(scope, ",")) {
		fmt.Println("performance-relevant scope: the Acceptance section must name the NumPy/PyTorch baseline and target (PROCESS.md rule 7)")
	}
	return nil
}

// appendListItem adds an item at the end of a list file, replacing a
// "(none open)" placeholder if present.
func appendListItem(root, name, entry string) error {
	lines, err := readLines(root, name)
	if err != nil {
		return err
	}
	var out []string
	replaced := false
	for _, l := range lines {
		if strings.TrimSpace(l) == "(none open)" {
			out = append(out, entry)
			replaced = true
			continue
		}
		out = append(out, l)
	}
	if !replaced {
		out = append(out, entry)
	}
	return os.WriteFile(filepath.Join(root, name), []byte(strings.Join(out, "\n")+"\n"), 0o644)
}

// removeListItem drops the open item for id; the placeholder returns when
// the list becomes empty.
func removeListItem(root, name, id string) (string, error) {
	lines, err := readLines(root, name)
	if err != nil {
		return "", err
	}
	var out []string
	var removed string
	items := 0
	for _, l := range lines {
		t := strings.TrimSpace(l)
		if strings.HasPrefix(t, "- [ ] "+id+" ") {
			removed = t
			continue
		}
		if strings.HasPrefix(t, "- [ ] ") {
			items++
		}
		out = append(out, l)
	}
	if items == 0 && name == "BUGS.md" {
		out = append(out, "(none open)")
	}
	return removed, os.WriteFile(filepath.Join(root, name), []byte(strings.Join(out, "\n")+"\n"), 0o644)
}

func runDone(root, id string) error {
	specs, errs := loadSpecs(root)
	if len(errs) > 0 {
		return fmt.Errorf("fix spec problems first: %v", errs[0])
	}
	var spec *Spec
	for i := range specs {
		if specs[i].ID == id {
			spec = &specs[i]
		}
	}
	if spec == nil {
		return fmt.Errorf("no spec with id %s", id)
	}
	if spec.Status != "open" {
		return fmt.Errorf("%s is already done", id)
	}
	data, err := os.ReadFile(filepath.Join(root, spec.Path))
	if err != nil {
		return err
	}
	text := strings.Replace(string(data), "\nstatus: open\n", "\nstatus: done\n", 1)
	if !strings.Contains(text, "\ndone: ") {
		text = strings.Replace(text, "\ncreated: ", "\ndone: "+today()+"\ncreated: ", 1)
	}
	newRel := path.Join("spec", "done", path.Base(spec.Path))
	if err := os.MkdirAll(filepath.Join(root, "spec", "done"), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(root, newRel), []byte(text), 0o644); err != nil {
		return err
	}
	if err := os.Remove(filepath.Join(root, spec.Path)); err != nil {
		return err
	}
	// git may not know the file yet; ignore errors from the rename bookkeeping
	exec.Command("git", "-C", root, "add", "-A", "--", spec.Path, newRel).Run()

	list, doneName := "TODO.md", "DONE.md"
	if strings.HasPrefix(id, "B-") {
		list, doneName = "BUGS.md", "BUGS-FIXED.md"
	}
	if _, err := removeListItem(root, list, id); err != nil {
		return err
	}
	lines, err := readLines(root, doneName)
	if err != nil {
		return err
	}
	entry := fmt.Sprintf("- %s %s — %s (%s)", today(), id, spec.Title, newRel)
	// insert after the header block (first blank line following a heading)
	inserted := false
	var out []string
	for i, l := range lines {
		out = append(out, l)
		if !inserted && i > 0 && strings.TrimSpace(l) == "" && i+1 < len(lines) && strings.HasPrefix(strings.TrimSpace(lines[i+1]), "- ") {
			out = append(out, entry)
			inserted = true
		}
	}
	if !inserted {
		out = append(out, "", entry)
	}
	if err := os.WriteFile(filepath.Join(root, doneName), []byte(strings.Join(out, "\n")+"\n"), 0o644); err != nil {
		return err
	}
	fmt.Printf("%s -> %s; %s updated, %s updated\n", spec.Path, newRel, list, doneName)
	return nil
}

func runInstall(root string) error {
	hook := filepath.Join(root, ".githooks", "pre-commit")
	if _, err := os.Stat(hook); err != nil {
		return errors.New(".githooks/pre-commit missing")
	}
	if err := os.Chmod(hook, 0o755); err != nil {
		return err
	}
	cmd := exec.Command("git", "config", "core.hooksPath", ".githooks")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git config: %v: %s", err, bytes.TrimSpace(out))
	}
	fmt.Println("git core.hooksPath = .githooks")
	return nil
}
