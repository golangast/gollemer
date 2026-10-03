package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// planFixture is a tiny project shaped like the deletor report feature:
// a Config struct, a flag parser, a flow that calls a worker per item,
// and a JSON-logging helper the issue names.
func planFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/plan\n\ngo 1.21\n")
	write("main.go", "package main\n\nimport \"example.com/plan/runner\"\n\nimport \"example.com/plan/worker\"\n\nfunc main() { runner.Run(worker.New()) }\n")
	write("config/config.go", "package config\n\n// Config holds the run options.\ntype Config struct {\n\tVerbose bool\n}\n")
	write("config/flags.go", "package config\n\n// GetFlags parses CLI flags into Config.\nfunc GetFlags() *Config { return &Config{} }\n")
	write("runner/run.go", `package runner

import "example.com/plan/utils"
import "example.com/plan/worker"

// Run processes every item.
func Run(w *worker.Worker) {
	for _, it := range list() {
		w.DoWork(it)
	}
	utils.WriteJSONLog(nil)
}

func list() []string { return nil }
`)
	write("worker/worker.go", `package worker

// Worker does one item of work.
type Worker struct{}

// New builds a Worker.
func New() *Worker { return &Worker{} }

// DoWork handles a single item.
func (w *Worker) DoWork(item string) {}
`)
	write("utils/utils.go", `package utils

// WriteJSONLog writes JSON-formatted logs.
func WriteJSONLog(m map[string]string) {}
`)
	return root
}

const planIssue = `Add Export Summary (JSON)

Export a summary of every processed item.

Add flags:
- ` + "`--summary <path>`" + ` (required path)

Complements your existing JSON logging (` + "`--log-json`" + `).`

// The plan names config, flags, flow, the issue-named helper, and the
// new file — the whole change, not one anchor.
func TestFeaturePlan(t *testing.T) {
	p, err := Analyze(planFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	out := p.GuideIssue(ExtractIssueConcepts(planIssue))
	for _, want := range []string{
		"TO ADD",
		"CONFIG", "config.go", "type Config struct", "`--summary`",
		"FLAGS", "GetFlags",
		"BEHAVIOR", "runner/run.go", "`Run`", "`DoWork`",
		"HELPERS", "WriteJSONLog", "the issue names `--log-json`",
		"NEW CODE", "utils/summary.go",
		"YOU ARE HERE:", "main → Run → yours here",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("plan missing %q\n%s", want, out)
		}
	}
}

// The retry fixture stays single-anchor: the title names the retry
// mechanism and it lives in the site's package.
func TestSiteMatchesTitle(t *testing.T) {
	p, err := Analyze(issueFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	c := ExtractIssueConcepts(issueText)
	var site *MechanismSite
	for _, m := range c.Mechanisms {
		if s := p.FindMechanism(m); s != nil && (site == nil || s.Score > site.Score) {
			site = s
		}
	}
	if !p.siteMatchesTitle(c, site) {
		t.Error("retry issue should match its httpclient site")
	}

	q, err := Analyze(planFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	c2 := ExtractIssueConcepts(planIssue)
	var site2 *MechanismSite
	for _, m := range c2.Mechanisms {
		if s := q.FindMechanism(m); s != nil && (site2 == nil || s.Score > site2.Score) {
			site2 = s
		}
	}
	if q.siteMatchesTitle(c2, site2) {
		t.Error("summary issue should NOT match its mechanism site")
	}
}

// Method calls through parameters typed by a project package resolve:
// Run(w *worker.Worker) calling w.DoWork() must edge to worker.DoWork.
func TestParamMethodCall(t *testing.T) {
	p, err := Analyze(planFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	var run *Func
	var doWork *Func
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			switch fn.Name {
			case "Run":
				run = fn
			case "DoWork":
				doWork = fn
			}
		}
	}
	if run == nil || doWork == nil {
		t.Fatal("fixture funcs missing")
	}
	found := false
	for _, id := range run.Calls {
		if id == doWork.ID {
			found = true
		}
	}
	if !found {
		t.Errorf("Run.Calls %v does not target DoWork", run.Calls)
	}
}

// dryRunFixture mirrors deletor issue #319: a tui package whose name
// matches the "(CLI + TUI)" parenthetical, plus a real config/flags
// pair and a CLI runner flow. The issue asks for a --dry-run flag, so
// the answer must be the multi-file plan, not the single tui anchor.
func dryRunFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/dryrun\n\ngo 1.21\n")
	write("main.go", "package main\n\nimport \"example.com/dryrun/runner\"\n\nimport \"example.com/dryrun/tui\"\n\nfunc main() { tui.NewApp(runner.RunCLI(nil)) }\n")
	write("config/config.go", "package config\n\n// Config holds the run options.\ntype Config struct {\n\tVerbose bool\n}\n")
	write("config/flags.go", "package config\n\n// GetFlags parses CLI flags into Config.\nfunc GetFlags() *Config { return &Config{} }\n")
	write("tui/app.go", `package tui

// App is the terminal UI.
type App struct{}

// NewApp builds the shared terminal UI.
func NewApp() *App { return &App{} }
`)
	write("runner/cli.go", `package runner

import "example.com/dryrun/config"
import "example.com/dryrun/worker"

// RunCLI processes every item from the command line.
func RunCLI(cfg *config.Config) {
	for _, it := range list() {
		worker.DoWork(it)
	}
}

func list() []string { return nil }
`)
	write("worker/worker.go", `package worker

import "os"

// DoWork handles a single item.
func DoWork(item string) { os.Remove(item) }
`)
	return root
}

const dryRunIssue = "Add `Dry-Run / Preview` mode (CLI + TUI)\n\n" +
	"Allow users to preview what would be deleted without modifying the filesystem.\n\n" +
	"### CLI activation\n- Add flag: `--dry-run`\n"

// A flag-asking "add" issue whose title parenthetical names the tui
// package must still get the feature plan: the --dry-run flag lives in
// config/flags, not in tui. Before the fix, siteMatchesTitle saw "tui"
// and suppressed the plan, leaving only the NewApp anchor.
func TestFlagIssueBeatsSurfaceMatch(t *testing.T) {
	p, err := Analyze(dryRunFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	c := ExtractIssueConcepts(dryRunIssue)
	if !flagsSignal(c) {
		t.Fatal("dry-run issue should raise the flags signal")
	}
	out := p.GuideIssue(c)
	for _, want := range []string{
		"CONFIG", "config.go", "type Config struct", "`--dry-run`",
		"FLAGS", "GetFlags",
		"BEHAVIOR", "RunCLI",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("plan missing %q\n%s", want, out)
		}
	}
	if strings.Contains(out, "wrap the work here") {
		t.Errorf("flag issue fell back to the single tui anchor\n%s", out)
	}
}
