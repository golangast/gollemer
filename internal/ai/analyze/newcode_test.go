package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// newCodeFixture builds a project with a worker package and one test
// file containing a benchmark. The issue names a benchmark directory
// that doesn't exist yet.
func newCodeFixture(t *testing.T) *Project {
	t.Helper()
	dir := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/nc\n\ngo 1.21\n")
	write("worker/worker.go", `package worker

// ScanFiles scans a directory recursively.
func ScanFiles(dir string) []string { return nil }

// FilterFiles filters scanned files.
func FilterFiles(in []string) []string { return nil }

func helper() {}
`)
	write("worker/worker_test.go", `package worker

import "testing"

func BenchmarkScanFiles(b *testing.B) {}
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestNewCodePlan(t *testing.T) {
	p := newCodeFixture(t)
	c := ExtractIssueConcepts("Add benchmark tests for worker module\n\n" +
		"Benchmark file scanning operations.\n\n" +
		"Create a new directory: `internal/tests/benchmark/worker/`")
	out := p.GuideIssue(c)
	for _, want := range []string{
		"NEW CODE",
		"internal/tests/benchmark/worker/",
		"TARGETS",
		"ScanFiles",
		"FilterFiles",
		"IMITATE",
		"worker_test.go",
		"(has benchmarks)",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("new-code plan missing %q:\n%s", want, out)
		}
	}
	// The flow plan must not fire for new-code issues.
	if strings.Contains(out, "BEHAVIOR") {
		t.Errorf("new-code plan should not render BEHAVIOR:\n%s", out)
	}
}

func TestNewCodePlanNeedsMissingPath(t *testing.T) {
	p := newCodeFixture(t)
	// No backticked path that doesn't exist -> no new-code plan.
	c := ExtractIssueConcepts("Add benchmark tests for worker module\n\nBenchmark file scanning operations.")
	if np := p.buildNewCodePlan(c); np != nil {
		t.Errorf("buildNewCodePlan should be nil without a missing path, got %+v", np)
	}
}

func TestNewCodePlanIgnoresSlashedPhrases(t *testing.T) {
	p := newCodeFixture(t)
	// "Dry-Run / Preview" has a slash but is a phrase, not a path.
	c := ExtractIssueConcepts("Add `Dry-Run / Preview` mode (CLI + TUI)\n\nAdd flag: `--dry-run`")
	if np := p.buildNewCodePlan(c); np != nil {
		t.Errorf("slashed phrase must not count as a path, got %+v", np)
	}
}

func TestStemPlurals(t *testing.T) {
	for in, want := range map[string]string{
		"files":    "file",
		"tests":    "test",
		"boxes":    "box",
		"watches":  "watch",
		"classes":  "class",
		"scanning": "scann",
		"scan":     "scan",
	} {
		if got := stem(in); got != want {
			t.Errorf("stem(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestIsPathWord(t *testing.T) {
	for w, want := range map[string]bool{
		"internal/util/mmap.go":       true,
		"internal/tests/benchmark/w/": true,
		"internal/objects/store.go":   true,
		"golang.org/x/sys/unix":       false, // module path
		"allocs/op":                   false, // metric, not a path
		"Dry-Run / Preview":           false, // phrase, not a path
		"https://example.com/x":       false, // URL
		"os.ReadFile":                 false, // call, not a path
	} {
		if got := isPathWord(w); got != want {
			t.Errorf("isPathWord(%q) = %v, want %v", w, got, want)
		}
	}
}

// replaceFixture: a store with os.ReadFile/io.ReadAll call sites and a
// util package; the issue names a new mmap.go plus the files to change.
func replaceFixture(t *testing.T) *Project {
	t.Helper()
	dir := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/rp\n\ngo 1.21\n")
	write("internal/util/util.go", "package util\n\nfunc Helper() {}\n")
	write("internal/store/store.go", `package store

import "os"
import "io"

func Put(path string) { os.ReadFile(path) }

func Get(r io.Reader) { io.ReadAll(r) }

func Other() {}
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestReplaceSites(t *testing.T) {
	p := replaceFixture(t)
	c := ExtractIssueConcepts("perf: use mmap\n\n" +
		"Replace `os.ReadFile` / `io.ReadAll` with mmap.\n\n" +
		"Add wrapper in `internal/util/mmap.go`. Use it in `internal/store/store.go`.")
	np := p.buildNewCodePlan(c)
	if np == nil {
		t.Fatal("buildNewCodePlan returned nil")
	}
	if len(np.newPaths) != 1 || np.newPaths[0] != "internal/util/mmap.go" {
		t.Errorf("newPaths = %v", np.newPaths)
	}
	if len(np.replace) != 2 {
		t.Fatalf("replace sites = %v, want Put and Get", np.replace)
	}
	names := map[string]bool{}
	for _, r := range np.replace {
		names[r.fn.Name] = true
	}
	if !names["Put"] || !names["Get"] {
		t.Errorf("replace sites missing Put/Get: %v", names)
	}
	if names["Other"] {
		t.Errorf("Other doesn't call the named calls: %v", names)
	}
	out := p.GuideIssue(c)
	for _, want := range []string{"REPLACE", "Put", "os.ReadFile", "Get", "io.ReadAll"} {
		if !strings.Contains(out, want) {
			t.Errorf("plan missing %q:\n%s", want, out)
		}
	}
}

func TestReplaceNeedsSignal(t *testing.T) {
	p := replaceFixture(t)
	// Names files and calls but no replacement language -> no REPLACE.
	c := ExtractIssueConcepts("perf: use mmap\n\n" +
		"Consider `os.ReadFile` usage.\n\n" +
		"Add wrapper in `internal/util/mmap.go`. See `internal/store/store.go`.")
	np := p.buildNewCodePlan(c)
	if np == nil {
		t.Fatal("buildNewCodePlan returned nil")
	}
	if len(np.replace) != 0 {
		t.Errorf("no replace signal — replace should be empty, got %v", np.replace)
	}
}

// cmdFixture: a cobra-style cmd package; the issue adds a new command.
func cmdFixture(t *testing.T) *Project {
	t.Helper()
	dir := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/cmdx\n\ngo 1.21\n")
	write("cmd/save.go", `package cmd

import "github.com/spf13/cobra"

var saveCmd = &cobra.Command{
	Use:   "save",
	Short: "Save a snapshot",
	RunE: func(cmd *cobra.Command, args []string) error {
		return nil
	},
}
`)
	write("cmd/restore.go", `package cmd

import "github.com/spf13/cobra"

var restoreCmd = &cobra.Command{
	Use:   "restore",
	Short: "Restore a snapshot",
	RunE: func(cmd *cobra.Command, args []string) error {
		return nil
	},
}
`)
	write("util/util.go", `package util

// Wipe removes everything under dir.
func Wipe(dir string) error { return nil }
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestSiblingCommands(t *testing.T) {
	p := cmdFixture(t)
	c := ExtractIssueConcepts("Add `mytool copy` command\n\nCreate `cmd/copy.go` with Cobra command `copyCmd`.")
	np := p.buildNewCodePlan(c)
	if np == nil {
		t.Fatal("buildNewCodePlan returned nil")
	}
	if len(np.follow) != 2 {
		t.Fatalf("follow = %v, want saveCmd and restoreCmd", np.follow)
	}
	for _, s := range np.follow {
		if !strings.Contains(s.label, "Cobra command") {
			t.Errorf("sibling %v missing Cobra label", s)
		}
	}
	if np.follow[0].intent == "" && np.follow[1].intent == "" {
		t.Errorf("follow intents empty: %v", np.follow)
	}
	intents := map[string]bool{}
	for _, s := range np.follow {
		intents[s.intent] = true
	}
	if !intents["Save a snapshot"] || !intents["Restore a snapshot"] {
		t.Errorf("intents missing Short descriptions: %v", np.follow)
	}
	// Sibling commands replace the generic package-API targets.
	out := p.GuideIssue(c)
	if strings.Contains(out, "TARGETS") {
		t.Errorf("TARGETS should be suppressed when FOLLOW fires:\n%s", out)
	}
	if !strings.Contains(out, "FOLLOW") {
		t.Errorf("plan missing FOLLOW:\n%s", out)
	}
}

func TestReuseFuncs(t *testing.T) {
	p := cmdFixture(t)
	c := ExtractIssueConcepts("Add `mytool copy` command\n\nCreate `cmd/copy.go`. Reuse `internal/util/util.go` Wipe() for cleanup.")
	np := p.buildNewCodePlan(c)
	if np == nil {
		t.Fatal("buildNewCodePlan returned nil")
	}
	if len(np.reuse) != 1 || np.reuse[0].Name != "Wipe" {
		t.Fatalf("reuse = %v, want [Wipe]", np.reuse)
	}
	out := p.GuideIssue(c)
	if !strings.Contains(out, "REUSE") || !strings.Contains(out, "Wipe removes everything") {
		t.Errorf("plan missing REUSE with intent:\n%s", out)
	}
}

func TestReuseNeedsSignal(t *testing.T) {
	p := cmdFixture(t)
	// Mentions Wipe() but no reuse language -> no REUSE section.
	c := ExtractIssueConcepts("Add `mytool copy` command\n\nCreate `cmd/copy.go`. Maybe Wipe() helps.")
	np := p.buildNewCodePlan(c)
	if np == nil {
		t.Fatal("buildNewCodePlan returned nil")
	}
	if len(np.reuse) != 0 {
		t.Errorf("no reuse signal — reuse should be empty, got %v", np.reuse)
	}
}
