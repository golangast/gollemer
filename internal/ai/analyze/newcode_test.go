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
