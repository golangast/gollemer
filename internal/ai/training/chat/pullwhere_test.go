package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

func TestPullTarget(t *testing.T) {
	cases := []struct {
		line             string
		owner, repo, num string
	}{
		{"https://github.com/pashkov256/deletor/pull/328", "pashkov256", "deletor", "328"},
		{"Find where to edit https://github.com/pashkov256/deletor/pull/328", "pashkov256", "deletor", "328"},
		{"where for https://github.com/owner/repo/pull/12", "owner", "repo", "12"},
		{"https://github.com/pashkov256/deletor/issues/320", "", "", ""},
		{"just some chat text", "", "", ""},
	}
	for _, c := range cases {
		o, r, n := pullTarget(c.line)
		if o != c.owner || r != c.repo || n != c.num {
			t.Errorf("pullTarget(%q) = %q/%q/%q, want %q/%q/%q",
				c.line, o, r, n, c.owner, c.repo, c.num)
		}
	}
}

func pullRenderFixture(t *testing.T) *analyze.Project {
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
	write("go.mod", "module example.com/pr\n\ngo 1.21\n")
	write("report/report.go", "package report\n\n// Writer writes JSON reports to disk.\ntype Writer struct{}\n")
	write("main.go", "package main\n\nfunc main() {}\n")
	p, err := analyze.Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRenderPull(t *testing.T) {
	p := pullRenderFixture(t)
	pr := &githubPull{Title: "Add report flag", Body: "Closes #320"}
	files := []githubPullFile{
		{Filename: "main.go", Status: "modified", Additions: 5, Deletions: 2},
		{Filename: "report/report.go", Status: "added", Additions: 120},
	}
	out := renderPull(p, "328", pr, files)
	for _, want := range []string{
		"PR #328: Add report flag",
		"Implements issue #320.",
		"2 FILES (+125 −2)",
		"NEW FILES",
		"CHANGED",
		"report/report.go",
		"the Writer struct",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("renderPull output missing %q\n---\n%s", want, out)
		}
	}
	// New files list before changed files.
	if strings.Index(out, "NEW FILES") > strings.Index(out, "CHANGED") {
		t.Errorf("NEW FILES should come before CHANGED\n---\n%s", out)
	}
}
