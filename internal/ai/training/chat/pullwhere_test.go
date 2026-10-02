package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

func TestPullTarget(t *testing.T) {	cases := []struct {
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

func TestListURLRes(t *testing.T) {
	// Index URLs match; specific-issue/PR URLs do not.
	for _, line := range []string{
		"https://github.com/pashkov256/deletor/issues",
		"https://github.com/pashkov256/deletor/issues/",
		"Find where to edit https://github.com/pashkov256/deletor/issues",
	} {
		if m := issueListRe.FindStringSubmatch(line); m == nil {
			t.Errorf("issueListRe should match %q", line)
		}
	}
	for _, line := range []string{
		"https://github.com/pashkov256/deletor/issues/320",
		"https://github.com/pashkov256/deletor/pull/328",
	} {
		if m := issueListRe.FindStringSubmatch(line); m != nil {
			t.Errorf("issueListRe should NOT match %q", line)
		}
	}
	for _, line := range []string{
		"https://github.com/pashkov256/deletor/pulls",
		"https://github.com/pashkov256/deletor/pulls/",
	} {
		if m := pullListRe.FindStringSubmatch(line); m == nil {
			t.Errorf("pullListRe should match %q", line)
		}
	}
	if m := pullListRe.FindStringSubmatch("https://github.com/pashkov256/deletor/pull/328"); m != nil {
		t.Errorf("pullListRe should NOT match a specific PR URL")
	}
}

func TestExpandBareNumber(t *testing.T) {
	old := lastList
	defer func() { lastList = old }()
	lastList = []listedRef{
		{kind: "issues", owner: "o", repo: "r", num: "320", title: "t"},
		{kind: "pull", owner: "o", repo: "r", num: "328", title: "t"},
	}
	if got := expandBareNumber("320"); got != "https://github.com/o/r/issues/320" {
		t.Errorf("expandBareNumber(320) = %q", got)
	}
	if got := expandBareNumber("328"); got != "https://github.com/o/r/pull/328" {
		t.Errorf("expandBareNumber(328) = %q", got)
	}
	if got := expandBareNumber("999"); got != "999" {
		t.Errorf("unknown number should pass through, got %q", got)
	}
	lastList = nil
	if got := expandBareNumber("320"); got != "320" {
		t.Errorf("empty list should pass through, got %q", got)
	}
}
