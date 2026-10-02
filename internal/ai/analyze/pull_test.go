package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func pullFixture(t *testing.T) string {
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
	write("report/report_test.go", "package report\n\nimport \"testing\"\n\nfunc TestWriter(t *testing.T) {}\n")
	write("main.go", "package main\n\nfunc main() {}\n")
	return root
}

func TestDescribeFile(t *testing.T) {
	p, err := Analyze(pullFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct{ file, want string }{
		{"report/report.go", "the Writer struct"},
		{"report/report_test.go", "tests for package report"},
		{"main.go", "part of package main"},
	}
	for _, c := range cases {
		got := p.DescribeFile(c.file)
		if !strings.Contains(got, c.want) {
			t.Errorf("DescribeFile(%q) = %q, want it to contain %q", c.file, got, c.want)
		}
	}
	if got := p.DescribeFile("report/report.go"); !strings.Contains(got, "writes JSON reports") {
		t.Errorf("DescribeFile should use the doc comment, got %q", got)
	}
	if got := p.DescribeFile("does/not/exist.go"); got != "" {
		t.Errorf("DescribeFile of unknown file = %q, want empty", got)
	}
}
