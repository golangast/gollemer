package analyze

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The component is one source of truth: the whole-file form the
// creations write is exactly the shared body inside a main, and
// both forms are valid Go.
func TestWebServerComponentConsistency(t *testing.T) {
	if !strings.Contains(WebServerSource, WebServerMainBody) {
		t.Fatal("WebServerSource does not contain WebServerMainBody verbatim — the two write paths would drift")
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "main.go", WebServerSource, parser.ParseComments)
	if err != nil {
		t.Fatalf("WebServerSource does not parse: %v", err)
	}
	if f.Name.Name != "main" {
		t.Errorf("component package = %s, want main", f.Name.Name)
	}
	for _, marker := range []string{"http.HandleFunc", "http.ListenAndServe", `"net/http"`, `"fmt"`} {
		if !strings.Contains(WebServerSource, marker) {
			t.Errorf("component missing %s", marker)
		}
	}
}

// Regression: the project panel rendered the line count as
// "Lines ~1", whose tilde reads as a minus at terminal sizes —
// "~1" was screenshotted as "−1" for a one-line file. The count
// is exact (Analyze sums each parsed file's lines), so it is
// shown as the plain number like its sibling stats.
func TestSummaryLinesShowsRealCount(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "jim"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "jim", "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if p.Lines != 1 {
		t.Fatalf("Lines = %d, want 1", p.Lines)
	}
	s := p.Summary()
	if !strings.Contains(s, "Lines") || !strings.Contains(s, "Lines 1") {
		t.Errorf("summary does not show the real count:\n%s", s)
	}
	if strings.Contains(s, "Lines ~") || strings.Contains(s, "Lines -") {
		t.Errorf("summary still decorates the count into a false negative:\n%s", s)
	}
}
