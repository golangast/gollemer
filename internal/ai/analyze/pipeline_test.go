package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The pipeline traces main's calls: the fixture's main calls store.New
// and Save, so both must appear under main.
func TestPipelineTracesMain(t *testing.T) {
	root := fixture(t)
	p, err := Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	out := p.Pipeline()
	if !strings.Contains(out, "PIPELINE:") {
		t.Errorf("Pipeline missing header:\n%s", out)
	}
	if !strings.Contains(out, "main") {
		t.Errorf("Pipeline missing main:\n%s", out)
	}
	// store.New is called by main; the tree must show it as a callee.
	lines := strings.Split(out, "---")
	_ = lines
	found := false
	for _, l := range strings.Split(out, "\n") {
		if strings.Contains(l, "▶") && strings.Contains(l, "store.New") {
			found = true
		}
	}
	if !found {
		t.Errorf("Pipeline should show store.New as a callee of main:\n%s", out)
	}
}

// The story narrates the run in plain words.
func TestStory(t *testing.T) {
	root := fixture(t)
	p, err := Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	out := p.Story()
	if !strings.Contains(out, "main()") {
		t.Errorf("Story should narrate main():\n%s", out)
	}
	if !strings.Contains(out, "main.go") {
		t.Errorf("Story should name main's file:\n%s", out)
	}
}

// A library (no func main) gets the API-surface treatment instead.
func TestPipelineLibrary(t *testing.T) {
	root := t.TempDir()
	write := func(rel, src string) {
		t.Helper()
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/lib\n\ngo 1.21\n")
	write("lib.go", "package lib\n\n// Greet says hello.\nfunc Greet() string { return \"hi\" }\n")
	p, err := Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	if out := p.Pipeline(); !strings.Contains(out, "library") {
		t.Errorf("library Pipeline should say so:\n%s", out)
	}
	if out := p.Story(); !strings.Contains(out, "library") {
		t.Errorf("library Story should say so:\n%s", out)
	}
}
