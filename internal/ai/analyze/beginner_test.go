package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// beginnerFixture builds a tiny project on disk: main calls Compute,
// a test exercises it, and Container depends on Widget.
func beginnerFixture(t *testing.T) *Project {
	t.Helper()
	root := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("main.go", "package demo\n\nfunc main() {\n\tx := Compute(2)\n\t_ = x\n}\n")
	write("lib.go", "package demo\n\n// Compute doubles a number.\nfunc Compute(n int) int { return n * 2 }\n")
	write("lib_test.go", "package demo\n\nimport \"testing\"\n\nfunc TestCompute(t *testing.T) {\n\tif Compute(2) != 4 {\n\t\tt.Fatal(\"bad\")\n\t}\n}\n")

	mainFn := &Func{ID: "demo.main", Name: "main", IsMain: true, File: "main.go", Line: 3, EndLine: 6, Calls: []string{"demo.Compute"}}
	compute := &Func{ID: "demo.Compute", Name: "Compute", Exported: true, File: "lib.go", Line: 4, EndLine: 4,
		Doc: "Compute doubles a number.", Callers: []string{"demo.main", "demo.TestCompute"}}
	testFn := &Func{ID: "demo.TestCompute", Name: "TestCompute", IsTest: true, File: "lib_test.go", Line: 5, EndLine: 9}
	get := &Func{ID: "demo.Container.Get", Name: "Get", Receiver: "*Container", File: "lib.go", Line: 4,
		Sig: "func (c *Container) Get() *Widget", Src: "func (c *Container) Get() *Widget { c.mu.Lock(); return c.w }"}
	container := &Type{Name: "Container", Kind: "struct", Exported: true, Doc: "Container holds everything.",
		Fields: []string{"items", "mu"}, Methods: []string{"Get", "Put"}}
	widget := &Type{Name: "Widget", Kind: "struct", Exported: true, Fields: []string{"Name"}}
	pkg := &Package{Name: "demo", Dir: "", Types: []*Type{container, widget},
		Funcs: []*Func{mainFn, compute, testFn, get}}
	p := &Project{Root: root, Module: "example.com/demo", Packages: []*Package{pkg}, Files: 3, Lines: 20}
	p.byID = map[string]*Func{"demo.main": mainFn, "demo.Compute": compute, "demo.TestCompute": testFn, "demo.Container.Get": get}
	return p
}

func TestExamplesText(t *testing.T) {
	p := beginnerFixture(t)
	out := p.ExamplesText(p.FuncByID("demo.Compute"))
	for _, want := range []string{
		"How Compute is really used",
		"called from 1 place(s)",
		"exercised by 1 test(s)",
		"Called by demo.main main.go:4",
		"x := Compute(2)",
		"In a test — TestCompute",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("ExamplesText missing %q\n---\n%s", want, out)
		}
	}
	// Uncalled exported function: explains the public surface.
	orphan := &Func{ID: "demo.Lonely", Name: "Lonely", Exported: true}
	out = p.ExamplesText(orphan)
	if !strings.Contains(out, "part of the public surface") {
		t.Errorf("orphan explanation wrong\n---\n%s", out)
	}
}

func TestGlossaryText(t *testing.T) {
	p := beginnerFixture(t)
	out := p.GlossaryText()
	for _, want := range []string{"Container — Container holds everything.", "Widget — a struct holding Name"} {
		if !strings.Contains(out, want) {
			t.Errorf("GlossaryText missing %q\n---\n%s", want, out)
		}
	}
}

func TestLearnPathOrder(t *testing.T) {
	p := beginnerFixture(t)
	out := p.LearnPathText()
	wi := strings.Index(out, "Widget")
	ci := strings.Index(out, "Container")
	if wi < 0 || ci < 0 || wi > ci {
		t.Errorf("Widget (dependency) must come before Container\n---\n%s", out)
	}
	if !strings.Contains(out, "mutexes (https://gobyexample.com/mutexes)") {
		t.Errorf("LearnPathText missing the mutex feature link\n---\n%s", out)
	}
	if !strings.Contains(out, "Builds on: Widget") {
		t.Errorf("LearnPathText missing Builds on line\n---\n%s", out)
	}
}

func TestLearnPathDuplicateNames(t *testing.T) {
	// Two packages may both define "Container" (izidic does). The
	// path builder must not panic or merge them.
	a := &Type{Name: "Container", Kind: "struct", Exported: true, PkgDir: "core"}
	bb := &Type{Name: "Container", Kind: "struct", Exported: true, PkgDir: "examples/di"}
	p := &Project{Packages: []*Package{
		{Name: "core", Dir: "core", Types: []*Type{a}},
		{Name: "di", Dir: "examples/di", Types: []*Type{bb}},
	}}
	p.byID = map[string]*Func{}
	out := p.LearnPathText()
	for _, want := range []string{"Container (in core)", "Container (in examples/di)"} {
		if !strings.Contains(out, want) {
			t.Errorf("duplicate-name learn path missing %q\n---\n%s", want, out)
		}
	}
}

func TestTourStops(t *testing.T) {
	p := beginnerFixture(t)
	stops := p.TourStops()
	if len(stops) < 5 {
		t.Fatalf("want at least 5 tour stops, got %d: %v", len(stops), stops)
	}
	if !strings.Contains(stops[0], "Welcome to example.com/demo") {
		t.Errorf("first stop should be the welcome\n---\n%s", stops[0])
	}
	joined := strings.Join(stops, "\n")
	for _, want := range []string{"Start where the program starts: main", "The cast", "engine room", "1 tests"} {
		if !strings.Contains(joined, want) {
			t.Errorf("tour missing %q\n---\n%s", want, joined)
		}
	}
}
