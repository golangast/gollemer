package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// typeMethodFixture builds a project with a Widget type, sibling
// methods, and an options struct following the *Option pattern.
func typeMethodFixture(t *testing.T) *Project {
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
	write("go.mod", "module example.com/tm\n\ngo 1.21\n")
	write("widget/widget.go", `package widget

type Widget struct {
	Name string
}

// Describe returns a summary of the widget.
func (w *Widget) Describe() string { return w.Name }

// Render draws the widget.
func (w *Widget) Render() string { return "" }
`)
	write("widget/opts.go", `package widget

// PaintOption customizes painting.
type PaintOption struct {
	Color string
}

// Paint paints the widget.
func (w *Widget) Paint(options ...PaintOption) {}
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestTypeMethodPlan(t *testing.T) {
	p := typeMethodFixture(t)
	c := ExtractIssueConcepts("Implement Info method on widget\n\n" +
		"Show column types and counts. Please implement the verbose options as well.")
	tp := p.buildTypeMethodPlan(c)
	if tp == nil {
		t.Fatal("buildTypeMethodPlan returned nil")
	}
	if tp.typ.Name != "Widget" {
		t.Errorf("target type = %s, want Widget", tp.typ.Name)
	}
	if tp.methodName != "Info" {
		t.Errorf("method name = %s, want Info", tp.methodName)
	}
	if tp.receiver != "w" {
		t.Errorf("receiver = %s, want w", tp.receiver)
	}
	out := p.GuideIssue(c)
	for _, want := range []string{
		"TARGET",
		"type Widget struct",
		"func (w *Widget) Info(verbose ...InfoOption)",
		"SIBLINGS",
		"Describe",
		"OPTIONS",
		"PaintOption",
		"InfoOption",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("type-method plan missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "BEHAVIOR") {
		t.Errorf("type-method plan should not render BEHAVIOR:\n%s", out)
	}
}

func TestTypeMethodPlanSkipsExisting(t *testing.T) {
	p := typeMethodFixture(t)
	c := ExtractIssueConcepts("Implement Describe method on widget\n\nAdd more detail.")
	if tp := p.buildTypeMethodPlan(c); tp != nil {
		t.Errorf("method already exists — plan should be nil, got %+v", tp)
	}
}

func TestTypeMethodPlanNeedsMethodWord(t *testing.T) {
	p := typeMethodFixture(t)
	// No "<Name> method" phrasing -> not a new-method issue.
	c := ExtractIssueConcepts("Add a widget frobnicate\n\nMake it faster.")
	if tp := p.buildTypeMethodPlan(c); tp != nil {
		t.Errorf("no method named — plan should be nil, got %+v", tp)
	}
}

func TestTypeMethodPlanIgnoresGenericVerbs(t *testing.T) {
	p := typeMethodFixture(t)
	// "Add method" names no method.
	c := ExtractIssueConcepts("Add method to widget\n\nSomething new.")
	if tp := p.buildTypeMethodPlan(c); tp != nil {
		t.Errorf("generic verb must not name a method, got %+v", tp)
	}
}
