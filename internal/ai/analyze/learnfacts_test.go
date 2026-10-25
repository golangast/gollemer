package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The structured learn path carries the same order, dependencies,
// and features LearnPathText renders — as data, for the chat's
// learn mode to walk one concept at a time.
func TestLearnPathStructured(t *testing.T) {
	p := beginnerFixture(t)
	concepts := p.LearnPath()
	if len(concepts) != 2 {
		t.Fatalf("LearnPath = %d concepts, want 2 (Widget, Container)", len(concepts))
	}
	if concepts[0].Name != "Widget" || concepts[1].Name != "Container" {
		t.Errorf("order = %s, %s — want Widget before Container (dependency first)",
			concepts[0].Name, concepts[1].Name)
	}
	c := concepts[1]
	if len(c.BuildsOn) != 1 || c.BuildsOn[0] != "Widget" {
		t.Errorf("Container.BuildsOn = %v, want [Widget]", c.BuildsOn)
	}
	foundMutex := false
	for _, f := range c.Features {
		if f[0] == "mutexes" && f[1] == "https://gobyexample.com/mutexes" {
			foundMutex = true
		}
	}
	if !foundMutex {
		t.Errorf("Container.Features = %v, want the mutexes feature", c.Features)
	}
	if c.Type == nil || c.Type.Name != "Container" {
		t.Errorf("concept's Type = %v, want the Container type", c.Type)
	}
}

// VarFacts exposes the varflow index as data: the same setters
// DataflowText renders for "where does AnswerCount get its data".
func TestVarFacts(t *testing.T) {
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
	write("go.mod", "module example.com/facts\n\ngo 1.21\n")
	write("main.go", `package main

// AnswerCount is how many answers have been loaded.
var AnswerCount int

// Limit caps the answers.
const Limit = 10

func main() { LoadAnswers() }

// LoadAnswers loads the answers.
func LoadAnswers() { AnswerCount = 42 }
`)
	p, err := Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	facts := p.VarFacts()
	byName := map[string]DataFact{}
	for _, f := range facts {
		byName[f.Name] = f
	}
	ac, ok := byName["AnswerCount"]
	if !ok {
		t.Fatalf("VarFacts missing AnswerCount: %+v", facts)
	}
	if ac.Kind != "var" || ac.Display != "main.AnswerCount" {
		t.Errorf("AnswerCount fact = %+v, want kind var, display main.AnswerCount", ac)
	}
	if len(ac.Setters) != 1 {
		t.Fatalf("AnswerCount setters = %+v, want exactly LoadAnswers", ac.Setters)
	}
	s := ac.Setters[0]
	if s.Short != "LoadAnswers" || s.Func != "main.LoadAnswers" || s.File != "main.go" || s.Line <= 0 {
		t.Errorf("setter = %+v, want main.LoadAnswers in main.go", s)
	}
	lim, ok := byName["Limit"]
	if !ok || lim.Kind != "const" || len(lim.Setters) != 0 {
		t.Errorf("Limit fact = %+v (present %v), want a const with no setters", lim, ok)
	}
	// The structured facts and the prose answer agree.
	if txt, ok := p.DataflowText("AnswerCount"); !ok || !strings.Contains(txt, "LoadAnswers") {
		t.Errorf("DataflowText disagrees with VarFacts:\n%s", txt)
	}
}
