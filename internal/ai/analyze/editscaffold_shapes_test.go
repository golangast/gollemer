package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// shapesProject is the fixture for the scaffold extension:
// one package holding every target-selection situation the
// add-field rule must read — a struct with a constructor, a
// struct with a most-called method, a struct with no
// constructor and no methods, a struct whose methods tie, a
// generic struct — plus a switch function to add a case to, a
// function to clone, a generic function to clone, and a main
// whose switch must NOT be scaffolded.
func shapesProject(t *testing.T) (string, *Project) {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"go.mod": "module example.com/shapes\n\ngo 1.21\n",
		"lib/lib.go": `package lib

type Config struct {
	Host string
	Port int
}

func NewConfig(host string) *Config {
	return &Config{Host: host, Port: 80}
}

func (c *Config) Label() string { return c.Host }

type Store struct {
	Name string
}

func (s *Store) Save(key string) error { return nil }

func (s *Store) Load(key string) (string, error) { return "", nil }

func PutOne(s *Store) { _ = s.Save("a") }

func PutTwo(s *Store) { _ = s.Save("b") }

func GetOne(s *Store) { _, _ = s.Load("a") }

type Orphan struct {
	X int
}

func UseOrphan(o Orphan) int { return o.X }

type Pair struct {
	V int
}

func (p *Pair) Left() int { return p.V }

func (p *Pair) Right() int { return p.V }

func CallLeft(p *Pair) int { return p.Left() }

func CallRight(p *Pair) int { return p.Right() }

type Bag[T any] struct {
	Items []T
}

func (b *Bag[T]) Add(v T) { b.Items = append(b.Items, v) }

func Greet(name string) string { return "hello " + name }

func First[T any](items []T) T {
	if len(items) > 0 {
		return items[0]
	}
	var zero T
	return zero
}
`,
		"lib/parse.go": `package lib

func Parse(mode string) string {
	switch mode {
	case "fast":
		return "fast lane"
	default:
		return "slow lane"
	}
}
`,
		"app/main.go": `package main

func main() {
	mode := "fast"
	switch mode {
	case "fast":
		println("fast")
	}
}
`,
	}
	for rel, src := range files {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	p, err := Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze shapes fixture: %v", err)
	}
	return dir, p
}

// planScaffoldFile returns the plan's scaffold file (the one
// ending in _test.go), or "" when the plan carries none.
func planScaffoldFile(plan *EditPlan) (string, string) {
	for _, f := range plan.Files {
		if strings.HasSuffix(f.Rel, "_test.go") {
			return f.Rel, string(f.After)
		}
	}
	return "", ""
}

func planNotes(plan *EditPlan) string { return strings.Join(plan.Notes, "\n") }

// vetAndTest runs vet over the fixture and then its whole
// suite — the no-verdict-poisoning check: with the scaffold
// files in place, everything still compiles and passes.
func vetAndTest(t *testing.T, dir string) {
	t.Helper()
	if out, ok := runGoIn(t, dir, "vet", "./..."); !ok {
		t.Errorf("fixture vet with scaffold in place:\n%s", out)
	}
	if out, ok := runGoIn(t, dir, "test", "./..."); !ok {
		t.Errorf("fixture suite with scaffold in place:\n%s", out)
	}
}

func TestScaffoldCloneFuncShape(t *testing.T) {
	dir, p := shapesProject(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddFunc, SiblingName: "Greet", NewName: "GreetLoudly"})
	if len(plan.Files) != 2 {
		t.Fatalf("clone plan files = %d, want 2 (edit + scaffold)", len(plan.Files))
	}
	rel, sc := planScaffoldFile(plan)
	if rel != "lib/lib_test.go" {
		t.Fatalf("clone scaffold file = %s, want lib/lib_test.go", rel)
	}
	for _, want := range []string{
		"func TestGreetLoudly(t *testing.T)",
		"function this edit copies from Greet",
		`name: "zero values"`,
		`name: "edge: name — non-empty string"`,
		"got := GreetLoudly(tc.args.name)",
		"t.Skip(\"scaffold: fill in the want/wantErr TODOs below",
		"nothing verifies this edit with it",
	} {
		if !scContains(sc, want) {
			t.Errorf("clone scaffold missing %q:\n%s", want, sc)
		}
	}
	if notes := planNotes(plan); !strings.Contains(notes, "test scaffold") ||
		!strings.Contains(notes, "not part of how the edit is verified") {
		t.Errorf("clone plan notes don't label the scaffold honestly:\n%s", notes)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	vetAndTest(t, dir)
	out, ok := runGoIn(t, dir, "test", "-v", "-run", "TestGreetLoudly", "./lib/")
	if !ok || !strings.Contains(out, "--- SKIP: TestGreetLoudly") {
		t.Errorf("clone scaffold should run as SKIP:\n%s", out)
	}
}

func TestScaffoldCloneMethodShape(t *testing.T) {
	dir, p := shapesProject(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddFunc, SiblingName: "Store.Save", NewName: "SaveAll"})
	_, sc := planScaffoldFile(plan)
	if sc == "" {
		t.Fatalf("method clone plan carries no scaffold; notes:\n%s", planNotes(plan))
	}
	for _, want := range []string{
		"func TestSaveAll(t *testing.T)",
		"s *Store",
		"err := tc.args.s.SaveAll(tc.args.key)",
		`s: &Store{Name: "test-string"},`,
	} {
		if !scContains(sc, want) {
			t.Errorf("method clone scaffold missing %q:\n%s", want, sc)
		}
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	vetAndTest(t, dir)
	out, ok := runGoIn(t, dir, "test", "-v", "-run", "TestSaveAll", "./lib/")
	if !ok || !strings.Contains(out, "--- SKIP: TestSaveAll") {
		t.Errorf("method clone scaffold should run as SKIP:\n%s", out)
	}
}

func TestScaffoldCloneGenericOmitted(t *testing.T) {
	_, p := shapesProject(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddFunc, SiblingName: "First", NewName: "FirstAgain"})
	if rel, _ := planScaffoldFile(plan); rel != "" {
		t.Errorf("generic clone carries a scaffold file (%s) — it must be omitted", rel)
	}
	if notes := planNotes(plan); !strings.Contains(notes, "No test scaffold added") || !strings.Contains(notes, "generic") {
		t.Errorf("generic clone plan doesn't explain the omission:\n%s", notes)
	}
}

func TestScaffoldAddCaseShape(t *testing.T) {
	dir, p := shapesProject(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddCase, FuncName: "Parse", CaseExpr: "backoff"})
	rel, sc := planScaffoldFile(plan)
	if rel != "lib/parse_test.go" {
		t.Fatalf("add-case scaffold file = %s, want lib/parse_test.go", rel)
	}
	for _, want := range []string{
		"func TestParse(t *testing.T)",
		`alongside the new "backoff" case this edit adds to its switch`,
		"cases below exercise Parse's inputs generally — add a case for",
		"got := Parse(tc.args.mode)",
		"t.Skip(\"scaffold: fill in the want/wantErr TODOs below",
	} {
		if !scContains(sc, want) {
			t.Errorf("add-case scaffold missing %q:\n%s", want, sc)
		}
	}
	notes := planNotes(plan)
	if !strings.Contains(notes, "as it stands with the new case in place") ||
		!strings.Contains(notes, "yours to add in the TODOs") ||
		!strings.Contains(notes, "not part of how the edit is verified") {
		t.Errorf("add-case plan note isn't the honest accompanies-edit wording:\n%s", notes)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	vetAndTest(t, dir)
	out, ok := runGoIn(t, dir, "test", "-v", "-run", "TestParse", "./lib/")
	if !ok || !strings.Contains(out, "--- SKIP: TestParse") {
		t.Errorf("add-case scaffold should run as SKIP:\n%s", out)
	}
}

func TestScaffoldAddCaseInMainOmitted(t *testing.T) {
	_, p := shapesProject(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddCase, FuncName: "main", CaseExpr: "slow"})
	if rel, _ := planScaffoldFile(plan); rel != "" {
		t.Errorf("add-case in main carries a scaffold file (%s) — main can't be called by a test", rel)
	}
	if notes := planNotes(plan); !strings.Contains(notes, "No test scaffold added") ||
		!strings.Contains(notes, "entry point") {
		t.Errorf("add-case in main doesn't explain the omission:\n%s", notes)
	}
}

func TestScaffoldAddFieldConstructor(t *testing.T) {
	dir, p := shapesProject(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddField, StructName: "Config", FieldName: "Timeout", FieldType: "int"})
	rel, sc := planScaffoldFile(plan)
	if rel != "lib/lib_test.go" {
		t.Fatalf("add-field scaffold file = %s, want lib/lib_test.go", rel)
	}
	for _, want := range []string{
		"func TestNewConfig(t *testing.T)",
		"constructor of Config, written alongside the Timeout field",
		"host string",
		"got := NewConfig(tc.args.host)",
		"t.Skip(\"scaffold: fill in the want/wantErr TODOs below",
	} {
		if !scContains(sc, want) {
			t.Errorf("constructor scaffold missing %q:\n%s", want, sc)
		}
	}
	if notes := planNotes(plan); !strings.Contains(notes, "The scaffold aims at NewConfig, Config's constructor.") {
		t.Errorf("add-field plan note doesn't name the chosen target and why:\n%s", notes)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	vetAndTest(t, dir)
	out, ok := runGoIn(t, dir, "test", "-v", "-run", "TestNewConfig", "./lib/")
	if !ok || !strings.Contains(out, "--- SKIP: TestNewConfig") {
		t.Errorf("constructor scaffold should run as SKIP:\n%s", out)
	}
}

func TestScaffoldAddFieldMostCalledMethod(t *testing.T) {
	dir, p := shapesProject(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddField, StructName: "Store", FieldName: "TTL", FieldType: "int"})
	_, sc := planScaffoldFile(plan)
	if sc == "" {
		t.Fatalf("add-field (Store) plan carries no scaffold; notes:\n%s", planNotes(plan))
	}
	for _, want := range []string{
		"func TestSave(t *testing.T)",
		"written alongside the TTL field this edit adds to Store",
		"s *Store",
		"key string",
		"err := tc.args.s.Save(tc.args.key)",
		`s: &Store{Name: "test-string"},`,
	} {
		if !scContains(sc, want) {
			t.Errorf("method scaffold missing %q:\n%s", want, sc)
		}
	}
	if notes := planNotes(plan); !strings.Contains(notes, "The scaffold aims at Store.Save, Store's most-called method (2 callers in the index") {
		t.Errorf("add-field plan note doesn't name the chosen target and why:\n%s", notes)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	vetAndTest(t, dir)
	out, ok := runGoIn(t, dir, "test", "-v", "-run", "TestSave", "./lib/")
	if !ok || !strings.Contains(out, "--- SKIP: TestSave") {
		t.Errorf("method scaffold should run as SKIP:\n%s", out)
	}
}

func TestScaffoldAddFieldNoTarget(t *testing.T) {
	_, p := shapesProject(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddField, StructName: "Orphan", FieldName: "Y", FieldType: "string"})
	if rel, _ := planScaffoldFile(plan); rel != "" {
		t.Errorf("targetless struct carries a scaffold file (%s) — a target would have been invented", rel)
	}
	notes := planNotes(plan)
	if !strings.Contains(notes, "No test scaffold added") ||
		!strings.Contains(notes, "no NewOrphan constructor and no methods") ||
		!strings.Contains(notes, "I won't invent a target") {
		t.Errorf("no-target plan note doesn't say why, honestly:\n%s", notes)
	}
}

func TestScaffoldAddFieldTiedMethods(t *testing.T) {
	_, p := shapesProject(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddField, StructName: "Pair", FieldName: "W", FieldType: "int"})
	if rel, _ := planScaffoldFile(plan); rel != "" {
		t.Errorf("tied-methods struct carries a scaffold file (%s) — the target would have been a guess", rel)
	}
	if notes := planNotes(plan); !strings.Contains(notes, "called equally often") ||
		!strings.Contains(notes, "I won't guess") {
		t.Errorf("tie plan note doesn't say why, honestly:\n%s", notes)
	}
}

func TestScaffoldAddFieldGenericMethodOmitted(t *testing.T) {
	_, p := shapesProject(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddField, StructName: "Bag", FieldName: "Cap", FieldType: "int"})
	if rel, _ := planScaffoldFile(plan); rel != "" {
		t.Errorf("generic-method struct carries a scaffold file (%s) — it must be omitted", rel)
	}
	if notes := planNotes(plan); !strings.Contains(notes, "No test scaffold added") ||
		!strings.Contains(notes, "Bag.Add is generic") {
		t.Errorf("generic-method plan doesn't explain the omission:\n%s", notes)
	}
}

func TestScaffoldAddServerExcluded(t *testing.T) {
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
	write("go.mod", "module example.com/serveronly\n\ngo 1.21\n")
	write("web/main.go", "package main\n")
	p, err := Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze server fixture: %v", err)
	}
	plan := mustPlan(t, p, EditRequest{Kind: EditAddServer, File: "web/main.go"})
	if rel, _ := planScaffoldFile(plan); rel != "" {
		t.Errorf("webserver plan carries a scaffold file (%s) — the shape is excluded", rel)
	}
	if notes := planNotes(plan); strings.Contains(notes, "scaffold") {
		t.Errorf("webserver plan mentions a scaffold — exclusion should be silent:\n%s", notes)
	}
}
