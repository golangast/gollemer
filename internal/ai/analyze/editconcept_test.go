package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The concept book (editconcept.go): rows are data. These
// tests pin the loader's discipline (a malformed row is
// skipped, the rest of the book still loads), the generic
// planner's two assembly classes, and the slot bindings the
// user's approved entries need — print line exactly as he
// specified it (fmt.Println("")), interface with methods only
// ever copied from a real type, never invented.

const testBookJSON = `{
  "concepts": [
    {"id": "print-line", "names": ["print line", "println"], "kind": "statement",
     "template": "fmt.Println(\"{text}\")", "imports": ["fmt"]},
    {"id": "interface", "names": ["interface"], "kind": "declaration",
     "template": "type {name} interface {\n{methods}\n}"}
  ]
}`

func testBook(t *testing.T) (printLine, iface ConceptEntry) {
	t.Helper()
	entries, skipped := ParseConceptBook([]byte(testBookJSON))
	if skipped != 0 || len(entries) != 2 {
		t.Fatalf("test book parsed to %d entries, %d skipped", len(entries), skipped)
	}
	return entries[0], entries[1]
}

// Loader discipline: every flavour of bad row is skipped and
// counted; the good rows around them still load.
func TestParseConceptBookSkipsMalformedRows(t *testing.T) {
	blob := []byte(`{"concepts": [
		{"id": "ok-one", "names": ["widget"], "kind": "statement", "template": "widget()"},
		42,
		{"id": "no-kind", "names": ["gadget"], "template": "gadget()"},
		{"id": "bad-kind", "names": ["gadget"], "kind": "spell", "template": "gadget()"},
		{"id": "no-names", "names": [], "kind": "statement", "template": "x()"},
		{"id": "no-template", "names": ["gadget"], "kind": "statement"},
		{"id": "ok-one", "names": ["dupe"], "kind": "statement", "template": "dupe()"},
		{"id": "ok-two", "names": ["sprocket"], "kind": "declaration", "template": "type {name} struct{}"}
	]}`)
	entries, skipped := ParseConceptBook(blob)
	if len(entries) != 2 {
		t.Fatalf("entries = %d, want 2 (the two good rows): %+v", len(entries), entries)
	}
	if skipped != 6 {
		t.Errorf("skipped = %d, want 6", skipped)
	}
	if entries[0].ID != "ok-one" || entries[1].ID != "ok-two" {
		t.Errorf("surviving rows = %q, %q", entries[0].ID, entries[1].ID)
	}
	if got, sk := ParseConceptBook([]byte("not json at all")); len(got) != 0 || sk != 0 {
		t.Errorf("garbage file = %d entries, %d skipped — want none of either", len(got), sk)
	}
}

// conceptFixture: a module with a program file (main with a
// statement, plus a helper), a bare package-clause file, and a
// store package whose type carries methods — one leaning on an
// imported package (time), plus an interface type of its own.
func conceptFixture(t *testing.T) (string, *Project) {
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
	write("go.mod", "module example.com/concepts\n\ngo 1.21\n")
	write("main.go", `package main

func Helper() string {
	return "helping"
}

func main() {
	println(Helper())
}
`)
	write("bare.go", "package main\n")
	write("store/store.go", `package store

import "time"

type Store struct{}

func (s *Store) Load(path string) (string, error) { return "", nil }

func (s *Store) Save(name string, data []byte) error { return nil }

func (s *Store) Expire(d time.Duration) bool { return false }

type Readable interface {
	Read(p []byte) (int, error)
	Close() error
}

type Empty struct{}

func (e *Empty) Nothing() {}

type Lonely struct{}
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return dir, p
}

func conceptPlan(t *testing.T, p *Project, req EditRequest) *EditPlan {
	t.Helper()
	plan, refusal := p.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("PlanEdit(%s) refused: %s", req.Kind, refusal)
	}
	return plan
}

func conceptRefusal(t *testing.T, p *Project, req EditRequest, want string) {
	t.Helper()
	plan, refusal := p.PlanEdit(req)
	if plan != nil {
		t.Fatalf("PlanEdit produced a plan, want refusal %q", want)
	}
	if !strings.Contains(refusal, want) {
		t.Errorf("refusal = %q, want it to say %q", refusal, want)
	}
}

// The user's exact spec: an unadorned print line lands as
// fmt.Println("") — empty string, fmt imported, at the end of
// main — and verifies against the re-analyzed project.
func TestConceptPrintLineDefault(t *testing.T) {
	dir, p := conceptFixture(t)
	printLine, _ := testBook(t)
	plan := conceptPlan(t, p, EditRequest{Kind: EditConcept, File: "main.go", Concept: &printLine})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, "\tfmt.Println(\"\")\n") {
		t.Errorf("print line not at the end of main:\n%s", after)
	}
	if !strings.Contains(after, `"fmt"`) {
		t.Errorf("fmt import missing:\n%s", after)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("plan does not verify against the applied project")
	}
}

// The saying slot: the user's words go inside the quotes, with
// Go string escaping doing the work — a quote and a newline in
// the text come out escaped, never raw.
func TestConceptPrintLineSayingEscapes(t *testing.T) {
	_, p := conceptFixture(t)
	printLine, _ := testBook(t)
	plan := conceptPlan(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &printLine,
		ConceptText: "say \"hi\"\nthen go",
	})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, `fmt.Println("say \"hi\"\nthen go")`) {
		t.Errorf("saying text not escaped into the literal:\n%s", after)
	}
}

// The in-<Func> slot: the statement lands in the named function,
// not in main.
func TestConceptPrintLineInNamedFunction(t *testing.T) {
	dir, p := conceptFixture(t)
	printLine, _ := testBook(t)
	plan := conceptPlan(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &printLine, ConceptFunc: "Helper",
	})
	after := string(plan.Files[0].After)
	helperAt := strings.Index(after, "func Helper")
	mainAt := strings.Index(after, "func main")
	printAt := strings.Index(after, `fmt.Println("")`)
	if !(helperAt >= 0 && printAt > helperAt && printAt < mainAt) {
		t.Errorf("print line did not land inside Helper:\n%s", after)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("plan does not verify (verify must aim at Helper, not main)")
	}
}

func TestConceptPrintLineMissingFunctionRefuses(t *testing.T) {
	_, p := conceptFixture(t)
	printLine, _ := testBook(t)
	conceptRefusal(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &printLine, ConceptFunc: "Nope",
	}, "has no function named \"Nope\"")
}

// A bare package-clause file has no main: the print line gets a
// main to live in (the add-server precedent), and the result
// parses and verifies.
func TestConceptPrintLineBareFileGrowsMain(t *testing.T) {
	dir, p := conceptFixture(t)
	printLine, _ := testBook(t)
	plan := conceptPlan(t, p, EditRequest{Kind: EditConcept, File: "bare.go", Concept: &printLine})
	after := string(plan.Files[0].After)
	for _, want := range []string{"func main() {", "\tfmt.Println(\"\")", `"fmt"`} {
		if !strings.Contains(after, want) {
			t.Errorf("bare-file plan missing %q:\n%s", want, after)
		}
	}
	if len(plan.Notes) == 0 || !strings.Contains(plan.Notes[0], "had no main") {
		t.Errorf("plan notes do not explain the grown main: %v", plan.Notes)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("bare-file plan does not verify")
	}
}

// Interface, bare: named, empty body, the TODO where methods
// go, and a plan note saying the methods are the user's — never
// invented.
func TestConceptInterfaceBareIsTODO(t *testing.T) {
	dir, p := conceptFixture(t)
	_, iface := testBook(t)
	plan := conceptPlan(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &iface, ConceptName: "Storer",
	})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, "type Storer interface {") {
		t.Errorf("declaration missing:\n%s", after)
	}
	if !strings.Contains(after, "// TODO: add the methods this interface needs") {
		t.Errorf("empty interface carries no TODO:\n%s", after)
	}
	joined := strings.Join(plan.Notes, " ")
	if !strings.Contains(joined, "never invent") {
		t.Errorf("plan notes do not own the empty method block: %v", plan.Notes)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("interface plan does not verify")
	}
}

func TestConceptInterfaceNamelessRefuses(t *testing.T) {
	_, p := conceptFixture(t)
	_, iface := testBook(t)
	conceptRefusal(t, p, EditRequest{Kind: EditConcept, File: "main.go", Concept: &iface},
		"needs a name")
}

func TestConceptInterfaceNameCollisionRefuses(t *testing.T) {
	_, p := conceptFixture(t)
	_, iface := testBook(t)
	conceptRefusal(t, p, EditRequest{
		Kind: EditConcept, File: "store/store.go", Concept: &iface, ConceptName: "Store",
	}, "already exists in package store")
}

// Interface like a struct type: the type's REAL method set,
// signatures verbatim from the index, and the import one of
// them leans on (time) copied in.
func TestConceptInterfaceLikeCopiesRealMethods(t *testing.T) {
	dir, p := conceptFixture(t)
	_, iface := testBook(t)
	plan := conceptPlan(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &iface,
		ConceptName: "Storer", ConceptLike: "Store",
	})
	after := string(plan.Files[0].After)
	for _, want := range []string{
		"type Storer interface {",
		"Load(path string) (string, error)",
		"Save(name string, data []byte) error",
		"Expire(d time.Duration) bool",
		`"time"`,
	} {
		if !strings.Contains(after, want) {
			t.Errorf("like-plan missing %q:\n%s", want, after)
		}
	}
	if strings.Contains(after, "TODO") {
		t.Errorf("like-plan kept the TODO:\n%s", after)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("like-plan does not verify")
	}
}

// Interface like an interface type: its declared methods are
// copied the same way.
func TestConceptInterfaceLikeInterface(t *testing.T) {
	_, p := conceptFixture(t)
	_, iface := testBook(t)
	plan := conceptPlan(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &iface,
		ConceptName: "ReadCloser", ConceptLike: "Readable",
	})
	after := string(plan.Files[0].After)
	for _, want := range []string{"Read(p []byte) (int, error)", "Close() error"} {
		if !strings.Contains(after, want) {
			t.Errorf("interface-like plan missing %q:\n%s", want, after)
		}
	}
}

func TestConceptInterfaceLikeMissingTypeRefuses(t *testing.T) {
	_, p := conceptFixture(t)
	_, iface := testBook(t)
	conceptRefusal(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &iface,
		ConceptName: "Storer", ConceptLike: "Nope",
	}, "couldn't find a type named \"Nope\"")
}

func TestConceptInterfaceLikeMethodlessTypeRefuses(t *testing.T) {
	_, p := conceptFixture(t)
	_, iface := testBook(t)
	conceptRefusal(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &iface,
		ConceptName: "X", ConceptLike: "Lonely",
	}, "has no methods")
}

// The planner is generic by KIND, proven at this level too: a
// statement entry the shipped book doesn't have assembles from
// its row alone.
func TestConceptPlannerThirdConceptFromData(t *testing.T) {
	entries, skipped := ParseConceptBook([]byte(`{"concepts": [
		{"id": "for-loop", "names": ["for loop"], "kind": "statement",
		 "template": "for i := 0; i < 3; i++ {\n\t// TODO: the loop body\n}"}
	]}`))
	if skipped != 0 || len(entries) != 1 {
		t.Fatalf("fixture row parsed to %d entries, %d skipped", len(entries), skipped)
	}
	_, p := conceptFixture(t)
	plan := conceptPlan(t, p, EditRequest{Kind: EditConcept, File: "main.go", Concept: &entries[0]})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, "for i := 0; i < 3; i++ {") {
		t.Errorf("for-loop row did not assemble:\n%s", after)
	}
}
