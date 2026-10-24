package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// errorfixFixture builds a syntactically valid module for the
// error-fix classes. Line numbers in main.go are load-bearing —
// the tests point compiler errors at them:
//
//	16: x := 42          (unused local, pure)
//	21: y := calc.Add    (unused local, call-valued)
//	30: }                (ReturnsNothing's closing brace)
//	37: }                (NamedReturns's closing brace)
//	43: }                (StructReturn's closing brace)
func errorfixFixture(t *testing.T) (string, *Project) {
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
	write("go.mod", "module example.com/errfix\n\ngo 1.21\n")
	write("calc/calc.go", `package calc

func Add(a, b int) int { return a + b }

type Point struct {
	X int
	Y int
}

func Origin() Point { return Point{} }
`)
	write("main.go", `package main

import (
	"fmt"
	"os"

	"example.com/errfix/calc"
)

func main() {
	fmt.Println(calc.Add(1, 2))
	_ = os.Args
}

func UnusedLocal() int {
	x := 42
	return 1
}

func CallLocal() int {
	y := calc.Add(1, 2)
	return 1
}

func ReturnsNothing() int {
	n := calc.Add(2, 3)
	if n > 0 {
		return n
	}
}

func NamedReturns() (out int, err error) {
	out = 5
	if out > 10 {
		return
	}
}

func StructReturn() calc.Point {
	if true {
		return calc.Point{X: 1}
	}
}
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze fixture: %v", err)
	}
	return dir, p
}

func writeFixtureFile(t *testing.T, dir, rel, src string) *Project {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, rel), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func mustFixPlan(t *testing.T, p *Project, req ErrorFixRequest) *EditPlan {
	t.Helper()
	plan, refusal := p.PlanErrorFix(req)
	if refusal != "" || plan == nil {
		t.Fatalf("PlanErrorFix(%+v) = refusal %q", req, refusal)
	}
	return plan
}

func wantFixRefusal(t *testing.T, p *Project, req ErrorFixRequest, substr string) {
	t.Helper()
	plan, refusal := p.PlanErrorFix(req)
	if plan != nil {
		t.Fatalf("PlanErrorFix(%+v) planned an unsafe fix: %s", req, plan.Summary)
	}
	if substr != "" && !strings.Contains(refusal, substr) {
		t.Fatalf("PlanErrorFix(%+v) refusal = %q, want it to mention %q", req, refusal, substr)
	}
}

// (a) The unused import leaves a paren block; the used one stays.
func TestErrorFixUnusedImport(t *testing.T) {
	dir, _ := errorfixFixture(t)
	p := writeFixtureFile(t, dir, "extra.go", "package main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n)\n\nfunc Extra() { fmt.Println(\"x\") }\n")
	plan := mustFixPlan(t, p, ErrorFixRequest{Class: ErrorFixUnusedImport, File: "extra.go", Line: 5, Name: "os"})
	applyPlanToDisk(t, dir, plan)
	got := readFixtureFile(t, dir, "extra.go")
	if strings.Contains(got, "\"os\"") {
		t.Errorf("unused import survived:\n%s", got)
	}
	if !strings.Contains(got, "\"fmt\"") {
		t.Errorf("used import was removed too:\n%s", got)
	}
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("verify false after import removal")
	}
}

// (a2) Removing the only import of a lone (non-paren) declaration
// removes the whole declaration.
func TestErrorFixUnusedImportLoneDecl(t *testing.T) {
	dir, _ := errorfixFixture(t)
	p := writeFixtureFile(t, dir, "lone.go", "package main\n\nimport \"os\"\n\nfunc Lone() {}\n")
	plan := mustFixPlan(t, p, ErrorFixRequest{Class: ErrorFixUnusedImport, File: "lone.go", Line: 3, Name: "os"})
	applyPlanToDisk(t, dir, plan)
	got := readFixtureFile(t, dir, "lone.go")
	if strings.Contains(got, "import") || strings.Contains(got, "\"os\"") {
		t.Errorf("lone import declaration survived:\n%s", got)
	}
}

// (b) A pure unused local is removed; a call-valued one is not —
// deleting the call could change what the program does.
func TestErrorFixUnusedVar(t *testing.T) {
	dir, p := errorfixFixture(t)
	plan := mustFixPlan(t, p, ErrorFixRequest{Class: ErrorFixUnusedVar, File: "main.go", Line: 16, Name: "x"})
	// y := calc.Add(1, 2) — a call: removing it could change what
	// the program does, so no fix. (Checked before applying the x
	// fix, which shifts the line numbers below it.)
	wantFixRefusal(t, p, ErrorFixRequest{Class: ErrorFixUnusedVar, File: "main.go", Line: 21, Name: "y"}, "comes from a call")
	applyPlanToDisk(t, dir, plan)
	got := readFixtureFile(t, dir, "main.go")
	if strings.Contains(got, "x := 42") {
		t.Errorf("unused declaration survived:\n%s", got)
	}
}

// (c) The misspelled call is fixed to the one near match in scope.
func TestErrorFixUndefinedSpelling(t *testing.T) {
	dir, _ := errorfixFixture(t)
	p := writeFixtureFile(t, dir, "spell.go", "package main\n\nfunc Helper(a, b int) int { return a + b }\n\nfunc Spell() int {\n\treturn Helpr(1, 2)\n}\n")
	// "return Helpr(1, 2)" is line 6; Helpr starts at column 9.
	plan := mustFixPlan(t, p, ErrorFixRequest{Class: ErrorFixUndefinedName, File: "spell.go", Line: 6, Col: 9, Name: "Helpr"})
	applyPlanToDisk(t, dir, plan)
	got := readFixtureFile(t, dir, "spell.go")
	if !strings.Contains(got, "return Helper(1, 2)") {
		t.Errorf("spelling not fixed at the site:\n%s", got)
	}
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("verify false after spelling fix")
	}
	// A name with no near match and no providing package: no fix.
	p = writeFixtureFile(t, dir, "zebra.go", "package main\n\nfunc UseZebra() int {\n\treturn Zebra()\n}\n")
	wantFixRefusal(t, p, ErrorFixRequest{Class: ErrorFixUndefinedName, File: "zebra.go", Line: 4, Col: 9, Name: "Zebra"}, "no near match")
}

// (d1) "undefined: calc" where calc is a project package the file
// never imported: the fix is the import alone.
func TestErrorFixUndefinedPackageImport(t *testing.T) {
	dir, _ := errorfixFixture(t)
	p := writeFixtureFile(t, dir, "use.go", "package main\n\nfunc UseCalc() int {\n\treturn calc.Add(1, 2)\n}\n")
	plan := mustFixPlan(t, p, ErrorFixRequest{Class: ErrorFixUndefinedName, File: "use.go", Line: 4, Col: 9, Name: "calc"})
	applyPlanToDisk(t, dir, plan)
	got := readFixtureFile(t, dir, "use.go")
	if !strings.Contains(got, "\"example.com/errfix/calc\"") {
		t.Errorf("import not added:\n%s", got)
	}
	if !strings.Contains(got, "return calc.Add(1, 2)") {
		t.Errorf("the use itself should be untouched for a package-name fix:\n%s", got)
	}
}

// (d2) "undefined: Origin" where Origin is one other package's
// exported symbol: add the import AND qualify the use.
func TestErrorFixUndefinedSymbolImport(t *testing.T) {
	dir, _ := errorfixFixture(t)
	p := writeFixtureFile(t, dir, "use2.go", "package main\n\nimport \"fmt\"\n\nfunc UsePoint() string {\n\treturn fmt.Sprint(Origin())\n}\n")
	plan := mustFixPlan(t, p, ErrorFixRequest{Class: ErrorFixUndefinedName, File: "use2.go", Line: 6, Name: "Origin"})
	applyPlanToDisk(t, dir, plan)
	got := readFixtureFile(t, dir, "use2.go")
	if !strings.Contains(got, "\"example.com/errfix/calc\"") {
		t.Errorf("import not added:\n%s", got)
	}
	if !strings.Contains(got, "calc.Origin()") {
		t.Errorf("use not qualified:\n%s", got)
	}
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("verify false after import+qualify fix")
	}
}

// (e) Missing returns: zero values for unnamed results, a bare
// return when every result is named, and a struct zero for an
// in-module struct type.
func TestErrorFixMissingReturn(t *testing.T) {
	dir, p := errorfixFixture(t)
	plan := mustFixPlan(t, p, ErrorFixRequest{Class: ErrorFixMissingReturn, File: "main.go", Line: 30})
	if !strings.Contains(plan.Files[0].Diff, "+\treturn 0") {
		t.Errorf("missing-return diff lacks the appended zero return:\n%s", plan.Files[0].Diff)
	}
	applyPlanToDisk(t, dir, plan)
	got := readFixtureFile(t, dir, "main.go")
	if !strings.Contains(got, "\treturn 0\n}") {
		t.Errorf("return not appended at the end of the function:\n%s", got)
	}
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("verify false after missing-return fix")
	}

	// NamedReturns's brace moved from 37 to 38 (one line added).
	p2, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	plan = mustFixPlan(t, p2, ErrorFixRequest{Class: ErrorFixMissingReturn, File: "main.go", Line: 38})
	if !strings.Contains(plan.Files[0].Diff, "+\treturn\n") {
		t.Errorf("named-results fix should append a bare return:\n%s", plan.Files[0].Diff)
	}
	applyPlanToDisk(t, dir, plan)

	// StructReturn's brace is now at 45.
	p3, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	plan = mustFixPlan(t, p3, ErrorFixRequest{Class: ErrorFixMissingReturn, File: "main.go", Line: 45})
	applyPlanToDisk(t, dir, plan)
	got = readFixtureFile(t, dir, "main.go")
	if !strings.Contains(got, "return calc.Point{}") {
		t.Errorf("struct zero return missing:\n%s", got)
	}
}

// Cut classes and unknown classes never plan.
func TestErrorFixNoPlanForCutClasses(t *testing.T) {
	_, p := errorfixFixture(t)
	for _, class := range []string{"too_many_arguments", "cannot_use", "", "syntax"} {
		if plan, _ := p.PlanErrorFix(ErrorFixRequest{Class: class, File: "main.go", Line: 1}); plan != nil {
			t.Errorf("class %q planned a fix — cut classes must not", class)
		}
	}
}
