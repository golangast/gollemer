package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Line placement (2026-11-01): a concept sentence may name
// the line its piece lands on or by. These tests pin the
// anchoring rules and every refusal against a fixture whose
// lines are counted in the source itself:
//
//	 1  package main
//	 2
//	 3  import (
//	 4  	"fmt"
//	 5  )
//	 6
//	 7  type Profile struct {
//	 8  	Name string
//	 9  }
//	10
//	11  func helper() {
//	12  	fmt.Println("helping")
//	13  }
//	14
//	15  func main() {
//	16  	fmt.Println("one")
//	17  	if true {
//	18  		fmt.Println("nested")
//	19  	}
//	20  	// a trailing comment
//	21  	fmt.Println("two")
//	22  }
//	23  func Empty() {}
//	24  func OneLine() { fmt.Println("x") }

var lineFixtureLines = []string{
	"package main",                          // 1
	"",                                      // 2
	"import (",                              // 3
	"\t\"fmt\"",                             // 4
	")",                                     // 5
	"",                                      // 6
	"type Profile struct {",                 // 7
	"\tName string",                         // 8
	"}",                                     // 9
	"",                                      // 10
	"func helper() {",                       // 11
	"\tfmt.Println(\"helping\")",            // 12
	"}",                                     // 13
	"",                                      // 14
	"func main() {",                         // 15
	"\tfmt.Println(\"one\")",                // 16
	"\tif true {",                           // 17
	"\t\tfmt.Println(\"nested\")",           // 18
	"\t}",                                   // 19
	"\t// a trailing comment",               // 20
	"\tfmt.Println(\"two\")",                // 21
	"}",                                     // 22
	"func Empty() {}",                       // 23
	"func OneLine() { fmt.Println(\"x\") }", // 24
}

func lineFixture(t *testing.T) (string, *Project) {
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
	write("go.mod", "module example.com/lines\n\ngo 1.21\n")
	write("main.go", strings.Join(lineFixtureLines, "\n")+"\n")
	write("bare.go", "package main\n")
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return dir, p
}

func placedStmt(t *testing.T, p *Project, line int, before bool) *EditPlan {
	t.Helper()
	printLine, _ := testBook(t)
	return conceptPlan(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &printLine,
		ConceptFunc: "main", PlaceLine: line, PlaceBefore: before,
	})
}

func order(t *testing.T, after string, subs ...string) {
	t.Helper()
	prev := -1
	for _, s := range subs {
		i := strings.Index(after, s)
		if i < 0 {
			t.Fatalf("after text lacks %q:\n%s", s, after)
		}
		if i < prev {
			t.Fatalf("%q out of order in:\n%s", s, after)
		}
		prev = i
	}
}

func TestPlacedStatementAfterLine(t *testing.T) {
	_, p := lineFixture(t)
	plan := placedStmt(t, p, 16, false)
	after := string(plan.Files[0].After)
	order(t, after, `fmt.Println("one")`, `fmt.Println("")`, "if true {")
	if !strings.Contains(plan.Summary, "after line 16: `fmt.Println(\"one\")`") {
		t.Errorf("summary does not quote the anchor: %q", plan.Summary)
	}
}

func TestPlacedStatementOnLineInsideMultiLineStatement(t *testing.T) {
	_, p := lineFixture(t)
	// Line 18 is inside the if statement (lines 17–19): the
	// piece anchors to the statement's end, never splitting it.
	plan := placedStmt(t, p, 18, false)
	after := string(plan.Files[0].After)
	order(t, after, `fmt.Println("nested")`, "\n\t}\n", `fmt.Println("")`, `fmt.Println("two")`)
	if !strings.Contains(plan.Summary, "after line 18:") {
		t.Errorf("summary does not quote line 18: %q", plan.Summary)
	}
}

func TestPlacedStatementBeforeLine(t *testing.T) {
	_, p := lineFixture(t)
	plan := placedStmt(t, p, 21, true)
	after := string(plan.Files[0].After)
	order(t, after, "// a trailing comment", `fmt.Println("")`, `fmt.Println("two")`)
	if !strings.Contains(plan.Summary, "before line 21: `fmt.Println(\"two\")`") {
		t.Errorf("summary does not quote the anchor: %q", plan.Summary)
	}
}

func TestPlacedStatementBeforeFirstLineOfBody(t *testing.T) {
	_, p := lineFixture(t)
	plan := placedStmt(t, p, 16, true)
	after := string(plan.Files[0].After)
	order(t, after, "func main() {", `fmt.Println("")`, `fmt.Println("one")`)
}

func TestPlacedStatementAfterCommentLine(t *testing.T) {
	_, p := lineFixture(t)
	// Line 20 is a comment between statements: "after" anchors
	// after it (the boundary after the preceding statement).
	plan := placedStmt(t, p, 20, false)
	after := string(plan.Files[0].After)
	order(t, after, "// a trailing comment", `fmt.Println("")`, `fmt.Println("two")`)
	if !strings.Contains(plan.Summary, "after line 20: `// a trailing comment`") {
		t.Errorf("summary does not quote the comment line: %q", plan.Summary)
	}
}

func TestPlacedStatementIntoEmptyOneLineFunc(t *testing.T) {
	_, p := lineFixture(t)
	printLine, _ := testBook(t)
	plan := conceptPlan(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &printLine,
		ConceptFunc: "Empty", PlaceLine: 23,
	})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, "func Empty() {\n\tfmt.Println(\"\")\n}") {
		t.Errorf("one-line empty body not opened around the piece:\n%s", after)
	}
}

func TestPlacedStatementOneLineBodyBeforeAndAfter(t *testing.T) {
	_, p := lineFixture(t)
	printLine, _ := testBook(t)
	plan := conceptPlan(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &printLine,
		ConceptFunc: "OneLine", PlaceLine: 24, PlaceBefore: true,
	})
	order(t, string(plan.Files[0].After), `fmt.Println("")`, `fmt.Println("x")`)
	plan = conceptPlan(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &printLine,
		ConceptFunc: "OneLine", PlaceLine: 24,
	})
	order(t, string(plan.Files[0].After), `fmt.Println("x")`, `fmt.Println("")`)
}

func TestPlacedStatementRefusals(t *testing.T) {
	_, p := lineFixture(t)
	printLine, _ := testBook(t)
	req := func(line int) EditRequest {
		return EditRequest{Kind: EditConcept, File: "main.go", Concept: &printLine,
			ConceptFunc: "main", PlaceLine: line}
	}
	conceptRefusal(t, p, req(25), "the file has 24 lines")
	conceptRefusal(t, p, req(12), "inside helper, which runs lines 11–13")
	conceptRefusal(t, p, req(12), "main, which runs lines 15–22")
	conceptRefusal(t, p, req(1), "line 1 is the package clause")
	conceptRefusal(t, p, req(14), "isn't inside any function")
	conceptRefusal(t, p, req(8), "inside the struct Profile (lines 7–9)")
	conceptRefusal(t, p, req(4), "inside the import block (lines 3–5)")
}

func TestPlacedStatementGrowMainRefused(t *testing.T) {
	_, p := lineFixture(t)
	printLine, _ := testBook(t)
	conceptRefusal(t, p, EditRequest{
		Kind: EditConcept, File: "bare.go", Concept: &printLine, PlaceLine: 1,
	}, "has no main yet")
	// The same ask without a line still grows the main.
	conceptPlan(t, p, EditRequest{Kind: EditConcept, File: "bare.go", Concept: &printLine})
}

// --- declarations at file scope ---------------------------------

func placedDecl(t *testing.T, p *Project, line int, before bool) *EditPlan {
	t.Helper()
	strct, _ := testBookNewKinds(t)
	return conceptPlan(t, p, EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &strct,
		ConceptName: "NewType", PlaceLine: line, PlaceBefore: before,
	})
}

func TestPlacedDeclarationBeforeFirstLine(t *testing.T) {
	_, p := lineFixture(t)
	plan := placedDecl(t, p, 7, true)
	after := string(plan.Files[0].After)
	order(t, after, "type NewType struct {", "type Profile struct {")
	if !strings.Contains(plan.Summary, "before line 7: `type Profile struct {`") {
		t.Errorf("summary does not quote the anchor: %q", plan.Summary)
	}
}

func TestPlacedDeclarationAfterLastLine(t *testing.T) {
	_, p := lineFixture(t)
	plan := placedDecl(t, p, 9, false)
	after := string(plan.Files[0].After)
	order(t, after, "type Profile struct {", "type NewType struct {", "func helper() {")
}

func TestPlacedDeclarationBetweenDeclarations(t *testing.T) {
	_, p := lineFixture(t)
	// Line 10 is blank, between Profile and helper: "before"
	// anchors before the following declaration.
	plan := placedDecl(t, p, 10, true)
	after := string(plan.Files[0].After)
	order(t, after, "type Profile struct {", "type NewType struct {", "func helper() {")
	// And "after" that blank line anchors after the
	// preceding declaration — the same boundary.
	plan = placedDecl(t, p, 10, false)
	after = string(plan.Files[0].After)
	order(t, after, "type Profile struct {", "type NewType struct {", "func helper() {")
}

func TestPlacedDeclarationRefusals(t *testing.T) {
	_, p := lineFixture(t)
	strct, _ := testBookNewKinds(t)
	req := func(line int, before bool) EditRequest {
		return EditRequest{Kind: EditConcept, File: "main.go", Concept: &strct,
			ConceptName: "NewType", PlaceLine: line, PlaceBefore: before}
	}
	conceptRefusal(t, p, req(25, false), "the file has 24 lines")
	conceptRefusal(t, p, req(8, false), "inside the struct Profile (lines 7–9)")
	conceptRefusal(t, p, req(8, false), "Aim at line 7")
	conceptRefusal(t, p, req(9, true), "inside the struct Profile (lines 7–9)")
	conceptRefusal(t, p, req(4, false), "inside the import block (lines 3–5)")
	conceptRefusal(t, p, req(1, false), "line 1 is the package clause")
	conceptRefusal(t, p, req(18, false), "inside the function main (lines 15–22)")
}

// --- instantiation ----------------------------------------------

func TestPlacedInstantiate(t *testing.T) {
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
	write("go.mod", "module example.com/lines\n\ngo 1.21\n")
	write("main.go", "package main\n\ntype Jim struct {\n\tName string\n}\n\nfunc main() {\n\tprintln(\"one\")\n\tprintln(\"two\")\n}\n")
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	// main's statements are lines 8 and 9; place after line 8.
	plan := conceptPlan(t, p, EditRequest{
		Kind: EditInstantiate, InstType: "Jim", InstFunc: "main", PlaceLine: 8,
	})
	after := string(plan.Files[0].After)
	order(t, after, `println("one")`, "jim := Jim{", `println("two")`)
	if !strings.Contains(plan.Summary, "after line 8: `println(\"one\")`") {
		t.Errorf("summary does not quote the anchor: %q", plan.Summary)
	}
	conceptRefusal(t, p, EditRequest{
		Kind: EditInstantiate, InstType: "Jim", InstFunc: "main", PlaceLine: 40,
	}, "past the end of")
	conceptRefusal(t, p, EditRequest{
		Kind: EditInstantiate, InstType: "Jim", InstFunc: "main", PlaceLine: 3,
	}, "inside the struct Jim (lines 3–5)")
}
