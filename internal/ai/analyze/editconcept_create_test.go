package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The declaration-create build (2026-11-01): a declaration
// that NAMES its file creates the file when it doesn't exist
// yet — same plan, gate, transaction and build check — under
// the bare-file create family's package rules. These tests
// pin the planner half: package choice, the main.go stub
// rules, the collision and mixed-folder refusals, and the
// line-placement and missing-folder refusals. The chat half
// (which targets reach here) is pinned in the chat package.

const testConnectBookJSON = `{
  "concepts": [
    {"id": "database-connect", "names": ["database connect", "database connection function"], "kind": "declaration",
     "template": "func {name}() (*sql.DB, error) {\n\tdb, err := sql.Open(\"TODO: driver name\", \"TODO: connection string\")\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\treturn db, nil\n}",
     "imports": ["context", "database/sql"]}
  ]
}`

func testConnectEntry(t *testing.T) ConceptEntry {
	t.Helper()
	entries, skipped := ParseConceptBook([]byte(testConnectBookJSON))
	if skipped != 0 || len(entries) != 1 {
		t.Fatalf("connect book parsed to %d entries, %d skipped", len(entries), skipped)
	}
	return entries[0]
}

func namedFilePlan(t *testing.T, p *Project, entry *ConceptEntry, name, file string) (*EditPlan, string) {
	t.Helper()
	return p.PlanEdit(EditRequest{
		Kind:        EditConcept,
		File:        file,
		Concept:     entry,
		ConceptName: name,
	})
}

func applyPlanFiles(t *testing.T, dir string, plan *EditPlan) {
	t.Helper()
	for _, f := range plan.Files {
		full := filepath.Join(dir, f.Rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, f.After, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// The workout's step 6 shape: api/ is a package-main folder
// (its webserver lives there), and the struct ask names a
// file that doesn't exist yet. The named file wins over the
// folder default's package-main refusal: the plan creates
// the file as package main, holding just the declaration.
func TestNamedFileCreateStructInMainPackageFolder(t *testing.T) {
	strct, _ := testBookNewKinds(t)
	dir, p := folderFixture(t, map[string]string{
		"api/main.go": "package main\n\nfunc main() {}\n",
	})
	plan, refusal := namedFilePlan(t, p, &strct, "Profile", "api/profile.go")
	if refusal != "" {
		t.Fatalf("named-file create refused: %s", refusal)
	}
	if len(plan.Files) != 1 {
		t.Fatalf("plan touches %d files, want 1", len(plan.Files))
	}
	ef := plan.Files[0]
	if ef.Rel != "api/profile.go" || ef.Before != nil {
		t.Errorf("plan file = %q (Before %v), want a creation of api/profile.go", ef.Rel, ef.Before != nil)
	}
	after := string(ef.After)
	for _, want := range []string{"package main", "type Profile struct"} {
		if !strings.Contains(after, want) {
			t.Errorf("created file missing %q:\n%s", want, after)
		}
	}
	if strings.Contains(after, "func main(") {
		t.Errorf("created file invented a main — the struct is the whole content:\n%s", after)
	}
	if !strings.Contains(plan.Summary, "Create ") || !strings.Contains(plan.Summary, "a new file in the api folder (package main)") ||
		!strings.Contains(plan.Summary, "with the struct Profile, from the concept book") {
		t.Errorf("summary does not speak the create idiom (file + package): %q", plan.Summary)
	}
	applyPlanFiles(t, dir, plan)
	after2, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(after2) {
		t.Errorf("plan verify failed against the project holding the created file")
	}
}

// A library folder lends its package: the created file joins
// the folder's one package, read off its files — here the
// package's name differs from the folder's, and the plan
// notes say so.
func TestNamedFileCreateFunctionInLibraryFolder(t *testing.T) {
	_, fn := testBookNewKinds(t)
	_, p := folderFixture(t, map[string]string{
		"lib/lib.go": "package library\n\nfunc Helper() {}\n",
	})
	plan, refusal := namedFilePlan(t, p, &fn, "Validate", "lib/validate.go")
	if refusal != "" {
		t.Fatalf("named-file create refused: %s", refusal)
	}
	after := string(plan.Files[0].After)
	if !strings.HasPrefix(after, "package library\n") {
		t.Errorf("created file does not carry the folder's package:\n%s", after)
	}
	if !strings.Contains(after, "func Validate() {") {
		t.Errorf("created file missing the declaration:\n%s", after)
	}
	if !strings.Contains(strings.Join(plan.Notes, "\n"), "package library, so the new file carries that package name") {
		t.Errorf("notes don't explain the package-name choice:\n%s", strings.Join(plan.Notes, "\n"))
	}
}

// An empty folder lends its own name as the package.
func TestNamedFileCreateEmptyFolderTakesFolderName(t *testing.T) {
	strct, _ := testBookNewKinds(t)
	dir, _ := folderFixture(t, nil)
	if err := os.MkdirAll(filepath.Join(dir, "store"), 0o755); err != nil {
		t.Fatal(err)
	}
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	plan, refusal := namedFilePlan(t, p, &strct, "Item", "store/item.go")
	if refusal != "" {
		t.Fatalf("named-file create refused: %s", refusal)
	}
	after := string(plan.Files[0].After)
	if !strings.HasPrefix(after, "package store\n") {
		t.Errorf("created file does not take the folder's name as its package:\n%s", after)
	}
	if !strings.Contains(strings.Join(plan.Notes, "\n"), "takes the folder's name as its package") {
		t.Errorf("notes don't explain the package-name choice:\n%s", strings.Join(plan.Notes, "\n"))
	}
}

// A root-level named file joins the root package.
func TestNamedFileCreateAtRoot(t *testing.T) {
	strct, _ := testBookNewKinds(t)
	_, p := folderFixture(t, nil)
	plan, refusal := namedFilePlan(t, p, &strct, "Profile", "profile.go")
	if refusal != "" {
		t.Fatalf("named-file create refused: %s", refusal)
	}
	after := string(plan.Files[0].After)
	if !strings.HasPrefix(after, "package main\n") {
		t.Errorf("root creation is not package main:\n%s", after)
	}
	if !strings.Contains(plan.Summary, "project folder") {
		t.Errorf("summary doesn't name the project folder: %q", plan.Summary)
	}
}

// A created main.go is a program file: beside the asked-for
// declaration it carries the empty func main stub the
// bare-file family starts every new main.go with.
func TestNamedFileCreateMainGoCarriesTheStub(t *testing.T) {
	strct, _ := testBookNewKinds(t)
	dir, _ := folderFixture(t, nil)
	if err := os.MkdirAll(filepath.Join(dir, "app2"), 0o755); err != nil {
		t.Fatal(err)
	}
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	plan, refusal := namedFilePlan(t, p, &strct, "Profile", "app2/main.go")
	if refusal != "" {
		t.Fatalf("main.go create refused: %s", refusal)
	}
	after := string(plan.Files[0].After)
	for _, want := range []string{"package main", "type Profile struct", "func main() {", "// TODO: write the body"} {
		if !strings.Contains(after, want) {
			t.Errorf("created main.go missing %q:\n%s", want, after)
		}
	}
	if strings.Index(after, "type Profile struct") > strings.Index(after, "func main()") {
		t.Errorf("the declaration should sit above the stub main:\n%s", after)
	}
	if !strings.Contains(strings.Join(plan.Notes, "\n"), "empty func main a new main.go starts with") {
		t.Errorf("notes don't own the stub:\n%s", strings.Join(plan.Notes, "\n"))
	}
}

// …unless the package already has its main in another file:
// then a second main.go is the family's refusal, naming the
// file that holds the main.
func TestNamedFileCreateMainGoRefusesWhenMainExists(t *testing.T) {
	strct, _ := testBookNewKinds(t)
	_, p := folderFixture(t, map[string]string{
		"api/server.go": "package main\n\nfunc main() {}\n",
	})
	_, refusal := namedFilePlan(t, p, &strct, "Profile", "api/main.go")
	for _, want := range []string{"server.go", "already defines func main", "another main.go can't be added"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("main-exists refusal missing %q: %q", want, refusal)
		}
	}
}

// When the declaration IS func main, there is no second stub:
// the file holds the one main it was asked for.
func TestNamedFileCreateDeclIsMainGetsNoStub(t *testing.T) {
	_, fn := testBookNewKinds(t)
	dir, _ := folderFixture(t, nil)
	if err := os.MkdirAll(filepath.Join(dir, "app2"), 0o755); err != nil {
		t.Fatal(err)
	}
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	plan, refusal := namedFilePlan(t, p, &fn, "main", "app2/main.go")
	if refusal != "" {
		t.Fatalf("func main create refused: %s", refusal)
	}
	after := string(plan.Files[0].After)
	if n := strings.Count(after, "func main()"); n != 1 {
		t.Errorf("created file holds %d func main declarations, want exactly 1:\n%s", n, after)
	}
}

// A name the package already declares refuses package-aware,
// wherever in the folder the existing declaration lives.
func TestNamedFileCreateNameCollisionRefuses(t *testing.T) {
	strct, _ := testBookNewKinds(t)
	_, p := folderFixture(t, map[string]string{
		"api/types.go": "package main\n\ntype Profile struct{}\n",
	})
	_, refusal := namedFilePlan(t, p, &strct, "Profile", "api/profile.go")
	for _, want := range []string{"Profile already exists in package main", "pick a different name"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("collision refusal missing %q: %q", want, refusal)
		}
	}
}

// A mixed-package folder refuses, naming both packages and
// the files carrying them.
func TestNamedFileCreateMixedFolderRefuses(t *testing.T) {
	strct, _ := testBookNewKinds(t)
	_, p := folderFixture(t, map[string]string{
		"mix/a.go": "package one\n\nfunc A() {}\n",
		"mix/b.go": "package two\n\nfunc B() {}\n",
	})
	_, refusal := namedFilePlan(t, p, &strct, "Profile", "mix/profile.go")
	for _, want := range []string{"mixes", "package one (", "a.go)", "package two (", "b.go)", "one package per folder"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("mixed-folder refusal missing %q: %q", want, refusal)
		}
	}
}

// A line placement can't aim into a file with no lines yet.
func TestNamedFileCreateLinePlacementRefuses(t *testing.T) {
	strct, _ := testBookNewKinds(t)
	_, p := folderFixture(t, map[string]string{
		"api/main.go": "package main\n\nfunc main() {}\n",
	})
	_, refusal := p.PlanEdit(EditRequest{
		Kind: EditConcept, File: "api/profile.go", Concept: &strct,
		ConceptName: "Profile", PlaceLine: 3,
	})
	if !strings.Contains(refusal, "doesn't exist yet, so it has no lines to aim at") {
		t.Errorf("line-placement refusal = %q", refusal)
	}
}

// A missing folder is the create family's first step, named
// plainly — the planner's own guard behind the chat's.
func TestNamedFileCreateMissingFolderRefuses(t *testing.T) {
	strct, _ := testBookNewKinds(t)
	_, p := folderFixture(t, nil)
	_, refusal := namedFilePlan(t, p, &strct, "Profile", "nowhere/profile.go")
	want := "there's no folder nowhere in this project — say \"create a folder nowhere\" first, then ask again"
	if refusal != want {
		t.Errorf("missing-folder refusal = %q, want %q", refusal, want)
	}
}

// The row's imports compose into the created file through
// the one import inserter, as they do in an existing file.
func TestNamedFileCreateComposesImports(t *testing.T) {
	connect := testConnectEntry(t)
	_, p := folderFixture(t, map[string]string{
		"store/seed.go": "package store\n\nfunc Seed() {}\n",
	})
	plan, refusal := namedFilePlan(t, p, &connect, "Store", "store/db.go")
	if refusal != "" {
		t.Fatalf("named-file create refused: %s", refusal)
	}
	after := string(plan.Files[0].After)
	for _, want := range []string{"package store", "\"context\"", "\"database/sql\"", "func Store() (*sql.DB, error)"} {
		if !strings.Contains(after, want) {
			t.Errorf("created file missing %q:\n%s", want, after)
		}
	}
}

// Statements never create: a statement aimed at a missing
// file keeps the read refusal it has today — creating the
// file is a declaration's licence only.
func TestStatementToMissingFileStillRefuses(t *testing.T) {
	printLine, _ := testBook(t)
	_, p := folderFixture(t, map[string]string{
		"api/main.go": "package main\n\nfunc main() {}\n",
	})
	plan, refusal := p.PlanEdit(EditRequest{
		Kind: EditConcept, File: "api/new.go", Concept: &printLine,
		ConceptFunc: "main",
	})
	if plan != nil || refusal == "" {
		t.Fatalf("statement to a missing file planned (%v) or answered silently (%q)", plan != nil, refusal)
	}
	if !strings.Contains(refusal, "couldn't read") {
		t.Errorf("statement refusal changed shape: %q", refusal)
	}
}
