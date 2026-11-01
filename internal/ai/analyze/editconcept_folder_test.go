package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The folder layout build (2026-11-01): a declaration row
// carrying a folder field, asked with no named file, creates
// <folder>/<name>.go as the folder's own package. The fixture
// row below exists NOWHERE in the shipped book — that is the
// point: the folder behaviour is read off the row's data by
// the generic planner, with no concept-specific code, exactly
// as the book's data-driven precedent demands.

const testFolderBookJSON = `{
  "concepts": [
    {"id": "gadget-guard", "names": ["gadget guard"], "kind": "declaration", "folder": "gadgets",
     "template": "func {name}(next http.HandlerFunc) http.HandlerFunc {\n\treturn next\n}", "imports": ["net/http"]},
    {"id": "gadget-note", "names": ["gadget note"], "kind": "statement", "folder": "gadgets",
     "template": "println(\"{text}\")"}
  ]
}`

func testFolderBook(t *testing.T) (guard, note ConceptEntry) {
	t.Helper()
	entries, skipped := ParseConceptBook([]byte(testFolderBookJSON))
	if skipped != 0 || len(entries) != 2 {
		t.Fatalf("folder book parsed to %d entries, %d skipped", len(entries), skipped)
	}
	if entries[0].Folder != "gadgets" || entries[1].Folder != "gadgets" {
		t.Fatalf("folder field did not survive the loader: %+v", entries)
	}
	return entries[0], entries[1]
}

// folderFixture: a module whose root holds one program file,
// plus whatever extra files the case maps in.
func folderFixture(t *testing.T, extra map[string]string) (string, *Project) {
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
	write("go.mod", "module example.com/folders\n\ngo 1.21\n")
	write("main.go", "package main\n\nfunc main() {}\n")
	for rel, src := range extra {
		write(rel, src)
	}
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return dir, p
}

func folderPlan(t *testing.T, p *Project, entry *ConceptEntry, name string) (*EditPlan, string) {
	t.Helper()
	return p.PlanEdit(EditRequest{
		Kind:          EditConcept,
		Concept:       entry,
		ConceptName:   name,
		ConceptFolder: entry.Folder,
	})
}

func TestFolderDefaultCreatesFileInNewFolder(t *testing.T) {
	guard, _ := testFolderBook(t)
	_, p := folderFixture(t, nil)

	plan, refusal := folderPlan(t, p, &guard, "Shield")
	if refusal != "" {
		t.Fatalf("folder-default plan refused: %s", refusal)
	}
	if len(plan.Files) != 1 {
		t.Fatalf("plan touches %d files, want 1", len(plan.Files))
	}
	ef := plan.Files[0]
	if ef.Rel != "gadgets/shield.go" {
		t.Errorf("created file = %q, want gadgets/shield.go", ef.Rel)
	}
	if ef.Before != nil {
		t.Errorf("created file carries Before bytes — it must be a creation")
	}
	after := string(ef.After)
	for _, want := range []string{
		"package gadgets",
		"\"net/http\"",
		"func Shield(next http.HandlerFunc) http.HandlerFunc {",
	} {
		if !strings.Contains(after, want) {
			t.Errorf("created file missing %q:\n%s", want, after)
		}
	}
	if !strings.Contains(plan.Summary, "Create the gadgets folder and") ||
		!strings.Contains(plan.Summary, "package gadgets") {
		t.Errorf("summary does not speak the create idiom (folder + file + package): %q", plan.Summary)
	}
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, "folder is new") {
		t.Errorf("notes don't say the folder is new:\n%s", notes)
	}
	if !strings.Contains(notes, "placed, not wired") {
		t.Errorf("notes don't carry the placed-not-wired contract:\n%s", notes)
	}
}

func TestFolderDefaultVerifySeesTheNewPackage(t *testing.T) {
	guard, _ := testFolderBook(t)
	dir, p := folderFixture(t, nil)
	plan, refusal := folderPlan(t, p, &guard, "Shield")
	if refusal != "" {
		t.Fatalf("folder-default plan refused: %s", refusal)
	}
	// Apply the plan's file set to the fixture on disk, then
	// re-analyze: the plan's own verify must find the new
	// package and the declaration inside it.
	for _, f := range plan.Files {
		full := filepath.Join(dir, f.Rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, f.After, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	after, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(after) {
		t.Errorf("plan verify failed against the project holding the new package")
	}
	pkg := after.pkgByDir("gadgets")
	if pkg == nil || pkg.Name != "gadgets" {
		t.Fatalf("after-project has no gadgets package: %+v", pkg)
	}
}

func TestFolderDefaultUsesTheFoldersRealPackageName(t *testing.T) {
	guard, _ := testFolderBook(t)
	_, p := folderFixture(t, map[string]string{
		"gadgets/thing.go": "package gizmos\n\nfunc Thing() {}\n",
	})
	plan, refusal := folderPlan(t, p, &guard, "Shield")
	if refusal != "" {
		t.Fatalf("folder-default plan refused: %s", refusal)
	}
	after := string(plan.Files[0].After)
	if !strings.HasPrefix(after, "package gizmos\n") {
		t.Errorf("created file does not carry the folder's real package name:\n%s", after)
	}
	if !strings.Contains(plan.Summary, "package gizmos") {
		t.Errorf("summary names the wrong package: %q", plan.Summary)
	}
	if !strings.Contains(strings.Join(plan.Notes, "\n"), "package gizmos, so the new file carries that package name") {
		t.Errorf("notes don't explain the package-name choice:\n%s", strings.Join(plan.Notes, "\n"))
	}
}

func TestFolderDefaultRefusesPackageMainFolder(t *testing.T) {
	guard, _ := testFolderBook(t)
	_, p := folderFixture(t, map[string]string{
		"gadgets/main.go": "package main\n\nfunc main() {}\n",
	})
	_, refusal := folderPlan(t, p, &guard, "Shield")
	if !strings.Contains(refusal, "package main") || !strings.Contains(refusal, "program") {
		t.Errorf("package-main folder refusal = %q", refusal)
	}
}

func TestFolderDefaultRefusesMixedPackageFolder(t *testing.T) {
	guard, _ := testFolderBook(t)
	_, p := folderFixture(t, map[string]string{
		"gadgets/a.go": "package one\n\nfunc A() {}\n",
		"gadgets/b.go": "package two\n\nfunc B() {}\n",
	})
	_, refusal := folderPlan(t, p, &guard, "Shield")
	for _, want := range []string{"mixes", "package one", "package two", "one package per folder"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("mixed-folder refusal missing %q: %q", want, refusal)
		}
	}
}

func TestFolderDefaultCollisionIsPackageAware(t *testing.T) {
	guard, _ := testFolderBook(t)
	_, p := folderFixture(t, map[string]string{
		// The collision lives in a DIFFERENT file of the same
		// package: a same-name declaration anywhere in the
		// folder refuses, exactly as a named-file ask into
		// the package would.
		"gadgets/other.go": "package gadgets\n\nfunc Shield() {}\n",
	})
	_, refusal := folderPlan(t, p, &guard, "Shield")
	if !strings.Contains(refusal, "Shield already exists in package gadgets") {
		t.Errorf("collision refusal = %q", refusal)
	}
}

func TestFolderDefaultJoinsAnExistingDeclFile(t *testing.T) {
	guard, _ := testFolderBook(t)
	_, p := folderFixture(t, map[string]string{
		"gadgets/shield.go": "package gadgets\n\nfunc Other() {}\n",
	})
	plan, refusal := folderPlan(t, p, &guard, "Shield")
	if refusal != "" {
		t.Fatalf("existing-file plan refused: %s", refusal)
	}
	ef := plan.Files[0]
	if ef.Rel != "gadgets/shield.go" || ef.Before == nil {
		t.Fatalf("existing-file plan is not an edit of gadgets/shield.go: %+v", ef)
	}
	after := string(ef.After)
	if !strings.Contains(after, "func Other() {}") ||
		!strings.Contains(after, "func Shield(next http.HandlerFunc) http.HandlerFunc {") {
		t.Errorf("joined file doesn't hold both declarations:\n%s", after)
	}
	if len(plan.Notes) == 0 || !strings.Contains(plan.Notes[0], "already exists (package gadgets)") {
		t.Errorf("plan doesn't say why this file was picked: %v", plan.Notes)
	}
}

func TestFolderDefaultRefusals(t *testing.T) {
	guard, note := testFolderBook(t)
	_, p := folderFixture(t, nil)

	// A statement row never folder-defaults, even when its
	// (hypothetical) row carries the field.
	_, refusal := p.PlanEdit(EditRequest{
		Kind: EditConcept, Concept: &note, ConceptFolder: note.Folder,
	})
	if !strings.Contains(refusal, "goes inside a function") {
		t.Errorf("statement folder refusal = %q", refusal)
	}

	// A folder value that isn't one plain name refuses.
	bad := guard
	bad.Folder = "a/b"
	_, refusal = p.PlanEdit(EditRequest{
		Kind: EditConcept, Concept: &bad, ConceptName: "Shield", ConceptFolder: bad.Folder,
	})
	if !strings.Contains(refusal, "can't use") {
		t.Errorf("bad-folder refusal = %q", refusal)
	}

	// No name: the guidance names the folder form.
	_, refusal = folderPlan(t, p, &guard, "")
	if !strings.Contains(refusal, "lands in the gadgets folder") {
		t.Errorf("unnamed refusal = %q", refusal)
	}

	// A line number can't aim into a file that doesn't
	// exist yet.
	_, refusal = p.PlanEdit(EditRequest{
		Kind: EditConcept, Concept: &guard, ConceptName: "Shield",
		ConceptFolder: guard.Folder, PlaceLine: 3,
	})
	if !strings.Contains(refusal, "has no lines to aim at") {
		t.Errorf("line-aimed refusal = %q", refusal)
	}

	// An unusable declaration name refuses as the named-file
	// path does.
	_, refusal = folderPlan(t, p, &guard, "not a name")
	if !strings.Contains(refusal, "isn't a name Go allows") {
		t.Errorf("bad-name refusal = %q", refusal)
	}
}

// The named-target-wins rule at the planner level: the same
// request WITH a file goes down the ordinary path and the
// folder field plays no part.
func TestFolderDefaultNamedFileWinsAtPlanner(t *testing.T) {
	guard, _ := testFolderBook(t)
	_, p := folderFixture(t, nil)
	plan, refusal := p.PlanEdit(EditRequest{
		Kind: EditConcept, File: "main.go", Concept: &guard, ConceptName: "Shield",
	})
	if refusal != "" {
		t.Fatalf("named-file plan refused: %s", refusal)
	}
	if len(plan.Files) != 1 || plan.Files[0].Rel != "main.go" {
		t.Fatalf("named-file plan files = %+v", plan.Files)
	}
	if !strings.Contains(string(plan.Files[0].After), "func Shield(next http.HandlerFunc) http.HandlerFunc {") {
		t.Errorf("declaration did not land in the named file:\n%s", plan.Files[0].After)
	}
}
