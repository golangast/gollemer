package analyze

import (
	"strings"
	"testing"
)

// Same-offset insert composition (2026-10-31): a target file
// that is only a package clause anchors BOTH the new import
// block (inserted "after the package clause" = at EOF) and an
// appended declaration at the same offset. The shared splice
// must compose them imports-first — Go forbids imports after
// declarations — no matter which order the planner listed
// them in. The Uses planner once merged its two inserts by
// hand to dodge this; the delegate takes the shared path
// unmerged, and these tests pin the helper's contract through
// the delegate and directly.

// TestDelegateIntoPackageClauseOnlyFile drives the shared
// path end to end: a cross-package delegate whose target file
// is exactly "package lib\n" (the delegate fixture's
// lib/other.go). At the trap, the plan either refused with a
// spurious internal error or stranded the import block after
// the wrapper, which does not parse.
func TestDelegateIntoPackageClauseOnlyFile(t *testing.T) {
	_, p := delegateProject(t, nil)
	plan := mustPlan(t, p, EditRequest{
		Kind:         EditDelegate,
		DelegateFile: "lib/other.go",
		SourceName:   "store.Load",
	})
	if len(plan.Files) == 0 || plan.Files[0].Rel != "lib/other.go" {
		t.Fatalf("first plan file = %+v, want the lib/other.go wrapper", plan.Files)
	}
	check := func(label, call string, after []byte) {
		t.Helper()
		if _, _, err := parseGoSrc(label, after); err != nil {
			t.Errorf("%s does not parse: %v\n%s", label, err, after)
			return
		}
		impAt := strings.Index(string(after), `"example.com/delegate/store"`)
		fnAt := strings.Index(string(after), "func Load(")
		if impAt < 0 || fnAt < 0 {
			t.Errorf("%s missing the import or the wrapper:\n%s", label, after)
			return
		}
		if impAt > fnAt {
			t.Errorf("%s strands the import after the wrapper:\n%s", label, after)
		}
		if !strings.Contains(string(after), call) {
			t.Errorf("%s wrapper body is not the delegate call:\n%s", label, after)
		}
	}
	check("primary form", "return store.Load(id)", plan.Files[0].After)
	if len(plan.AltFiles) > 0 {
		// The retry form calls through its aliased import.
		check("retry form", "return storepkg.Load(id)", plan.AltFiles[0].After)
	}
}

// TestApplyTextEditsSameOffsetInserts pins the helper's
// contract directly: two inserts at one offset compose with
// the import edit first, whichever order they were passed in,
// and two plain inserts keep the order they were passed in.
func TestApplyTextEditsSameOffsetInserts(t *testing.T) {
	src := []byte("package lib\n")
	fset, fileAST, err := parseGoSrc("lib.go", src)
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	impEdits := insertImportSpecs(fileAST, fset, src, []string{`"example.com/delegate/store"`})
	if len(impEdits) != 1 || impEdits[0].start != len(src) {
		t.Fatalf("import edits = %+v, want one insert at EOF (%d)", impEdits, len(src))
	}
	appendEdit := textEdit{start: len(src), end: len(src), text: "\nfunc Load() {}\n"}

	for name, edits := range map[string][]textEdit{
		"imports listed first": {impEdits[0], appendEdit},
		"imports listed last":  {appendEdit, impEdits[0]},
	} {
		out := string(applyTextEdits(src, edits))
		impAt := strings.Index(out, "import (")
		fnAt := strings.Index(out, "func Load(")
		if impAt < 0 || fnAt < 0 || impAt > fnAt {
			t.Errorf("%s: imports do not precede the declaration:\n%s", name, out)
		}
		if _, _, err := parseGoSrc("out.go", []byte(out)); err != nil {
			t.Errorf("%s: composed output does not parse: %v\n%s", name, err, out)
		}
	}

	// Two plain inserts at one offset keep their passed order.
	a := textEdit{start: len(src), end: len(src), text: "\nfunc A() {}\n"}
	b := textEdit{start: len(src), end: len(src), text: "\nfunc B() {}\n"}
	out := string(applyTextEdits(src, []textEdit{a, b}))
	if strings.Index(out, "func A()") > strings.Index(out, "func B()") {
		t.Errorf("plain same-offset inserts lost their passed order:\n%s", out)
	}

	// Unrelated offsets still splice as before.
	multi := []byte("package lib\n\nfunc A() {}\n")
	got := applyTextEdits(multi, []textEdit{
		{start: 0, end: 0, text: "// hi\n"},
		{start: len(multi), end: len(multi), text: "func B() {}\n"},
	})
	if !strings.HasPrefix(string(got), "// hi\npackage lib") || !strings.HasSuffix(string(got), "func B() {}\n") {
		t.Errorf("distinct-offset edits changed behavior:\n%s", got)
	}
}
