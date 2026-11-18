package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The taught-declaration planner (edittaught.go): the template
// arrives as data from the teach seat — already renamed — and
// is assembled like a concept-book declaration: appended at
// file scope or created as a new file under the bare-file
// package rules, imports dressed, collisions refused, the plan
// carrying the user's own provenance note.

func taughtPlan(t *testing.T, p *Project, req EditRequest) *EditPlan {
	t.Helper()
	req.Kind = EditTaughtDecl
	plan, refusal := p.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("PlanEdit(taught_decl) refused: %s", refusal)
	}
	return plan
}

func TestTaughtFuncAppendsWithImportsAndNote(t *testing.T) {
	_, p := conceptFixture(t)
	plan := taughtPlan(t, p, EditRequest{
		File:             "bare.go",
		TaughtText:       "func Signup(w http.ResponseWriter, r *http.Request) {\n\tfmt.Fprintln(w, \"hi\")\n}",
		TaughtName:       "Signup",
		TaughtDeclKind:   "function",
		TaughtKindPhrase: "signup form",
		TaughtImports:    `"fmt" "net/http"`,
		TaughtNote:       "From your example Jake in jim/main.go, taught Nov 3, 2026.",
	})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, "func Signup(w http.ResponseWriter, r *http.Request)") {
		t.Errorf("taught func not appended:\n%s", after)
	}
	if !strings.Contains(after, `"net/http"`) || !strings.Contains(after, `"fmt"`) {
		t.Errorf("imports not dressed:\n%s", after)
	}
	found := false
	for _, n := range plan.Notes {
		if strings.Contains(n, "From your example Jake in jim/main.go") {
			found = true
		}
	}
	if !found {
		t.Errorf("provenance note missing: %v", plan.Notes)
	}
	if !strings.Contains(plan.Summary, "signup form") || !strings.Contains(plan.Summary, "Signup") {
		t.Errorf("summary does not name kind and name: %q", plan.Summary)
	}
}

func TestTaughtTypeAppendsAndVerifies(t *testing.T) {
	dir, p := conceptFixture(t)
	plan := taughtPlan(t, p, EditRequest{
		File:             "bare.go",
		TaughtText:       "type Cart struct {\n\tItems []string\n\tNext  *Cart\n}",
		TaughtName:       "Cart",
		TaughtDeclKind:   "type",
		TaughtKindPhrase: "cart",
	})
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("taught type plan does not verify against the applied project")
	}
}

func TestTaughtNameCollisionRefused(t *testing.T) {
	_, p := conceptFixture(t)
	_, refusal := p.PlanEdit(EditRequest{
		Kind:             EditTaughtDecl,
		File:             "bare.go",
		TaughtText:       "func Helper() string { return \"x\" }",
		TaughtName:       "Helper",
		TaughtDeclKind:   "function",
		TaughtKindPhrase: "helper",
	})
	if !strings.Contains(refusal, "Helper already exists in package main") {
		t.Errorf("collision refusal = %q", refusal)
	}
}

func TestTaughtBadNameRefused(t *testing.T) {
	_, p := conceptFixture(t)
	_, refusal := p.PlanEdit(EditRequest{
		Kind:             EditTaughtDecl,
		File:             "bare.go",
		TaughtText:       "func X() {}",
		TaughtName:       "not a name",
		TaughtDeclKind:   "function",
		TaughtKindPhrase: "thing",
	})
	if !strings.Contains(refusal, "isn't a name Go allows") {
		t.Errorf("bad-name refusal = %q", refusal)
	}
}

// A taught declaration aimed at a file that doesn't exist yet
// creates it under the folder's package rules — here an empty
// folder, so the new file takes the folder's name as its
// package.
func TestTaughtNamedFileCreateInEmptyFolder(t *testing.T) {
	dir, _ := conceptFixture(t)
	if err := os.MkdirAll(filepath.Join(dir, "shop"), 0o755); err != nil {
		t.Fatal(err)
	}
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	plan := taughtPlan(t, fresh, EditRequest{
		File:             "shop/cart.go",
		TaughtText:       "type Cart struct {\n\tItems []string\n}",
		TaughtName:       "Cart",
		TaughtDeclKind:   "type",
		TaughtKindPhrase: "cart",
	})
	if len(plan.Files) != 1 || plan.Files[0].Before != nil {
		t.Fatalf("create plan should have one new file, got %+v", plan.Files)
	}
	after := string(plan.Files[0].After)
	if !strings.HasPrefix(after, "package shop\n") {
		t.Errorf("new file package = %q", strings.SplitN(after, "\n", 2)[0])
	}
	if !strings.Contains(after, "type Cart struct") {
		t.Errorf("taught decl missing from new file:\n%s", after)
	}
	applyPlan(t, dir, plan)
	re, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(re) {
		t.Error("create plan does not verify")
	}
}

// A taught function created as a new main.go carries the
// family's empty-main stub beside it (never instead of it).
func TestTaughtNamedFileCreateMainGoGetsStub(t *testing.T) {
	dir, _ := conceptFixture(t)
	if err := os.MkdirAll(filepath.Join(dir, "prog"), 0o755); err != nil {
		t.Fatal(err)
	}
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	plan := taughtPlan(t, fresh, EditRequest{
		File:             "prog/main.go",
		TaughtText:       "func Greet() string { return \"hi\" }",
		TaughtName:       "Greet",
		TaughtDeclKind:   "function",
		TaughtKindPhrase: "greeter",
	})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, "package main") || !strings.Contains(after, "func Greet()") || !strings.Contains(after, "func main() {") {
		t.Errorf("new main.go should carry the decl and the stub:\n%s", after)
	}
}

// An aliased import spec survives the trip through the
// request: the spec is one unit ("db "database/sql""), not
// two tokens.
func TestTaughtAppendAliasedImport(t *testing.T) {
	_, p := conceptFixture(t)
	plan := taughtPlan(t, p, EditRequest{
		File:             "bare.go",
		TaughtText:       "func NoRows() error {\n\treturn db.ErrNoRows\n}",
		TaughtName:       "NoRows",
		TaughtDeclKind:   "function",
		TaughtKindPhrase: "row check",
		TaughtImports:    `db "database/sql"`,
		TaughtNote:       "From your example Load in jim/db.go, taught Nov 4, 2026.",
	})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, `db "database/sql"`) {
		t.Fatalf("aliased import not carried whole:\n%s", after)
	}
}
