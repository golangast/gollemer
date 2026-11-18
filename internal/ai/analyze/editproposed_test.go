package analyze

import (
	"strings"
	"testing"
)

// The model-proposed declaration planner (editproposed.go):
// the rendered text arrives as data and is assembled like a
// concept-book declaration — appended at file scope, imports
// dressed, name collisions refused — with the plan carrying
// the note that owns its origin before any gate sees it.

func proposedPlan(t *testing.T, p *Project, req EditRequest) *EditPlan {
	t.Helper()
	req.Kind = EditProposedDecl
	plan, refusal := p.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("PlanEdit(proposed_decl) refused: %s", refusal)
	}
	return plan
}

// A proposed struct lands at the end of the file, the plan
// names the model as its source, and the applied project
// verifies.
func TestProposedStructPlanAndVerify(t *testing.T) {
	dir, p := conceptFixture(t)
	plan := proposedPlan(t, p, EditRequest{
		File:         "main.go",
		ProposedText: "type Point struct { X int; Y string }",
		ProposedName: "Point",
		ProposedKind: "struct",
	})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, "type Point struct {") || !strings.Contains(after, "X int") {
		t.Errorf("proposed struct not appended:\n%s", after)
	}
	if len(plan.Notes) != 1 || !strings.Contains(plan.Notes[0], "proposed by my small trained model") {
		t.Errorf("plan does not own the model's authorship: notes = %v", plan.Notes)
	}
	if !strings.Contains(plan.Summary, "as proposed by the model") {
		t.Errorf("summary does not name the model: %q", plan.Summary)
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

// A proposed method lands on its receiver in the file that
// declares the type, imports its types lean on are added, and
// the method form verifies (its name is NOT a package-level
// collision even though the type name is taken — by the type
// itself).
func TestProposedMethodPlanAndVerify(t *testing.T) {
	dir, p := conceptFixture(t)
	plan := proposedPlan(t, p, EditRequest{
		File:             "store/store.go",
		ProposedText:     "func ( s * Store ) ExpireSoon ( d time.Duration ) bool { panic ( \"not implemented\" ) }",
		ProposedName:     "ExpireSoon",
		ProposedKind:     "func",
		ProposedReceiver: "Store",
		ProposedImports:  "time",
	})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, "ExpireSoon") {
		t.Errorf("proposed method not appended:\n%s", after)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("method plan does not verify against the applied project")
	}
}

// The imports the declaration's types reference are added to
// a file that lacks them, the concept family's way.
func TestProposedImportsLand(t *testing.T) {
	_, p := conceptFixture(t)
	plan := proposedPlan(t, p, EditRequest{
		File:            "main.go",
		ProposedText:    "type Window struct { Span time.Duration }",
		ProposedName:    "Window",
		ProposedKind:    "struct",
		ProposedImports: "time",
	})
	if after := string(plan.Files[0].After); !strings.Contains(after, `"time"`) {
		t.Errorf("time import not added:\n%s", after)
	}
}

// Refusals: no text, an illegal name, and a name the package
// already declares — each in the planner family's voice, and
// never a plan.
func TestProposedRefusals(t *testing.T) {
	_, p := conceptFixture(t)
	for _, c := range []struct {
		req  EditRequest
		want string
	}{
		{EditRequest{File: "main.go", ProposedName: "Point", ProposedKind: "struct"},
			"none was proposed"},
		{EditRequest{File: "main.go", ProposedText: "type X struct{}", ProposedName: "not a name", ProposedKind: "struct"},
			"isn't a name Go allows"},
		{EditRequest{File: "main.go", ProposedText: "type Helper struct{}", ProposedName: "Helper", ProposedKind: "struct"},
			"already exists in package main"},
		{EditRequest{File: "", ProposedText: "type Point struct{}", ProposedName: "Point", ProposedKind: "struct"},
			"needs a file"},
	} {
		c.req.Kind = EditProposedDecl
		plan, refusal := p.PlanEdit(c.req)
		if plan != nil {
			t.Errorf("PlanEdit(%+v) produced a plan, want refusal %q", c.req, c.want)
		}
		if !strings.Contains(refusal, c.want) {
			t.Errorf("PlanEdit(%+v) refusal = %q, want it to say %q", c.req, refusal, c.want)
		}
	}
}
