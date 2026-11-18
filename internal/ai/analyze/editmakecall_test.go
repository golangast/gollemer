package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The make-and-call planner: one ask, two files. The callee
// is the book's function row planned by the concept planner;
// the caller is assembled from the PLANNED declaration.

func makeCallEntry() *ConceptEntry {
	return &ConceptEntry{
		ID:       "function",
		Kind:     ConceptDeclaration,
		Names:    []string{"function", "func"},
		Template: "func {name}() {\n\t// TODO: write the body\n}",
	}
}

func makeCallProject(t *testing.T, extra map[string]string) (string, *Project) {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"go.mod": "module example.com/mc\n\ngo 1.21\n",
		"api/main.go": `package main

import "fmt"

func main() {
	fmt.Println("api")
}
`,
		"api/report.go": "package main\n",
		"lib/lib.go": `package lib

func Helper() int { return 1 }
`,
		"web/web.go": `package web

func Serve() {}
`,
	}
	for rel, src := range extra {
		files[rel] = src
	}
	for rel, src := range files {
		abs := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(abs, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	proj, err := Analyze(dir)
	if err != nil || proj == nil {
		t.Fatalf("Analyze: %v", err)
	}
	return dir, proj
}

func makeCallReq(callee, target, caller, host string) EditRequest {
	return EditRequest{
		Kind:           EditMakeCall,
		File:           callee,
		Concept:        makeCallEntry(),
		ConceptName:    "Validate",
		MakeCallTarget: target,
		MakeCallCaller: caller,
		MakeCallHost:   host,
	}
}

func planFilesByRel(plan *EditPlan) map[string]EditFile {
	out := map[string]EditFile{}
	for _, f := range plan.Files {
		out[f.Rel] = f
	}
	return out
}

// The headline: callee created new in api/, caller a new
// function in the same package's other file.
func TestMakeCallNamedCallerSamePackage(t *testing.T) {
	_, proj := makeCallProject(t, nil)
	plan, refusal := proj.PlanEdit(makeCallReq("api/validate.go", "api/report.go", "Report", ""))
	if refusal != "" || plan == nil {
		t.Fatalf("refused: %q", refusal)
	}
	if len(plan.Files) != 2 {
		t.Fatalf("plan touches %d files, want 2", len(plan.Files))
	}
	byRel := planFilesByRel(plan)
	callee := byRel["api/validate.go"]
	if callee.Before != nil {
		t.Errorf("callee file should be a creation (Before nil)")
	}
	if !strings.Contains(string(callee.After), "func Validate() {") {
		t.Errorf("callee After missing the declaration:\n%s", callee.After)
	}
	caller := byRel["api/report.go"]
	if !strings.Contains(string(caller.After), "func Report() {") || !strings.Contains(string(caller.After), "Validate()") {
		t.Errorf("caller After missing Report calling Validate:\n%s", caller.After)
	}
	if strings.Contains(plan.Summary, "TODO") {
		t.Errorf("summary shouldn't carry the TODO note: %q", plan.Summary)
	}
	joined := strings.Join(plan.Notes, "\n")
	if !strings.Contains(joined, "empty scaffold") {
		t.Errorf("plan notes must own the scaffold body; notes:\n%s", joined)
	}
	if !strings.Contains(joined, "One plan covers both files") {
		t.Errorf("plan notes must own the joint rollback; notes:\n%s", joined)
	}
}

// Cross-package: the caller in a library package imports and
// qualifies the callee's package.
func TestMakeCallNamedCallerCrossPackage(t *testing.T) {
	_, proj := makeCallProject(t, nil)
	req := makeCallReq("lib/checks.go", "web/web.go", "ServeCheck", "")
	plan, refusal := proj.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("refused: %q", refusal)
	}
	byRel := planFilesByRel(plan)
	caller := byRel["web/web.go"]
	for _, want := range []string{`"example.com/mc/lib"`, "func ServeCheck() {", "lib.Validate()"} {
		if !strings.Contains(string(caller.After), want) {
			t.Errorf("cross-package caller missing %q:\n%s", want, caller.After)
		}
	}
}

// Bare form: the call lands at the end of the target's main.
func TestMakeCallBareHostedByMain(t *testing.T) {
	_, proj := makeCallProject(t, nil)
	req := makeCallReq("api/validate.go", "api/main.go", "", "")
	plan, refusal := proj.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("refused: %q", refusal)
	}
	byRel := planFilesByRel(plan)
	main := byRel["api/main.go"]
	if !strings.Contains(string(main.After), "fmt.Println(\"api\")\n\tValidate()\n}") {
		t.Errorf("call not at the end of main:\n%s", main.After)
	}
}

// Bare form where main lives in ANOTHER file of the target's
// package: the call lands there and the plan says so.
func TestMakeCallBareMainInSiblingFile(t *testing.T) {
	_, proj := makeCallProject(t, nil)
	req := makeCallReq("api/validate.go", "api/report.go", "", "")
	plan, refusal := proj.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("refused: %q", refusal)
	}
	byRel := planFilesByRel(plan)
	main, ok := byRel["api/main.go"]
	if !ok {
		t.Fatalf("plan doesn't touch api/main.go, the main's home: %v", plan.Files)
	}
	if !strings.Contains(string(main.After), "Validate()") {
		t.Errorf("call not in the sibling main:\n%s", main.After)
	}
	if !strings.Contains(strings.Join(plan.Notes, "\n"), "api/main.go") || !strings.Contains(strings.Join(plan.Notes, "\n"), "the call landed there, in the same package") {
		t.Errorf("plan must name where the call landed; notes:\n%s", strings.Join(plan.Notes, "\n"))
	}
}

// Explicit host form.
func TestMakeCallFromHost(t *testing.T) {
	_, proj := makeCallProject(t, nil)
	req := makeCallReq("web/checks.go", "web/web.go", "", "Serve")
	plan, refusal := proj.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("refused: %q", refusal)
	}
	byRel := planFilesByRel(plan)
	web := byRel["web/web.go"]
	if !strings.Contains(string(web.After), "func Serve() {\n\tValidate()\n}") {
		t.Errorf("call not inside Serve:\n%s", web.After)
	}
}

// Bare form with no main anywhere in the target's package:
// refusal spells both working forms and invents nothing.
func TestMakeCallBareNoMainRefuses(t *testing.T) {
	_, proj := makeCallProject(t, nil)
	req := makeCallReq("lib/checks.go", "lib/lib.go", "", "")
	plan, refusal := proj.PlanEdit(req)
	if plan != nil {
		t.Fatalf("planned despite no host:\n%s", plan.Summary)
	}
	for _, want := range []string{"no main in", "lib/lib.go or its package to hold the call", "call it from <Func> in lib/lib.go", "add a function named <Caller> in lib/lib.go that calls it"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("refusal missing %q:\n%s", want, refusal)
		}
	}
}

// The declaration's own refusals hold verbatim: a callee name
// that already exists in the target package.
func TestMakeCallCalleeNameExistsRefuses(t *testing.T) {
	_, proj := makeCallProject(t, map[string]string{
		"api/validate.go": "package main\n\nfunc Validate() {}\n",
	})
	plan, refusal := proj.PlanEdit(makeCallReq("api/other.go", "api/report.go", "Report", ""))
	if plan != nil {
		t.Fatalf("planned despite the name collision")
	}
	if !strings.Contains(refusal, "Validate already exists in package main") {
		t.Errorf("refusal isn't the declaration's own:\n%s", refusal)
	}
}

// An unexported callee name aimed across packages refuses the
// way the uses route refuses it.
func TestMakeCallUnexportedCrossPackageRefuses(t *testing.T) {
	_, proj := makeCallProject(t, nil)
	req := makeCallReq("lib/checks.go", "web/web.go", "ServeCheck", "")
	req.ConceptName = "validate"
	plan, refusal := proj.PlanEdit(req)
	if plan != nil {
		t.Fatalf("planned despite the unexported cross-package callee")
	}
	if !strings.Contains(refusal, "isn't exported") {
		t.Errorf("refusal:\n%s", refusal)
	}
}

// A caller named exactly like the callee, same package:
// the plan's own declaration would collide.
func TestMakeCallCallerNamedLikeCalleeRefuses(t *testing.T) {
	_, proj := makeCallProject(t, nil)
	req := makeCallReq("api/validate.go", "api/report.go", "Validate", "")
	plan, refusal := proj.PlanEdit(req)
	if plan != nil {
		t.Fatalf("planned despite caller==callee in one package")
	}
	if !strings.Contains(refusal, "can't hold two of them") {
		t.Errorf("refusal:\n%s", refusal)
	}
}

// The target file must exist (the calling side's rule).
func TestMakeCallMissingTargetRefuses(t *testing.T) {
	_, proj := makeCallProject(t, nil)
	req := makeCallReq("api/validate.go", "api/absent.go", "Report", "")
	plan, refusal := proj.PlanEdit(req)
	if plan != nil {
		t.Fatalf("planned despite the missing target")
	}
	if !strings.Contains(refusal, "api/absent.go") || !strings.Contains(refusal, "in this project — the calling side goes into a file that already exists") {
		t.Errorf("refusal:\n%s", refusal)
	}
}

// Verify: applied to a real tree (by hand, the way the spine
// applies), the merged plan verifies.
func TestMakeCallVerifyAfterApply(t *testing.T) {
	dir, proj := makeCallProject(t, nil)
	plan, refusal := proj.PlanEdit(makeCallReq("api/validate.go", "api/report.go", "Report", ""))
	if refusal != "" {
		t.Fatalf("refused: %q", refusal)
	}
	for _, f := range plan.Files {
		abs := filepath.Join(dir, f.Rel)
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(abs, f.After, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	fresh, err := Analyze(dir)
	if err != nil || fresh == nil {
		t.Fatalf("re-analyze: %v", err)
	}
	if !plan.Verify(fresh) {
		t.Errorf("merged plan doesn't verify against the applied tree")
	}
}
