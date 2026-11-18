package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// sigFixture builds a module exercising every signature shape:
// grouped parameters, nested (recursive-shaped) calls, call
// sites across two packages plus a test file, shadowing traps
// for renames, an interface-dictated method, a function taken as
// a value, variadic + spread, context callers with varied names,
// and a type whose import has to be copied into the decl file.
func sigFixture(t *testing.T) (string, *Project) {
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
	write("go.mod", "module example.com/sig\n\ngo 1.21\n")
	write("calc/calc.go", `package calc

func Combine(a, b int, label string) int {
	return a + b
}

func Mul(a, b int) int { return a * b }

func Scale(factor int, values []int) []int {
	out := make([]int, len(values))
	double := func(x int) int { return x * factor }
	for i, v := range values {
		out[i] = double(v)
	}
	if factor > 10 {
		factor := 99
		_ = factor
	}
	return out
}

func Other(factor int) int { return factor * 2 }

func Route(mode string, level int, strict bool) string {
	if strict {
		return mode
	}
	return ""
}

func Div(a, b int) (q int, err error) {
	if b == 0 {
		return 0, nil
	}
	return a / b, nil
}

var DefaultLimit = 10

func Fetch(key string) string { return key }

func FetchCaller() string { return Fetch("k") }

func FetchRemote(key string) string { return key }

func Pair[T any](first T, second T) (T, T) { return first, second }
`)
	write("calc/calc_test.go", `package calc

import "testing"

func TestCombine(t *testing.T) {
	if Combine(3, 4, "y") == 0 {
		t.Fatal("zero")
	}
}
`)
	write("calc/inc.go", `package calc

func Inc(x int) int { return x + 1 }
`)
	write("calc/incuse.go", `package calc

import "context"

func IncTwice(ctx context.Context, x int) int {
	return Inc(x) + Inc(x)
}
`)
	write("calc/variadic.go", `package calc

import "context"

func SumAll(nums ...int) int {
	s := 0
	for _, n := range nums {
		s += n
	}
	return s
}

func TotalAll(ctx context.Context, xs []int) int {
	return SumAll(xs...)
}
`)
	write("calc/plain.go", `package calc

func WaitFor(key string) bool { return key != "" }
`)
	write("calc/calltime.go", `package calc

import "time"

var WaitUnit = time.Second

func GoWait() bool { return WaitFor("x") }
`)
	write("app/app.go", `package app

import "example.com/sig/calc"

var MulOp = calc.Mul

func UseCombine() int {
	return calc.Combine(1, 2, "x")
}

func UseNested() int {
	return calc.Combine(calc.Combine(1, 2, "i"), 5, "o")
}

func UseRoute() string {
	return calc.Route("fast", 3, true)
}

func UseFetchRemote(key string) string {
	return calc.FetchRemote(key)
}
`)
	write("flow/flow.go", `package flow

import "context"

func StepOK(name string, count int) string {
	if count > 0 {
		return StepOK(name, count-1)
	}
	return name
}

func CallerA(ctx context.Context) string { return StepOK("a", 1) }

func CallerB(c context.Context) string { return StepOK("b", 2) }

func CallerC() string {
	ctx := context.Background()
	_ = ctx
	return StepOK("c", 3)
}

func StepBad(name string) string { return name }

func CallerNoCtx() string { return StepBad("x") }

func CallerWithCtx(ctx context.Context) string { return StepBad("y") }
`)
	write("store/store.go", `package store

type Saver interface {
	Save(name string, data []byte) error
}

type FileSaver struct{}

func (FileSaver) Save(name string, data []byte) error { return nil }

var _ Saver = FileSaver{}
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze fixture: %v", err)
	}
	return dir, p
}

// sigFileAfter returns a plan's After text for one file.
func sigFileAfter(t *testing.T, plan *EditPlan, rel string) string {
	t.Helper()
	for _, f := range plan.Files {
		if f.Rel == rel {
			return string(f.After)
		}
	}
	var rels []string
	for _, f := range plan.Files {
		rels = append(rels, f.Rel)
	}
	t.Fatalf("plan has no file %s (has %s)", rel, strings.Join(rels, ", "))
	return ""
}

// --- (a) rename a parameter, with shadowing traps ---

func TestSigRenameParamShadowing(t *testing.T) {
	dir, p := sigFixture(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditRenameParam, FuncName: "Scale", SigOld: "factor", SigNew: "multiplier"})
	after := sigFileAfter(t, plan, "calc/calc.go")
	for _, want := range []string{
		"func Scale(multiplier int, values []int)",
		"return x * multiplier",                            // closure capture renamed
		"if multiplier > 10 {",                             // direct use renamed
		"factor := 99",                                     // the shadowing local is NOT the parameter
		"_ = factor",                                       // its use stays bound to it
		"func Other(factor int) int { return factor * 2 }", // another function's same-named param untouched
	} {
		if !strings.Contains(after, want) {
			t.Errorf("renamed file missing %q:\n%s", want, after)
		}
	}
	if strings.Contains(after, "double := func(x int) int { return x * factor }") {
		t.Errorf("closure capture was not renamed:\n%s", after)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	if !plan.Verify(reanalyze(t, dir)) {
		t.Fatal("verify failed after rename")
	}
}

func TestSigRenameParamRefusals(t *testing.T) {
	_, p := sigFixture(t)
	// A named result holds the name — parameters and results
	// share one scope in Go.
	mustRefuse(t, p, EditRequest{Kind: EditRenameParam, FuncName: "Div", SigOld: "a", SigNew: "q"}, "named result")
	// The body declares the name itself at top level.
	mustRefuse(t, p, EditRequest{Kind: EditRenameParam, FuncName: "Scale", SigOld: "factor", SigNew: "double"}, "declares double itself")
	// No such parameter.
	mustRefuse(t, p, EditRequest{Kind: EditRenameParam, FuncName: "Scale", SigOld: "nope", SigNew: "x"}, "no parameter named")
	// Same name.
	mustRefuse(t, p, EditRequest{Kind: EditRenameParam, FuncName: "Scale", SigOld: "factor", SigNew: "factor"}, "already named")
	// Blanking a used parameter.
	mustRefuse(t, p, EditRequest{Kind: EditRenameParam, FuncName: "Scale", SigOld: "factor", SigNew: "_"}, "no name to hold")
}

// A parameter rename on an interface method is fine: names are
// not part of a type, so no interface dictates them. (The
// reorder/add forms on the same method refuse — see below.)
func TestSigRenameParamOnInterfaceMethod(t *testing.T) {
	dir, p := sigFixture(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditRenameParam, FuncName: "FileSaver.Save", SigOld: "name", SigNew: "path"})
	after := sigFileAfter(t, plan, "store/store.go")
	if !strings.Contains(after, "func (FileSaver) Save(path string, data []byte) error") {
		t.Fatalf("method declaration not renamed:\n%s", after)
	}
	if !strings.Contains(after, "Save(name string, data []byte) error") {
		t.Fatalf("the interface itself must stay untouched:\n%s", after)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	if !plan.Verify(reanalyze(t, dir)) {
		t.Fatal("verify failed after method param rename")
	}
}

// --- (b) swap, grouped params, sites across packages + test file ---

func TestSigSwapGroupedAndNested(t *testing.T) {
	dir, p := sigFixture(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditReorderParams, FuncName: "Combine", SigOld: "a", SigNew: "b"})
	if !strings.Contains(plan.Summary, "func Combine(a, b int, label string) int becomes func Combine(b int, a int, label string) int") {
		t.Errorf("summary lacks the old → new signature: %s", plan.Summary)
	}
	decl := sigFileAfter(t, plan, "calc/calc.go")
	if !strings.Contains(decl, "func Combine(b int, a int, label string) int") {
		t.Errorf("grouped declaration not expanded + swapped:\n%s", decl)
	}
	// Byte-exact outside the edited spans: the whole caller file
	// is asserted verbatim, including the nested call permuted
	// inside the outer call's argument.
	wantApp := `package app

import "example.com/sig/calc"

var MulOp = calc.Mul

func UseCombine() int {
	return calc.Combine(2, 1, "x")
}

func UseNested() int {
	return calc.Combine(5, calc.Combine(2, 1, "i"), "o")
}

func UseRoute() string {
	return calc.Route("fast", 3, true)
}

func UseFetchRemote(key string) string {
	return calc.FetchRemote(key)
}
`
	if got := sigFileAfter(t, plan, "app/app.go"); got != wantApp {
		t.Errorf("app/app.go after swap:\n%s\nwant:\n%s", got, wantApp)
	}
	test := sigFileAfter(t, plan, "calc/calc_test.go")
	if !strings.Contains(test, `Combine(4, 3, "y")`) {
		t.Errorf("test-file call site not swapped:\n%s", test)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	if !plan.Verify(reanalyze(t, dir)) {
		t.Fatal("verify failed after swap")
	}
}

// --- (c) move first / last / before / after ---

func TestSigMoveParams(t *testing.T) {
	cases := []struct {
		name     string
		req      EditRequest
		wantDecl string
		wantCall string
	}{
		{"first", EditRequest{Kind: EditReorderParams, FuncName: "Route", SigOld: "level", SigMove: "first"},
			"func Route(level int, mode string, strict bool) string", `calc.Route(3, "fast", true)`},
		{"last", EditRequest{Kind: EditReorderParams, FuncName: "Route", SigOld: "mode", SigMove: "last"},
			"func Route(level int, strict bool, mode string) string", `calc.Route(3, true, "fast")`},
		{"before", EditRequest{Kind: EditReorderParams, FuncName: "Route", SigOld: "strict", SigMove: "before", SigNew: "mode"},
			"func Route(strict bool, mode string, level int) string", `calc.Route(true, "fast", 3)`},
		{"after", EditRequest{Kind: EditReorderParams, FuncName: "Route", SigOld: "mode", SigMove: "after", SigNew: "level"},
			"func Route(level int, mode string, strict bool) string", `calc.Route(3, "fast", true)`},
		{"already first", EditRequest{Kind: EditReorderParams, FuncName: "Route", SigOld: "mode", SigMove: "first"},
			"", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir, p := sigFixture(t)
			if c.wantDecl == "" {
				mustRefuse(t, p, c.req, "leaves")
				return
			}
			plan := mustPlan(t, p, c.req)
			if got := sigFileAfter(t, plan, "calc/calc.go"); !strings.Contains(got, c.wantDecl) {
				t.Errorf("declaration: got\n%s\nwant substring %q", got, c.wantDecl)
			}
			if got := sigFileAfter(t, plan, "app/app.go"); !strings.Contains(got, c.wantCall) {
				t.Errorf("call site: got\n%s\nwant substring %q", got, c.wantCall)
			}
			applyPlan(t, dir, plan)
			buildFixture(t, dir)
			if !plan.Verify(reanalyze(t, dir)) {
				t.Fatal("verify failed after move")
			}
		})
	}
}

// --- (d) refusals: spread site, function as value, interface ---

func TestSigReorderRefusals(t *testing.T) {
	_, p := sigFixture(t)
	// Spread call site, named: SumAll is variadic and TotalAll
	// spreads a slice into it.
	r := mustRefuse(t, p, EditRequest{Kind: EditReorderParams, FuncName: "SumAll", SigOld: "nums", SigMove: "first"}, "spreads a slice")
	if !strings.Contains(r, "calc/variadic.go") {
		t.Errorf("spread refusal should name the site file: %s", r)
	}
	// The function is taken as a value in app.
	r = mustRefuse(t, p, EditRequest{Kind: EditReorderParams, FuncName: "Mul", SigOld: "a", SigNew: "b"}, "assigned as a value")
	if !strings.Contains(r, "app/app.go") {
		t.Errorf("value-use refusal should name the site: %s", r)
	}
	// An interface dictates this method's signature.
	r = mustRefuse(t, p, EditRequest{Kind: EditReorderParams, FuncName: "FileSaver.Save", SigOld: "name", SigNew: "data"}, "interface")
	if !strings.Contains(r, "Saver") {
		t.Errorf("interface refusal should name the interface: %s", r)
	}
	// Generic target.
	mustRefuse(t, p, EditRequest{Kind: EditReorderParams, FuncName: "Pair", SigOld: "first", SigNew: "second"}, "generic")
	// Swapping a parameter with itself.
	mustRefuse(t, p, EditRequest{Kind: EditReorderParams, FuncName: "Combine", SigOld: "a", SigNew: "a"}, "same parameter")
}

func TestSigAddRefusesInterfaceMethod(t *testing.T) {
	_, p := sigFixture(t)
	r := mustRefuse(t, p, EditRequest{Kind: EditAddParam, FuncName: "FileSaver.Save", SigType: "int", SigName: "n", SigUsing: "0"}, "interface")
	if !strings.Contains(r, "Saver") {
		t.Errorf("interface refusal should name the interface: %s", r)
	}
}

// A build-tagged file the typed load cannot cover mentions the
// target: the plan refuses rather than change the signature
// blind.
func TestSigUncoveredFileRefuses(t *testing.T) {
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
	write("go.mod", "module example.com/sigx\n\ngo 1.21\n")
	write("calc/calc.go", `package calc

func Combine(a, b int) int { return a + b }

func UseIt() int { return Combine(1, 2) }
`)
	write("calc/excluded.go", "//go:build never\n\npackage calc\n\nfunc Excluded() int { return Combine(3, 4) }\n")
	p, err := Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze fixture: %v", err)
	}
	mustRefuse(t, p, EditRequest{Kind: EditReorderParams, FuncName: "Combine", SigOld: "a", SigNew: "b"}, "isn't part of this platform's build")
}

// --- (e) add a context parameter ---

func TestSigAddContext(t *testing.T) {
	dir, p := sigFixture(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddParam, FuncName: "StepOK", SigType: "context.Context"})
	after := sigFileAfter(t, plan, "flow/flow.go")
	for _, want := range []string{
		"func StepOK(ctx context.Context, name string, count int) string",
		"return StepOK(ctx, name, count-1)", // recursive site passes the new parameter itself
		`StepOK(ctx, "a", 1)`,               // caller's ctx parameter
		`StepOK(c, "b", 2)`,                 // caller's context under another name
		`StepOK(ctx, "c", 3)`,               // caller's local ctx
	} {
		if !strings.Contains(after, want) {
			t.Errorf("add-context result missing %q:\n%s", want, after)
		}
	}
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, "passes c — its own context parameter") {
		t.Errorf("manifest should show the per-site derived arguments:\n%s", notes)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	if !plan.Verify(reanalyze(t, dir)) {
		t.Fatal("verify failed after add-context")
	}
}

func TestSigAddContextOneCallerLacksIt(t *testing.T) {
	_, p := sigFixture(t)
	r := mustRefuse(t, p, EditRequest{Kind: EditAddParam, FuncName: "StepBad", SigType: "context.Context"}, "no context is in scope")
	if !strings.Contains(r, "flow/flow.go") || !strings.Contains(r, "CallerNoCtx") {
		t.Errorf("refusal should name the failing site and its function: %s", r)
	}
	if !strings.Contains(r, "everywhere or nowhere") {
		t.Errorf("refusal should state the all-or-nothing rule: %s", r)
	}
}

func TestSigAddContextImportAndVariadic(t *testing.T) {
	dir, p := sigFixture(t)
	// The declaration's file has no imports at all: the plan
	// adds "context" there, copied from the standard spelling.
	plan := mustPlan(t, p, EditRequest{Kind: EditAddParam, FuncName: "Inc", SigType: "context.Context"})
	decl := sigFileAfter(t, plan, "calc/inc.go")
	if !strings.Contains(decl, `"context"`) || !strings.Contains(decl, "func Inc(ctx context.Context, x int) int") {
		t.Errorf("decl file should gain the import and the parameter:\n%s", decl)
	}
	use := sigFileAfter(t, plan, "calc/incuse.go")
	if !strings.Contains(use, "return Inc(ctx, x) + Inc(ctx, x)") {
		t.Errorf("both call sites should gain the argument:\n%s", use)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	if !plan.Verify(reanalyze(t, dir)) {
		t.Fatal("verify failed after add-context with import")
	}

	// A variadic function takes the new FIRST parameter fine —
	// even at a spread call site.
	dir2, p2 := sigFixture(t)
	plan2 := mustPlan(t, p2, EditRequest{Kind: EditAddParam, FuncName: "SumAll", SigType: "context.Context"})
	after := sigFileAfter(t, plan2, "calc/variadic.go")
	if !strings.Contains(after, "func SumAll(ctx context.Context, nums ...int) int") {
		t.Errorf("variadic declaration:\n%s", after)
	}
	if !strings.Contains(after, "return SumAll(ctx, xs...)") {
		t.Errorf("spread site should gain the context ahead of the spread:\n%s", after)
	}
	applyPlan(t, dir2, plan2)
	buildFixture(t, dir2)
	if !plan2.Verify(reanalyze(t, dir2)) {
		t.Fatal("verify failed after add-context on variadic")
	}

	// Already has a context.
	mustRefuse(t, p, EditRequest{Kind: EditAddParam, FuncName: "IncTwice", SigType: "context.Context"}, "already takes a context")
}

// --- (f) add a parameter with the user's expression ---

func TestSigAddUsing(t *testing.T) {
	dir, p := sigFixture(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddParam, FuncName: "Fetch", SigType: "int", SigName: "limit", SigUsing: "DefaultLimit"})
	decl := sigFileAfter(t, plan, "calc/calc.go")
	if !strings.Contains(decl, "func Fetch(key string, limit int) string") {
		t.Errorf("declaration:\n%s", decl)
	}
	if !strings.Contains(decl, `return Fetch("k", DefaultLimit)`) {
		t.Errorf("call site should carry the expression verbatim:\n%s", decl)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	if !plan.Verify(reanalyze(t, dir)) {
		t.Fatal("verify failed after add-using")
	}
}

func TestSigAddUsingUnsatisfiableSite(t *testing.T) {
	_, p := sigFixture(t)
	// DefaultLimit lives in calc; the only caller of FetchRemote
	// is in app, where the name means nothing.
	r := mustRefuse(t, p, EditRequest{Kind: EditAddParam, FuncName: "FetchRemote", SigType: "int", SigName: "limit", SigUsing: "DefaultLimit"}, "isn't in scope")
	if !strings.Contains(r, "app/app.go") {
		t.Errorf("refusal should name the unsatisfiable site's file: %s", r)
	}
}

func TestSigAddUsingCopiesImport(t *testing.T) {
	dir, p := sigFixture(t)
	plan := mustPlan(t, p, EditRequest{Kind: EditAddParam, FuncName: "WaitFor", SigType: "time.Duration", SigName: "wait", SigUsing: "time.Second"})
	decl := sigFileAfter(t, plan, "calc/plain.go")
	if !strings.Contains(decl, `"time"`) {
		t.Errorf("decl file should gain the time import copied from elsewhere in the project:\n%s", decl)
	}
	if !strings.Contains(decl, "func WaitFor(key string, wait time.Duration) bool") {
		t.Errorf("declaration:\n%s", decl)
	}
	site := sigFileAfter(t, plan, "calc/calltime.go")
	if !strings.Contains(site, `WaitFor("x", time.Second)`) {
		t.Errorf("call site:\n%s", site)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	if !plan.Verify(reanalyze(t, dir)) {
		t.Fatal("verify failed after add-using with copied import")
	}
}

func TestSigAddNoUsingGuidance(t *testing.T) {
	_, p := sigFixture(t)
	mustRefuse(t, p, EditRequest{Kind: EditAddParam, FuncName: "Fetch", SigType: "int", SigName: "limit", SigNoUsing: true}, "using <expression>")
}
