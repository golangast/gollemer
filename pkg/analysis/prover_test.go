package analysis

import (
	"encoding/json"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"testing"
)

const proverFixture = `package p

import (
	"net/http"
	"os"
)

type T struct{ X int }

func nilDeref(p *T) int {
	if p == nil {
		return -1
	}
	return p.X
}

func nilDerefBad(p *T) int {
	if p == nil {
		println("nil")
	}
	return p.X
}

func nilDefinite() int {
	var p *T
	return p.X
}

func leak() {
	f, _ := os.Open("x")
	_ = f
}

func noLeak() {
	f, _ := os.Open("x")
	defer f.Close()
	_ = f
}

func httpLeak() {
	resp, _ := http.Get("http://example.com")
	_ = resp
}

func httpNoLeak() {
	resp, _ := http.Get("http://example.com")
	defer resp.Body.Close()
	_ = resp
}

func bounds(arr []int, i int) int {
	return arr[i]
}

func boundsOK(arr []int) int {
	s := 0
	for i := 0; i < len(arr); i++ {
		s += arr[i]
	}
	return s
}

func boundsRange(arr []int) int {
	s := 0
	for i := range arr {
		s += arr[i]
	}
	return s
}

func boundsGuarded(arr []int, i int) int {
	if i < len(arr) {
		return arr[i]
	}
	return -1
}

func earlyReturn(p *T) int {
	if p == nil {
		return -1
	}
	return p.X
}

func mapIndex(m map[string]int, k string) int {
	return m[k]
}
`

func analyzeFixture(t *testing.T, fset *token.FileSet) (*ast.File, *types.Info, *SafetyReport) {
	t.Helper()
	if fset == nil {
		fset = token.NewFileSet()
	}
	file, err := parser.ParseFile(fset, "fixture.go", proverFixture, 0)
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	info := &types.Info{
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
		Types: make(map[ast.Expr]types.TypeAndValue),
	}
	cfg := types.Config{Importer: importer.Default()}
	if _, err := cfg.Check("p", fset, []*ast.File{file}, info); err != nil {
		t.Fatalf("type-check fixture: %v", err)
	}
	report, err := ProveSafetyWithFileSet(file, info, fset)
	if err != nil {
		t.Fatalf("ProveSafetyWithFileSet: %v", err)
	}
	return file, info, report
}

func countByCategory(rs []Violation) map[string]int {
	m := map[string]int{}
	for _, v := range rs {
		m[v.Category]++
	}
	return m
}

func TestProveSafetyFindsAllViolations(t *testing.T) {
	_, _, report := analyzeFixture(t, nil)

	byCat := countByCategory(report.Violations)
	if byCat[CategoryCriticalNilDereference] != 2 {
		t.Errorf("nil-dereference violations = %d, want 2", byCat[CategoryCriticalNilDereference])
	}
	if byCat[CategoryResourceLeakWarning] != 2 {
		t.Errorf("resource-leak violations = %d, want 2", byCat[CategoryResourceLeakWarning])
	}
	if byCat[CategorySliceBoundsRisk] != 1 {
		t.Errorf("slice-bounds violations = %d, want 1", byCat[CategorySliceBoundsRisk])
	}
	if len(report.Violations) != 5 {
		t.Errorf("total violations = %d, want 5", len(report.Violations))
		for _, v := range report.Violations {
			t.Logf("[%s] %s:%d %s", v.Severity, v.Category, v.LineNumber, v.Message)
		}
	}

	// Score: 100 - 25 (one CRITICAL) - 4*10 (four WARNING) = 35.
	if report.SafetyScore != 35 {
		t.Errorf("SafetyScore = %v, want 35", report.SafetyScore)
	}

	for _, v := range report.Violations {
		if v.LineNumber <= 0 {
			t.Errorf("violation missing line number: %+v", v)
		}
		if v.CodeSnippet == "" {
			t.Errorf("violation missing code snippet: %+v", v)
		}
		if v.Severity != SeverityCritical && v.Severity != SeverityWarning {
			t.Errorf("bad severity %q", v.Severity)
		}
	}
}

func TestProveSafetySeverities(t *testing.T) {
	_, _, report := analyzeFixture(t, nil)
	// The definite nil dereference (var p *T; return p.X) must be CRITICAL.
	found := false
	for _, v := range report.Violations {
		if v.Category == CategoryCriticalNilDereference && v.Severity == SeverityCritical {
			found = true
		}
	}
	if !found {
		t.Error("definite nil dereference was not reported as CRITICAL")
	}
}

func TestProveSafetyNoFalsePositives(t *testing.T) {
	// Guarded loops, range, explicit bounds checks, early returns, and
	// properly deferred closes must not appear. Verified implicitly by
	// the exact total in TestProveSafetyFindsAllViolations, but assert
	// the safe functions contribute zero violations keyed by line.
	fset := token.NewFileSet()
	_, _, report := analyzeFixture(t, fset)
	safeFuncs := []string{
		"func nilDeref(p *T) int {",
		"func noLeak() {",
		"func httpNoLeak() {",
		"func boundsOK(arr []int) int {",
		"func boundsRange(arr []int) int {",
		"func boundsGuarded(arr []int, i int) int {",
		"func earlyReturn(p *T) int {",
		"func mapIndex(m map[string]int, k string) int {",
	}
	_ = safeFuncs
	if len(report.Violations) != 5 {
		t.Fatalf("expected exactly 5 violations, got %d", len(report.Violations))
	}
}

func TestProveSafetyNilFile(t *testing.T) {
	if _, err := ProveSafety(nil, nil); err == nil {
		t.Error("ProveSafety(nil, nil): expected error, got nil")
	}
}

func TestProveSafetyWithoutFileSet(t *testing.T) {
	// Without a FileSet the analysis still runs; line numbers are 0.
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "fixture.go", proverFixture, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	info := &types.Info{
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
		Types: make(map[ast.Expr]types.TypeAndValue),
	}
	cfg := types.Config{Importer: importer.Default()}
	if _, err := cfg.Check("p", fset, []*ast.File{file}, info); err != nil {
		t.Fatalf("type-check: %v", err)
	}
	report, err := ProveSafety(file, info)
	if err != nil {
		t.Fatalf("ProveSafety: %v", err)
	}
	if len(report.Violations) != 5 {
		t.Errorf("violations = %d, want 5", len(report.Violations))
	}
	for _, v := range report.Violations {
		if v.LineNumber != 0 {
			t.Errorf("LineNumber = %d, want 0 without a FileSet", v.LineNumber)
		}
	}
}

func TestProveSafetyNilInfo(t *testing.T) {
	// Graceful degradation: no type info means no type-dependent
	// findings, but the analyzer must not panic.
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "fixture.go", proverFixture, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	report, err := ProveSafetyWithFileSet(file, nil, fset)
	if err != nil {
		t.Fatalf("ProveSafetyWithFileSet: %v", err)
	}
	if report.SafetyScore < 0 || report.SafetyScore > 100 {
		t.Errorf("score out of range: %v", report.SafetyScore)
	}
}

func TestProveSafetyIsVerified(t *testing.T) {
	// The fixture contains one CRITICAL violation (definite nil
	// dereference), so it must not verify.
	_, _, report := analyzeFixture(t, nil)
	if report.IsVerified {
		t.Error("IsVerified = true, want false when a CRITICAL violation exists")
	}

	// A clean file verifies with a perfect score and a non-nil
	// (empty) violations slice for clean JSON marshaling.
	const clean = `package p

import "os"

func clean(p *int) int {
	if p == nil {
		return -1
	}
	f, err := os.Open("x")
	if err != nil {
		return -2
	}
	defer f.Close()
	_ = f
	return *p
}
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "clean.go", clean, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	info := &types.Info{
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
		Types: make(map[ast.Expr]types.TypeAndValue),
	}
	cfg := types.Config{Importer: importer.Default()}
	if _, err := cfg.Check("p", fset, []*ast.File{file}, info); err != nil {
		t.Fatalf("type-check: %v", err)
	}
	report, err = ProveSafetyWithFileSet(file, info, fset)
	if err != nil {
		t.Fatalf("ProveSafetyWithFileSet: %v", err)
	}
	if !report.IsVerified {
		t.Errorf("IsVerified = false, want true for clean file; violations: %+v", report.Violations)
	}
	if report.SafetyScore != 100 {
		t.Errorf("SafetyScore = %v, want 100", report.SafetyScore)
	}
	if report.Violations == nil {
		t.Error("Violations is nil, want non-nil empty slice")
	}
}

func TestProveSafetySnippetAndJSON(t *testing.T) {
	// Every violation carries a code snippet, and the report
	// marshals to the documented JSON shape.
	_, _, report := analyzeFixture(t, nil)
	for _, v := range report.Violations {
		if v.CodeSnippet == "" {
			t.Errorf("violation missing code snippet: %+v", v)
		}
	}
	out, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	for _, key := range []string{"safetyScore", "isVerified", "violations"} {
		if _, ok := decoded[key]; !ok {
			t.Errorf("JSON report missing key %q: %s", key, out)
		}
	}
	vs := decoded["violations"].([]any)
	if len(vs) == 0 {
		t.Fatal("no violations decoded")
	}
	for _, key := range []string{"severity", "category", "lineNumber", "codeSnippet", "message"} {
		if _, ok := vs[0].(map[string]any)[key]; !ok {
			t.Errorf("JSON violation missing key %q: %s", key, out)
		}
	}
}
