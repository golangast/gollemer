package xray

import (
	"context"
	"encoding/json"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

func testCtx() context.Context { return context.Background() }

func checkResponse(t *testing.T, resp *XRayResponse) {
	t.Helper()
	if strings.TrimSpace(resp.GeneratedCode) == "" {
		t.Error("empty GeneratedCode")
	}
	if strings.TrimSpace(resp.BeginnerExplanation) == "" {
		t.Error("empty BeginnerExplanation")
	}
	if strings.TrimSpace(resp.Analogy) == "" {
		t.Error("empty Analogy")
	}
	if resp.SafetyBadge.NullSafetyStatus == "" || resp.SafetyBadge.AllocEstimate == "" {
		t.Errorf("incomplete SafetyBadge: %+v", resp.SafetyBadge)
	}
	again, err := format.Source([]byte(resp.GeneratedCode))
	if err != nil {
		t.Fatalf("GeneratedCode does not parse: %v", err)
	}
	if string(again) != resp.GeneratedCode {
		t.Error("GeneratedCode is not gofmt-stable")
	}
	if _, err := json.Marshal(resp); err != nil {
		t.Errorf("response does not marshal to JSON: %v", err)
	}
}

func TestSynthesizeWithXRayBuildPrompts(t *testing.T) {
	eng := NewEngine()

	resp, err := eng.SynthesizeWithXRay(testCtx(), XRayRequest{
		Query: "Create an HTTP server on port 8080 with a JSON health check",
	})
	if err != nil {
		t.Fatalf("http server: %v", err)
	}
	checkResponse(t, resp)
	for _, want := range []string{`":8080"`, "/healthz", `"status":"ok"`, "application/json"} {
		if !strings.Contains(resp.GeneratedCode, want) {
			t.Errorf("http server code missing %q", want)
		}
	}
	if len(resp.VisualSequence) == 0 {
		t.Error("http server: empty VisualSequence")
	}
	if got := resp.SafetyBadge.NullSafetyStatus; got != "clean" && got != "warnings" && got != "critical" {
		t.Errorf("bad NullSafetyStatus %q", got)
	}

	resp, err = eng.SynthesizeWithXRay(testCtx(), XRayRequest{
		Query: "Create a worker pool using channels and WaitGroup",
	})
	if err != nil {
		t.Fatalf("worker pool: %v", err)
	}
	checkResponse(t, resp)
	if !strings.Contains(resp.GeneratedCode, "sync.WaitGroup") {
		t.Error("worker pool code missing sync.WaitGroup")
	}
	if !strings.Contains(resp.Analogy, "conveyor belt") && !strings.Contains(resp.Analogy, "factory line") {
		t.Errorf("worker pool analogy off-topic: %q", resp.Analogy)
	}
}

const authCode = `package auth

import "fmt"

// HandleLogin processes a login attempt end to end.
func HandleLogin(user string) {
	fmt.Println("login attempt for", user)
	if ValidateJWT(user) {
		QueryDB(user)
	}
}

// ValidateJWT checks the token signature.
func ValidateJWT(user string) bool {
	fmt.Println("validating token for", user)
	return user != ""
}

// QueryDB loads the user row.
func QueryDB(user string) {
	fmt.Println("querying database for", user)
}
`

func TestSynthesizeWithXRayTrace(t *testing.T) {
	eng := NewEngine()
	resp, err := eng.SynthesizeWithXRay(testCtx(), XRayRequest{
		Query:         "Trace how user authentication flows from handler to database",
		TargetPackage: "auth",
		ContextCode:   authCode,
	})
	if err != nil {
		t.Fatalf("trace: %v", err)
	}
	checkResponse(t, resp)
	wantTitles := []string{"Step 1: Handle login", "Step 2: Validate JWT", "Step 3: Query DB"}
	if len(resp.VisualSequence) != len(wantTitles) {
		t.Fatalf("steps = %d, want %d: %+v", len(resp.VisualSequence), len(wantTitles), resp.VisualSequence)
	}
	for i, want := range wantTitles {
		s := resp.VisualSequence[i]
		if s.Title != want {
			t.Errorf("step %d title = %q, want %q", i+1, s.Title, want)
		}
		if s.StepNumber != i+1 {
			t.Errorf("step %d number = %d", i+1, s.StepNumber)
		}
		if s.FilePath != "auth" {
			t.Errorf("step %d filePath = %q, want auth", i+1, s.FilePath)
		}
		if s.CodeSnippet == "" || s.Description == "" {
			t.Errorf("step %d missing snippet/description", i+1)
		}
	}
	// Doc comments become the descriptions.
	if !strings.Contains(resp.VisualSequence[0].Description, "login attempt end to end") {
		t.Errorf("step 1 description = %q, want doc comment", resp.VisualSequence[0].Description)
	}
	if !strings.Contains(resp.BeginnerExplanation, "HandleLogin") {
		t.Errorf("explanation missing entry: %q", resp.BeginnerExplanation)
	}
}

func TestSynthesizeWithXRayTraceNeedsCode(t *testing.T) {
	_, err := SynthesizeWithXRay(testCtx(), XRayRequest{Query: "trace the auth flow"})
	if err == nil {
		t.Error("trace without ContextCode: want error")
	}
}

func TestSynthesizeWithXRayErrors(t *testing.T) {
	if _, err := SynthesizeWithXRay(testCtx(), XRayRequest{Query: "   "}); err == nil {
		t.Error("empty query: want error")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := SynthesizeWithXRay(ctx, XRayRequest{Query: "create an http server"}); err == nil {
		t.Error("cancelled context: want error")
	}
	if _, err := SynthesizeWithXRay(testCtx(), XRayRequest{Query: "bake a cake"}); err == nil {
		t.Error("unknown build request: want error")
	}
}

func TestTraceExecutionPath(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "auth.go", authCode, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	steps, err := TraceExecutionPath(file, "HandleLogin")
	if err != nil {
		t.Fatalf("TraceExecutionPath: %v", err)
	}
	if len(steps) != 3 {
		t.Fatalf("steps = %d, want 3", len(steps))
	}
	if steps[0].Title != "Step 1: Handle login" {
		t.Errorf("title = %q", steps[0].Title)
	}
	if steps[0].FilePath != "snippet.go" {
		t.Errorf("filePath = %q", steps[0].FilePath)
	}
	if steps[1].Subtitle != "function" {
		t.Errorf("subtitle = %q", steps[1].Subtitle)
	}
	// Case-insensitive entry resolution.
	if _, err := TraceExecutionPath(file, "handlelogin"); err != nil {
		t.Errorf("case-insensitive entry: %v", err)
	}
}

func TestTraceExecutionPathUnknownEntry(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "auth.go", authCode, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, err = TraceExecutionPath(file, "Nope")
	if err == nil {
		t.Fatal("unknown entry: want error")
	}
	if !strings.Contains(err.Error(), "HandleLogin") {
		t.Errorf("error should list candidates: %v", err)
	}
}

func TestTraceExecutionPathCycle(t *testing.T) {
	src := `package main

func ping() { pong() }
func pong() { ping() }
func main() { ping() }
`
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "cyc.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	steps, err := TraceExecutionPath(file, "main")
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 3 {
		t.Errorf("cycle: steps = %d, want 3 (main, ping, pong)", len(steps))
	}
}

func TestGetConceptAnalogy(t *testing.T) {
	cases := map[string]string{
		"interface": "A wall outlet spec: defines what plug shape is required without caring how the electricity is generated.",
		"struct":    "A concrete physical appliance: holds data and implements the plug.",
		"goroutine": "An independent worker at a factory line: lightweight and managed by Go's runtime scheduler.",
		"channel":   "A conveyor belt: moves data safely between workers without needing explicit locks.",
		"Channel":   "A conveyor belt: moves data safely between workers without needing explicit locks.",
		" CHANNEL ": "A conveyor belt: moves data safely between workers without needing explicit locks.",
	}
	for in, want := range cases {
		if got := GetConceptAnalogy(in); got != want {
			t.Errorf("GetConceptAnalogy(%q) = %q, want %q", in, got, want)
		}
	}
	if got := GetConceptAnalogy("quantum"); got == "" {
		t.Error("unknown primitive: want fallback, got empty")
	}
}

func TestHumanize(t *testing.T) {
	cases := map[string]string{
		"parseJSONBody": "Parse JSON body",
		"ValidateJWT":   "Validate JWT",
		"HandleLogin":   "Handle login",
		"main":          "Main",
		"readLines":     "Read lines",
		"QueryDB":       "Query DB",
	}
	for in, want := range cases {
		if got := humanize(in); got != want {
			t.Errorf("humanize(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestEngineMethodsMatchPackageFuncs(t *testing.T) {
	eng := NewEngine()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "a.go", "package a\nfunc A() {}\n", 0)
	if err != nil {
		t.Fatal(err)
	}
	var _ *ast.File = file
	if _, err := eng.TraceExecutionPath(file, "A"); err != nil {
		t.Errorf("method TraceExecutionPath: %v", err)
	}
	if eng.GetConceptAnalogy("struct") != GetConceptAnalogy("struct") {
		t.Error("method GetConceptAnalogy differs from package func")
	}
}
