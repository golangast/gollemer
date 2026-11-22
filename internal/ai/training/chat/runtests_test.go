package chat

// The test seat's pins: the grammar's claims and declines
// (questions above all), the parser against golden `go test`
// output of every shape, the report's exact wording, and
// one live fixture run through the claim chain — a real
// tiny module whose tests really run, never the gollemer
// repo itself.

import (
	"fmt"
	"strings"
	"testing"
)

func TestParseRunTestsClaims(t *testing.T) {
	cases := []struct {
		line string
		want string
	}{
		{"run the tests", ""},
		{"run tests", ""},
		{"test the project", ""},
		{"run the tests.", ""},
		{"can you run the tests", ""},
		{"please run the tests in api", "api"},
		{"run the tests in api", "api"},
		{"run tests in internal/ai", "internal/ai"},
		{"run the tests for api", "api"},
		{"run the tests in ./api/", "api"},
	}
	for _, tc := range cases {
		got, ok := parseRunTests(tc.line)
		if !ok {
			t.Errorf("%q did not claim", tc.line)
			continue
		}
		// The subtree target comes back as typed (minus a
		// trailing ./ or / the claim cleans); compare the
		// cleaned form the claim will resolve.
		want := tc.want
		if got != want && strings.TrimSuffix(strings.TrimPrefix(got, "./"), "/") != want {
			t.Errorf("%q target = %q, want %q", tc.line, got, want)
		}
	}
}

func TestParseRunTestsDeclines(t *testing.T) {
	for _, line := range []string{
		// Questions about tests are not this claim.
		"do the tests pass?",
		"did the tests pass",
		"are the tests passing",
		"why did the test fail",
		"which test failed",
		// Neighbours' sentences.
		"run the program in main.go", // run-the-program's
		"run make test",              // the makefile brain's
		"run it",                     // the beginner five's
		"run the server jim",         // run-the-server's
		// Near misses.
		"test the waters",
		"test the project again",
		"run the tests in", // no folder named
		"run the tests for",
		"run all the tests",
		"testing the project",
		"the tests pass",
	} {
		if got, ok := parseRunTests(line); ok {
			t.Errorf("%q claimed with target %q — it must not", line, got)
		}
	}
}

// --- the parser, against golden output -----------------------------

const goTestAllPassOut = `ok  	example.com/fix/calc	0.012s
?   	example.com/fix/cmd	[no test files]
ok  	example.com/fix/api	0.008s
`

const goTestFailureOut = `ok  	example.com/fix/calc	0.010s
--- FAIL: TestAdd (0.00s)
    api_test.go:12: expected 3, got 4
FAIL
FAIL	example.com/fix/api	0.006s
FAIL
`

const goTestBuildFailOut = `# example.com/fix/calc
./calc/calc.go:5:2: undefined: Addd
./calc/calc.go:9:1: missing return
FAIL	example.com/fix/calc [build failed]
FAIL
`

const goTestPanicOut = `--- FAIL: TestBoom (0.00s)
panic: runtime error: index out of range [3] with length 2 [recovered]
	panic: runtime error: index out of range [3] with length 2

goroutine 6 [running]:
testing.tRunner.func1.2({0x5a8a40, 0x7b8b60})
	/usr/local/go/src/testing/testing.go:1234 +0x123
panic({0x5a8a40, 0x7b8b60})
	/usr/local/go/src/runtime/panic.go:123 +0x45
example.com/fix/api.Boom(...)
	/home/u/fix/api/api.go:20 +0x38
example.com/fix/api.TestBoom(0x7b8b60)
	/home/u/fix/api/api_test.go:10 +0x56
exit status 2
FAIL	example.com/fix/api	0.006s
FAIL
`

func TestParseGoTestOutputAllPass(t *testing.T) {
	rep := parseGoTestOutput(goTestAllPassOut, "example.com/fix", "/home/u/fix")
	if rep.NoResults {
		t.Fatal("parsed as no-results")
	}
	if got := rep.RanPackages(); got != 2 {
		t.Errorf("ran packages = %d, want 2", got)
	}
	if got := rep.FailedPackages(); got != 0 {
		t.Errorf("failed packages = %d, want 0", got)
	}
	out := renderGoTestReport(rep, "")
	if !strings.Contains(out, "All the tests pass — 2 packages ran their tests and every one is green.") {
		t.Errorf("green line wrong:\n%s", out)
	}
	if !strings.Contains(out, "1 other package has no test files") {
		t.Errorf("no-test-files note missing:\n%s", out)
	}
}

func TestParseGoTestOutputFailure(t *testing.T) {
	rep := parseGoTestOutput(goTestFailureOut, "example.com/fix", "/home/u/fix")
	if rep.FailedPackages() != 1 || rep.RanPackages() != 2 {
		t.Fatalf("failed/ran = %d/%d, want 1/2", rep.FailedPackages(), rep.RanPackages())
	}
	pkg := rep.Packages[1]
	if pkg.Path != "example.com/fix/api" || pkg.Dir != "api" || len(pkg.Failures) != 1 {
		t.Fatalf("package = %+v", pkg)
	}
	f := pkg.Failures[0]
	if f.Test != "TestAdd" || f.File != "api_test.go" || f.Line != 12 || f.Message != "expected 3, got 4" {
		t.Errorf("failure = %+v", f)
	}
}

// runTestsFixture is the tiny module the render and live
// tests share: calc (whose Add carries a doc comment the
// context clause can quote) and api.
func runTestsFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/fix\n\ngo 1.21\n")
	writeFile(t, dir, "calc/calc.go", "package calc\n\n// Add returns the sum of two integers.\nfunc Add(a, b int) int { return a + b }\n")
	writeFile(t, dir, "calc/calc_test.go", "package calc\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {\n\tif Add(1, 2) != 3 {\n\t\tt.Fatal(\"bad sum\")\n\t}\n}\n")
	writeFile(t, dir, "api/api.go", "package api\n\nfunc Boom() int {\n\txs := []int{1, 2}\n\treturn xs[3]\n}\n")
	return dir
}

func TestRenderGoTestFailureWithContextClause(t *testing.T) {
	dir := runTestsFixture(t)
	withChatSession(t, dir, true)
	rep := parseGoTestOutput(goTestFailureOut, "example.com/fix", dir)
	out := renderGoTestReport(rep, "")
	for _, want := range []string{
		"The tests ran — 1 of 2 packages failed.",
		"example.com/fix/api:",
		"TestAdd failed at api/api_test.go:12: expected 3, got 4. (Add returns the sum of two integers.)",
		"Start with TestAdd in api/api_test.go — it's the first failure.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report missing %q:\n%s", want, out)
		}
	}
}

func TestParseGoTestOutputBuildFailure(t *testing.T) {
	rep := parseGoTestOutput(goTestBuildFailOut, "example.com/fix", "/home/u/fix")
	if rep.FailedPackages() != 1 {
		t.Fatalf("failed = %d, want 1", rep.FailedPackages())
	}
	pkg := rep.Packages[0]
	if pkg.Status != "buildfail" || len(pkg.BuildErrs) != 2 {
		t.Fatalf("package = %+v", pkg)
	}
	be := pkg.BuildErrs[0]
	if be.File != "calc/calc.go" || be.Line != 5 || be.Col != 2 || be.Message != "undefined: Addd" {
		t.Errorf("build error = %+v", be)
	}
	out := renderGoTestReport(rep, "")
	for _, want := range []string{
		"example.com/fix/calc doesn't compile:",
		"BUILD calc/calc.go:5:2: undefined: Addd — in plain words: that name doesn't exist in this package",
		"BUILD calc/calc.go:9:1: missing return — in plain words: the function promises a result",
		"Start with calc/calc.go:5 — nothing runs until the build passes.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report missing %q:\n%s", want, out)
		}
	}
}

func TestParseGoTestOutputPanic(t *testing.T) {
	rep := parseGoTestOutput(goTestPanicOut, "example.com/fix", "/home/u/fix")
	if rep.FailedPackages() != 1 {
		t.Fatalf("failed = %d, want 1", rep.FailedPackages())
	}
	pkg := rep.Packages[0]
	if pkg.Panic == nil {
		t.Fatalf("no panic parsed: %+v", pkg)
	}
	if pkg.Panic.Message != "runtime error: index out of range [3] with length 2" {
		t.Errorf("panic message = %q", pkg.Panic.Message)
	}
	// The first PROJECT frame wins over the testing/runtime
	// frames above it.
	if pkg.Panic.Func != "Boom" || pkg.Panic.File != "api/api.go" || pkg.Panic.Line != 20 {
		t.Errorf("panic frame = %+v, want Boom at api/api.go:20", pkg.Panic)
	}
	out := renderGoTestReport(rep, "")
	for _, want := range []string{
		"TestBoom panicked: runtime error: index out of range [3] with length 2.",
		"The first frame in your code is Boom, at api/api.go:20.",
		"Start with Boom in api/api.go — that's where the panic surfaced.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "TestBoom failed.") || strings.Contains(out, "TestBoom failed at") {
		t.Errorf("the panicking test was also rendered as a plain failure:\n%s", out)
	}
}

func TestRenderGoTestTimeoutPartial(t *testing.T) {
	dir := runTestsFixture(t)
	withChatSession(t, dir, true)
	rep := parseGoTestOutput(goTestFailureOut, "example.com/fix", dir)
	rep.TimedOut = true
	out := renderGoTestReport(rep, "")
	if !strings.Contains(out, "The tests didn't finish — I stopped them after 4m0s") {
		t.Errorf("timeout headline wrong:\n%s", out)
	}
	if !strings.Contains(out, "Here's what had already failed by then:") || !strings.Contains(out, "TestAdd failed at api/api_test.go:12") {
		t.Errorf("partial results missing:\n%s", out)
	}
	rep2 := parseGoTestOutput(goTestAllPassOut, "example.com/fix", dir)
	rep2.TimedOut = true
	out2 := renderGoTestReport(rep2, "")
	if !strings.Contains(out2, "Nothing had reported a failure yet.") {
		t.Errorf("timeout-with-nothing headline wrong:\n%s", out2)
	}
}

func TestRenderGoTestNoResults(t *testing.T) {
	rep := parseGoTestOutput("go: cannot find main module, but found .git/config in /home/u/fix\n", "example.com/fix", "/home/u/fix")
	if !rep.NoResults {
		t.Fatal("expected NoResults")
	}
	out := renderGoTestReport(rep, "")
	if !strings.Contains(out, "I couldn't run the tests: go: cannot find main module") {
		t.Errorf("no-results line wrong:\n%s", out)
	}
}

func TestGoTestBuildGloss(t *testing.T) {
	cases := map[string]string{
		"undefined: Addd":                       "doesn't exist in this package",
		"\"fmt\" imported and not used":         "imports a package it never uses",
		"x declared and not used":               "set but never used",
		"cannot use \"a\" (type string) as int": "different type is needed",
		"missing return":                        "reaches the end without returning",
		"some wholly unknown diagnostic":        "",
	}
	for msg, want := range cases {
		got := goTestBuildGloss(msg)
		if want == "" {
			if got != "" {
				t.Errorf("gloss(%q) = %q, want none", msg, got)
			}
			continue
		}
		if !strings.Contains(got, want) {
			t.Errorf("gloss(%q) = %q, want it to mention %q", msg, got, want)
		}
	}
}

// --- the claim, live -----------------------------------------------

// echoPromptYes is the scripted gate answer that echoes
// its prompt the way the real LineReader does, so the
// gate's wording is visible to assert on (runprogram's
// pattern).
func echoPromptYes(prompt string) (string, error) {
	fmt.Print(prompt)
	return "y", nil
}

// One real run through the chain: the fixture's calc test
// passes and its api test fails, and the report says both.
func TestRunTestsLiveFixtureRun(t *testing.T) {
	dir := runTestsFixture(t)
	writeFile(t, dir, "api/api_test.go", "package api\n\nimport \"testing\"\n\nfunc TestBoom(t *testing.T) {\n\tif Boom() != 0 {\n\t\tt.Fatal(\"never reached\")\n\t}\n}\n")
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("run the tests", echoPromptYes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through the whole claim chain:\n%s", out)
	}
	if !strings.Contains(out, "Run it? go test ./... (the whole project)") {
		t.Errorf("the run gate did not show the command:\n%s", out)
	}
	if !strings.Contains(out, "The tests ran — 1 of 2 packages failed.") {
		t.Errorf("the failure headline missing:\n%s", out)
	}
	if !strings.Contains(out, "panicked: runtime error: index out of range") {
		t.Errorf("the panic was not explained:\n%s", out)
	}
}

// Questions keep their fate: the claim itself declines
// them, so the brains that own them still get them.
func TestRunTestsQuestionsFallThrough(t *testing.T) {
	dir := runTestsFixture(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	for _, line := range []string{"do the tests pass?", "why did the test fail", "did the tests pass"} {
		if runRunTests(line, yes, profile) {
			t.Errorf("%q was claimed by the test seat", line)
		}
	}
}

// The gate is real: a no runs nothing and says so.
func TestRunTestsGateDecline(t *testing.T) {
	dir := runTestsFixture(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("run the tests", no, profile)
	})
	if !handled {
		t.Fatalf("the line fell through:\n%s", out)
	}
	if !strings.Contains(out, "OK — not run.") {
		t.Errorf("the decline was not the answer:\n%s", out)
	}
	if strings.Contains(out, "The tests ran") {
		t.Errorf("tests ran despite the no:\n%s", out)
	}
}

// A subtree the project doesn't have is refused by name,
// with the folders that do exist offered instead.
func TestRunTestsUnknownFolder(t *testing.T) {
	dir := runTestsFixture(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("run the tests in nosuch", yes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through:\n%s", out)
	}
	if !strings.Contains(out, "There's no folder called nosuch in this project.") {
		t.Errorf("the refusal missing:\n%s", out)
	}
	if !strings.Contains(out, "api") || !strings.Contains(out, "calc") {
		t.Errorf("the real folders were not named:\n%s", out)
	}
}

// A subtree run scopes the pattern and the report.
func TestRunTestsSubtreeRun(t *testing.T) {
	dir := runTestsFixture(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("run the tests in calc", echoPromptYes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through:\n%s", out)
	}
	if !strings.Contains(out, "Run it? go test ./calc/... (the tests in calc/)") {
		t.Errorf("the subtree gate text wrong:\n%s", out)
	}
	if !strings.Contains(out, "All the tests in calc/ pass — 1 package ran their tests and every one is green.") {
		t.Errorf("the subtree green line wrong:\n%s", out)
	}
}
