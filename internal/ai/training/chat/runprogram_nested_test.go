package chat

import (
	"strings"
	"testing"
)

// The nested-module run failure (John's live failure,
// 2026-11-10: "run the server jim" where jim is a wasm app —
// its own module). The compiler's wall — the main module
// "does not contain package …" — used to be narrated with
// the ordinary build-failure tail, "Fix those, then say 'run
// the program …' again", which sends the user back to an ask
// that can never work. These tests pin the honest narration
// on a reproduced failure, and pin the ordinary narration
// byte-for-byte beside it.

const nestedInnerMain = `package main

import "fmt"

func main() {
	fmt.Println(greeting())
}
`

func TestRunServerNestedModuleNarration(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "inner/go.mod", "module example.com/inner\n\ngo 1.21\n")
	writeFile(t, dir, "inner/main.go", nestedInnerMain)
	writeFile(t, dir, "inner/help.go", "package main\n\nfunc greeting() string {\n\treturn \"hello from inner\"\n}\n")
	withRunSession(t, dir)
	yes := func(string) (string, error) { return "y", nil }

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the server inner", yes, &UserProfile{}) })
	if !handled {
		t.Fatalf("\"run the server inner\" was not claimed:\n%s", out)
	}
	for _, want := range []string{
		"is its own Go module",
		"inner/go.mod",
		"walled off from this project",
		"cd inner && go run .",
		"does not contain package", // the compiler's own words, kept
	} {
		if !strings.Contains(out, want) {
			t.Errorf("nested-module narration missing %q:\n%s", want, out)
		}
	}
	for _, banned := range []string{"Fix those", "It ran fine", "is running in the background"} {
		if strings.Contains(out, banned) {
			t.Errorf("nested-module narration contains %q — the stranded-user shape:\n%s", banned, out)
		}
	}
	if sessionProgram != nil {
		t.Errorf("a nested-module failure left a program in the session")
	}
}

// The ordinary compile failure keeps its narration exactly:
// the same tail, word for word, that it has always had.
func TestRunProgramOrdinaryFailureNarrationUnchanged(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "bad/main.go", "package main\n\nfunc main() {\n\tnope()\n}\n")
	withRunSession(t, dir)
	yes := func(string) (string, error) { return "y", nil }

	out := captureStdout(t, func() { runProgramClaim("bad/main.go", yes, &UserProfile{}) })
	if !strings.Contains(out, "undefined: nope") {
		t.Errorf("the compiler's words were lost:\n%s", out)
	}
	wantTail := `Fix those, then say "run the program bad/main.go" again.`
	if !strings.HasSuffix(strings.TrimRight(out, "\n"), wantTail) {
		t.Errorf("ordinary failure narration changed; want it to end with %q:\n%s", wantTail, out)
	}
	if strings.Contains(out, "its own Go module") {
		t.Errorf("an ordinary failure was narrated as the nested-module wall:\n%s", out)
	}
}

// The detection itself: two witnesses — the compiler's
// words and a go.mod really in an ancestor folder — or it
// isn't the wall.
func TestNestedModuleBuildDetection(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "inner/go.mod", "module example.com/inner\n\ngo 1.21\n")
	writeFile(t, dir, "deep/go.mod", "module example.com/deep\n\ngo 1.21\n")
	wall := "main module (example.com/runfixture) does not contain package example.com/runfixture/inner\n"

	if modDir, ok := nestedModuleBuild(dir, "inner/main.go", wall); !ok || modDir != "inner" {
		t.Errorf("nestedModuleBuild(inner) = (%q, %v), want (\"inner\", true)", modDir, ok)
	}
	if modDir, ok := nestedModuleBuild(dir, "deep/sub/main.go", wall); !ok || modDir != "deep" {
		t.Errorf("nestedModuleBuild(deep/sub) = (%q, %v), want (\"deep\", true) — the module root, not the file's folder", modDir, ok)
	}
	if _, ok := nestedModuleBuild(dir, "inner/main.go", "inner/main.go:2:1: undefined: nope\n"); ok {
		t.Errorf("an ordinary compile error was read as the wall")
	}
	if _, ok := nestedModuleBuild(dir, "plain/main.go", wall); ok {
		t.Errorf("the wall was claimed with no go.mod in any ancestor folder")
	}
	if _, ok := nestedModuleBuild(dir, "main.go", wall); ok {
		t.Errorf("a root program was read as nested")
	}
}
