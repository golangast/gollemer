package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The make-and-call claim's chat face: the joined sentence
// is claimed beside the uses, the callee's file resolves
// under the declaration rules (creatable in an existing
// folder), the caller's file must exist, and the shared
// spine lands both halves behind one gate — or neither.

func makeCallChatFixture(t *testing.T, extra map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"go.mod": "module example.com/mcchat\n\ngo 1.21\n",
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
		writeFile(t, dir, rel, src)
	}
	return dir
}

func TestParseMakeCallForms(t *testing.T) {
	claims := []struct {
		line, name, callee, target, caller, host string
	}{
		{"make a function named Validate in api/validate.go and call it in api/main.go", "Validate", "api/validate.go", "api/main.go", "", ""},
		{"make a function Validate in api/validate.go and call it in api/main.go", "Validate", "api/validate.go", "api/main.go", "", ""},
		{"add a function named Check in lib/checks.go and call it from Serve in web/web.go", "Check", "lib/checks.go", "web/web.go", "", "Serve"},
		{"create a function named Validate in api/validate.go and add a function named Report in api/report.go that calls it", "Validate", "api/validate.go", "api/report.go", "Report", ""},
		{"put a function named Validate in api/validate.go and make a function Report in api/report.go that calls it", "Validate", "api/validate.go", "api/report.go", "Report", ""},
		// "then" joins the halves as "and" does (2026-11-05).
		{"make a function named Foo in jim/a.go then call it in jim/b.go", "Foo", "jim/a.go", "jim/b.go", "", ""},
		{"add a function named Foo in jim/a.go then call it from main in jim/b.go", "Foo", "jim/a.go", "jim/b.go", "", "main"},
		{"make a function named Foo in jim/a.go then add a function named Bar in jim/b.go that calls it", "Foo", "jim/a.go", "jim/b.go", "Bar", ""},
	}
	for _, c := range claims {
		name, callee, target, caller, host, ok := parseMakeCall(c.line)
		if !ok {
			t.Errorf("parseMakeCall(%q) didn't claim", c.line)
			continue
		}
		if name != c.name || callee != c.callee || target != c.target || caller != c.caller || host != c.host {
			t.Errorf("parseMakeCall(%q) = (%q, %q, %q, %q, %q), want (%q, %q, %q, %q, %q)",
				c.line, name, callee, target, caller, host, c.name, c.callee, c.target, c.caller, c.host)
		}
	}
	notMine := []string{
		"add a function named Validate to api/validate.go",                                                   // declaration alone
		"add a function named Report in api/report.go that uses Validate from api/validate.go",               // uses line alone
		"make a function named Validate in api/validate.go",                                                  // half a sentence
		"make a function named Validate in api/validate.go and call it in api/validate.go",                   // same file twice
		"make a function named Validate in api/validate.go and call it in ./api/validate.go",                 // same file, second spelling
		"make a file called api/validate.go and add a function named Validate to it",                         // created-target compound
		"make a function named Validate in api/validate.go and call Validate in api/main.go",                 // no "it"
		"make a function named Validate in api/validate.go and call it on api/main.go",                       // wrong preposition
		"add a function named Report in api/report.go that calls Validate from api/validate.go",              // the delegate's shape
		"make a function named Validate in api/validate.go and add a function named Report in api/report.go", // no "that calls it"
		"what does the function Validate do",                                                                 // a question
	}
	for _, line := range notMine {
		if _, _, _, _, _, ok := parseMakeCall(line); ok {
			t.Errorf("parseMakeCall(%q) claimed a line that isn't its shape", line)
		}
	}
}

// The headline through the whole claim chain: both files
// land, the project builds (the spine checks), and the
// caller really calls the callee.
func TestMakeCallNamedCallerEndToEnd(t *testing.T) {
	dir := makeCallChatFixture(t, nil)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !claimChain("make a function named Validate in api/validate.go and add a function named Report in api/report.go that calls it", yes, nil) {
			t.Error("the make-and-call line wasn't claimed")
		}
	})
	// The plan is shown before the gate (the prompt itself
	// goes to the ask function, not stdout): the summary and
	// BOTH files' diffs.
	for _, want := range []string{"Make Validate in", "call it in", "--- a/api/validate.go", "--- a/api/report.go", "One plan covers both files"} {
		if !strings.Contains(out, want) {
			t.Errorf("plan output missing %q:\n%s", want, out)
		}
	}
	callee := readBack(t, dir, "api/validate.go")
	if !strings.Contains(callee, "func Validate() {") {
		t.Errorf("callee didn't land:\n%s", callee)
	}
	caller := readBack(t, dir, "api/report.go")
	if !strings.Contains(caller, "func Report() {") || !strings.Contains(caller, "Validate()") {
		t.Errorf("caller didn't land calling the callee:\n%s", caller)
	}
}

// The bare form, hosted by the target's main.
func TestMakeCallBareEndToEnd(t *testing.T) {
	dir := makeCallChatFixture(t, nil)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !claimChain("make a function named Validate in api/validate.go and call it in api/main.go", yes, nil) {
			t.Error("the bare make-and-call line wasn't claimed")
		}
	})
	main := readBack(t, dir, "api/main.go")
	if !strings.Contains(main, "fmt.Println(\"api\")\n\tValidate()\n}") {
		t.Errorf("call not at the end of main:\n%s", main)
	}
	if _, err := os.Stat(filepath.Join(dir, "api", "validate.go")); err != nil {
		t.Errorf("callee file missing: %v (output:\n%s)", err, out)
	}
}

// Cross-package end to end: the caller imports and
// qualifies the callee's package.
func TestMakeCallCrossPackageEndToEnd(t *testing.T) {
	dir := makeCallChatFixture(t, nil)
	withServerProject(t, dir)
	captureStdout(t, func() {
		if !claimChain("make a function named Check in lib/checks.go and add a function named ServeCheck in web/web.go that calls it", yes, nil) {
			t.Error("the cross-package line wasn't claimed")
		}
	})
	caller := readBack(t, dir, "web/web.go")
	for _, want := range []string{`"example.com/mcchat/lib"`, "lib.Check()"} {
		if !strings.Contains(caller, want) {
			t.Errorf("cross-package caller missing %q:\n%s", want, caller)
		}
	}
}

// A "no" at the one gate writes NEITHER file.
func TestMakeCallDeclinedWritesNothing(t *testing.T) {
	dir := makeCallChatFixture(t, nil)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !claimChain("make a function named Validate in api/validate.go and add a function named Report in api/report.go that calls it", no, nil) {
			t.Error("the make-and-call line wasn't claimed")
		}
	})
	if !strings.Contains(out, "Not applied — every file is exactly as it was") {
		t.Errorf("the no wasn't honoured in the output:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "api", "validate.go")); !os.IsNotExist(err) {
		t.Errorf("callee file exists after a declined gate (err=%v)", err)
	}
	if got := readBack(t, dir, "api/report.go"); got != "package main\n" {
		t.Errorf("caller file changed after a declined gate:\n%s", got)
	}
}

// A callee whose name already exists: the declaration's own
// refusal answers, and nothing is written.
func TestMakeCallNameExistsRefuses(t *testing.T) {
	dir := makeCallChatFixture(t, map[string]string{
		"api/validate.go": "package main\n\nfunc Validate() {}\n",
	})
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !claimChain("make a function named Validate in api/other.go and add a function named Report in api/report.go that calls it", yes, nil) {
			t.Error("the make-and-call line wasn't claimed")
		}
	})
	if !strings.Contains(out, "Validate already exists in package main") {
		t.Errorf("the declaration's refusal didn't surface:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "api", "other.go")); !os.IsNotExist(err) {
		t.Errorf("a file was created despite the refusal (err=%v)", err)
	}
}

// The name-exists pin from the two-step world: a uses line
// for the same pair still plans exactly as before (the new
// claim didn't steal it — parseMakeCall leaves it alone,
// pinned in TestParseMakeCallForms — and the uses route
// still lands it once the declaration exists).
func TestMakeCallTwoStepStillWorks(t *testing.T) {
	dir := makeCallChatFixture(t, nil)
	withServerProject(t, dir)
	captureStdout(t, func() {
		if !claimChain("add a function named Validate to api/validate.go", yes, nil) {
			t.Error("the declaration line wasn't claimed")
		}
		if !claimChain("add a function named Report in api/report.go that uses Validate from api/validate.go", yes, nil) {
			t.Error("the uses line wasn't claimed")
		}
	})
	caller := readBack(t, dir, "api/report.go")
	if !strings.Contains(caller, "func Report() {") || !strings.Contains(caller, "Validate()") {
		t.Errorf("two-step caller didn't land:\n%s", caller)
	}
}
