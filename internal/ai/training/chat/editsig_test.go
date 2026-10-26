package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// Every signature sentence form parses to the request it names.
func TestParseSignatureForms(t *testing.T) {
	cases := []struct {
		line string
		want analyze.EditRequest
	}{
		{"rename the factor parameter of Scale to multiplier",
			analyze.EditRequest{Kind: analyze.EditRenameParam, FuncName: "Scale", SigOld: "factor", SigNew: "multiplier"}},
		{"rename Scale's factor parameter to multiplier",
			analyze.EditRequest{Kind: analyze.EditRenameParam, FuncName: "Scale", SigOld: "factor", SigNew: "multiplier"}},
		{"rename the factor parameter of Scale to multiplier.",
			analyze.EditRequest{Kind: analyze.EditRenameParam, FuncName: "Scale", SigOld: "factor", SigNew: "multiplier"}},
		{"swap the a and b parameters of Combine",
			analyze.EditRequest{Kind: analyze.EditReorderParams, FuncName: "Combine", SigOld: "a", SigNew: "b"}},
		{"swap the name and data parameters of FileSaver.Save",
			analyze.EditRequest{Kind: analyze.EditReorderParams, FuncName: "FileSaver.Save", SigOld: "name", SigNew: "data"}},
		{"move the level parameter of Route first",
			analyze.EditRequest{Kind: analyze.EditReorderParams, FuncName: "Route", SigOld: "level", SigMove: "first"}},
		{"move the mode parameter of Route last",
			analyze.EditRequest{Kind: analyze.EditReorderParams, FuncName: "Route", SigOld: "mode", SigMove: "last"}},
		{"move the strict parameter of Route before mode",
			analyze.EditRequest{Kind: analyze.EditReorderParams, FuncName: "Route", SigOld: "strict", SigMove: "before", SigNew: "mode"}},
		{"move the strict parameter of Route after level",
			analyze.EditRequest{Kind: analyze.EditReorderParams, FuncName: "Route", SigOld: "strict", SigMove: "after", SigNew: "level"}},
		{"add a context parameter to StepOK",
			analyze.EditRequest{Kind: analyze.EditAddParam, FuncName: "StepOK", SigType: "context.Context"}},
		{"add a context parameter named c to StepOK",
			analyze.EditRequest{Kind: analyze.EditAddParam, FuncName: "StepOK", SigType: "context.Context", SigName: "c"}},
		{"add an int parameter named limit to Fetch using DefaultLimit",
			analyze.EditRequest{Kind: analyze.EditAddParam, FuncName: "Fetch", SigType: "int", SigName: "limit", SigUsing: "DefaultLimit"}},
		{"add a time.Duration parameter named wait to WaitFor using time.Second",
			analyze.EditRequest{Kind: analyze.EditAddParam, FuncName: "WaitFor", SigType: "time.Duration", SigName: "wait", SigUsing: "time.Second"}},
		{"add an int parameter named limit to Fetch",
			analyze.EditRequest{Kind: analyze.EditAddParam, FuncName: "Fetch", SigType: "int", SigName: "limit", SigNoUsing: true}},
	}
	for _, c := range cases {
		got, ok := parseAssembledEdit(c.line)
		if !ok {
			t.Errorf("parseAssembledEdit(%q) not claimed", c.line)
			continue
		}
		if got != c.want {
			t.Errorf("parseAssembledEdit(%q) = %+v, want %+v", c.line, got, c.want)
		}
	}
}

// Near-miss sentences keep their old owners: the field form, the
// identity rename, the clone form, and disk-move lines are not
// signature changes.
func TestParseSignatureNearMisses(t *testing.T) {
	if got, ok := parseAssembledEdit("add a Timeout field of type int to Config"); !ok ||
		got.Kind != analyze.EditAddField || got.FieldName != "Timeout" || got.StructName != "Config" {
		t.Errorf("field sentence = %+v, %v — want the EditAddField claim", got, ok)
	}
	if got, ok := parseAssembledEdit("add a function like Greet named GreetLoudly"); !ok ||
		got.Kind != analyze.EditAddFunc || got.NewName != "GreetLoudly" {
		t.Errorf("clone sentence = %+v, %v — want the EditAddFunc claim", got, ok)
	}
	for _, line := range []string{
		"rename Save to Store",               // the identity rename's sentence
		"swap the parameters of Combine",     // no parameter names — not a claim
		"move the jim folder to the archive", // a disk move (boundary answers it)
		"add a parameter to Fetch",           // no type, no name — not a claim
	} {
		if req, ok := parseAssembledEdit(line); ok {
			t.Errorf("parseAssembledEdit(%q) claimed as %+v — that sentence belongs to another brain", line, req)
		}
	}
}

// sigChatFixture builds the small two-package module the flow
// tests below run against.
func sigChatFixture(t *testing.T) string {
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
	write("go.mod", "module example.com/chatsig\n\ngo 1.21\n")
	write("calc/calc.go", `package calc

func Combine(a, b int, label string) int { return a + b }

func FetchRemote(key string) string { return key }
`)
	write("app/app.go", `package app

import "example.com/chatsig/calc"

func UseCombine() int {
	return calc.Combine(1, 2, "x")
}

func UseFetch(key string) string {
	return calc.FetchRemote(key)
}
`)
	return dir
}

// (g) Freshness interplay: a new caller file appears outside the
// chat between analyze and the request — the a2a2f6f re-read
// fires before planning, and the applied plan covers the file
// the stale index never saw.
func TestSignatureEditFreshnessReRead(t *testing.T) {
	dir := sigChatFixture(t)
	withEditsProject(t, dir)
	// The external edit: a brand-new caller, written behind the
	// chat's back after the analyze.
	more := filepath.Join(dir, "app", "more.go")
	if err := os.WriteFile(more, []byte("package app\n\nimport \"example.com/chatsig/calc\"\n\nfunc UseMore() int {\n\treturn calc.Combine(10, 20, \"fresh\")\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out := captureStdout(t, func() {
		if !runAssembledEdit("swap the a and b parameters of Combine", yes) {
			t.Error("signature sentence not claimed")
		}
	})
	if !strings.Contains(out, "You changed files outside the chat since I last read this project — I re-read it before planning.") {
		t.Errorf("the freshness re-read line should print before planning:\n%s", out)
	}
	if got := readBack(t, dir, "app/more.go"); !strings.Contains(got, `calc.Combine(20, 10, "fresh")`) {
		t.Errorf("the new caller file should be in the applied plan:\n%s", got)
	}
	if got := readBack(t, dir, "calc/calc.go"); !strings.Contains(got, "func Combine(b int, a int, label string) int") {
		t.Errorf("declaration not swapped:\n%s", got)
	}
}

// (h) Transaction interplay: a using-expression of the wrong
// type makes the build fail after the write — every file of the
// set comes back byte- and mode-identical.
func TestSignatureEditBuildFailureRestoresSet(t *testing.T) {
	dir := sigChatFixture(t)
	withEditsProject(t, dir)
	appFile := filepath.Join(dir, "app", "app.go")
	if err := os.Chmod(appFile, 0o640); err != nil {
		t.Fatal(err)
	}
	calcBefore := readBack(t, dir, "calc/calc.go")
	appBefore := readBack(t, dir, "app/app.go")
	out := captureStdout(t, func() {
		if !runAssembledEdit(`add an int parameter named limit to FetchRemote using "text"`, yes) {
			t.Error("signature sentence not claimed")
		}
	})
	if !strings.Contains(out, "restored") {
		t.Errorf("a failed build should report the restore:\n%s", out)
	}
	if got := readBack(t, dir, "calc/calc.go"); got != calcBefore {
		t.Errorf("calc/calc.go not restored byte-identically:\n%s", got)
	}
	if got := readBack(t, dir, "app/app.go"); got != appBefore {
		t.Errorf("app/app.go not restored byte-identically:\n%s", got)
	}
	fi, err := os.Stat(appFile)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode().Perm() != 0o640 {
		t.Errorf("app/app.go mode = %v, want 0640 preserved through the rollback", fi.Mode().Perm())
	}
	if _, err := os.Stat(filepath.Join(dir, ".gollemer-transaction.json")); !os.IsNotExist(err) {
		t.Errorf("transaction manifest left behind after rollback: %v", err)
	}
}
