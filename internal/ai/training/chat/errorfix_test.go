package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// errorfixChatFixture builds a project with three independent
// compile errors in three files — one per shipped fix class that
// the flow tests exercise end to end.
func errorfixChatFixture(t *testing.T) string {
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
	write("go.mod", "module example.com/chaterr\n\ngo 1.21\n")
	write("greet.go", `package main

func Greet() string { return "hi" }

func main() { println(Greet()) }
`)
	write("broken_import.go", `package main

import "os"

func Nothing() {}
`)
	write("broken_var.go", `package main

func Unused() int {
	x := 42
	return 1
}
`)
	write("broken_spell.go", `package main

func Spell() string {
	return Greeet()
}
`)
	return dir
}

// Compiler lines classify into fix classes; cut classes (type
// mismatches, argument counts) and non-errors parse to nothing.
func TestParseCompilerError(t *testing.T) {
	cases := []struct {
		line      string
		wantClass string
		wantName  string
		wantLine  int
		wantCol   int
	}{
		{`./broken_import.go:3:1: imported and not used: "os"`, analyze.ErrorFixUnusedImport, "os", 3, 1},
		{`broken_import.go:3:1: "os" imported and not used`, analyze.ErrorFixUnusedImport, "os", 3, 1},
		{`./broken_var.go:4:2: declared and not used: x`, analyze.ErrorFixUnusedVar, "x", 4, 2},
		{`./broken_var.go:4:2: x declared and not used`, analyze.ErrorFixUnusedVar, "x", 4, 2},
		{`./broken_spell.go:4:9: undefined: Greeet`, analyze.ErrorFixUndefinedName, "Greeet", 4, 9},
		{`./main.go:12:1: missing return at end of function`, analyze.ErrorFixMissingReturn, "", 12, 1},
		{`./main.go:12: missing return at end of function`, analyze.ErrorFixMissingReturn, "", 12, 0},
	}
	for _, c := range cases {
		req, _, ok := parseCompilerError(c.line)
		if !ok {
			t.Errorf("parseCompilerError(%q) not claimed", c.line)
			continue
		}
		if req.Class != c.wantClass || req.Name != c.wantName || req.Line != c.wantLine || req.Col != c.wantCol {
			t.Errorf("parseCompilerError(%q) = %+v, want class %s name %q at %d:%d", c.line, req, c.wantClass, c.wantName, c.wantLine, c.wantCol)
		}
	}
	// Cut classes and non-errors never parse into a fix request.
	// (A dotted `undefined: fmt.Println` DOES classify — the
	// planner refuses it later; the flow test covers that.)
	for _, line := range []string{
		`./greet.go:3:1: too many arguments in call to Greet: have (int) want ()`,
		`./greet.go:3:1: cannot use "x" (value of type string) as int value in argument to Greet`,
		`./greet.go:3:1: mismatched types string and int`,
		"what does Greet do",
	} {
		if req, _, ok := parseCompilerError(line); ok {
			t.Errorf("parseCompilerError(%q) claimed %+v — cut classes and non-errors must not parse", line, req)
		}
	}
}

// The three-error project, fixed one paste at a time: each fix is
// kept (the baseline comparison knows the other errors were already
// there), and the last one leaves the project building clean.
func TestErrorFixOneAtATime(t *testing.T) {
	dir := errorfixChatFixture(t)
	withEditsProject(t, dir)

	if !runErrorFix(`./broken_import.go:3:1: imported and not used: "os"`, yes) {
		t.Fatal("import fix not offered")
	}
	if got := readBack(t, dir, "broken_import.go"); strings.Contains(got, "\"os\"") {
		t.Fatalf("unused import survived the fix:\n%s", got)
	}
	// The other two errors are untouched by the first fix.
	if got := readBack(t, dir, "broken_var.go"); !strings.Contains(got, "x := 42") {
		t.Fatal("the first fix touched another error's file")
	}
	if !runErrorFix(`./broken_var.go:4:2: declared and not used: x`, yes) {
		t.Fatal("unused-var fix not offered")
	}
	if got := readBack(t, dir, "broken_var.go"); strings.Contains(got, "x := 42") {
		t.Fatalf("unused declaration survived the fix:\n%s", got)
	}
	if !runErrorFix(`./broken_spell.go:4:9: undefined: Greeet`, yes) {
		t.Fatal("spelling fix not offered")
	}
	if got := readBack(t, dir, "broken_spell.go"); !strings.Contains(got, "return Greet()") {
		t.Fatalf("spelling not fixed:\n%s", got)
	}
	// And now the whole project really does build.
	if out, ok := buildProject(dir); !ok {
		t.Fatalf("project should build clean after all three fixes:\n%s", out)
	}
}

// Declining a fix leaves the file byte-identical.
func TestErrorFixDeclineLeavesBytes(t *testing.T) {
	dir := errorfixChatFixture(t)
	withEditsProject(t, dir)
	before := readBack(t, dir, "broken_import.go")
	if !runErrorFix(`./broken_import.go:3:1: imported and not used: "os"`, no) {
		t.Fatal("import fix not offered")
	}
	if after := readBack(t, dir, "broken_import.go"); after != before {
		t.Fatal("declined error fix changed the file")
	}
}

// A fix that plans but makes the build worse — the near match has
// the wrong signature — is rolled back byte-identical: the flow
// never leaves a "fixed" claim standing on a red build it caused.
func TestErrorFixThatBreaksBuildRollsBack(t *testing.T) {
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
	write("go.mod", "module example.com/chaterr2\n\ngo 1.21\n")
	write("greet.go", `package main

func Greet() string { return "hi" }

func main() { println(Greet()) }
`)
	write("spell.go", `package main

func Spell() string {
	return Greeet("extra")
}
`)
	withEditsProject(t, dir)
	before := readBack(t, dir, "spell.go")
	if !runErrorFix(`./spell.go:4:9: undefined: Greeet`, yes) {
		t.Fatal("spelling fix not offered (it should be — and then rolled back)")
	}
	if after := readBack(t, dir, "spell.go"); after != before {
		t.Fatalf("a fix that broke the build was not rolled back:\n%s", after)
	}
}

// Cut classes reach no gate: translation only, exactly as before.
func TestErrorFixCutClassesNoOffer(t *testing.T) {
	dir := errorfixChatFixture(t)
	withEditsProject(t, dir)
	asked := false
	flagAsk := func(string) (string, error) { asked = true; return "y", nil }
	for _, line := range []string{
		`./greet.go:3:1: too many arguments in call to Greet: have (int) want ()`,
		`./greet.go:3:1: cannot use 1 (value of type int) as string value in return statement`,
		`./greet.go:3:1: mismatched types string and int`,
		`./broken_spell.go:4:9: undefined: fmt.Println`,
	} {
		if runErrorFix(line, flagAsk) {
			t.Errorf("runErrorFix(%q) offered a fix — cut classes must stay translation-only", line)
		}
	}
	if asked {
		t.Fatal("a cut-class error reached the [y/n] gate")
	}
}
