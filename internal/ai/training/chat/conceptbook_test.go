package chat

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The concept book + composer (conceptbook.go): concepts are
// data, the grammar is composed once. These tests pin the
// claim matrix against the shipped book, the data-only proof
// (a third concept registered purely as a JSON row), the
// end-to-end fills in the live-session states, and the ordering
// against the bespoke shapes and the boundary.

// withConceptEntries swaps the loaded book for a test's, the
// advocates-test pattern (cache + a spent Once, restored after).
func withConceptEntries(t *testing.T, entries []analyze.ConceptEntry) {
	t.Helper()
	oldCache, oldOnce := conceptCache, conceptOnce
	t.Cleanup(func() { conceptCache, conceptOnce = oldCache, oldOnce })
	conceptCache = entries
	conceptOnce = &sync.Once{}
	conceptOnce.Do(func() {})
}

func defaultBookEntries(t *testing.T) []analyze.ConceptEntry {
	t.Helper()
	entries, skipped := analyze.ParseConceptBook([]byte(defaultConceptBook))
	if skipped != 0 || len(entries) != 4 {
		t.Fatalf("default book = %d entries, %d skipped", len(entries), skipped)
	}
	return entries
}

// The snapshot in the repo and the compiled default are the
// same book — checked row for row so they cannot drift.
func TestConceptBookFileMatchesDefault(t *testing.T) {
	blob, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "data", "concepts.json"))
	if err != nil {
		t.Fatalf("data/concepts.json not readable from the chat package: %v", err)
	}
	fromFile, skipped := analyze.ParseConceptBook(blob)
	if skipped != 0 {
		t.Fatalf("data/concepts.json has %d malformed rows", skipped)
	}
	if want := defaultBookEntries(t); !reflect.DeepEqual(fromFile, want) {
		t.Errorf("data/concepts.json rows drifted from the compiled default:\nfile: %+v\ndefault: %+v", fromFile, want)
	}
}

// The claim matrix over the shipped book: articles × entries ×
// slot combinations.
func TestParseConceptClaims(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct {
		line string
		id   string
		name string
		text string
		fn   string
		like string
		tgt  string
	}{
		{"add a print line to jim/main.go", "print-line", "", "", "", "", "jim/main.go"},
		{"add print line to jim/main.go", "print-line", "", "", "", "", "jim/main.go"},
		{"add the print line to the file jim/main.go", "print-line", "", "", "", "", "jim/main.go"},
		{"put a print line in jim/main.go", "print-line", "", "", "", "", "jim/main.go"},
		{"add a println to jim/main.go", "print-line", "", "", "", "", "jim/main.go"},
		{`add a print line saying "hello world" to jim/main.go`, "print-line", "", "hello world", "", "", "jim/main.go"},
		{`add a print line that says "hi" to jim/main.go`, "print-line", "", "hi", "", "", "jim/main.go"},
		{`add a print line saying "say \"hi\" now" in Helper to jim/main.go`, "print-line", "", `say "hi" now`, "Helper", "", "jim/main.go"},
		{"add a print line in Helper to jim/main.go", "print-line", "", "", "Helper", "", "jim/main.go"},
		{"add a print line to folder jim", "print-line", "", "", "", "", "jim"},
		{"add a print line", "print-line", "", "", "", "", ""},
		{"add an interface named Reader to store.go", "interface", "Reader", "", "", "", "store.go"},
		{"add an interface named Reader like Store to store.go", "interface", "Reader", "", "", "Store", "store.go"},
		{"add the interface like Store named Reader to store.go", "interface", "Reader", "", "", "Store", "store.go"},
		{"add an interface to jim/main.go", "interface", "", "", "", "", "jim/main.go"},
		{"add an interface", "interface", "", "", "", "", ""},
		{"add a struct named Config to store.go", "struct", "Config", "", "", "", "store.go"},
		{"add the struct named Config to jim/main.go", "struct", "Config", "", "", "", "jim/main.go"},
		{"put a struct type named Config in store.go", "struct", "Config", "", "", "", "store.go"},
		{"add a struct to jim/main.go", "struct", "", "", "", "", "jim/main.go"},
		{"add a function named Fetch to main.go", "function", "Fetch", "", "", "", "main.go"},
		{"add a func named Fetch to main.go", "function", "Fetch", "", "", "", "main.go"},
		{"add the function named Fetch to the file main.go", "function", "Fetch", "", "", "", "main.go"},
		{"add a function to jim/main.go", "function", "", "", "", "", "jim/main.go"},
	}
	for _, c := range cases {
		req, ok := parseConceptFrom(c.line, book)
		if !ok {
			t.Errorf("parseConcept(%q) not claimed", c.line)
			continue
		}
		if req.entry.ID != c.id || req.name != c.name || req.text != c.text ||
			req.inFunc != c.fn || req.like != c.like || req.target != c.tgt {
			t.Errorf("parseConcept(%q) = {id:%q name:%q text:%q fn:%q like:%q tgt:%q}, want id %q name %q text %q fn %q like %q tgt %q",
				c.line, req.entry.ID, req.name, req.text, req.inFunc, req.like, req.target,
				c.id, c.name, c.text, c.fn, c.like, c.tgt)
		}
	}
}

// Non-claims: the bespoke shapes' sentences, questions, and
// lines whose slot zone carries an unknown word (never silently
// dropped) stay with their owners. Two pins here tightened when
// "function" became a book entry: the delegate's line (its
// behaviour-clause tail is not a path) and the clone's (its
// "like" names a function to copy, and the function row's
// template has no {methods} slot to spend it on) — the composer
// refuses both at parse level, on top of the chain order that
// already serves them first.
func TestParseConceptNonClaims(t *testing.T) {
	book := defaultBookEntries(t)
	for _, line := range []string{
		"add a function in jim/main.go that calls Load", // the delegate's
		"add a function like Greet named GreetLoudly",   // the clone's
		"add a webserver to jim/main.go",                // add-server's
		"add the webserver to jim/main.go",
		"add a Timeout field of type int to Config", // the field form's
		"add a plum case to Describe",               // the case form's
		"add a context parameter to StepOK",         // the signature form's
		"register thanks in Greetings",              // the register form's
		"create a file jim/main.go",                 // creations' verb
		"how do i add a print line to jim/main.go",  // a question
		"add a getter for Timeout",                  // other brains'
		"add a method to jim/main.go",               // unknown concept: the boundary's
		"add a print line quickly to jim/main.go",   // unknown slot word
		"add a print line to",                       // (prep with nothing: not a claim at parse level? see below)
	} {
		if line == "add a print line to" {
			// The target clause opens and names nothing: the
			// claim holds with an empty target (the flow answers
			// with guidance), like the add-server designator.
			if req, ok := parseConceptFrom(line, book); !ok || req.target != "" {
				t.Errorf("parseConcept(%q) = %+v, %v — want claimed with empty target", line, req, ok)
			}
			continue
		}
		if req, ok := parseConceptFrom(line, book); ok {
			t.Errorf("parseConcept(%q) claimed %+v — that line is not the composer's", line, req)
		}
	}
}

// The data-driven proof: a THIRD concept exists only as a JSON
// row in a fixture book. No Go code knows what a "for loop" is;
// the row alone makes its sentence work end to end.
func TestConceptThirdConceptPureData(t *testing.T) {
	entries, skipped := analyze.ParseConceptBook([]byte(`{"concepts": [
		{"id": "print-line", "names": ["print line"], "kind": "statement",
		 "template": "fmt.Println(\"{text}\")", "imports": ["fmt"]},
		{"id": "for-loop", "names": ["for loop", "loop"], "kind": "statement",
		 "template": "for i := 0; i < 3; i++ {\n\t// TODO: what the loop does is yours to write\n}"}
	]}`))
	if skipped != 0 || len(entries) != 2 {
		t.Fatalf("fixture book = %d entries, %d skipped", len(entries), skipped)
	}
	withConceptEntries(t, entries)

	for _, analyzed := range []bool{true, false} {
		root := liveSessionRoot(t)
		withChatSession(t, root, analyzed)
		profile := &UserProfile{}
		createJimThroughTheChain(t, profile)
		line := "add a for loop to jim/main.go"
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled {
			t.Errorf("analyzed=%v: %q fell through the claim chain:\n%s", analyzed, line, out)
			continue
		}
		requireNoAnalyzeOutput(t, line, out)
		got := readBack(t, root, "jim/main.go")
		if !strings.Contains(got, "for i := 0; i < 3; i++ {") || !strings.Contains(got, "func main()") {
			t.Errorf("analyzed=%v: the data-only concept did not assemble into jim/main.go:\n%s\noutput:\n%s", analyzed, got, out)
		}
		if !strings.Contains(out, "from the concept book") {
			t.Errorf("analyzed=%v: answer does not own the concept book:\n%s", analyzed, out)
		}
	}
}

// The user's own line, now served: "add print line to
// jim/main.go" in both session states fills the bare file with a
// main carrying fmt.Println("") and its import — the boundary
// never speaks.
func TestConceptPrintLineLiveSession(t *testing.T) {
	for _, analyzed := range []bool{true, false} {
		root := liveSessionRoot(t)
		withChatSession(t, root, analyzed)
		profile := &UserProfile{}
		createJimThroughTheChain(t, profile)
		line := "add print line to jim/main.go"
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled {
			t.Errorf("analyzed=%v: %q fell through the claim chain:\n%s", analyzed, line, out)
			continue
		}
		requireNoAnalyzeOutput(t, line, out)
		got := readBack(t, root, "jim/main.go")
		for _, want := range []string{"func main() {", "\tfmt.Println(\"\")", `"fmt"`} {
			if !strings.Contains(got, want) {
				t.Errorf("analyzed=%v: jim/main.go missing %q:\n%s", analyzed, want, got)
			}
		}
		if strings.Contains(out, "I can't add") {
			t.Errorf("analyzed=%v: the boundary answered a served concept:\n%s", analyzed, out)
		}
		if !strings.Contains(out, "built the project and checked") {
			t.Errorf("analyzed=%v: spine build/verify not reported:\n%s", analyzed, out)
		}
	}
}

// Gated "no": nothing is written, and the decline is the
// spine's own words.
func TestConceptDeclineWritesNothing(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a print line to jim/main.go", no, profile) })
	if !handled {
		t.Fatalf("declined concept line not claimed:\n%s", out)
	}
	if !strings.Contains(out, "Not applied") {
		t.Errorf("decline not reported:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); got != "package main\n" {
		t.Errorf("declined concept wrote to jim/main.go:\n%s", got)
	}
}

// Composer guidance, through the chain: a declaration with no
// name and a concept with no target are claimed and answered in
// composer terms, writing nothing.
func TestConceptGuidanceThroughChain(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	for _, c := range []struct{ line, want string }{
		{"add an interface to jim/main.go", "an interface needs a name"},
		{"add a struct to jim/main.go", "a struct needs a name"},
		{"add an interface named Reader", "doesn't name the file or folder"},
		{"add a print line to", "doesn't name the file or folder"},
	} {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(c.line, yes, profile) })
		if !handled {
			t.Errorf("%q fell through the claim chain:\n%s", c.line, out)
			continue
		}
		if !strings.Contains(out, c.want) {
			t.Errorf("%q guidance does not say %q:\n%s", c.line, c.want, out)
		}
	}
	if got := readBack(t, root, "jim/main.go"); got != "package main\n" {
		t.Errorf("guidance wrote to jim/main.go:\n%s", got)
	}
}

// Interface end to end against a project that has the type:
// named + like, the real methods land in the target file.
func TestConceptInterfaceEndToEnd(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/conceptiface\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {}\n")
	writeFile(t, dir, "store.go", `package main

type Store struct{}

func (s *Store) Load(path string) (string, error) { return "", nil }

func (s *Store) Save(name string, data []byte) error { return nil }
`)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	line := "add an interface named Storer like Store to main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("interface line fell through the claim chain:\n%s", out)
	}
	got := readBack(t, dir, "main.go")
	for _, want := range []string{
		"type Storer interface {",
		"Load(path string) (string, error)",
		"Save(name string, data []byte) error",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("main.go missing %q:\n%s\noutput:\n%s", want, got, out)
		}
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
}

// A nameless function ask through the chain: the composer's
// guidance names all THREE working forms — its own named
// scaffold (with the user's file filled in), the delegate, and
// the clone — and writes nothing.
func TestConceptFunctionNamelessGuidanceNamesAllThreeForms(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	line := "add a function to jim/main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	if !strings.Contains(out, "gollemer [composer]>") {
		t.Errorf("%q was not answered by the composer:\n%s", line, out)
	}
	for _, want := range []string{
		`"add a function named <Name> to jim/main.go"`,
		`"add a function in jim/main.go that calls <something that exists>"`,
		`"copy <existing function> as <new name>"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("nameless-function guidance does not name %q:\n%s", want, out)
		}
	}
	if got := readBack(t, root, "jim/main.go"); got != "package main\n" {
		t.Errorf("guidance wrote to jim/main.go:\n%s", got)
	}
}

// Function end to end in the live session: the scaffold lands
// in jim/main.go, the plan's note owns the emptiness before the
// gate, and the spine reports its build/verify. The print line
// goes first — the beginner's own order — because it grows the
// main a bare package-main file needs before ANY declaration
// can join it (the spine's build is absolute: a package main
// with no main function does not build, before the edit or
// after it, for every shape — the interface included).
func TestConceptFunctionScaffoldLiveSession(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}
	line := "add a function named Fetch to jim/main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	got := readBack(t, root, "jim/main.go")
	for _, want := range []string{"func Fetch() {", "// TODO: write the body"} {
		if !strings.Contains(got, want) {
			t.Errorf("jim/main.go missing %q:\n%s\noutput:\n%s", want, got, out)
		}
	}
	if !strings.Contains(out, "This is an empty scaffold") {
		t.Errorf("the plan did not own the scaffold's emptiness:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
}

// The beginner workflow the struct row exists for, in one
// session: add the struct, then add a field to it with the
// shipped field form — the second sentence reads the project
// the first edit left behind. The print line goes first (the
// beginner's own order, and the step that grows the main a
// bare package-main file needs before any declaration can
// join it — see TestConceptFunctionScaffoldLiveSession).
func TestConceptStructThenFieldInOneSession(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a struct named Config to jim/main.go", yes, profile) })
	if !handled {
		t.Fatalf("struct line fell through the claim chain:\n%s", out)
	}
	got := readBack(t, root, "jim/main.go")
	for _, want := range []string{
		"type Config struct {",
		`// TODO: add fields — or say "add a <Name> field of type <Type> to Config"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("jim/main.go missing %q after the struct add:\n%s\noutput:\n%s", want, got, out)
		}
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("struct add: spine build/verify not reported:\n%s", out)
	}

	out = captureStdout(t, func() { handled = claimChain("add a Timeout field of type int to Config", yes, profile) })
	if !handled {
		t.Fatalf("field line fell through the claim chain:\n%s", out)
	}
	got = readBack(t, root, "jim/main.go")
	if !strings.Contains(got, "Timeout int") {
		t.Errorf("the field did not land in the struct the same session added:\n%s\noutput:\n%s", got, out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("field add: spine build/verify not reported:\n%s", out)
	}
}

// Ordering against the neighbours, through the chain: the
// webserver and delegate lines keep their shapes (no concept
// marker in their answers), and an unknown concept still lands
// on the boundary.
func TestConceptOrderingAgainstNeighbours(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a webserver to jim/main.go", yes, profile) })
	if !handled || strings.Contains(out, "concept book") {
		t.Errorf("webserver line stolen or unhandled:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); got != analyze.WebServerSource {
		t.Errorf("webserver line did not fill the component:\n%s", got)
	}

	out = captureStdout(t, func() { handled = claimChain("add a method to jim/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "I can't add a method yet") {
		t.Errorf("unknown concept did not land on the boundary:\n%s", out)
	}
}
