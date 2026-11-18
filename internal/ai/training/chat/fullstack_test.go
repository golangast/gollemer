package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The full-stack claim (fullstack.go): "add a full stack
// app called <thing> in <folder> with fields …" grows one
// thing through every layer in one gated plan. These
// tests drive the whole flow through claimChain — the
// parse, the two-step for an app-less folder, the fields
// guidance, the landing with its in-transaction wasm
// compile, the gate's no, the session-folder reading, and
// the rollback when the app stops compiling.

// pinGoDirsToDisk points the Go build's caches at the
// real disk for a test whose landing compiles the wasm
// app: this VM's test HOME sits on a 512MB /tmp tmpfs
// (the package TestMain), and a toolchain download plus
// build cache there fills it mid-suite and fake-fails
// the compile ("no space left on device"). A real HOME
// is real disk; this only keeps the test honest.
func pinGoDirsToDisk(t *testing.T) {
	t.Helper()
	t.Setenv("GOCACHE", "/home/hatch/.cache/go-build")
	t.Setenv("GOMODCACHE", "/home/hatch/go/pkg/mod")
	if err := os.MkdirAll("/home/hatch/workspace/fullstack-probe/gotmp", 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOTMPDIR", "/home/hatch/workspace/fullstack-probe/gotmp")
}

func TestParseFullStack(t *testing.T) {
	claims := []struct {
		line   string
		thing  string
		folder string
		fields string // the parsed fields, name:kind joined
		state  string
	}{
		{"add a full stack app called notes in jim with fields title, pages number and done yes-no", "notes", "jim", "title:text,pages:number,done:yesno", "ok"},
		{"add a full-stack app called notes in jim with fields title", "notes", "jim", "title:text", "ok"},
		{"make a fullstack app called books with fields name", "books", "", "name:text", "ok"},
		{"add the full stack app called notes in the folder jim with fields title and done yesno", "notes", "jim", "title:text,done:yesno", "ok"},
		{"add a full stack app called notes in jim", "notes", "jim", "", "missing"},
		{"build a full stack app called notes in jim/app with fields body text", "notes", "jim/app", "body:text", "ok"},
	}
	for _, tc := range claims {
		q, ok := parseFullStack(tc.line)
		if !ok {
			t.Errorf("parseFullStack(%q) not claimed", tc.line)
			continue
		}
		var got []string
		for _, f := range q.fields {
			got = append(got, f.Name+":"+f.Kind)
		}
		if q.thing != tc.thing || q.folder != tc.folder || strings.Join(got, ",") != tc.fields || q.fieldsState != tc.state {
			t.Errorf("parseFullStack(%q) = %+v (fields %s), want thing %q folder %q fields %q state %q", tc.line, q, strings.Join(got, ","), tc.thing, tc.folder, tc.fields, tc.state)
		}
	}
	declines := []string{
		"add a wasm app in jim",                                      // the app claim's
		"add a full stack of pancakes",                               // no app word, no called
		"add a full stack app in jim",                                // no called
		"add a stack app called notes in jim",                        // no "full"
		"add a card component to jim",                                // the component claim's
		"add a full stack app called notes in jim with pages number", // no "fields" word
		"serve the wasm app in jim",                                  // the serve claim's
		"what is a full stack app",                                   // a question, not this verb shape
	}
	for _, line := range declines {
		if _, ok := parseFullStack(line); ok {
			t.Errorf("parseFullStack(%q) claimed a line that isn't its sentence", line)
		}
	}
	// A bad kind word inside the clause is claimed
	// guidance, not a decline: the shape matched.
	q, ok := parseFullStack("add a full stack app called notes in jim with fields title wobbly")
	if !ok || q.fieldsState == "ok" || q.fieldsState == "missing" {
		t.Errorf("bad kind word: claimed %v, state %q — want a claimed problem", ok, q.fieldsState)
	}
}

// The landing, end to end through the chain: folder, app,
// stack — the stack's sentence lands the backend beside a
// created server, the frontend in the app, and the
// confirmation names the whole stack. The landing runs
// the wasm compile inside the transaction, so this also
// proves the generated frontend compiles as landed.
func TestFullStackLandsThroughTheChain(t *testing.T) {
	pinGoDirsToDisk(t)
	yes := func(string) (string, error) { return "y", nil }
	dir := staticAssetsFixture(t, staticAssetsFiles(nil))
	out := captureStdout(t, func() {
		claimChain("create a folder jim", yes, &UserProfile{})
		claimChain("add a wasm app in jim", yes, &UserProfile{})
		if !claimChain("add a full stack app called notes in jim with fields title, pages number and done yes-no", yes, &UserProfile{}) {
			t.Error("the full-stack sentence was not claimed")
		}
	})
	for _, want := range []string{"/api/notes", "the notes table", "rqlited -node-id=1 data/", "The notes stack is grown", "I built the project and checked that it's really there."} {
		if !strings.Contains(out, want) {
			t.Errorf("landing output is missing %q:\n%s", want, out)
		}
	}
	for rel, wants := range map[string][]string{
		"server/notes.go": {`http.HandleFunc("/api/notes"`, "CREATE TABLE IF NOT EXISTS notes"},
		"server/main.go":  {"setupNotes()", `http.Dir("jim")`},
		"jim/notes.go":    {"func notesSection(", "notesSave"},
		"jim/main.go":     {"notesSection(doc)"},
		"jim/index.html":  {"/* full-stack sections"},
	} {
		body := readBack(t, dir, rel)
		for _, want := range wants {
			if !strings.Contains(body, want) {
				t.Errorf("%s is missing %q", rel, want)
			}
		}
	}
	// Asked again, the same thing is already there.
	out = captureStdout(t, func() {
		claimChain("add a full stack app called notes in jim with fields title, pages number and done yes-no", yes, &UserProfile{})
	})
	if !strings.Contains(out, "already there") {
		t.Errorf("the repeat ask should answer already-there:\n%s", out)
	}
}

// The gate's no lands nothing: no server, no backend, no
// frontend, the app untouched.
func TestFullStackGateDeclinedLandsNothing(t *testing.T) {
	yes := func(string) (string, error) { return "y", nil }
	no := func(string) (string, error) { return "n", nil }
	dir := staticAssetsFixture(t, staticAssetsFiles(nil))
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, &UserProfile{})
		claimChain("add a wasm app in jim", yes, &UserProfile{})
		claimChain("add a full stack app called notes in jim with fields title", no, &UserProfile{})
	})
	for _, rel := range []string{"server/notes.go", "server/main.go", "jim/notes.go"} {
		if _, err := os.Stat(filepath.Join(dir, rel)); !os.IsNotExist(err) {
			t.Errorf("the declined gate left %s behind", rel)
		}
	}
	if body := readBack(t, dir, "jim/main.go"); strings.Contains(body, "notesSection") {
		t.Error("the declined gate edited the app's main.go")
	}
}

// A folder with no wasm app meets the family's two-step,
// in the open, and nothing is created for it.
func TestFullStackNoAppTwoStep(t *testing.T) {
	yes := func(string) (string, error) { return "y", nil }
	dir := staticAssetsFixture(t, staticAssetsFiles(nil))
	out := captureStdout(t, func() {
		claimChain("create a folder jim", yes, &UserProfile{})
		if !claimChain("add a full stack app called notes in jim with fields title", yes, &UserProfile{}) {
			t.Error("the full-stack sentence was not claimed")
		}
	})
	if !strings.Contains(out, `"add a wasm app in jim"`) {
		t.Errorf("the two-step should name the app sentence:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "server")); !os.IsNotExist(err) {
		t.Error("the two-step created a server")
	}
}

// No fields clause: the claim answers with the shape of
// the whole sentence, before any project question.
func TestFullStackMissingFieldsGuidance(t *testing.T) {
	yes := func(string) (string, error) { return "y", nil }
	staticAssetsFixture(t, staticAssetsFiles(nil))
	out := captureStdout(t, func() {
		if !claimChain("add a full stack app called notes in jim", yes, &UserProfile{}) {
			t.Error("the full-stack sentence was not claimed")
		}
	})
	if !strings.Contains(out, "names no fields") || !strings.Contains(out, "with fields title, pages number and done yes-no") {
		t.Errorf("the fields guidance is wrong:\n%s", out)
	}
}

// The folderless sentence reads the session's app — the
// reading is printed, and the stack lands in that app.
func TestFullStackSessionFolder(t *testing.T) {
	pinGoDirsToDisk(t)
	yes := func(string) (string, error) { return "y", nil }
	dir := staticAssetsFixture(t, staticAssetsFiles(nil))
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, &UserProfile{})
		claimChain("add a wasm app in jim", yes, &UserProfile{})
	})
	out := captureStdout(t, func() {
		if !claimChain("add a full stack app called books with fields name", yes, &UserProfile{}) {
			t.Error("the folderless full-stack sentence was not claimed")
		}
	})
	if !strings.Contains(out, "Reading the app as the wasm app in jim") {
		t.Errorf("the session reading was not printed:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "jim", "books.go")); err != nil {
		t.Errorf("the stack did not land in the session's app: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "server", "books.go")); err != nil {
		t.Errorf("the stack's backend did not land: %v", err)
	}
}

// The rollback: the app is broken by hand after it was
// built (a wasm file that doesn't compile), so the
// landing's in-transaction wasm check fails — the whole
// stack rolls back, the server files never land, and the
// app's own files keep their bytes.
func TestFullStackRollsBackWhenTheAppStopsCompiling(t *testing.T) {
	pinGoDirsToDisk(t)
	yes := func(string) (string, error) { return "y", nil }
	dir := staticAssetsFixture(t, staticAssetsFiles(nil))
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, &UserProfile{})
		claimChain("add a wasm app in jim", yes, &UserProfile{})
	})
	mainBefore := readBack(t, dir, "jim/main.go")
	writeFile(t, dir, "jim/broken.go", "//go:build js && wasm\n\npackage main\n\nfunc broken( {\n")
	out := captureStdout(t, func() {
		if !claimChain("add a full stack app called notes in jim with fields title", yes, &UserProfile{}) {
			t.Error("the full-stack sentence was not claimed")
		}
	})
	if !strings.Contains(out, "doesn't compile") || !strings.Contains(out, "restored your files") {
		t.Errorf("the rollback narration is wrong:\n%s", out)
	}
	for _, rel := range []string{"server/notes.go", "server/main.go", "jim/notes.go"} {
		if _, err := os.Stat(filepath.Join(dir, rel)); !os.IsNotExist(err) {
			t.Errorf("the rollback left %s behind", rel)
		}
	}
	if got := readBack(t, dir, "jim/main.go"); got != mainBefore {
		t.Error("the rollback changed the app's main.go")
	}
}
