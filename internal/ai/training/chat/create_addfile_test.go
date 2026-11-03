package chat

import (
	"path/filepath"
	"strings"
	"testing"
)

// The index-aware routing fix, ITEM 2 (D1): "add the file
// jake/database/db.go" used to be claimed by the write
// boundary's refusal — it created nothing, and the refusal
// never named the create sentence. Two moves fix it:
//
//   - parseCreate borrows the book's verb "add" for the file
//     case alone (and the article slot learns "the"), so the
//     file-shaped add-line rides the existing create flow —
//     same plan, gate, package rules, and exists refusal.
//   - A file-itself line that still reaches the boundary
//     (an unparseable tail kept the create grammar off it)
//     gets a generic answer whose suggestions LEAD with the
//     create sentence for its own target.

// The widened grammar, at parse level: the add forms and the
// "the"-article forms parse to the same createReq the file
// case has always produced.
func TestParseCreateAddFileForms(t *testing.T) {
	cases := []struct {
		line string
		want createReq
	}{
		{"add the file jake/database/db.go", createReq{file: "jake/database/db.go"}},
		{"add a file jake/database/db.go", createReq{file: "jake/database/db.go"}},
		{"add file jake/database/db.go", createReq{file: "jake/database/db.go"}},
		{"add a file named jake/database/db.go", createReq{file: "jake/database/db.go"}},
		{"add a file db.go in folder jake/database", createReq{file: "db.go", loc: "jake/database"}},
		{"add a file srv.go with a webserver", createReq{file: "srv.go", kind: "webserver"}},
		{"create the file jane/main.go", createReq{file: "jane/main.go"}},
		{"make the file jane/main.go", createReq{file: "jane/main.go"}},
		{"make a file jane/main.go", createReq{file: "jane/main.go"}},
		{"create the folder jim", createReq{dir: true, name: "jim"}},
	}
	for _, c := range cases {
		got, ok := parseCreate(c.line)
		if !ok {
			t.Errorf("parseCreate(%q) not claimed", c.line)
			continue
		}
		if got != c.want {
			t.Errorf("parseCreate(%q) = %+v, want %+v", c.line, got, c.want)
		}
	}
}

// Under "add", ONLY the file case is borrowed: the folder
// case and the bare-path case refuse, and the concept book's
// own file-shaped sentences fall out of the file grammar on
// their "to" tail — their owner does not change.
func TestParseCreateAddNonClaims(t *testing.T) {
	for _, line := range []string{
		"add a folder jim",                                // folders are create|make only
		"add folder jim",                                  // same, no article
		"add a directory jim",                             // the directory spelling too
		"add the folder jim",                              // the new article does not widen "add"
		"add jane/main.go",                                // the bare path under "add" keeps falling through
		"add the jane/main.go",                            // same, behind the article
		"add a file path to jim/main.go",                  // the book's file-paths row
		"add a file exists to jim/main.go",                // the book's file-exists row
		"add a file exists check in epsilon to pieces.go", // the handtest spelling of that row
		"add file reading to jim/main.go",                 // the book's read-file row
		"add file paths to jim/main.go",                   // the book's file-paths row, plural
		"make a file called api/validate.go and add a function named Validate to it", // created-target compound: never the file grammar's
	} {
		if req, ok := parseCreate(line); ok {
			t.Errorf("parseCreate(%q) claimed as %+v — that sentence is not the borrowed file case's", line, req)
		}
	}
}

// D1 verbatim, and its symmetric forms, through the whole
// claim chain: jake/database/ already holds a package-database
// file, and the ask lands db.go beside it carrying that
// package — behind the create plan and its gate.
func TestAddFileD1ThroughChain(t *testing.T) {
	for _, line := range []string{
		"add the file jake/database/db.go",
		"add a file jake/database/db.go",
		"create the file jake/database/db.go",
		"make the file jake/database/db.go",
	} {
		root := liveSessionRoot(t)
		withChatSession(t, root, true)
		writeFile(t, root, "jake/database/database.go", "package database\n")
		profile := &UserProfile{}
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled {
			t.Fatalf("%q fell through the whole claim chain:\n%s", line, out)
		}
		if !strings.Contains(out, "◆ CREATE") || !strings.Contains(out, "jake/database/db.go") {
			t.Errorf("%q did not show the create plan for its file:\n%s", line, out)
		}
		if strings.Contains(out, "I can't add") {
			t.Errorf("%q was answered by the write boundary instead of creating:\n%s", line, out)
		}
		if got := readBack(t, root, "jake/database/db.go"); got != "package database\n" {
			t.Errorf("%q created db.go as %q, want the folder's package clause %q", line, got, "package database\n")
		}
	}
}

// Repeating the add-file ask for a file that is already there
// meets the standing exists refusal, word for word, and the
// file is untouched.
func TestAddFileExistingRefused(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	writeFile(t, root, "jake/database/database.go", "package database\n")
	if !runCreate("add the file jake/database/db.go", yes) {
		t.Fatal("first add-file ask not claimed")
	}
	before := readBack(t, root, "jake/database/db.go")
	var handled bool
	out := captureStdout(t, func() { handled = runCreate("add the file jake/database/db.go", yes) })
	if !handled {
		t.Fatal("repeat add-file ask not claimed")
	}
	if !strings.Contains(out, "jake/database/db.go is already there — I never overwrite what's yours") {
		t.Errorf("exists refusal not said in its standing words:\n%s", out)
	}
	if after := readBack(t, root, "jake/database/db.go"); after != before {
		t.Errorf("existing file changed: before %q, after %q", before, after)
	}
}

// A missing folder under the borrowed verb refuses exactly
// like the create forms do — the folder is a step of its own.
func TestAddFileMissingFolderRefused(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	var handled bool
	out := captureStdout(t, func() { handled = runCreate("add the file newdir/x.go", yes) })
	if !handled {
		t.Fatal("add-file ask not claimed")
	}
	if !strings.Contains(out, "there's no folder newdir in this project") || !strings.Contains(out, `"create a folder newdir" first`) {
		t.Errorf("missing-folder refusal not said:\n%s", out)
	}
	if exists(filepath.Join(root, "newdir")) {
		t.Error("the folder was silently created")
	}
}

// The collision pins at chain level: the book's two
// file-shaped rows still land their own pieces in jim/main.go
// — the borrowed verb did not steal them.
func TestAddFileBookCollisionsStillLand(t *testing.T) {
	cases := []struct {
		line   string
		id     string
		marker string
	}{
		{"add a file path to jim/main.go", "file-paths", "filepath.Join"},
		{"add a file exists to jim/main.go", "file-exists", "os.Stat"},
	}
	book := defaultBookEntries(t)
	for _, c := range cases {
		if req, ok := parseConceptFrom(c.line, book); !ok || req.entry.ID != c.id {
			t.Errorf("parseConcept(%q) = %v, %v — want the book's %q row", c.line, req, ok, c.id)
		}
		root := liveSessionRoot(t)
		withChatSession(t, root, true)
		profile := &UserProfile{}
		createJimThroughTheChain(t, profile)
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(c.line, yes, profile) })
		if !handled {
			t.Fatalf("%q fell through the whole claim chain:\n%s", c.line, out)
		}
		if !strings.Contains(out, "built the project and checked") {
			t.Errorf("%q did not land through the spine:\n%s", c.line, out)
		}
		if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, c.marker) {
			t.Errorf("%q: jim/main.go does not contain %q after the book landed it:\n%s", c.line, c.marker, got)
		}
		if exists(filepath.Join(root, "jim", "path")) || exists(filepath.Join(root, "jim", "exists")) {
			t.Errorf("%q: a creation landed beside the book piece", c.line)
		}
	}
}

// The boundary lead: a file-itself line whose tail keeps the
// create grammar off it still reaches the boundary, is
// flagged as the file itself, and its generic answer leads
// with the create sentence for its own target — the standing
// alternatives ride along after it, unchanged.
func TestWriteBoundaryFileItselfLead(t *testing.T) {
	const wantAnswer = `I can't add that yet — that's not one of the pieces I can assemble today. To make the file itself, say "create a file jake/database/db.go". What I can add to jake/database/db.go today: a webserver — "add a webserver to jake/database/db.go" — or a function copied from a real one — "add a function in jake/database/db.go that calls <Name>"`
	for _, line := range []string{
		"add the file jake/database/db.go please", // trailing word: the create grammar refuses the tail
		"add a file to jake/database/db.go",       // the file word names no file for the grammar ("to" is read as the name, then the path is a stray tail)
	} {
		if _, ok := parseCreate(line); ok {
			t.Errorf("parseCreate(%q) claimed — the boundary lead test needs a real fall-through", line)
		}
		req, ok := parseWriteBoundary(line)
		if !ok {
			t.Errorf("parseWriteBoundary(%q) did not claim a file-itself line", line)
			continue
		}
		if req.kind != wbGeneric || !req.fileItself || req.target != "jake/database/db.go" {
			t.Errorf("parseWriteBoundary(%q) = kind %q, fileItself %v, target %q — want generic, true, jake/database/db.go", line, req.kind, req.fileItself, req.target)
		}
		if got := req.answer(); got != wantAnswer {
			t.Errorf("answer(%q) =\n%s\nwant:\n%s", line, got, wantAnswer)
		}
		var handled bool
		out := captureStdout(t, func() { handled = runWriteBoundary(line) })
		if !handled {
			t.Errorf("runWriteBoundary(%q) not claimed", line)
			continue
		}
		lead := strings.Index(out, `To make the file itself, say "create a file jake/database/db.go".`)
		alts := strings.Index(out, "What I can add to jake/database/db.go today:")
		if lead < 0 || alts < 0 || lead > alts {
			t.Errorf("runWriteBoundary(%q): the create lead is not the first suggestion:\n%s", line, out)
		}
	}
}

// Every other generic boundary answer stays byte-identical:
// a non-file-itself line keeps its exact standing text, with
// no create lead inserted.
func TestWriteBoundaryGenericAnswerUnchangedForNonFileItself(t *testing.T) {
	req, ok := parseWriteBoundary("add an interface to jim/main.go")
	if !ok {
		t.Fatal("interface line not claimed")
	}
	if req.fileItself {
		t.Error("interface line flagged as the file itself")
	}
	const want = `I can't add an interface yet — that's not one of the pieces I can assemble today. What I can add to jim/main.go today: a webserver — "add a webserver to jim/main.go" — or a function copied from a real one — "add a function in jim/main.go that calls <Name>"`
	if got := req.answer(); got != want {
		t.Errorf("generic answer changed:\ngot:\n%s\nwant:\n%s", got, want)
	}
}
