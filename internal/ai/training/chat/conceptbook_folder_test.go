package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The folder layout build (2026-11-01), end to end through
// the claim chain: a declaration ask for a folder-mapped
// family with NO file named creates <folder>/<name>.go as
// its own package — folder auto-created, one gate, one
// transaction, the spine's whole-project build covering the
// new package. A named target always wins.

// folderSessionRoot: a small module with a program file.
func folderSessionRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/foldersession\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {}\n")
	return dir
}

func dirExists(dir string) bool {
	st, err := os.Stat(dir)
	return err == nil && st.IsDir()
}

func TestFolderDefaultFormHandlerEndToEnd(t *testing.T) {
	root := folderSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a form handler named Login", yes, profile) })
	if !handled {
		t.Fatalf("folder-default ask fell through the chain:\n%s", out)
	}
	for _, want := range []string{
		"Create the handlers folder and", "package handlers",
		"placed, not wired", "built the project and checked",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	got := readBack(t, root, "handlers/login.go")
	for _, want := range []string{
		"package handlers",
		"func Login(w http.ResponseWriter, r *http.Request) {",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("handlers/login.go missing %q:\n%s", want, got)
		}
	}
	if dirExists(filepath.Join(root, "security")) || dirExists(filepath.Join(root, "middleware")) {
		t.Errorf("a folder nobody asked for was created")
	}
}

func TestFolderDefaultSecurityFamilyEndToEnd(t *testing.T) {
	root := folderSessionRoot(t)
	withChatSession(t, root, false) // the cold session: no prior analyze
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a cors middleware named Cors", yes, profile) })
	if !handled {
		t.Fatalf("folder-default cors ask fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Fatalf("spine build/verify not reported:\n%s", out)
	}
	got := readBack(t, root, "security/cors.go")
	if !strings.HasPrefix(got, "package security\n") ||
		!strings.Contains(got, "func Cors(next http.HandlerFunc) http.HandlerFunc {") {
		t.Errorf("security/cors.go is not the cors middleware as its own package:\n%s", got)
	}
}

func TestFolderDefaultMiddlewareFamilyEndToEnd(t *testing.T) {
	root := folderSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a middleware named Logger", yes, profile) })
	if !handled {
		t.Fatalf("folder-default middleware ask fell through:\n%s", out)
	}
	got := readBack(t, root, "middleware/logger.go")
	if !strings.HasPrefix(got, "package middleware\n") ||
		!strings.Contains(got, "func Logger(next http.HandlerFunc) http.HandlerFunc {") {
		t.Errorf("middleware/logger.go is not the middleware as its own package:\n%s", got)
	}
}

// THE PIN: a named file always wins — the folder field never
// redirects an ask that names its target.
func TestFolderDefaultNamedTargetWins(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	out := captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatalf("print line did not grow jim's main:\n%s", out)
	}

	var handled bool
	out = captureStdout(t, func() { handled = claimChain("add a cors middleware named Cors to jim/main.go", yes, profile) })
	if !handled {
		t.Fatalf("named-target cors ask fell through:\n%s", out)
	}
	got := readBack(t, root, "jim/main.go")
	if !strings.Contains(got, "func Cors(next http.HandlerFunc) http.HandlerFunc {") {
		t.Errorf("the cors middleware did not land in the NAMED file:\n%s", got)
	}
	if dirExists(filepath.Join(root, "security")) {
		t.Errorf("the folder default fired despite the named target — security/ exists")
	}
	if _, err := os.Stat(filepath.Join(root, "security", "cors.go")); err == nil {
		t.Errorf("security/cors.go was created despite the named target")
	}
}

// The build-coverage proof: the project's build FAILS before
// the edit (main names a handler that doesn't exist yet) and
// can only pass if the spine's `go build ./...` compiles the
// NEW package. Success here is impossible if the new package
// were skipped.
func TestFolderDefaultBuildCoversTheNewPackage(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "go.mod", "module example.com/foldersession\n\ngo 1.21\n")
	writeFile(t, root, "main.go", `package main

import "example.com/foldersession/handlers"

var _ = handlers.Login

func main() {}
`)
	withChatSession(t, root, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a form handler named Login", yes, profile) })
	if !handled {
		t.Fatalf("build-coverage ask fell through:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Fatalf("the build did not cover the new package (or the edit rolled back):\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "handlers", "login.go")); err != nil {
		t.Errorf("handlers/login.go missing after a verified build: %v", err)
	}
}

// The rollback side of the same proof: the new package makes
// the build fail (main calls the handler with the wrong
// shape), so the transaction restores everything — including
// removing the folder the plan created.
func TestFolderDefaultBuildFailureRemovesTheCreatedFolder(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "go.mod", "module example.com/foldersession\n\ngo 1.21\n")
	writeFile(t, root, "main.go", `package main

import "example.com/foldersession/handlers"

func main() {
	handlers.Login()
}
`)
	withChatSession(t, root, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a form handler named Login", yes, profile) })
	if !handled {
		t.Fatalf("rollback ask fell through:\n%s", out)
	}
	if !strings.Contains(out, "That edit doesn't build") {
		t.Fatalf("expected the build-failure rollback:\n%s", out)
	}
	if dirExists(filepath.Join(root, "handlers")) {
		t.Errorf("the created handlers folder survived the rollback")
	}
	got := readBack(t, root, "main.go")
	if !strings.Contains(got, "handlers.Login()") {
		t.Errorf("main.go was not restored:\n%s", got)
	}
}

// A declined gate writes nothing — and creates no folder.
func TestFolderDefaultDeclineLeavesNothing(t *testing.T) {
	root := folderSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a form handler named Login", no, profile) })
	if !handled {
		t.Fatalf("declined ask fell through:\n%s", out)
	}
	if !strings.Contains(out, "Not applied") {
		t.Fatalf("expected the decline line:\n%s", out)
	}
	if dirExists(filepath.Join(root, "handlers")) {
		t.Errorf("a declined plan created the handlers folder")
	}
}

func TestFolderDefaultExistingFolderRealPackage(t *testing.T) {
	root := folderSessionRoot(t)
	writeFile(t, root, "handlers/old.go", "package webforms\n\nfunc Old() {}\n")
	withChatSession(t, root, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a form handler named Login", yes, profile) })
	if !handled {
		t.Fatalf("existing-folder ask fell through:\n%s", out)
	}
	if !strings.Contains(out, "package webforms") {
		t.Errorf("output doesn't name the folder's real package:\n%s", out)
	}
	got := readBack(t, root, "handlers/login.go")
	if !strings.HasPrefix(got, "package webforms\n") {
		t.Errorf("new file does not carry the folder's real package name:\n%s", got)
	}
}

func TestFolderDefaultRefusalsThroughTheChain(t *testing.T) {
	cases := []struct {
		name  string
		files map[string]string
		line  string
		want  []string
	}{
		{
			name:  "package main folder",
			files: map[string]string{"handlers/main.go": "package main\n\nfunc main() {}\n"},
			line:  "add a form handler named Login",
			want:  []string{"package main", "program"},
		},
		{
			name: "mixed packages",
			files: map[string]string{
				"handlers/a.go": "package one\n\nfunc A() {}\n",
				"handlers/b.go": "package two\n\nfunc B() {}\n",
			},
			line: "add a form handler named Login",
			want: []string{"mixes", "package one", "package two"},
		},
		{
			name:  "collision in the folder package",
			files: map[string]string{"handlers/other.go": "package handlers\n\nfunc Login(w http.ResponseWriter, r *http.Request) {}\n"},
			line:  "add a form handler named Login",
			want:  []string{"Login already exists in package handlers"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root := folderSessionRoot(t)
			for rel, src := range c.files {
				writeFile(t, root, rel, src)
			}
			// The collision fixture's import needs net/http
			// for its own handler signature.
			if c.name == "collision in the folder package" {
				writeFile(t, root, "handlers/other.go", "package handlers\n\nimport \"net/http\"\n\nfunc Login(w http.ResponseWriter, r *http.Request) {}\n")
			}
			withChatSession(t, root, true)
			profile := &UserProfile{}
			var handled bool
			out := captureStdout(t, func() { handled = claimChain(c.line, yes, profile) })
			if !handled {
				t.Fatalf("refusal case %q fell through:\n%s", c.line, out)
			}
			for _, want := range c.want {
				if !strings.Contains(out, want) {
					t.Errorf("refusal missing %q:\n%s", want, out)
				}
			}
			if _, err := os.Stat(filepath.Join(root, "handlers", "login.go")); err == nil {
				t.Errorf("a refused ask still created handlers/login.go")
			}
		})
	}
}

func TestFolderDefaultGuidanceBoundaries(t *testing.T) {
	root := folderSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}

	// Unnamed folder-family ask: the guidance names the
	// folder form.
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a cors middleware", yes, profile) })
	if !handled || !strings.Contains(out, "needs a name") || !strings.Contains(out, "security folder") {
		t.Errorf("unnamed cors guidance = handled %v:\n%s", handled, out)
	}

	// A statement row never folder-defaults: the no-target
	// guidance stands exactly as before.
	out = captureStdout(t, func() { handled = claimChain("add a body limit", yes, profile) })
	if !handled || !strings.Contains(out, "doesn't name the file or folder") {
		t.Errorf("statement no-target guidance = handled %v:\n%s", handled, out)
	}

	// An unmapped declaration keeps the old guidance too.
	out = captureStdout(t, func() { handled = claimChain("add a struct named Point", yes, profile) })
	if !handled || !strings.Contains(out, "doesn't name the file or folder") {
		t.Errorf("unmapped declaration guidance = handled %v:\n%s", handled, out)
	}
	if dirExists(filepath.Join(root, "security")) || dirExists(filepath.Join(root, "handlers")) {
		t.Errorf("guidance answers created folders")
	}
}

// The phrasing route carries the no-file form with the
// SHIPPED classifier artifact: the model supplies the family
// and the name, the extractor renders a canonical with no
// path in it, and the folder is resolved deterministically
// from the book when the piece executes. Measured on the
// folders judge (data/training/phrasing_heldout_folders
// .jsonl, 362 items, leak checks 0): 99.5% slot-exact
// single-unseen, 100% single-seen, 100% compound-created.
// (A retrain on the extended corpus was measured and NOT
// shipped: it traded these numbers down — folders singles
// 97.7, rqlite 97.5 — against zero baseline movement here,
// so the artifact stays byte-identical to b393117.)
func TestFolderDefaultPhrasingRouteShapes(t *testing.T) {
	withPhrasingModel(t)
	cases := []struct {
		line string
		want []string
	}{
		// A messy single: no grammar reads "kindly add a
		// form", and the canonical carries no target.
		{"kindly add a form named Orbit",
			[]string{"add a form handler named Orbit"}},
		// Created compounds on a folder first piece, in the
		// forms whose second piece carries the folder
		// address: the address is the binding's, never a
		// shared target for the first piece.
		{"place a chain middleware named Nudge and after that insert a rqlite create inside it to middleware/nudge.go",
			[]string{"add a middleware chain named Nudge", "add a rqlite create table in Nudge to middleware/nudge.go"}},
		{"make a form handler named Login and add a switch in Login to handlers/login.go",
			[]string{"add a form handler named Login", "add a switch statement in Login to handlers/login.go"}},
	}
	for _, c := range cases {
		got, ok := PhrasingRoute(c.line)
		if !ok {
			t.Errorf("PhrasingRoute(%q) not claimed", c.line)
			continue
		}
		if strings.Join(got, " ## ") != strings.Join(c.want, " ## ") {
			t.Errorf("PhrasingRoute(%q) = %q, want %q", c.line, got, c.want)
		}
	}
}

// The phrasing layer executing a folder piece end to end:
// the routed canonical runs through runConcept, which
// applies the folder default — folder created, declaration
// placed, spine-verified.
func TestFolderDefaultPhrasingEndToEnd(t *testing.T) {
	withPhrasingModel(t)
	root := folderSessionRoot(t)
	withChatSession(t, root, true)

	var handled bool
	out := captureStdout(t, func() { handled = runPhrasingLayer("kindly add a form named Orbit", yes) })
	if !handled {
		t.Fatalf("phrasing folder ask fell through:\n%s", out)
	}
	got := readBack(t, root, "handlers/orbit.go")
	if !strings.HasPrefix(got, "package handlers\n") ||
		!strings.Contains(got, "func Orbit(w http.ResponseWriter, r *http.Request) {") {
		t.Errorf("the phrasing folder piece did not create handlers/orbit.go:\n%s", got)
	}
}

// The data-driven proof at the chat level: a book row that
// exists nowhere in the shipped book drives the whole
// behaviour — folder, package, file — with no new code.
func TestFolderDefaultFixtureRowDrivesIt(t *testing.T) {
	entries, skipped := analyze.ParseConceptBook([]byte(`{
	  "concepts": [
	    {"id": "wizard-handler", "names": ["wizard handler"], "kind": "declaration", "folder": "wizards",
	     "template": "func {name}(w http.ResponseWriter, r *http.Request) {\n\t// TODO: cast something\n}", "imports": ["net/http"]}
	  ]
	}`))
	if skipped != 0 || len(entries) != 1 {
		t.Fatalf("fixture book = %d entries, %d skipped", len(entries), skipped)
	}
	withConceptEntries(t, entries)

	root := folderSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a wizard handler named Cast", yes, profile) })
	if !handled {
		t.Fatalf("fixture-row ask fell through:\n%s", out)
	}
	got := readBack(t, root, "wizards/cast.go")
	if !strings.HasPrefix(got, "package wizards\n") ||
		!strings.Contains(got, "func Cast(w http.ResponseWriter, r *http.Request) {") {
		t.Errorf("the fixture row did not drive a folder creation:\n%s", got)
	}
}
