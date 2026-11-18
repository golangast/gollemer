package chat

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The bare-file fix (2026-11-01), from John's live screenshot:
// "create folder jane" worked, but "create jane/main.go" fell
// through the whole claim chain into the intent layer's
// "review a function for problems?" prompt, and "create
// main.go in folder jane" got the add boundary's answer about
// a different main.go. The create family now claims the
// path-shaped bare forms itself.

// The new forms parse to the requests they name — the same
// createReq the file-word forms produce.
func TestParseCreateBarePathForms(t *testing.T) {
	cases := []struct {
		line string
		want createReq
	}{
		{"create jane/main.go", createReq{file: "jane/main.go"}},
		{"create jim/store.go", createReq{file: "jim/store.go"}},
		{"create main.go", createReq{file: "main.go"}},
		{"create a jane/main.go", createReq{file: "jane/main.go"}},
		{"make jane/main.go", createReq{file: "jane/main.go"}},
		{"create main.go in folder jane", createReq{file: "main.go", loc: "jane"}},
		{"create main.go in the folder jane", createReq{file: "main.go", loc: "jane"}},
		{"create a file jane/main.go", createReq{file: "jane/main.go"}},
		{"create a file main.go in folder jane", createReq{file: "main.go", loc: "jane"}},
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

// Bare NON-.go paths and qualified tails are not this grammar:
// they keep falling through to whatever owned them before.
func TestParseCreateBarePathNonClaims(t *testing.T) {
	for _, line := range []string{
		"create notes.txt",                               // bare non-.go path: not claimed (as before)
		"create jane/notes.txt",                          // same, with a folder in the path
		"create main.go with a webserver",                // a kind tail needs the file-word form
		"create main.go in folder jane with a webserver", // same
		"create main.go in folder",                       // the folder never arrives
		"create main.go in main",                         // a tail that is not "in folder"
		"create main.go please",                          // trailing words are not a tail
	} {
		if req, ok := parseCreate(line); ok {
			t.Errorf("parseCreate(%q) claimed as %+v — that sentence is not the bare-file grammar's", line, req)
		}
		if runCreate(line, yes) {
			t.Errorf("runCreate(%q) claimed a non-bare-file sentence", line)
		}
	}
}

// John's two screenshot lines, verbatim, through the claim
// chain in BOTH session states: the create seat claims them
// (claimChain mirrors the unified loop, where tryCreate sits
// ahead of the intent dispatch — a handled line here can never
// reach the "review a function?" prompt), the file lands as
// the minimal program, and the project still builds.
func TestBareCreateScreenshotLinesThroughChain(t *testing.T) {
	for _, analyzed := range []bool{true, false} {
		for _, line := range []string{"create jane/main.go", "create main.go in folder jane"} {
			root := liveSessionRoot(t)
			withChatSession(t, root, analyzed)
			profile := &UserProfile{}
			var folderOK bool
			captureStdout(t, func() { folderOK = claimChain("create a folder jane", yes, profile) })
			if !folderOK {
				t.Fatalf("analyzed=%v: folder line not claimed", analyzed)
			}
			var handled bool
			out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
			if !handled {
				t.Fatalf("analyzed=%v: %q fell through the whole claim chain:\n%s", analyzed, line, out)
			}
			if strings.Contains(out, "review a function") {
				t.Errorf("analyzed=%v: %q reached the intent layer's review prompt:\n%s", analyzed, line, out)
			}
			if !strings.Contains(out, "◆ CREATE") || !strings.Contains(out, "it builds") {
				t.Errorf("analyzed=%v: %q was not a build-verified creation:\n%s", analyzed, line, out)
			}
			if got := readBack(t, root, "jane/main.go"); got != createdMainStub {
				t.Errorf("analyzed=%v: %q created %q, want the minimal program %q", analyzed, line, got, createdMainStub)
			}
			cmd := exec.Command("go", "build", "./...")
			cmd.Dir = root
			if bout, err := cmd.CombinedOutput(); err != nil {
				t.Errorf("analyzed=%v: project does not build after %q: %v\n%s", analyzed, line, err, bout)
			}
		}
	}
}

// A bare .go creation never makes its folder silently: the
// missing folder is refused with the line that creates it, and
// nothing appears. Non-.go files and starter kinds keep the
// old make-the-folder behavior — pinned here so the scoping
// stays honest.
func TestBareCreateMissingFolderRefused(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	for _, line := range []string{
		"create jane/main.go",
		"create main.go in folder jane",
		"create a file jane/main.go",
		"create a file main.go in folder jane",
	} {
		var handled bool
		out := captureStdout(t, func() { handled = runCreate(line, yes) })
		if !handled {
			t.Fatalf("%q not claimed", line)
		}
		if !strings.Contains(out, "there's no folder jane in this project") || !strings.Contains(out, `"create a folder jane"`) {
			t.Errorf("%q: missing-folder refusal not said:\n%s", line, out)
		}
		if exists(filepath.Join(root, "jane")) {
			t.Errorf("%q: the folder was silently created", line)
		}
	}

	// The untouched neighbors: a non-.go file still makes its
	// folder as part of the one plan, exactly as before.
	if !runCreate("create a file notes.txt in folder newdir", yes) {
		t.Fatal("non-.go file ask not claimed")
	}
	if !exists(filepath.Join(root, "newdir", "notes.txt")) {
		t.Error("non-.go creation lost its make-the-folder behavior")
	}
}

// The package clause, settled from the disk: siblings lend
// theirs, an empty folder lends its own name, the root falls
// back to main — and a main.go is package main wherever it
// lands in an empty folder, carrying its empty func main.
func TestBareCreatePackageRules(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	if err := os.MkdirAll(filepath.Join(root, "jim"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "api"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, root, "api/api.go", "package api\n\n// One returns one.\nfunc One() int { return 1 }\n")

	if !runCreate("create jim/store.go", yes) {
		t.Fatal("empty-folder bare ask not claimed")
	}
	if got := readBack(t, root, "jim/store.go"); got != "package jim\n" {
		t.Errorf("jim/store.go = %q, want the folder's own package clause", got)
	}
	if !runCreate("create api/extra.go", yes) {
		t.Fatal("sibling-folder bare ask not claimed")
	}
	if got := readBack(t, root, "api/extra.go"); got != "package api\n" {
		t.Errorf("api/extra.go = %q, want the sibling package clause", got)
	}
	if !runCreate("create store.go", yes) {
		t.Fatal("root bare ask not claimed")
	}
	if got := readBack(t, root, "store.go"); got != "package main\n" {
		t.Errorf("store.go = %q, want the root fallback, package main", got)
	}
}

// A folder holding two packages refuses a bare creation,
// naming both — a new file has to join exactly one package.
func TestBareCreateMixedPackagesRefused(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	writeFile(t, root, "mix/a.go", "package alpha\n")
	writeFile(t, root, "mix/b.go", "package beta\n")
	var handled bool
	out := captureStdout(t, func() { handled = runCreate("create mix/x.go", yes) })
	if !handled {
		t.Fatal("mixed-folder ask not claimed")
	}
	if !strings.Contains(out, "more than one package") || !strings.Contains(out, "alpha") || !strings.Contains(out, "beta") {
		t.Errorf("mixed-package refusal does not name the packages:\n%s", out)
	}
	if exists(filepath.Join(root, "mix", "x.go")) {
		t.Error("a file was created in a mixed-package folder")
	}
}

// A package that already has its main program refuses another
// main.go, naming the file that holds func main. A func main
// in a DIFFERENT shape — a method or another package — is not
// a conflict and never fires this.
func TestBareCreateDuplicateMainRefused(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	writeFile(t, root, "jane/server.go", "package main\n\nfunc main() {}\n")
	for _, line := range []string{"create jane/main.go", "create main.go in folder jane"} {
		var handled bool
		out := captureStdout(t, func() { handled = runCreate(line, yes) })
		if !handled {
			t.Fatalf("%q not claimed", line)
		}
		if !strings.Contains(out, "jane/server.go") || !strings.Contains(out, "already defines func main") {
			t.Errorf("%q: duplicate-main refusal does not name the holder:\n%s", line, out)
		}
		if exists(filepath.Join(root, "jane", "main.go")) {
			t.Errorf("%q: a second main.go was created", line)
		}
	}
}

// The path form honors the same fences as every creation: an
// existing file is refused untouched, and a no at the gate
// writes nothing.
func TestBareCreateExistingAndDecline(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	if err := os.MkdirAll(filepath.Join(root, "jane"), 0o755); err != nil {
		t.Fatal(err)
	}
	var handled bool
	out := captureStdout(t, func() { handled = runCreate("create jane/main.go", no) })
	if !handled {
		t.Fatal("path-form ask not claimed")
	}
	if exists(filepath.Join(root, "jane", "main.go")) {
		t.Errorf("decline wrote the file; output:\n%s", out)
	}
	if !strings.Contains(out, "Not created — nothing was written") {
		t.Errorf("decline not acknowledged:\n%s", out)
	}

	if !runCreate("create jane/main.go", yes) {
		t.Fatal("path-form ask not claimed on the yes")
	}
	out = captureStdout(t, func() { handled = runCreate("create jane/main.go", yes) })
	if !handled || !strings.Contains(out, "already there") {
		t.Errorf("second ask not refused as existing:\n%s", out)
	}
	if got := readBack(t, root, "jane/main.go"); got != createdMainStub {
		t.Errorf("existing file was overwritten: %q", got)
	}
}
