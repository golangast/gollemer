package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// withCreateRoot points the creation root at dir the way an open
// project would: lastAnalyzeRoot set, no analyzed project object
// (runCreate resolves the root from the globals, like the loop).
func withCreateRoot(t *testing.T, dir string) {
	t.Helper()
	oldProj, oldRoot, oldURoot := lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot
	t.Cleanup(func() { lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot = oldProj, oldRoot, oldURoot })
	lastAnalyzeProject = nil
	lastAnalyzeRoot = dir
	unifiedProjectRoot = dir
}

func writeFile(t *testing.T, dir, rel, src string) {
	t.Helper()
	full := filepath.Join(dir, rel)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
}

func exists(path string) bool {
	_, err := os.Lstat(path)
	return err == nil
}

// The screenshot lines and the documented grammar parse to the
// requests they name.
func TestParseCreateForms(t *testing.T) {
	cases := []struct {
		line string
		want createReq
	}{
		{"create a folder jim", createReq{dir: true, name: "jim"}},
		{"create folder named jim", createReq{dir: true, name: "jim"}},
		{"create a folder named jim", createReq{dir: true, name: "jim"}},
		{"make a folder jim", createReq{dir: true, name: "jim"}},
		{"make a directory called jim", createReq{dir: true, name: "jim"}},
		{"create a folder jim.", createReq{dir: true, name: "jim"}},
		{"create a file in folder jim with a webserver", createReq{file: "main.go", loc: "jim", kind: "webserver"}},
		{"create a file main.go", createReq{file: "main.go"}},
		{"create file server.go with a webserver", createReq{file: "server.go", kind: "webserver"}},
		{"create a file server.go in folder jim", createReq{file: "server.go", loc: "jim"}},
		{"create a file server.go with a webserver in folder jim", createReq{file: "server.go", loc: "jim", kind: "webserver"}},
		{"create a file named main.go", createReq{file: "main.go"}},
		{"create a file notes.txt", createReq{file: "notes.txt"}},
		{"create a file srv.go with a web server", createReq{file: "srv.go", kind: "webserver"}},
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

// Lines owned by other features — the resources claim's goal
// statements, the composer's imperatives, function writing, plain
// chat — must never parse as creations.
func TestParseCreateNonClaims(t *testing.T) {
	for _, line := range []string{
		"i want to build a webserver",              // resources: goal, no create verb/target
		"how do i build a webserver",               // resources
		"building a webserver",                     // resources
		"resources for webservers",                 // resources
		"build api for /users",                     // the composer's imperative
		"create a function that reverses a string", // gocode's line
		"create a new project",                     // scaffolding is not v1
		"make chat",
		"make a cake",
		"what does this program do",
	} {
		if req, ok := parseCreate(line); ok {
			t.Errorf("parseCreate(%q) claimed as %+v — that sentence belongs to another feature", line, req)
		}
		if runCreate(line, yes) {
			t.Errorf("runCreate(%q) claimed a non-creation sentence", line)
		}
	}
	// The other direction: the owners still own their lines.
	if _, _, ok := parseBuildGoal("i want to build a webserver"); !ok {
		t.Error("resources lost its goal statement")
	}
}

// The boundary family parses — imperative, disk-shaped only.
func TestParseBoundaryForms(t *testing.T) {
	cases := []struct {
		line, verb, cmd string
	}{
		{"delete the file server.go", "delete", "rm server.go"},
		{"delete folder jim", "delete", "rm -rf jim"},
		{"remove file a.go", "delete", "rm a.go"},
		{"remove the folder oldstuff", "delete", "rm -rf oldstuff"},
		{"move the file a.go to folder jim", "move", "mv a.go jim"},
		{"move file a.go to b.go", "move", "mv a.go b.go"},
		{"rename the file old.go to new.go", "rename", "mv old.go new.go"},
		{"rename the folder jim to james", "rename", "mv jim james"},
		{"install golang.org/x/tools/cmd/goimports", "install", "go install golang.org/x/tools/cmd/goimports@latest"},
		{"download https://example.com/x.zip", "download", "curl -LO https://example.com/x.zip"},
	}
	for _, c := range cases {
		got, ok := parseBoundary(c.line)
		if !ok {
			t.Errorf("parseBoundary(%q) not claimed", c.line)
			continue
		}
		if got.verb != c.verb || got.cmd != c.cmd {
			t.Errorf("parseBoundary(%q) = {%s %q}, want {%s %q}", c.line, got.verb, got.cmd, c.verb, c.cmd)
		}
	}
}

// Disk actions without the disk shape — and questions — are not
// the boundary's, and the code rename keeps its real owner.
func TestParseBoundaryNonClaims(t *testing.T) {
	for _, line := range []string{
		"rename Foo to Bar", // the rename feature's line
		"delete my account",
		"how do i delete a file",
		"remove the plum case from Describe",
		"move the meeting to friday",
		"what does delete mean",
	} {
		if b, ok := parseBoundary(line); ok {
			t.Errorf("parseBoundary(%q) claimed as %+v — not a disk-shaped ask", line, b)
		}
	}
	if _, ok := parseRenameSentence("rename Foo to Bar"); !ok {
		t.Error("the rename feature lost its own sentence")
	}
	if runCreate("rename Foo to Bar", yes) {
		t.Error("runCreate claimed a code rename")
	}
}

// Yes at the gate: the folder appears; the plan was shown first.
func TestCreateFolderAccept(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	var handled bool
	out := captureStdout(t, func() { handled = runCreate("create a folder jim", yes) })
	if !handled {
		t.Fatal("folder ask not claimed")
	}
	if !exists(filepath.Join(root, "jim")) {
		t.Fatalf("folder not created; output:\n%s", out)
	}
	if !strings.Contains(out, "◆ CREATE") || !strings.Contains(out, "jim/") {
		t.Errorf("plan with the exact path was not shown:\n%s", out)
	}
	if !strings.Contains(out, "Done — created jim/") {
		t.Errorf("success not reported:\n%s", out)
	}
}

// No (and an EOF where the answer would be) writes nothing.
func TestCreateDeclineWritesNothing(t *testing.T) {
	for name, ask := range map[string]editAsk{"no": no, "eof": eof} {
		root := t.TempDir()
		withCreateRoot(t, root)
		var handled bool
		out := captureStdout(t, func() { handled = runCreate("create a file in folder jim with a webserver", ask) })
		if !handled {
			t.Fatalf("%s: ask not claimed", name)
		}
		if exists(filepath.Join(root, "jim")) {
			t.Errorf("%s: something was created despite the gate answer; output:\n%s", name, out)
		}
		if !strings.Contains(out, "Not created — nothing was written") {
			t.Errorf("%s: decline not acknowledged:\n%s", name, out)
		}
	}
}

// The second screenshot line, verbatim: the file names itself
// from the kind, and folder + file ride ONE plan behind ONE gate.
func TestCreateFileInNewFolderWebserver(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	var handled bool
	out := captureStdout(t, func() { handled = runCreate("create a file in folder jim with a webserver", yes) })
	if !handled {
		t.Fatal("webserver ask not claimed")
	}
	full := filepath.Join(root, "jim", "main.go")
	if !exists(full) {
		t.Fatalf("webserver file not created; output:\n%s", out)
	}
	body := readBack(t, root, "jim/main.go")
	for _, marker := range []string{"package main", "net/http", "http.HandleFunc", "ListenAndServe"} {
		if !strings.Contains(body, marker) {
			t.Errorf("webserver file missing %s:\n%s", marker, body)
		}
		// The gate shows the FULL content before the yes — the
		// plan is the only other place these markers can appear.
		if !strings.Contains(out, marker) {
			t.Errorf("plan did not show %s before the gate:\n%s", marker, out)
		}
	}
	if !strings.Contains(out, "it builds") {
		t.Errorf("verification not reported (the flow compiles the file):\n%s", out)
	}
	if exists(filepath.Join(root, "jim", ".gollemer-create-check")) {
		t.Error("verification binary left behind")
	}
}

// A kindless .go file gets its package clause from the siblings in
// an existing package folder, and package main in a bare root.
func TestCreateKindlessGoPackageClause(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	writeFile(t, root, "shop/shop.go", "package shop\n\n// Greet says hi.\nfunc Greet() string { return \"hi\" }\n")
	if !runCreate("create a file extra.go in folder shop", yes) {
		t.Fatal("kindless file ask not claimed")
	}
	if got := readBack(t, root, "shop/extra.go"); got != "package shop\n" {
		t.Errorf("extra.go = %q, want the sibling package clause", got)
	}

	root2 := t.TempDir()
	withCreateRoot(t, root2)
	out := captureStdout(t, func() { runCreate("create a file main.go", yes) })
	if got := readBack(t, root2, "main.go"); got != "package main\n" {
		t.Errorf("main.go = %q, want package main; output:\n%s", got, out)
	}
	if !strings.Contains(out, "Done — created main.go") {
		t.Errorf("bare main clause not accepted as a starting point:\n%s", out)
	}
}

// Existing targets are refused, untouched — file and folder alike.
func TestCreateExistingRefused(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	writeFile(t, root, "jim/keep.txt", "mine\n")
	writeFile(t, root, "notes.txt", "mine\n")
	for _, line := range []string{"create a folder jim", "create a file notes.txt"} {
		var handled bool
		out := captureStdout(t, func() { handled = runCreate(line, yes) })
		if !handled {
			t.Fatalf("%q not claimed", line)
		}
		if !strings.Contains(out, "already there") {
			t.Errorf("%q: refusal not said:\n%s", line, out)
		}
	}
	if got := readBack(t, root, "notes.txt"); got != "mine\n" {
		t.Errorf("existing file changed: %q", got)
	}
	if !exists(filepath.Join(root, "jim", "keep.txt")) {
		t.Error("existing folder contents disturbed")
	}
}

// Escapes — ".." and absolute paths outside the root — are
// refused, and nothing appears outside.
func TestCreateEscapeRefused(t *testing.T) {
	parent := t.TempDir()
	root := filepath.Join(parent, "proj")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	withCreateRoot(t, root)
	absOutside := filepath.Join(parent, "abs-evil")
	for _, line := range []string{"create a folder ../evil", "create a folder " + absOutside} {
		var handled bool
		out := captureStdout(t, func() { handled = runCreate(line, yes) })
		if !handled {
			t.Fatalf("%q not claimed", line)
		}
		if !strings.Contains(out, "Not creating that") {
			t.Errorf("%q: refusal not said:\n%s", line, out)
		}
	}
	if exists(filepath.Join(parent, "evil")) || exists(absOutside) {
		t.Error("something was created outside the root")
	}
}

// A symlinked folder inside the project must not smuggle a file
// outside it.
func TestCreateSymlinkEscapeRefused(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(root, "link")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	withCreateRoot(t, root)
	out := captureStdout(t, func() { runCreate("create a file x.go in folder link", yes) })
	if !strings.Contains(out, "Not creating that") {
		t.Errorf("symlink escape not refused:\n%s", out)
	}
	if exists(filepath.Join(outside, "x.go")) {
		t.Error("file created outside the root through the symlink")
	}
}

// A webserver asked into an existing library package is refused
// before the gate: it could never compile where it lands.
func TestCreateWebserverInLibraryPackageRefused(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	writeFile(t, root, "lib/lib.go", "package lib\n\n// Thing is a thing.\nfunc Thing() int { return 1 }\n")
	out := captureStdout(t, func() { runCreate("create a file srv.go in folder lib with a webserver", yes) })
	if !strings.Contains(out, "main package of its own") {
		t.Errorf("library-package refusal not said:\n%s", out)
	}
	if exists(filepath.Join(root, "lib", "srv.go")) {
		t.Error("webserver written into a library package")
	}
}

// A creation whose package doesn't build is removed again —
// exactly what the operation made, nothing of the user's.
func TestCreateVerifyFailureRemoves(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	writeFile(t, root, "go.mod", "module example.com/verifyfail\n\ngo 1.21\n")
	writeFile(t, root, "pkg/broken.go", "package pkg\n\nfunc Broken( {\n")
	out := captureStdout(t, func() { runCreate("create a file ok.go in folder pkg", yes) })
	if exists(filepath.Join(root, "pkg", "ok.go")) {
		t.Errorf("unbuildable creation left in place; output:\n%s", out)
	}
	if !strings.Contains(out, "doesn't build") || !strings.Contains(out, "removed it again") {
		t.Errorf("rollback not reported with the reason:\n%s", out)
	}
	if !exists(filepath.Join(root, "pkg", "broken.go")) {
		t.Error("the user's own file was removed — only the creation may go")
	}
}

// Inside a module, a kindless file joining a healthy library
// package is verified by building that package.
func TestCreateInModuleLibraryPackage(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	writeFile(t, root, "go.mod", "module example.com/modlib\n\ngo 1.21\n")
	writeFile(t, root, "pkg/lib.go", "package pkg\n\n// One returns one.\nfunc One() int { return 1 }\n")
	out := captureStdout(t, func() { runCreate("create a file extra.go in folder pkg", yes) })
	if got := readBack(t, root, "pkg/extra.go"); got != "package pkg\n" {
		t.Errorf("extra.go = %q, want the package clause; output:\n%s", got, out)
	}
	if !strings.Contains(out, "it builds") {
		t.Errorf("package build not reported:\n%s", out)
	}
	if exists(filepath.Join(root, "pkg", ".gollemer-create-check")) {
		t.Error("verification binary left behind")
	}
}

// Boundary answers answer — and perform nothing.
func TestCreateBoundaryAnswersPerformNothing(t *testing.T) {
	root := t.TempDir()
	withCreateRoot(t, root)
	writeFile(t, root, "server.go", "package main\n")
	var handled bool
	out := captureStdout(t, func() { handled = runCreate("delete the file server.go", yes) })
	if !handled {
		t.Fatal("delete ask not claimed")
	}
	if !exists(filepath.Join(root, "server.go")) {
		t.Error("the boundary answer deleted the file")
	}
	if !strings.Contains(out, "can't delete") || !strings.Contains(out, "rm server.go") {
		t.Errorf("boundary answer incomplete:\n%s", out)
	}

	out = captureStdout(t, func() { handled = runCreate("move the file server.go to folder jim", yes) })
	if !handled {
		t.Fatal("move ask not claimed")
	}
	if !exists(filepath.Join(root, "server.go")) || exists(filepath.Join(root, "jim")) {
		t.Error("the boundary answer moved something")
	}
	if !strings.Contains(out, "mv server.go jim") {
		t.Errorf("move boundary missing the command:\n%s", out)
	}
}

// With no project open, the root falls back to the folder the chat
// started in.
func TestCreateRootFallback(t *testing.T) {
	root := t.TempDir()
	oldProj, oldRoot, oldURoot := lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot
	t.Cleanup(func() { lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot = oldProj, oldRoot, oldURoot })
	lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot = nil, "", root
	out := captureStdout(t, func() { runCreate("create a folder viafallback", yes) })
	if !exists(filepath.Join(root, "viafallback")) {
		t.Errorf("fallback root not used; output:\n%s", out)
	}
}
