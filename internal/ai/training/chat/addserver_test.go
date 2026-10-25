package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// withServerProject points the chat globals at dir the way
// creations do (root set, no analyzed project) so runAddServer
// resolves its root and analyzes quietly itself.
func withServerProject(t *testing.T, dir string) {
	t.Helper()
	oldProj, oldRoot, oldURoot := lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot
	t.Cleanup(func() { lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot = oldProj, oldRoot, oldURoot })
	lastAnalyzeProject = nil
	lastAnalyzeRoot = dir
	unifiedProjectRoot = dir
}

// serverFixture is the live-session shape, generalized: a module
// whose jim/ folder holds whatever main.go the test writes. The
// root package file matters: with jim as the ONLY package, the
// spine's `go build ./...` would try to write a "jim" executable
// into the root and collide with the jim/ directory — real
// projects (and John's) have more than the one package.
func serverFixture(t *testing.T, jimMain string) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/addsrv\n\ngo 1.21\n")
	writeFile(t, dir, "doc.go", "package addsrv\n")
	writeFile(t, dir, "jim/main.go", jimMain)
	return dir
}

// The grammar parses — and only the grammar.
func TestParseAddServerForms(t *testing.T) {
	claims := map[string]string{
		"add a webserver to the jim/main.go": "jim/main.go",
		"add a webserver to jim/":            "jim/",
		"add webserver to jim/main.go":       "jim/main.go",
		"put a web server into jim/main.go":  "jim/main.go",
		"add a web server in jim":            "jim",
		"Add a webserver to jim/main.go.":    "jim/main.go",
	}
	for line, want := range claims {
		got, ok := parseAddServer(line)
		if !ok {
			t.Errorf("parseAddServer(%q) not claimed", line)
			continue
		}
		if got != want {
			t.Errorf("parseAddServer(%q) = %q, want %q", line, got, want)
		}
	}
	for _, line := range []string{
		"i want to build a webserver",           // resources' goal statement
		"create a file srv.go with a webserver", // creations' line
		"analyze jim/",                          // the analyzer's line
		"what does main do",                     // a question
		"add a case for plum to Describe",       // an assembled edit
		"add a webserver",                       // no target
		"add a webserver to",                    // no target
		"add a server to jim/main.go",           // not the component's name
		"add a webserver for me",                // no to/in/into
		"the webserver is down",                 // not an imperative
	} {
		if got, ok := parseAddServer(line); ok {
			t.Errorf("parseAddServer(%q) claimed target %q — that sentence belongs to another feature", line, got)
		}
	}
	// The owners keep their lines.
	if _, _, ok := parseBuildGoal("i want to build a webserver"); !ok {
		t.Error("resources lost its goal statement")
	}
}

// The first screenshot line, verbatim, against the file it named:
// a bare package clause becomes the whole component, and the
// spine's build check passes.
func TestAddServerScreenshotFileLine(t *testing.T) {
	root := serverFixture(t, "package main\n")
	withServerProject(t, root)
	var handled bool
	out := captureStdout(t, func() { handled = runAddServer("add a webserver to the jim/main.go", yes) })
	if !handled {
		t.Fatal("screenshot line not claimed")
	}
	body := readBack(t, root, "jim/main.go")
	if body != analyze.WebServerSource {
		t.Errorf("bare clause did not become the shared component:\n%s", body)
	}
	for _, marker := range []string{"net/http", "http.HandleFunc", "ListenAndServe"} {
		if !strings.Contains(body, marker) {
			t.Errorf("result missing %s", marker)
		}
		if !strings.Contains(out, marker) {
			t.Errorf("the pre-gate diff did not show %s:\n%s", marker, out)
		}
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
	if _, ok := buildProject(root); !ok {
		t.Error("the package does not build after the edit")
	}
}

// The second screenshot line, verbatim: a folder target resolves
// to its main.go instead of dumping an analyze report.
func TestAddServerScreenshotFolderLine(t *testing.T) {
	root := serverFixture(t, "package main\n")
	withServerProject(t, root)
	out := captureStdout(t, func() { runAddServer("add a webserver to jim/", yes) })
	if got := readBack(t, root, "jim/main.go"); got != analyze.WebServerSource {
		t.Errorf("folder target did not resolve to jim/main.go and fill it:\n%s\noutput:\n%s", got, out)
	}
}

// No at the gate: the file is byte-identical.
func TestAddServerDeclineByteIdentical(t *testing.T) {
	root := serverFixture(t, "package main\n")
	withServerProject(t, root)
	before := readBack(t, root, "jim/main.go")
	out := captureStdout(t, func() { runAddServer("add a webserver to the jim/main.go", no) })
	if got := readBack(t, root, "jim/main.go"); got != before {
		t.Errorf("declined edit changed the file: %q", got)
	}
	if !strings.Contains(out, "Not applied — every file is exactly as it was") {
		t.Errorf("decline not acknowledged:\n%s", out)
	}
}

// An empty main is filled with the component's body, imports
// added, and the project builds. (One patient per fixture
// project: a package-main file without a main never builds, so
// two unedited patients in one project would fail each other's
// build gate for reasons that predate the edit.)
func TestAddServerEmptyMainFilled(t *testing.T) {
	root := t.TempDir()
	withServerProject(t, root)
	writeFile(t, root, "go.mod", "module example.com/addsrv2\n\ngo 1.21\n")
	writeFile(t, root, "doc.go", "package addsrv2\n")
	writeFile(t, root, "app/main.go", "package main\n\nfunc main() {\n}\n")

	out := captureStdout(t, func() { runAddServer("add a webserver to app/main.go", yes) })
	body := readBack(t, root, "app/main.go")
	if !strings.Contains(body, "http.ListenAndServe") || !strings.Contains(body, `"net/http"`) {
		t.Errorf("empty main not filled:\n%s\noutput:\n%s", body, out)
	}
	if _, ok := buildProject(root); !ok {
		t.Error("project does not build after the edit")
	}
}

// A main-less package-main file grows a main carrying the
// component; its existing code is undisturbed.
func TestAddServerNoMainAppends(t *testing.T) {
	root := t.TempDir()
	withServerProject(t, root)
	writeFile(t, root, "go.mod", "module example.com/addsrv2b\n\ngo 1.21\n")
	writeFile(t, root, "doc.go", "package addsrv2b\n")
	writeFile(t, root, "tool/tool.go", "package main\n\n// Helper helps.\nfunc Helper() int { return 1 }\n")

	out := captureStdout(t, func() { runAddServer("add a webserver to tool/tool.go", yes) })
	body := readBack(t, root, "tool/tool.go")
	if !strings.Contains(body, "func main()") || !strings.Contains(body, "http.ListenAndServe") {
		t.Errorf("main-less file did not grow a main:\n%s\noutput:\n%s", body, out)
	}
	if !strings.Contains(body, "func Helper()") {
		t.Errorf("existing code disturbed:\n%s", body)
	}
	if _, ok := buildProject(root); !ok {
		t.Error("project does not build after the edit")
	}
}

// A main that already works is refused — with the component's
// exact lines handed over — and the file is untouched.
func TestAddServerNonEmptyMainRefused(t *testing.T) {
	src := "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"already alive\")\n}\n"
	root := serverFixture(t, src)
	withServerProject(t, root)
	out := captureStdout(t, func() { runAddServer("add a webserver to jim/main.go", yes) })
	for _, want := range []string{"already does its own work", "http.HandleFunc", "http.ListenAndServe", "hello — your webserver is running"} {
		if !strings.Contains(out, want) {
			t.Errorf("by-hand refusal missing %q:\n%s", want, out)
		}
	}
	if got := readBack(t, root, "jim/main.go"); got != src {
		t.Errorf("refused edit changed the file:\n%s", got)
	}
}

// A library file is refused with the nearest real capability —
// the project's own main file when it has one.
func TestAddServerLibraryRefused(t *testing.T) {
	root := t.TempDir()
	withServerProject(t, root)
	writeFile(t, root, "go.mod", "module example.com/addsrv3\n\ngo 1.21\n")
	writeFile(t, root, "main.go", "package main\n\nfunc main() {}\n")
	writeFile(t, root, "lib/lib.go", "package lib\n\n// Thing is a thing.\nfunc Thing() int { return 1 }\n")
	out := captureStdout(t, func() { runAddServer("add a webserver to lib/lib.go", yes) })
	if !strings.Contains(out, "package lib") || !strings.Contains(out, "package main") {
		t.Errorf("library refusal not said:\n%s", out)
	}
	if !strings.Contains(out, "add a webserver to main.go") {
		t.Errorf("library refusal does not name the project's main file:\n%s", out)
	}
	if got := readBack(t, root, "lib/lib.go"); !strings.Contains(got, "package lib") || strings.Contains(got, "ListenAndServe") {
		t.Errorf("library file disturbed:\n%s", got)
	}
}

// A file that already serves is refused.
func TestAddServerAlreadyServingRefused(t *testing.T) {
	root := serverFixture(t, analyze.WebServerSource)
	withServerProject(t, root)
	out := captureStdout(t, func() { runAddServer("add a webserver to jim/main.go", yes) })
	if !strings.Contains(out, "already has a webserver") {
		t.Errorf("already-serving refusal not said:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); got != analyze.WebServerSource {
		t.Error("already-serving file changed")
	}
}

// Folder targets: no suitable file names the create command;
// several candidates are named, never silently picked; one
// package-main file under another name resolves.
func TestAddServerFolderResolution(t *testing.T) {
	root := t.TempDir()
	withServerProject(t, root)
	writeFile(t, root, "go.mod", "module example.com/addsrv4\n\ngo 1.21\n")

	// No Go files at all in bob/ — the create line is the answer.
	writeFile(t, root, "bob/notes.txt", "hi\n")
	out := captureStdout(t, func() { runAddServer("add a webserver to bob/", yes) })
	if !strings.Contains(out, "create a file bob/main.go with a webserver") {
		t.Errorf("no-suitable-file refusal does not name the create command:\n%s", out)
	}
	if exists(filepath.Join(root, "bob", "main.go")) {
		t.Error("refusal created the file anyway")
	}

	// Two package-main files in duo/ — ambiguity is named. (The
	// pair is shaped to build as-is, so only the resolution —
	// never a build failure — can stop the uno/ edit below.)
	writeFile(t, root, "duo/a.go", "package main\n\nfunc main() {}\n")
	writeFile(t, root, "duo/b.go", "package main\n\n// Helper helps.\nfunc Helper() {}\n")
	out = captureStdout(t, func() { runAddServer("add a webserver to duo/", yes) })
	if !strings.Contains(out, "more than one main-package file") || !strings.Contains(out, "a.go") || !strings.Contains(out, "b.go") {
		t.Errorf("ambiguous-folder refusal does not name the candidates:\n%s", out)
	}

	// One package-main file under another name in uno/ — resolves.
	writeFile(t, root, "uno/app.go", "package main\n")
	out = captureStdout(t, func() { runAddServer("add a webserver to uno/", yes) })
	if got := readBack(t, root, "uno/app.go"); got != analyze.WebServerSource {
		t.Errorf("single-candidate folder did not resolve to app.go:\n%s\noutput:\n%s", got, out)
	}
}

// A missing file names the creation that would write it whole.
func TestAddServerMissingFileSuggestsCreate(t *testing.T) {
	root := serverFixture(t, "package main\n")
	withServerProject(t, root)
	out := captureStdout(t, func() { runAddServer("add a webserver to nope/main.go", yes) })
	if !strings.Contains(out, "create a file nope/main.go with a webserver") {
		t.Errorf("missing-file answer does not name the create command:\n%s", out)
	}
}

// Escapes out of the root are refused and create nothing.
func TestAddServerEscapeRefused(t *testing.T) {
	parent := t.TempDir()
	root := filepath.Join(parent, "proj")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	withServerProject(t, root)
	writeFile(t, parent, "outside.go", "package main\n")
	out := captureStdout(t, func() { runAddServer("add a webserver to ../outside.go", yes) })
	if !strings.Contains(out, "Not adding that") {
		t.Errorf("escape not refused:\n%s", out)
	}
	if got := readBack(t, parent, "outside.go"); got != "package main\n" {
		t.Errorf("file outside the root was edited: %q", got)
	}
}

// The non-collision set, end to end: analyze questions and the
// neighbouring write claims behave exactly as before.
func TestAddServerNonCollisions(t *testing.T) {
	root := serverFixture(t, "package main\n")
	withServerProject(t, root)
	for _, line := range []string{
		"analyze jim/",
		"what does main do",
		"i want to build a webserver",
		"create a file srv.go with a webserver",
		"how do i build a webserver",
	} {
		if runAddServer(line, yes) {
			t.Errorf("runAddServer claimed %q — not its sentence", line)
		}
	}
	// Creations still own theirs (declined here, so nothing is
	// written — the claim is what is being proven).
	if !runCreate("create a file srv.go with a webserver", no) {
		t.Error("creations lost their line")
	}
	if exists(filepath.Join(root, "srv.go")) {
		t.Error("the declined creation wrote a file")
	}
}
