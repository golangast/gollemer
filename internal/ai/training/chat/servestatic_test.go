package chat

import (
	"strings"
	"testing"
)

// The serve-static grammar, at the parse level: the brief's
// ask forms claim, with John's own sentence among them, and
// the concept book's targeted lines are never stolen — the
// boundary is the "to <file>" tail this grammar never takes.
func TestParseServeStaticClaims(t *testing.T) {
	claims := []struct {
		line, folder string
	}{
		{"serve static files", ""},
		{"serve the static files", ""},
		{"serve static assets", ""},
		{"serve static files from jim/temp", "jim/temp"},
		{"serve static files from the jim/temp folder", "jim/temp"},
		{"add static file serving", ""},
		{"add static file serving from jim/temp", "jim/temp"},
		{"serve the jim/temp folder", "jim/temp"},
		{"serve the temp folder", "temp"},
		{"serve jim/temp folder", "jim/temp"},
		{"let the server serve images and css", ""},
		{"update the server to serve images, js, css", ""}, // John's words
		{"update the server to serve images and css", ""},
		{"make the server serve files", ""},
	}
	for _, tc := range claims {
		got, ok := parseServeStatic(tc.line)
		if !ok {
			t.Errorf("parseServeStatic(%q) not claimed", tc.line)
			continue
		}
		if got.folder != tc.folder {
			t.Errorf("parseServeStatic(%q) = %+v, want folder %q", tc.line, got, tc.folder)
		}
	}
}

func TestParseServeStaticNonClaims(t *testing.T) {
	for _, line := range []string{
		// The concept book's rows keep their claims: every
		// one carries the "to <file>" tail.
		"serve static files to main.go",
		"add static files to main.go",
		"add serve static files to jim/main.go",
		"add static file serving to main.go",
		"add an html template to main.go",
		// The knowledge question keeps the book's answer.
		"how do I serve static files",
		// The neighbours' sentences.
		"serve index.html",
		"serve index.html in main.go",
		"add a server that serves index.html",
		"add a webserver to main.go",
		// Shapes that name nothing, or carry a longer tail.
		"serve the folder",
		"serve static",
		"update the server to serve index.html to main.go",
		"let the server serve index.html",
		"update the server",
	} {
		if got, ok := parseServeStatic(line); ok {
			t.Errorf("parseServeStatic(%q) claimed %+v — not its sentence", line, got)
		}
	}
}

// staticAssetsFixture builds a fixture project and seats
// the chat in it (analyzed, chdir'd — the clarifyFixture
// pattern, so LinkPath renders the short project-relative
// paths the confirmations speak in).
func staticAssetsFixture(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, content := range files {
		writeFile(t, root, rel, content)
	}
	clarifyPhrasing(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	return root
}

// staticAssetsBase is the fixture module both families'
// tests share: a go.mod and a root package file.
var staticAssetsBase = map[string]string{
	"go.mod": "module example.com/staticassets\n\ngo 1.21\n",
	"doc.go": "package staticassets\n",
}

func staticAssetsFiles(extra map[string]string) map[string]string {
	out := map[string]string{}
	for rel, src := range staticAssetsBase {
		out[rel] = src
	}
	for rel, src := range extra {
		out[rel] = src
	}
	return out
}

const staticJimMain = `package main

import "net/http"

func Jake(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("jake"))
}

func main() {
	http.HandleFunc("/jake", Jake)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("root"))
	})
	http.ListenAndServe(":8080", nil)
}
`

// The theft surface, pinned at parse level for EVERY book
// row: the every-row sweep's sentences ("add a <name> to
// jim/main.go", and the declaration variant) must never be
// claimed by either new grammar, for any of a row's names.
// (The sweep itself also proves the rows still LAND; this
// pin isolates the one thing this package could break —
// claim order — without the sweep's per-row builds.)
func TestStaticAssetsParsersStealNoBookLine(t *testing.T) {
	book := defaultBookEntries(t)
	if len(book) == 0 {
		t.Fatal("the concept book is empty — the pin would prove nothing")
	}
	for _, entry := range book {
		for _, name := range entry.Names {
			for _, line := range []string{
				"add a " + name + " to jim/main.go",
				"add a " + name + " named Rowtest to jim/main.go",
				"add " + name + " to jim/main.go",
			} {
				if got, ok := parseWebpage(line); ok {
					t.Errorf("parseWebpage stole book row %q's line %q: %+v", entry.ID, line, got)
				}
				if got, ok := parseServeStatic(line); ok {
					t.Errorf("parseServeStatic stole book row %q's line %q: %+v", entry.ID, line, got)
				}
			}
		}
	}
}

// The jim fixture: John's server shape in jim/main.go, a
// temp folder beside it holding web files.
func staticJimFixture(t *testing.T) string {
	t.Helper()
	return staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go":        staticJimMain,
		"jim/temp/style.css": "body {}\n",
		"jim/temp/script.js": "console.log(1)\n",
	}))
}

const staticRegLineForJimTemp = `http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("jim/temp"))))`

// The pair journey: make the webpage in jim/temp, then
// John's serve-static sentence — the folder defaults to the
// webpage's folder from the session, no question asked.
func TestServeStaticSessionDefaultFolder(t *testing.T) {
	root := staticJimFixture(t)
	profile := &UserProfile{}
	captureStdout(t, func() { claimChain("add a basic webpage to jim/temp/index.html", yes, profile) })
	out := captureStdout(t, func() { claimChain("update the server to serve images, js, css", yes, profile) })
	if hasPendingQuestion() {
		t.Error("the session-default folder still asked a question")
	}
	got := readBack(t, root, "jim/main.go")
	if !strings.Contains(got, staticRegLineForJimTemp) {
		t.Errorf("registration did not land in jim/main.go:\n%s", got)
	}
	if strings.Index(got, staticRegLineForJimTemp) > strings.Index(got, "http.ListenAndServe") {
		t.Errorf("registration is not before the serve statement:\n%s", got)
	}
	if !strings.Contains(out, "Done — The server in jim/main.go now serves the jim/temp folder's files at /static/.") {
		t.Errorf("landing confirmation missing:\n%s", out)
	}
	// Asking again is a nothing-to-add answer, never a
	// second registration.
	out = captureStdout(t, func() { claimChain("serve static files", yes, profile) })
	if !strings.Contains(out, "already serves the jim/temp folder at /static/") {
		t.Errorf("second ask was not the already-serving answer:\n%s", out)
	}
	if n := strings.Count(readBack(t, root, "jim/main.go"), `"/static/"`); n != 1 {
		t.Errorf("the route was registered %d times, want exactly 1", n)
	}
}

// A named folder claims directly, in John's other form.
func TestServeStaticNamedFolder(t *testing.T) {
	root := staticJimFixture(t)
	profile := &UserProfile{}
	captureStdout(t, func() { claimChain("serve the jim/temp folder", yes, profile) })
	if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, staticRegLineForJimTemp) {
		t.Errorf("named-folder registration did not land:\n%s", got)
	}
}

// No session default and two candidate folders: ONE
// question, and the number answer completes the ask.
func TestServeStaticFolderQuestion(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go":        staticJimMain,
		"jim/temp/style.css": "body {}\n",
		"assets/app.css":     "body {}\n",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("serve static files", yes, profile) })
	for _, want := range []string{"Which folder should the server serve at /static/?", "1. assets", "2. jim/temp"} {
		if !strings.Contains(out, want) {
			t.Errorf("folder question missing %q:\n%s", want, out)
		}
	}
	if !hasPendingQuestion() {
		t.Fatal("the folder-less ask set no pending question")
	}
	captureStdout(t, func() { claimChain("2", yes, profile) })
	if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, staticRegLineForJimTemp) {
		t.Errorf("the answered ask did not register jim/temp:\n%s", got)
	}
}

// The refusals, plainly worded: no server (add one first),
// a missing folder (create it first), and a route already
// held by another handler (both named).
func TestServeStaticRefusalNoServer(t *testing.T) {
	staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/style.css": "body {}\n",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("serve the jim/temp folder", yes, profile) })
	if !strings.Contains(out, "no webserver in this project yet") || !strings.Contains(out, "add a webserver to") {
		t.Errorf("no-server refusal wrong:\n%s", out)
	}
}

func TestServeStaticRefusalMissingFolder(t *testing.T) {
	staticJimFixture(t)
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("serve the nope folder", yes, profile) })
	if !strings.Contains(out, "create a folder nope") {
		t.Errorf("missing-folder refusal wrong:\n%s", out)
	}
}

func TestServeStaticRefusalRouteTaken(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": `package main

import "net/http"

func Other(w http.ResponseWriter, r *http.Request) {}

func main() {
	http.HandleFunc("/static/", Other)
	http.ListenAndServe(":8080", nil)
}
`,
		"jim/temp/style.css": "body {}\n",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("serve the jim/temp folder", yes, profile) })
	if !strings.Contains(out, "/static/ is already taken") || !strings.Contains(out, "Other") {
		t.Errorf("route-taken refusal wrong:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); strings.Contains(got, "FileServer") {
		t.Errorf("a file server was added over a taken route:\n%s", got)
	}
}
