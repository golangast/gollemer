package chat

import (
	"strings"
	"testing"
)

// The serve-a-file claim: John's four live wordings and the
// family's natural forms parse exactly; the neighbours'
// lines (the concept book's handler rows, serve-static,
// serve-html, the wiring family) never parse here; and the
// flow lands the handler + registration through the spine,
// asks its one question when the file is ambiguous, and
// answers honestly when the file or the folder isn't there.

func TestParseServeFileJohnsWordings(t *testing.T) {
	cases := []struct {
		line string
		want serveFileRequest
	}{
		{"add a handler that serves a file in jim",
			serveFileRequest{target: "jim"}},
		{"add a handler named jim that returns a file in jim",
			serveFileRequest{name: "jim", target: "jim"}},
		{"add a handler named jim that returns a file in jim/main.go",
			serveFileRequest{name: "jim", target: "jim/main.go"}},
		{"serve the file index.html from a handler in jim",
			serveFileRequest{file: "index.html", target: "jim"}},
		{"add a handler that serves logo.png in jim",
			serveFileRequest{file: "logo.png", target: "jim"}},
		{"add a handler that serves the file index.html in jim",
			serveFileRequest{file: "index.html", target: "jim"}},
		{"add a handler that serves an html file in jim",
			serveFileRequest{htmlPage: true, target: "jim"}},
		{"add a handler named Logo that serves assets/logo.png",
			serveFileRequest{name: "Logo", file: "assets/logo.png"}},
		{"serve the file from a handler in jim",
			serveFileRequest{target: "jim"}},
		{"add a handler that serves logo.png in jim at /img/logo",
			serveFileRequest{file: "logo.png", target: "jim", route: "/img/logo"}},
		{"make a handler named Data that returns data.json in jim",
			serveFileRequest{name: "Data", file: "data.json", target: "jim"}},
	}
	for _, c := range cases {
		got, ok := parseServeFile(c.line)
		if !ok {
			t.Errorf("parseServeFile(%q) declined, want %+v", c.line, c.want)
			continue
		}
		if got != c.want {
			t.Errorf("parseServeFile(%q) = %+v, want %+v", c.line, got, c.want)
		}
	}
}

func TestParseServeFileDeclinesNeighbours(t *testing.T) {
	lines := []string{
		// The book's handler rows carry a kind word before
		// "handler" — the book's lines, never this grammar's.
		"add a form handler named Contact to jim/main.go",
		"add a session handler to jim/main.go",
		"add a middleware handler to jim/main.go",
		// serve-html's and add-server's shapes.
		"add a server that serves index.html",
		"serve index.html to jim",
		"add a webserver to jim/main.go",
		// serve-static's folder lines.
		"serve static files from jim",
		"serve the static files in jim/temp",
		// The wiring family's lines (no "that serves" clause).
		"add a handler named Jake to jim/main.go",
		"connect the Jake handler to the webserver",
		// A file spec with no file shape to it.
		"add a handler that serves static files in jim",
		"add a handler that serves the logo in jim",
		// A longer tail is another ask glued on.
		"add a handler that serves logo.png in jim and a footer",
		// Questions keep their owners.
		"how do I serve a file",
		"what handler serves the logo",
	}
	for _, line := range lines {
		if got, ok := parseServeFile(line); ok {
			t.Errorf("parseServeFile(%q) claimed %+v — a neighbour's line", line, got)
		}
	}
}

// The theft surface at parse level for EVERY book row: the
// sweep's sentences must never be claimed by this grammar.
func TestParseServeFileStealsNoBookLine(t *testing.T) {
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
				if got, ok := parseServeFile(line); ok {
					t.Errorf("parseServeFile stole book row %q's line %q: %+v", entry.ID, line, got)
				}
			}
		}
	}
}

const serveFileJimServer = `package main

import "net/http"

func main() {
	http.ListenAndServe(":8080", nil)
}
`

func serveFileFixture(t *testing.T, files map[string]string) string {
	t.Helper()
	return staticAssetsFixture(t, staticAssetsFiles(files))
}

// The named-file landing: handler + registration composed
// onto jim's server in one gated plan, the partner
// confirmation naming the route, and the second ask the
// already-serving answer.
func TestServeFileLandingComposes(t *testing.T) {
	root := serveFileFixture(t, map[string]string{
		"jim/main.go":  serveFileJimServer,
		"jim/logo.png": "PNG",
	})
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("add a handler that serves logo.png in jim", yes, profile) })
	got := readBack(t, root, "jim/main.go")
	if !strings.Contains(got, `http.HandleFunc("/logo.png", LogoHandler)`) {
		t.Errorf("registration did not land:\n%s", got)
	}
	if !strings.Contains(got, `http.ServeFile(w, r, "jim/logo.png")`) {
		t.Errorf("handler did not land:\n%s", got)
	}
	if strings.Index(got, "HandleFunc") > strings.Index(got, "ListenAndServe") {
		t.Errorf("registration is not before the serve statement:\n%s", got)
	}
	if !strings.Contains(out, "Done — The handler LogoHandler is in jim/main.go — it serves jim/logo.png at /logo.png.") {
		t.Errorf("landing confirmation missing:\n%s", out)
	}
	if !strings.Contains(out, "content type") {
		t.Errorf("the ServeFile content-type note is missing:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("add a handler that serves logo.png in jim", yes, profile) })
	if !strings.Contains(out, "already serves") {
		t.Errorf("second ask was not the already-serving answer:\n%s", out)
	}
	if n := strings.Count(readBack(t, root, "jim/main.go"), "func LogoHandler("); n != 1 {
		t.Errorf("LogoHandler declared %d times, want 1", n)
	}
}

// John's exact third wording: the name is the sentence's
// ("jim"), the target is the file itself, and the unnamed
// file resolves to the folder's one servable file.
func TestServeFileJohnsFileTargetWording(t *testing.T) {
	root := serveFileFixture(t, map[string]string{
		"jim/main.go":    serveFileJimServer,
		"jim/index.html": "<h1>hi</h1>\n",
	})
	profile := &UserProfile{}
	captureStdout(t, func() { claimChain("add a handler named jim that returns a file in jim/main.go", yes, profile) })
	got := readBack(t, root, "jim/main.go")
	if !strings.Contains(got, "func jim(") || !strings.Contains(got, `http.ServeFile(w, r, "jim/index.html")`) {
		t.Errorf("John's wording did not land the jim handler serving index.html:\n%s", got)
	}
	if !strings.Contains(got, `http.HandleFunc("/", jim)`) {
		t.Errorf("index.html should answer at its folder's address:\n%s", got)
	}
}

// The unnamed file with several candidates asks ONE
// question; the answer completes the original ask.
func TestServeFileUnnamedFileAsksOnce(t *testing.T) {
	root := serveFileFixture(t, map[string]string{
		"jim/main.go":   serveFileJimServer,
		"jim/logo.png":  "PNG",
		"jim/style.css": "body {}\n",
	})
	out := clarifyRun(t, "add a handler that serves a file in jim")
	if !strings.Contains(out, "which one should the handler serve") {
		t.Errorf("the ambiguity did not ask:\n%s", out)
	}
	if !hasPendingQuestion() {
		t.Fatal("no question is pending after the ambiguous ask")
	}
	if strings.Contains(readBack(t, root, "jim/main.go"), "ServeFile") {
		t.Error("something was written before the question was answered")
	}
	clarifyRun(t, "logo.png")
	if hasPendingQuestion() {
		t.Error("the answered question is still pending")
	}
	got := readBack(t, root, "jim/main.go")
	if !strings.Contains(got, `http.ServeFile(w, r, "jim/logo.png")`) {
		t.Errorf("the picked file was not served:\n%s", got)
	}
}

// The honest states: a file that isn't there, a folder
// that isn't there, and a target inside its own module.
func TestServeFileHonestRefusals(t *testing.T) {
	root := serveFileFixture(t, map[string]string{
		"jim/main.go":      serveFileJimServer,
		"app/go.mod":       "module example.com/app\n\ngo 1.21\n",
		"app/main.go":      "package main\n\nfunc main() {}\n",
		"app/index.html":   "<h1>app</h1>\n",
		"app/wasm_exec.js": "// wasm\n",
		"app/logo.png":     "PNG",
	})
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("add a handler that serves nope.png in jim", yes, profile) })
	if !strings.Contains(out, "no nope.png in jim") || !strings.Contains(out, "really there") {
		t.Errorf("missing-file answer wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("add a handler that serves logo.png in nowhere", yes, profile) })
	if !strings.Contains(out, "no folder nowhere") || !strings.Contains(out, "create a folder nowhere") {
		t.Errorf("missing-folder answer wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("add a handler that serves logo.png in app", yes, profile) })
	if !strings.Contains(out, "wasm app") || !strings.Contains(out, "serve the wasm app in app") {
		t.Errorf("nested-module answer wrong:\n%s", out)
	}
	if strings.Contains(readBack(t, root, "app/main.go"), "ServeFile") {
		t.Error("a handler was written into the wasm app's module")
	}
	if strings.Contains(readBack(t, root, "jim/main.go"), "ServeFile") {
		t.Error("a handler was written behind a refusal")
	}
}

// serveFileHandlerName's derivations, pinned.
func TestServeFileHandlerNameDerivation(t *testing.T) {
	cases := map[string]string{
		"logo.png":      "LogoHandler",
		"index.html":    "IndexHandler",
		"my-report.pdf": "MyReportHandler",
		"style.css":     "StyleHandler",
		"404.html":      "Serve404Handler",
		"data.json":     "DataHandler",
	}
	for base, want := range cases {
		if got := serveFileHandlerName(base); got != want {
			t.Errorf("serveFileHandlerName(%q) = %q, want %q", base, got, want)
		}
	}
}
