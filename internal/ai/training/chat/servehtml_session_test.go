package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// End-to-end tests for the serve-a-page recipe, driven through
// the unified loop's claim chain in its own order (claimChain,
// addserver_session_test.go), in scratch projects on disk —
// the brief's four states: an empty project, a page that
// already exists, a Go target already serving, and a gate
// decline that must write nothing.

// emptyServeRoot is the empty project: a go.mod and nothing
// else — no Go packages at all.
func emptyServeRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/emptyserve\n\ngo 1.21\n")
	return dir
}

func fileExists(t *testing.T, dir, rel string) bool {
	t.Helper()
	_, err := os.Stat(filepath.Join(dir, rel))
	return err == nil
}

// The headline case: an empty project, the brief's own words,
// both artifacts, the build verified.
func TestServeHTMLEmptyProject(t *testing.T) {
	root := emptyServeRoot(t)
	withChatSession(t, root, false)
	profile := &UserProfile{}
	line := "serve an html file"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	requireNoAnalyzeOutput(t, line, out)
	if !fileExists(t, root, "index.html") {
		t.Fatalf("the page was not created:\n%s", out)
	}
	if got := readBack(t, root, "index.html"); !strings.Contains(got, "TODO: your page") {
		t.Errorf("the page is not the starter page:\n%s", got)
	}
	goSrc := readBack(t, root, "main.go")
	for _, want := range []string{`http.HandleFunc("/index.html"`, `http.ServeFile(w, r, "index.html")`, `http.ListenAndServe(":8080", nil)`} {
		if !strings.Contains(goSrc, want) {
			t.Errorf("the program is missing %q:\n%s", want, goSrc)
		}
	}
	if n := strings.Count(goSrc, "ListenAndServe"); n != 1 {
		t.Errorf("the program has %d ListenAndServe calls, want 1", n)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
	if !strings.Contains(out, "Run it from the project root") {
		t.Errorf("the plan does not say where to run it from:\n%s", out)
	}
}

// John's own phrasing of the ask, aimed at a folder created
// through the chat itself, in his analyzed-first session state.
func TestServeHTMLFolderTargetThroughChain(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	line := "build a server that serves index.html to jim"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	requireNoAnalyzeOutput(t, line, out)
	if got := readBack(t, root, "jim/index.html"); !strings.Contains(got, "TODO: your page") {
		t.Errorf("jim/index.html is not the starter page:\n%s", got)
	}
	goSrc := readBack(t, root, "jim/main.go")
	if !strings.Contains(goSrc, `http.ServeFile(w, r, "index.html")`) {
		t.Errorf("jim/main.go does not serve the page:\n%s", goSrc)
	}
	if !strings.Contains(out, "Run it from the jim folder") {
		t.Errorf("the plan does not say to run it from jim:\n%s", out)
	}
}

// serveProgramRoot is a project whose program lives at its
// root: a go.mod and a root main.go the test writes itself.
// (liveSessionRoot's doc.go is a library package — a root
// main.go beside it would be two packages in one folder, and
// the spine would rightly refuse to build it.)
func serveProgramRoot(t *testing.T) string {
	t.Helper()
	return emptyServeRoot(t)
}

// The page already exists: untouched, byte for byte; only the
// wiring is written, and the plan says so.
func TestServeHTMLExistingPageUntouched(t *testing.T) {
	root := serveProgramRoot(t)
	mine := "<html><body>my own words</body></html>\n"
	writeFile(t, root, "main.go", "package main\n")
	writeFile(t, root, "index.html", mine)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	line := "serve index.html"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	if got := readBack(t, root, "index.html"); got != mine {
		t.Errorf("the existing page was overwritten:\n%s", got)
	}
	if !strings.Contains(out, "already exists — it is not changed") {
		t.Errorf("the plan does not say the page is untouched:\n%s", out)
	}
	if got := readBack(t, root, "main.go"); !strings.Contains(got, "http.ServeFile") {
		t.Errorf("the wiring was not written:\n%s", got)
	}
}

// The Go target already runs a server (the shared component):
// the recipe composes — one registration, no second server.
func TestServeHTMLComposesWithRunningServer(t *testing.T) {
	root := serveProgramRoot(t)
	writeFile(t, root, "main.go", analyze.WebServerSource)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	line := "add a server that serves page.html to main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	goSrc := readBack(t, root, "main.go")
	if n := strings.Count(goSrc, "ListenAndServe"); n != 1 {
		t.Errorf("composed main.go has %d ListenAndServe calls, want 1:\n%s", n, goSrc)
	}
	if !strings.Contains(goSrc, `http.HandleFunc("/page.html"`) ||
		!strings.Contains(goSrc, `"hello — your webserver is running"`) {
		t.Errorf("composition lost code or the registration:\n%s", goSrc)
	}
	if !fileExists(t, root, "page.html") {
		t.Errorf("the page was not created:\n%s", out)
	}
	if !strings.Contains(out, "no second ListenAndServe is added") {
		t.Errorf("the plan does not own the composition out loud:\n%s", out)
	}
}

// A server on a custom mux: claimed, refused with the reason
// and the line to paste — and nothing written, not even the
// page.
func TestServeHTMLCustomMuxRefusedWritesNothing(t *testing.T) {
	root := serveProgramRoot(t)
	muxSrc := `package main

import (
	"fmt"
	"net/http"
)

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "hi")
	})
	if err := http.ListenAndServe(":8080", mux); err != nil {
		fmt.Println("stopped:", err)
	}
}
`
	writeFile(t, root, "main.go", muxSrc)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	line := "serve index.html to main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	if !strings.Contains(out, "won't add a second ListenAndServe") {
		t.Errorf("the refusal does not name the reason:\n%s", out)
	}
	if got := readBack(t, root, "main.go"); got != muxSrc {
		t.Errorf("the refused ask changed main.go:\n%s", got)
	}
	if fileExists(t, root, "index.html") {
		t.Errorf("the refused ask created the page")
	}
}

// A gate decline writes nothing at all — neither half.
func TestServeHTMLDeclineWritesNothing(t *testing.T) {
	root := emptyServeRoot(t)
	withChatSession(t, root, false)
	profile := &UserProfile{}
	line := "serve an html file"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, no, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	if !strings.Contains(out, "Not applied") {
		t.Errorf("the decline was not reported:\n%s", out)
	}
	if fileExists(t, root, "index.html") || fileExists(t, root, "main.go") {
		t.Errorf("the decline wrote a file anyway")
	}
}

// The started-but-unnamed target: claimed, answered in serve
// terms, nothing written.
func TestServeHTMLUnnamedTargetAnswered(t *testing.T) {
	root := emptyServeRoot(t)
	withChatSession(t, root, false)
	profile := &UserProfile{}
	line := "serve index.html to the file"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	if !strings.Contains(out, "doesn't finish") {
		t.Errorf("the answer is not in serve terms:\n%s", out)
	}
	if fileExists(t, root, "index.html") || fileExists(t, root, "main.go") {
		t.Errorf("the unnamed-target ask wrote a file")
	}
}

// The neighbours' lines through the whole chain: the concept
// book's static-files and html-template sentences are claimed
// by their own claim (declined at its gate here), never by
// this recipe — no page appears, and no ServeFile wiring is
// written under a serve-html plan.
func TestServeHTMLNeighboursKeepTheirClaims(t *testing.T) {
	for _, line := range []string{
		"add static files to main.go",
		"add an html template to main.go",
	} {
		root := serveProgramRoot(t)
		writeFile(t, root, "main.go", "package main\n\nfunc main() {}\n")
		withChatSession(t, root, true)
		profile := &UserProfile{}
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, no, profile) })
		if !handled {
			t.Errorf("%q was not claimed by the chain at all:\n%s", line, out)
			continue
		}
		if strings.Contains(out, "Serve index.html") || strings.Contains(out, "starter page") {
			t.Errorf("%q was answered by the serve-html recipe:\n%s", line, out)
		}
		if fileExists(t, root, "index.html") {
			t.Errorf("%q created a page — the recipe stole the line", line)
		}
	}
}
