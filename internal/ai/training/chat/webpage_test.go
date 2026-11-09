package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The webpage grammar, at the parse level: the brief's ask
// forms claim with the slots the flow needs. John's exact
// line is the regression pin — it was refused live on
// 2026-11-09 because no piece owned it.
func TestParseWebpageClaims(t *testing.T) {
	claims := []struct {
		line, template, file string
		fileMissing          bool
	}{
		{"add a webpage to jim/temp/index.html", "", "jim/temp/index.html", false}, // John's line, verbatim
		{"create a webpage at jim/temp/index.html", "", "jim/temp/index.html", false},
		{"add a responsive webpage to jim/temp/index.html", "responsive", "jim/temp/index.html", false},
		{"add a basic webpage to index.html", "basic", "index.html", false},
		{"add a simple web page to index.html", "basic", "index.html", false},
		{"add a webpage with assets to jim/temp/index.html", "assets", "jim/temp/index.html", false},
		{"make a new webpage in jim/temp/index.html", "", "jim/temp/index.html", false},
		{"put a webpage into the file page.html", "", "page.html", false},
		{"create a webpage to jim/temp/index.html", "", "jim/temp/index.html", false},
		{"add a webpage", "", "", true},
		{"add a responsive webpage", "responsive", "", true},
		{"add a webpage to", "", "", true},
		{"add a webpage to the file", "", "", true},
	}
	for _, tc := range claims {
		got, ok := parseWebpage(tc.line)
		if !ok {
			t.Errorf("parseWebpage(%q) not claimed", tc.line)
			continue
		}
		if got.template != tc.template || got.file != tc.file || got.fileMissing != tc.fileMissing {
			t.Errorf("parseWebpage(%q) = %+v, want template=%q file=%q fileMissing=%v", tc.line, got, tc.template, tc.file, tc.fileMissing)
		}
	}
}

// The canonical sentences a clarifying answer completes
// into parse back with exactly their slots.
func TestWebpageCanonicalRoundTrip(t *testing.T) {
	for _, template := range []string{"", "basic", "responsive", "assets"} {
		line := webpageCanonical(template, "jim/temp/index.html")
		got, ok := parseWebpage(line)
		if !ok || got.template != template || got.file != "jim/temp/index.html" || got.fileMissing {
			t.Errorf("canonical %q parsed as %+v, %v", line, got, ok)
		}
	}
}

// Not this claim's: above all the page family's kind
// sentences — the boundary is pinned from BOTH sides (the
// family's own parses must keep them too).
func TestParseWebpageNonClaims(t *testing.T) {
	for _, line := range []string{
		// The page family's kinds keep their grammar.
		"create a homepage",
		"create a home page",
		"create a form page",
		"create an about page",
		"create a contact page",
		"create an item list page",
		"create a details page named item",
		// The neighbours' sentences.
		"add a form handler named Jake to main.go",
		"create a file jim/temp/index.html",
		"add a server that serves index.html",
		"serve index.html",
		"add a webserver to main.go",
		// Prose and compounds.
		"what is a webpage",
		"add a webpage and a server to main.go",
		"add a webpage to jim/temp/index.html and serve it",
		// Two templates in one sentence mean nothing single.
		"add a responsive webpage with assets to jim/temp/index.html",
	} {
		if got, ok := parseWebpage(line); ok {
			t.Errorf("parseWebpage(%q) claimed %+v — not its sentence", line, got)
		}
	}
	// The other side of the boundary: the page family's own
	// grammar still claims its kinds.
	for line, kind := range map[string]string{
		"create a homepage":    "home",
		"create a form page":   "form",
		"create an about page": "about",
	} {
		q, ok := parsePageFamily(line)
		if !ok || q.kind != kind {
			t.Errorf("parsePageFamily(%q) = %+v, %v — want kind %q", line, q, ok, kind)
		}
	}
	// And the family's grammar does not take the webpage
	// piece's sentence either.
	if q, ok := parsePageFamily("add a webpage to jim/temp/index.html"); ok {
		t.Errorf("parsePageFamily claimed the webpage line: %+v", q)
	}
}

// The live-session journey, end to end through the claim
// chain: the empty file created first (the way John's was),
// then his exact line, the template question, the number
// answer, the gate, the landing.
func TestWebpageJohnsSessionEndToEnd(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("create a folder jim/temp", yes, profile)
		claimChain("create a file jim/temp/index.html", yes, profile)
	})
	if got := readBack(t, root, "jim/temp/index.html"); got != "" {
		t.Fatalf("the created page file is not empty: %q\n%s", got, out)
	}
	out = captureStdout(t, func() {
		claimChain("add a webpage to jim/temp/index.html", yes, profile)
	})
	for _, want := range []string{"which template should it be?", "1. basic", "2. responsive", "3. with assets"} {
		if !strings.Contains(out, want) {
			t.Errorf("template question missing %q:\n%s", want, out)
		}
	}
	if !hasPendingQuestion() {
		t.Fatal("the unnamed template set no pending question")
	}
	out = captureStdout(t, func() {
		claimChain("2", yes, profile)
	})
	if hasPendingQuestion() {
		t.Error("the completed ask left a question pending")
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageContent("responsive", "index") {
		t.Errorf("the picked template did not land:\n%s", got)
	}
	if !strings.Contains(out, "Done — The responsive webpage is in jim/temp/index.html.") {
		t.Errorf("landing confirmation missing:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked that it's really there") {
		t.Errorf("verification sentence missing:\n%s", out)
	}
}

// A named template asks NO question and lands directly.
func TestWebpageNamedTemplateAsksNoQuestion(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/keep.css": "body {}\n",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a basic webpage to jim/temp/index.html", yes, profile)
	})
	if hasPendingQuestion() {
		t.Error("a named template set a pending question")
	}
	if strings.Contains(out, "which template") {
		t.Errorf("a named template still asked:\n%s", out)
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageContent("basic", "index") {
		t.Errorf("the named template did not land:\n%s", got)
	}
	if !strings.Contains(out, "Done — The basic webpage is in jim/temp/index.html.") {
		t.Errorf("landing confirmation missing:\n%s", out)
	}
}

// The file rules, said plainly and before any question: a
// page with words in it is never overwritten, a non-.html
// target refuses by name, and a missing folder is the
// create-folder two-step.
func TestWebpageRefusals(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": "<html><body>mine</body></html>\n",
		"jim/temp/notes.txt":  "",
	}))
	profile := &UserProfile{}

	out := captureStdout(t, func() { claimChain("add a webpage to jim/temp/index.html", yes, profile) })
	if !strings.Contains(out, "already has a page in it") || !strings.Contains(out, "never overwrite") {
		t.Errorf("non-empty refusal wrong:\n%s", out)
	}
	if hasPendingQuestion() {
		t.Error("a refused ask set a pending question")
	}
	if got := readBack(t, root, "jim/temp/index.html"); !strings.Contains(got, "mine") {
		t.Errorf("the existing page was touched:\n%s", got)
	}

	out = captureStdout(t, func() { claimChain("add a webpage to jim/temp/notes.txt", yes, profile) })
	if !strings.Contains(out, "isn't an .html file") {
		t.Errorf("non-html refusal wrong:\n%s", out)
	}

	out = captureStdout(t, func() { claimChain("add a webpage to nowhere/index.html", yes, profile) })
	if !strings.Contains(out, "create a folder nowhere") {
		t.Errorf("missing-folder refusal wrong:\n%s", out)
	}
}

// No file named: the question asks in webpage terms, and a
// bare path answer completes the ask (with the template the
// original sentence named riding along).
func TestWebpageFileQuestionCompletes(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/keep.css": "body {}\n",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("add a responsive webpage", yes, profile) })
	if !strings.Contains(out, "which file should the webpage go in?") {
		t.Errorf("file question missing:\n%s", out)
	}
	if !hasPendingQuestion() {
		t.Fatal("the file-less ask set no pending question")
	}
	captureStdout(t, func() { claimChain("jim/temp/index.html", yes, profile) })
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageContent("responsive", "index") {
		t.Errorf("the answered ask did not land the named template:\n%s", got)
	}
}
