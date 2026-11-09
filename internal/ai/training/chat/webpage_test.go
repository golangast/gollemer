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
		// The template wording (John, later on 2026-11-09):
		// the same asks with "template" as the object.
		{"add the template to jim/temp/index.html", "", "jim/temp/index.html", false}, // John's line, verbatim
		{"add a template to jim/temp/index.html", "", "jim/temp/index.html", false},
		{"add the basic template to jim/temp/index.html", "basic", "jim/temp/index.html", false},
		{"add a simple template to index.html", "basic", "index.html", false},
		{"add the responsive template to jim/temp/index.html", "responsive", "jim/temp/index.html", false},
		{"add a template with assets to jim/temp/index.html", "assets", "jim/temp/index.html", false},
		{"add the with assets template to jim/temp/index.html", "assets", "jim/temp/index.html", false},
		{"use the template for jim/temp/index.html", "", "jim/temp/index.html", false},
		{"use the basic template for jim/temp/index.html", "basic", "jim/temp/index.html", false},
		{"add the template", "", "", true},
		{"add the basic template", "basic", "", true},
		// The layout templates, in the template wording
		// and the webpage wording alike — spaced,
		// hyphenated, singular and plural column forms.
		{"add a two column template to jim/temp/index.html", "two-columns", "jim/temp/index.html", false},
		{"add a two-column template to jim/temp/index.html", "two-columns", "jim/temp/index.html", false},
		{"add a two-columns template to jim/temp/index.html", "two-columns", "jim/temp/index.html", false},
		{"add a three columns template to jim/temp/index.html", "three-columns", "jim/temp/index.html", false},
		{"add a three-column template to jim/temp/index.html", "three-columns", "jim/temp/index.html", false},
		{"add a grid template to jim/temp/index.html", "grid", "jim/temp/index.html", false},
		{"add a rows template to jim/temp/index.html", "rows", "jim/temp/index.html", false},
		{"use the grid template for jim/temp/index.html", "grid", "jim/temp/index.html", false},
		{"add a two column webpage to jim/temp/index.html", "two-columns", "jim/temp/index.html", false},
		{"add a three columns webpage to jim/temp/index.html", "three-columns", "jim/temp/index.html", false},
		{"add a grid webpage to jim/temp/index.html", "grid", "jim/temp/index.html", false},
		{"add a rows webpage to jim/temp/index.html", "rows", "jim/temp/index.html", false},
		// Material, in both wordings.
		{"add a material template to jim/temp/index.html", "material", "jim/temp/index.html", false},
		{"add a material webpage to jim/temp/index.html", "material", "jim/temp/index.html", false},
		{"use the material template for jim/temp/index.html", "material", "jim/temp/index.html", false},
		// The dropped article — the same ruling the
		// component piece got live (2026-11-09): this parse
		// already read the article as optional, and these
		// rows pin that it stays so, in every wording.
		{"add webpage to jim/temp/index.html", "", "jim/temp/index.html", false},
		{"add responsive webpage to jim/temp/index.html", "responsive", "jim/temp/index.html", false},
		{"add template to jim/temp/index.html", "", "jim/temp/index.html", false},
		{"add two column template to jim/temp/index.html", "two-columns", "jim/temp/index.html", false},
		{"add material template to jim/temp/index.html", "material", "jim/temp/index.html", false},
		{"add grid webpage to index.html", "grid", "index.html", false},
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
	for _, template := range []string{"", "basic", "responsive", "assets", "two-columns", "three-columns", "grid", "rows", "material"} {
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
		// The template wording with a .go target is the
		// concept book's html/template sentence — a Go
		// statement for a function — never this piece's,
		// however the template is named.
		"add the template to main.go",
		"add a template to jim/main.go",
		"add template to main.go", // dropped article opens no .go door either
		"add the basic template to main.go",
		"add a two column template to main.go",
		"add a material template to main.go",
		"use the template for main.go",
		"add an html template to main.go",
		"add html template to main.go",
		// "use" owns only the template wording.
		"use a webpage for jim/temp/index.html",
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
	// The other side of the .go boundary: the concept book
	// still owns its html/template sentence (its canonical
	// example), exactly as before this grammar grew the
	// template wording.
	ps, ok := CanonicalSlots("add an html template to main.go")
	if !ok || ps.Family != "concept-html-template" {
		t.Errorf("CanonicalSlots(book's html template line) = %+v, %v — want family concept-html-template", ps, ok)
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

// John's second live line, verbatim, end to end: the
// template wording claims the ask for the .html file he
// just made, the question lists all seven templates, and
// the number answer lands the picked layout.
func TestWebpageTemplateWordingJohnsLine(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim/temp", yes, profile)
		claimChain("create a file jim/temp/index.html", yes, profile)
	})
	out := captureStdout(t, func() {
		claimChain("add the template to jim/temp/index.html", yes, profile)
	})
	for _, want := range []string{"which template should it be?", "1. basic", "2. responsive", "3. with assets", "4. two columns", "5. three columns", "6. grid", "7. rows", "8. material"} {
		if !strings.Contains(out, want) {
			t.Errorf("template question missing %q:\n%s", want, out)
		}
	}
	if !hasPendingQuestion() {
		t.Fatal("the unnamed template set no pending question")
	}
	out = captureStdout(t, func() {
		claimChain("4", yes, profile)
	})
	if hasPendingQuestion() {
		t.Error("the completed ask left a question pending")
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageContent("two-columns", "index") {
		t.Errorf("the picked layout did not land:\n%s", got)
	}
	if !strings.Contains(out, "Done — The two-column webpage is in jim/temp/index.html.") {
		t.Errorf("landing confirmation missing:\n%s", out)
	}
}

// A named layout in the template wording asks NO question
// and lands directly.
func TestWebpageNamedLayoutAsksNoQuestion(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/keep.css": "body {}\n",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a two column template to jim/temp/index.html", yes, profile)
	})
	if hasPendingQuestion() {
		t.Error("a named layout set a pending question")
	}
	if strings.Contains(out, "which template") {
		t.Errorf("a named layout still asked:\n%s", out)
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageContent("two-columns", "index") {
		t.Errorf("the named layout did not land:\n%s", got)
	}
	if !strings.Contains(out, "Done — The two-column webpage is in jim/temp/index.html.") {
		t.Errorf("landing confirmation missing:\n%s", out)
	}
}

// The material template, named in the template wording:
// no question, and the page that lands is the from-scratch
// Material page.
func TestWebpageMaterialTemplateLands(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/keep.css": "body {}\n",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a material template to jim/temp/index.html", yes, profile)
	})
	if hasPendingQuestion() {
		t.Error("a named template set a pending question")
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageContent("material", "index") {
		t.Errorf("the material page did not land:\n%s", got)
	}
	if !strings.Contains(out, "Done — The material webpage is in jim/temp/index.html.") {
		t.Errorf("landing confirmation missing:\n%s", out)
	}
}

// The counted grid, at the parse level: every count form
// claims as the grid with its counts — digits or words,
// singular or plural, the "by" form in front of the name,
// the "with" clause and John's bare pair behind the
// object. Columns alone default the rows to 2, here at
// parse, so every later stage states the same counts.
// Out-of-range counts still parse as counts — the flow
// refuses them by name.
func TestParseWebpageCountedGridClaims(t *testing.T) {
	claims := []struct {
		line     string
		cols     int
		rows     int
		file     string
		missing  bool
	}{
		{"add template 2 columns 3 rows to jim/temp/index.html", 2, 3, "jim/temp/index.html", false}, // John's line, verbatim
		{"add a grid template with 2 columns and 3 rows to jim/temp/index.html", 2, 3, "jim/temp/index.html", false},
		{"add a grid template with 3 columns to jim/temp/index.html", 3, 2, "jim/temp/index.html", false}, // rows default
		{"add a 2 by 3 grid template to jim/temp/index.html", 2, 3, "jim/temp/index.html", false},
		{"add a grid webpage with 2 columns and 3 rows to jim/temp/index.html", 2, 3, "jim/temp/index.html", false},
		{"add grid template with two columns and three rows to jim/temp/index.html", 2, 3, "jim/temp/index.html", false},
		{"add a two by three grid template to jim/temp/index.html", 2, 3, "jim/temp/index.html", false},
		{"add template 1 column 1 row to jim/temp/index.html", 1, 1, "jim/temp/index.html", false},
		{"add a grid template with 1 column to jim/temp/index.html", 1, 2, "jim/temp/index.html", false},
		{"use the grid template with 2 columns and 3 rows for jim/temp/index.html", 2, 3, "jim/temp/index.html", false},
		{"add template 2 columns 3 rows", 2, 3, "", true},
		// The grid as its own object: the counts directly
		// behind the name, no object word between.
		{"add the grid 2 columns 3 rows to jim/temp/index.html", 2, 3, "jim/temp/index.html", false}, // John's seam line, verbatim
		{"add a grid 2 columns 3 rows to jim/temp/index.html", 2, 3, "jim/temp/index.html", false},
		{"add grid 2 columns 3 rows to jim/temp/index.html", 2, 3, "jim/temp/index.html", false},
		{"add the grid two columns and three rows to jim/temp/index.html", 2, 3, "jim/temp/index.html", false},
		{"add the grid 3 columns to jim/temp/index.html", 3, 2, "jim/temp/index.html", false}, // rows default
		{"add the grid 2 columns 3 rows", 2, 3, "", true},
		// Out of range: still this piece's sentence.
		{"add a grid template with 7 columns to jim/temp/index.html", 7, 2, "jim/temp/index.html", false},
		{"add the grid 7 columns to jim/temp/index.html", 7, 2, "jim/temp/index.html", false},
		{"add template 0 columns 3 rows to jim/temp/index.html", 0, 3, "jim/temp/index.html", false},
		{"add template 2 columns 9 rows to jim/temp/index.html", 2, 9, "jim/temp/index.html", false},
		{"add a grid template with seven columns to jim/temp/index.html", 7, 2, "jim/temp/index.html", false},
	}
	for _, tc := range claims {
		got, ok := parseWebpage(tc.line)
		if !ok {
			t.Errorf("parseWebpage(%q) not claimed", tc.line)
			continue
		}
		if got.template != "grid" || !got.gridCounted || got.gridCols != tc.cols || got.gridRows != tc.rows ||
			got.file != tc.file || got.fileMissing != tc.missing {
			t.Errorf("parseWebpage(%q) = %+v, want grid %dx%d file=%q missing=%v", tc.line, got, tc.cols, tc.rows, tc.file, tc.missing)
		}
	}
	// The counted canonical parses back with exactly its
	// counts.
	line := webpageCanonicalCounted(2, 3, "jim/temp/index.html")
	got, ok := parseWebpage(line)
	if !ok || got.template != "grid" || got.gridCols != 2 || got.gridRows != 3 || got.file != "jim/temp/index.html" {
		t.Errorf("counted canonical %q parsed as %+v, %v", line, got, ok)
	}
}

// The counted grid's boundary: a lone column count with no
// grid named names nothing (the two-columns layout owns
// its wording, in front of the object), counts ride on no
// other template, a "by" pair sizes only the grid, and a
// rows-only clause has no columns to stand on.
func TestParseWebpageCountedGridNonClaims(t *testing.T) {
	for _, line := range []string{
		"add template 2 columns to jim/temp/index.html", // John's pair, halved: declines
		"add template 2 columns to main.go",
		"add a webpage with 2 columns and 3 rows to jim/temp/index.html", // the pair names the grid only in the template wording
		"add a basic template with 2 columns and 3 rows to jim/temp/index.html",
		"add a rows template with 2 columns and 3 rows to jim/temp/index.html",
		"add a 2 by 3 template to jim/temp/index.html",
		"add a 2 by 3 rows template to jim/temp/index.html",
		"add a grid template with 3 rows to jim/temp/index.html",
		"add a grid template 2 columns 3 rows with 2 columns to jim/temp/index.html",
		// The grid-object form keeps the family's seating:
		// no counts behind the name names nothing, a rows-
		// only clause has no columns, a second clause is a
		// second sentence, the "use" wording never takes
		// it, and a .go target stays the book's.
		"add the grid to jim/temp/index.html",
		"add the grid 3 rows to jim/temp/index.html",
		"add the grid 2 columns 3 rows with 2 columns to jim/temp/index.html",
		"use the grid 2 columns 3 rows for jim/temp/index.html",
		"add the grid 2 columns 3 rows to main.go",
	} {
		if got, ok := parseWebpage(line); ok {
			t.Errorf("parseWebpage(%q) claimed %+v — not its sentence", line, got)
		}
	}
}

// John's counted line, verbatim, end to end: no question,
// the plan states the counts it will use, and the page
// that lands is the 2x3 grid.
func TestWebpageCountedGridJohnsLine(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": "",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add template 2 columns 3 rows to jim/temp/index.html", yes, profile)
	})
	if hasPendingQuestion() {
		t.Error("a counted ask set a pending question")
	}
	if strings.Contains(out, "which template") {
		t.Errorf("a counted ask still asked which template:\n%s", out)
	}
	for _, want := range []string{"grid webpage with 2 columns and 3 rows", "6 boxes"} {
		if !strings.Contains(out, want) {
			t.Errorf("the plan did not state %q:\n%s", want, out)
		}
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageGridContent(2, 3, "index") {
		t.Errorf("the counted grid did not land:\n%s", got)
	}
	if !strings.Contains(out, "Done — The grid webpage is in jim/temp/index.html.") {
		t.Errorf("landing confirmation missing:\n%s", out)
	}
}

// Columns alone: the plan states the defaulted rows, and
// the 3x2 grid lands.
func TestWebpageCountedGridColumnsOnly(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": "",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a grid template with 3 columns to jim/temp/index.html", yes, profile)
	})
	if !strings.Contains(out, "grid webpage with 3 columns and 2 rows") {
		t.Errorf("the plan did not state the defaulted rows:\n%s", out)
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageGridContent(3, 2, "index") {
		t.Errorf("the columns-only grid did not land as 3x2:\n%s", got)
	}
}

// Out-of-range counts refuse in the piece's own words,
// naming the range and the offending count — and nothing
// lands.
func TestWebpageCountedGridOutOfRange(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": "",
	}))
	profile := &UserProfile{}
	for _, tc := range []struct{ line, want string }{
		{"add a grid template with 7 columns to jim/temp/index.html", "7 columns"},
		{"add template 0 columns 3 rows to jim/temp/index.html", "0 columns"},
		{"add template 2 columns 9 rows to jim/temp/index.html", "9 rows"},
	} {
		out := captureStdout(t, func() { claimChain(tc.line, yes, profile) })
		if !strings.Contains(out, "a counted grid can have 1 to 6 columns and 1 to 6 rows") || !strings.Contains(out, tc.want) {
			t.Errorf("%q: refusal wrong:\n%s", tc.line, out)
		}
		if hasPendingQuestion() {
			t.Errorf("%q: a refused ask set a pending question", tc.line)
		}
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != "" {
		t.Errorf("a refused ask landed a page:\n%s", got)
	}
}

// A counted ask with no file named: the file question
// completes it with the counts riding along.
func TestWebpageCountedGridFileQuestion(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/keep.css": "body {}\n",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("add template 2 columns 3 rows", yes, profile) })
	if !strings.Contains(out, "which file should the webpage go in?") {
		t.Errorf("file question missing:\n%s", out)
	}
	if !hasPendingQuestion() {
		t.Fatal("the file-less counted ask set no pending question")
	}
	captureStdout(t, func() { claimChain("jim/temp/index.html", yes, profile) })
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageGridContent(2, 3, "index") {
		t.Errorf("the answered ask did not land the counted grid:\n%s", got)
	}
}

// The seam, end to end: one sentence, three targets, and
// the target decides what the counts mean. Aimed at a
// templated page's body file, John's grid-object line
// lands a section in that body (the body section's own
// confirmation, header styles once, footer untouched);
// aimed at an empty plain page it is the whole counted
// grid; aimed at a page with words in it, it is the
// never-overwrite refusal and the file is byte-identical.
func TestWebpageCountedGridRoutesByTarget(t *testing.T) {
	profile := &UserProfile{}

	// A scaffold body target: a section, not a page.
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	captureStdout(t, func() {
		claimChain("add a templated page called home in jim", yes, profile)
	})
	footerBefore := readBack(t, root, "jim/templates/footer.html")
	out := captureStdout(t, func() {
		claimChain("add the grid 2 columns 3 rows to jim/templates/home.html", yes, profile)
	})
	if !strings.Contains(out, "Done — The home page's body in jim/templates/home.html now has a section of 2 columns and 3 rows.") {
		t.Errorf("the seam line did not land as a body section:\n%s", out)
	}
	body := readBack(t, root, "jim/templates/home.html")
	if n := strings.Count(body, "TODO: box"); n != 6 {
		t.Errorf("the routed section holds %d boxes, want 6", n)
	}
	for _, call := range []string{"{{template \"header\" .}}", "{{template \"footer\" .}}"} {
		if !strings.Contains(body, call) {
			t.Errorf("the body lost its %s call — it was replaced, not grown:\n%s", call, body)
		}
	}
	if got := readBack(t, root, "jim/templates/footer.html"); got != footerBefore {
		t.Error("the footer file changed — the section belongs to the body only")
	}
	header := readBack(t, root, "jim/templates/header.html")
	for _, probe := range []string{"/* Columns-and-rows sections in the body", ".body-grid-2 { grid-template-columns: repeat(2, 1fr); }"} {
		if n := strings.Count(header, probe); n != 1 {
			t.Errorf("header holds %q %d times, want exactly 1", probe, n)
		}
	}

	// An empty plain page: the whole counted grid.
	root = staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": "",
	}))
	captureStdout(t, func() {
		claimChain("add the grid 2 columns 3 rows to jim/temp/index.html", yes, profile)
	})
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageGridContent(2, 3, "index") {
		t.Errorf("the seam line on an empty page did not land the counted grid:\n%s", got)
	}

	// A page with words in it: refused, byte-identical.
	mine := "<html><body>mine</body></html>\n"
	root = staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": mine,
	}))
	out = captureStdout(t, func() {
		claimChain("add the grid 2 columns 3 rows to jim/temp/index.html", yes, profile)
	})
	if !strings.Contains(out, "already has a page in it") || !strings.Contains(out, "never overwrite") {
		t.Errorf("the seam line on a written page was not refused:\n%s", out)
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != mine {
		t.Errorf("the written page changed:\n%s", got)
	}
}

// A template ask aimed at one part of a templated page:
// the templated-page explanation, not the overwrite
// refusal — and nothing on disk moves. John's line is the
// lead pin: he asked it three times live (2026-11-09) and
// the generic refusal taught him nothing, because a
// template can never land in one part of a page and the
// page's design already lives in its header.
func TestWebpageTemplateAskOnScaffoldPartExplains(t *testing.T) {
	profile := &UserProfile{}
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	captureStdout(t, func() {
		claimChain("add a templated page called home in jim", yes, profile)
	})
	scaffoldFiles := []string{"jim/templates/header.html", "jim/templates/footer.html", "jim/templates/home.html", "jim/homepage.go"}
	before := map[string]string{}
	for _, rel := range scaffoldFiles {
		before[rel] = readBack(t, root, rel)
	}

	// John's line verbatim, on the body.
	out := captureStdout(t, func() {
		claimChain("add the material template to jim/templates/home.html", yes, profile)
	})
	for _, want := range []string{
		"jim/templates/home.html is the body of the home templated page in jim",
		"its Material styles, lives in the header file, jim/templates/header.html",
		"already covers the whole page",
		"\"add the grid 2 columns 3 rows to jim/templates/home.html\"",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the body explanation is missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "never overwrite") {
		t.Errorf("the body got the overwrite refusal instead of the explanation:\n%s", out)
	}
	if hasPendingQuestion() {
		t.Error("an explained ask set a pending question")
	}

	// Every template kind gets the same answer shape on
	// the body — the file, not the template named, decides.
	for _, line := range []string{
		"add a basic webpage to jim/templates/home.html",
		"add a responsive template to jim/templates/home.html",
		"add a webpage with assets to jim/templates/home.html",
	} {
		out = captureStdout(t, func() { claimChain(line, yes, profile) })
		if !strings.Contains(out, "is the body of the home templated page") || strings.Contains(out, "never overwrite") {
			t.Errorf("%q did not get the body explanation:\n%s", line, out)
		}
	}

	// The header: the explanation names the design's home
	// (the header itself) and points at the body.
	out = captureStdout(t, func() {
		claimChain("add the basic template to jim/templates/header.html", yes, profile)
	})
	for _, want := range []string{
		"jim/templates/header.html is the shared header of the templated pages in jim",
		"\"add the grid 2 columns 3 rows to jim/templates/home.html\"",
		"the home page's body",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the header explanation is missing %q:\n%s", want, out)
		}
	}

	// The footer: the design lives in the header, and
	// sections point at the body the same way.
	out = captureStdout(t, func() {
		claimChain("add the basic template to jim/templates/footer.html", yes, profile)
	})
	for _, want := range []string{
		"jim/templates/footer.html is the shared footer of the templated pages in jim",
		"its Material styles, lives in the header file, jim/templates/header.html",
		"\"add the grid 2 columns 3 rows to jim/templates/home.html\"",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the footer explanation is missing %q:\n%s", want, out)
		}
	}

	// Nothing moved: every scaffold file byte-identical.
	for _, rel := range scaffoldFiles {
		if got := readBack(t, root, rel); got != before[rel] {
			t.Errorf("%s changed under an explanation:\n%s", rel, got)
		}
	}
}

// The drift guard for the explanation above: the one
// sentence it names for a body is walked through the real
// dispatch on a fixture scaffold and must land as a
// section — an answer may never promise a line that
// doesn't work (the capability seat's rule). The component
// sentence is deliberately NOT named by the explanation:
// the component planner refuses a body fragment (it has
// no </body> to sit inside), and the second half of this
// pin keeps that refusal on record so the explanation is
// never "fixed" into promising it.
func TestWebpageScaffoldGuidanceLineLands(t *testing.T) {
	profile := &UserProfile{}
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	captureStdout(t, func() {
		claimChain("add a templated page called home in jim", yes, profile)
	})
	out := captureStdout(t, func() {
		claimChain("add the grid 2 columns 3 rows to jim/templates/home.html", yes, profile)
	})
	if !strings.Contains(out, "Done — The home page's body in jim/templates/home.html now has a section of 2 columns and 3 rows.") {
		t.Errorf("the named guidance line did not land as a section:\n%s", out)
	}
	if n := strings.Count(readBack(t, root, "jim/templates/home.html"), "TODO: box"); n != 6 {
		t.Errorf("the landed section holds %d boxes, want 6", n)
	}
	// The component sentence on the same body: claimed,
	// and refused by its own planner — it does not land,
	// so the explanation must never name it.
	out = captureStdout(t, func() {
		claimChain("add a card to jim/templates/home.html", yes, profile)
	})
	if !strings.Contains(out, "no </body> in jim/templates/home.html") {
		t.Errorf("the component line on a body fragment answered unexpectedly:\n%s", out)
	}
	if n := strings.Count(readBack(t, root, "jim/templates/home.html"), "TODO: box"); n != 6 {
		t.Errorf("the component line changed the body: %d boxes, want the 6 from the section", n)
	}
}

// The seam's other side, pinned byte-identical: a template
// ask on a filled PLAIN page keeps the standing refusal
// exactly, and the same ask on an empty page lands the
// template as it always did.
func TestWebpageTemplateAskOnPlainPageUnchanged(t *testing.T) {
	mine := "<html><body>mine</body></html>\n"
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": mine,
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add the material template to jim/temp/index.html", yes, profile)
	})
	want := "Not making that webpage: jim/temp/index.html already has a page in it — I never overwrite what's yours. If you want a fresh template page there instead, empty the file yourself (or remove it) and ask again."
	if !strings.Contains(out, want) {
		t.Errorf("the plain-page refusal changed:\n%s", out)
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != mine {
		t.Errorf("the written page changed:\n%s", got)
	}

	root = staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": "",
	}))
	out = captureStdout(t, func() {
		claimChain("add the material template to jim/temp/index.html", yes, profile)
	})
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageContent("material", "index") {
		t.Errorf("the material template did not land on the empty page:\n%s", got)
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
