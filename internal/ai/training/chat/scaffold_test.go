package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The scaffold grammar, at parse: both object wordings
// claim with name and folder; a missing name or folder
// still claims (the flow asks); neighbours' shapes decline.
func TestParseTemplatedPage(t *testing.T) {
	claims := []struct {
		line, name, folder string
	}{
		{"add a templated page called home in jim", "home", "jim"},
		{"add a go template page called home in jim", "home", "jim"},
		{"add a template page called home in jim", "home", "jim"},
		{"create templated page called about-us in site", "about-us", "site"},
		{"add a templated page called home in the folder jim", "home", "jim"},
		{"add a templated page in jim", "", "jim"},
		{"add a templated page called home", "home", ""},
		{"add a templated page called in jim", "", "jim"},
	}
	for _, c := range claims {
		q, ok := parseTemplatedPage(c.line)
		if !ok || q.name != c.name || q.folder != c.folder {
			t.Errorf("parseTemplatedPage(%q) = %+v, %v; want name=%q folder=%q", c.line, q, ok, c.name, c.folder)
		}
		if rt, ok := parseTemplatedPage(q.canonical()); q.name != "" && q.folder != "" && (!ok || rt != q) {
			t.Errorf("canonical %q does not round-trip: %+v, %v", q.canonical(), rt, ok)
		}
	}
	for _, line := range []string{
		"add a webpage to jim/temp/index.html",
		"add a template to jim/temp/index.html",
		"add a templated page called home in jim carefully",
		"what is a templated page",
		"add a templated stylesheet called home in jim",
		"add a page called home in jim",
	} {
		if q, ok := parseTemplatedPage(line); ok {
			t.Errorf("parseTemplatedPage(%q) claimed %+v — not its sentence", line, q)
		}
	}
}

// The ask lands: four files, the partner confirmation
// naming body, header, footer, and handler, the wiring
// next step the planner verified — and the folder's
// existing server byte-identical (no route was forced).
func TestTemplatedPageLands(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a templated page called home in jim", yes, profile)
	})
	for _, want := range []string{
		"Done — The home templated page is ready: its body is in jim/templates/home.html, the shared header in jim/templates/header.html, the shared footer in jim/templates/footer.html, and its handler HomePage in jim/homepage.go.",
		"I built the project and checked that it's really there.",
		"\"connect the HomePage handler to the webserver\" will do it.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("confirmation missing %q:\n%s", want, out)
		}
	}
	header := readBack(t, root, "jim/templates/header.html")
	if !strings.Contains(header, analyze.MaterialNavigationBarSnippet) || !strings.Contains(header, `{{define "header"}}`) {
		t.Errorf("header file isn't the shared header with the nav:\n%s", header)
	}
	footer := readBack(t, root, "jim/templates/footer.html")
	if !strings.Contains(footer, analyze.MaterialFooterSnippet) {
		t.Errorf("footer file isn't the shared footer:\n%s", footer)
	}
	body := readBack(t, root, "jim/templates/home.html")
	if !strings.Contains(body, `{{template "header" .}}`) || !strings.Contains(body, `{{template "footer" .}}`) {
		t.Errorf("body file doesn't call the header and footer:\n%s", body)
	}
	handler := readBack(t, root, "jim/homepage.go")
	if !strings.Contains(handler, `template.ParseFiles("jim/templates/home.html", "jim/templates/header.html", "jim/templates/footer.html")`) {
		t.Errorf("handler doesn't parse the three files by their project-root paths:\n%s", handler)
	}
	if got := readBack(t, root, "jim/main.go"); got != staticJimMain {
		t.Errorf("the existing server was touched — no route may be forced:\n%s", got)
	}
}

// The name is required: without it the claim asks one
// question, and the one-word answer lands the scaffold.
func TestTemplatedPageNameAsked(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a templated page in jim", yes, profile)
	})
	if !strings.Contains(out, "doesn't give the page a name") || !hasPendingQuestion() {
		t.Fatalf("the unnamed ask was not claimed with a question:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "jim", "templates")); !os.IsNotExist(err) {
		t.Fatal("files were written before the name was given")
	}
	out = captureStdout(t, func() {
		claimChain("home", yes, profile)
	})
	if !strings.Contains(out, "Done — The home templated page is ready") {
		t.Errorf("the name answer did not land the scaffold:\n%s", out)
	}
}

// And the folder: unnamed, the claim asks; the answer
// completes the same scaffold.
func TestTemplatedPageFolderAsked(t *testing.T) {
	staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a templated page called home", yes, profile)
	})
	if !strings.Contains(out, "doesn't name the folder") || !hasPendingQuestion() {
		t.Fatalf("the folderless ask was not claimed with a question:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimChain("jim", yes, profile)
	})
	if !strings.Contains(out, "Done — The home templated page is ready") {
		t.Errorf("the folder answer did not land the scaffold:\n%s", out)
	}
}

// The body-section grammar, at parse: digits and words,
// columns-only (rows settle at 2), the bare shape (counts
// unset — the parser's 2x2 default rides in the ask), and
// the named-page form. The webpage's counted grid and other
// neighbours decline.
func TestParseScaffoldSection(t *testing.T) {
	claims := []struct {
		line         string
		cols, rows   int
		counted      bool
		page, folder string
	}{
		{"add 2 columns and 3 rows to the body in jim", 2, 3, true, "", "jim"},
		{"add two columns and three rows to the body in jim", 2, 3, true, "", "jim"},
		{"add 3 columns to the body in jim", 3, 2, true, "", "jim"},
		{"add columns and rows to the body in jim", 2, 2, false, "", "jim"},
		{"add 2 columns and 3 rows to the body of home in jim", 2, 3, true, "home", "jim"},
		{"add 2 columns and 3 rows to the body of the home page in jim", 2, 3, true, "home", "jim"},
		// The counted grid's own word in front of the
		// counts (or the bare pair): the same section.
		{"add the grid 2 columns and 3 rows to the body in jim", 2, 3, true, "", "jim"},
		{"add a grid 2 columns and 3 rows to the body in jim", 2, 3, true, "", "jim"},
		{"add the grid two columns and three rows to the body of home in jim", 2, 3, true, "home", "jim"},
		{"add the grid 3 columns to the body in jim", 3, 2, true, "", "jim"},
		{"add the grid columns and rows to the body in jim", 2, 2, false, "", "jim"},
	}
	for _, c := range claims {
		q, ok := parseScaffoldSection(c.line)
		if !ok || q.cols != c.cols || q.rows != c.rows || q.counted != c.counted || q.page != c.page || q.folder != c.folder {
			t.Errorf("parseScaffoldSection(%q) = %+v, %v; want cols=%d rows=%d counted=%v page=%q folder=%q",
				c.line, q, ok, c.cols, c.rows, c.counted, c.page, c.folder)
		}
		if rt, ok := parseScaffoldSection(q.canonical()); !ok || rt.cols != q.cols || rt.rows != q.rows || rt.page != q.page || rt.folder != q.folder {
			t.Errorf("canonical %q does not round-trip: %+v, %v", q.canonical(), rt, ok)
		}
	}
	for _, line := range []string{
		"add template 2 columns 3 rows to jim/temp/index.html",
		"add 2 columns and 3 rows to jim/temp/index.html",
		"add a grid template with 2 columns and 3 rows to jim/temp/index.html",
		"add 2 columns and 3 rows to the page in jim",
		"add a templated page called home in jim",
		// The grid word alone names no counts, and the
		// seam line's direct file target is the webpage
		// grammar's routing, not this one's.
		"add the grid to the body in jim",
		"add the grid 2 columns 3 rows to jim/templates/home.html",
	} {
		if q, ok := parseScaffoldSection(line); ok {
			t.Errorf("parseScaffoldSection(%q) claimed %+v — not its sentence", line, q)
		}
	}
}

// John's shape lands the section in the body file only:
// the confirmation states the counts, the footer is
// byte-identical, the header's styles land once — and a
// second ask appends a second section below the first.
func TestScaffoldSectionLands(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("add a templated page called home in jim", yes, profile)
	})
	footerBefore := readBack(t, root, "jim/templates/footer.html")
	out := captureStdout(t, func() {
		claimChain("add 2 columns and 3 rows to the body in jim", yes, profile)
	})
	for _, want := range []string{
		"section of 2 columns and 3 rows (6 boxes)",
		"Done — The home page's body in jim/templates/home.html now has a section of 2 columns and 3 rows.",
		"I built the project and checked that it's really there.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("section landing missing %q:\n%s", want, out)
		}
	}
	body := readBack(t, root, "jim/templates/home.html")
	if n := strings.Count(body, "TODO: box"); n != 6 {
		t.Errorf("body holds %d section boxes, want 6", n)
	}
	if got := readBack(t, root, "jim/templates/footer.html"); got != footerBefore {
		t.Error("the footer file changed — the section belongs to the body only")
	}
	headerBefore := readBack(t, root, "jim/templates/header.html")
	captureStdout(t, func() {
		claimChain("add 2 columns and 3 rows to the body in jim", yes, profile)
	})
	body = readBack(t, root, "jim/templates/home.html")
	if n := strings.Count(body, "class=\"body-grid "); n != 2 {
		t.Errorf("after a second ask the body holds %d sections, want 2", n)
	}
	if got := readBack(t, root, "jim/templates/header.html"); got != headerBefore {
		t.Error("the second section rewrote the header — its styles land once")
	}
	for _, probe := range []string{"/* Columns-and-rows sections in the body", ".body-grid-2 { grid-template-columns: repeat(2, 1fr); }"} {
		if n := strings.Count(headerBefore, probe); n != 1 {
			t.Errorf("header holds %q %d times, want exactly 1", probe, n)
		}
	}
}

// The grid-word form lands the same section in the same
// body: John's composition of the counted grid's word
// with this piece's sentence.
func TestScaffoldSectionGridWordLands(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("add a templated page called home in jim", yes, profile)
	})
	out := captureStdout(t, func() {
		claimChain("add the grid 2 columns and 3 rows to the body in jim", yes, profile)
	})
	if !strings.Contains(out, "Done — The home page's body in jim/templates/home.html now has a section of 2 columns and 3 rows.") {
		t.Errorf("the grid-word section did not land:\n%s", out)
	}
	if n := strings.Count(readBack(t, root, "jim/templates/home.html"), "TODO: box"); n != 6 {
		t.Errorf("body holds %d section boxes, want 6", n)
	}
}

// The settled defaults, pinned: no counts at all lands
// 2 columns and 2 rows, stated in the confirmation;
// columns alone lands rows of 2.
func TestScaffoldSectionDefaults(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("add a templated page called home in jim", yes, profile)
	})
	out := captureStdout(t, func() {
		claimChain("add columns and rows to the body in jim", yes, profile)
	})
	if !strings.Contains(out, "now has a section of 2 columns and 2 rows.") {
		t.Errorf("the bare ask didn't land the stated 2x2 default:\n%s", out)
	}
	if n := strings.Count(readBack(t, root, "jim/templates/home.html"), "TODO: box"); n != 4 {
		t.Errorf("the 2x2 section holds %d boxes, want 4", n)
	}
	out = captureStdout(t, func() {
		claimChain("add 3 columns to the body in jim", yes, profile)
	})
	if !strings.Contains(out, "now has a section of 3 columns and 2 rows.") {
		t.Errorf("columns alone didn't settle rows at 2:\n%s", out)
	}
}

// The section's refusals and question: out-of-range counts
// refuse like the counted grid; no scaffold in the folder
// is the two-step; several bodies and no name asks which
// page — and the pick lands in that page's body only.
func TestScaffoldSectionRefusalsAndPick(t *testing.T) {
	profile := &UserProfile{}
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	out := captureStdout(t, func() {
		claimChain("add 2 columns and 3 rows to the body in jim", yes, profile)
	})
	if !strings.Contains(out, "there's no templated page in jim yet") || !strings.Contains(out, "add a templated page called <name> in jim") {
		t.Errorf("no-scaffold answer isn't the two-step:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimChain("add 9 columns and 3 rows to the body in jim", yes, profile)
	})
	if !strings.Contains(out, "1 to 6 columns and 1 to 6 rows") || !strings.Contains(out, "9 columns") {
		t.Errorf("out-of-range refusal missing:\n%s", out)
	}
	captureStdout(t, func() {
		claimChain("add a templated page called home in jim", yes, profile)
	})
	writeFile(t, root, "jim/templates/about.html", readBack(t, root, "jim/templates/home.html"))
	out = captureStdout(t, func() {
		claimChain("add 2 columns and 2 rows to the body in jim", yes, profile)
	})
	if !strings.Contains(out, "more than one templated page") || !hasPendingQuestion() {
		t.Fatalf("the two-body ask was not claimed with a question:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimChain("about", yes, profile)
	})
	if !strings.Contains(out, "The about page's body in jim/templates/about.html now has a section of 2 columns and 2 rows.") {
		t.Errorf("the pick didn't land in the about body:\n%s", out)
	}
	if strings.Contains(readBack(t, root, "jim/templates/home.html"), "body-grid") {
		t.Error("the section also landed in the home body — the pick names one page")
	}
}

// The chat-level refusals: a missing folder is a two-step,
// an existing header refuses naming it, and a bad name
// refuses in the planner's words — nothing written in any
// of them.
func TestTemplatedPageChatRefusals(t *testing.T) {
	profile := &UserProfile{}
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	out := captureStdout(t, func() {
		claimChain("add a templated page called home in nook", yes, profile)
	})
	if !strings.Contains(out, "there's no folder nook") || !strings.Contains(out, "create a folder nook") {
		t.Errorf("missing-folder refusal isn't the two-step:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimChain("add a templated page called 2home in jim", yes, profile)
	})
	if !strings.Contains(out, "isn't a plain page name") {
		t.Errorf("bad-name refusal missing:\n%s", out)
	}
	root2 := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go":               staticJimMain,
		"jim/templates/header.html": "mine\n",
	}))
	_ = root
	out = captureStdout(t, func() {
		claimChain("add a templated page called home in jim", yes, profile)
	})
	if !strings.Contains(out, "jim/templates/header.html is already there") {
		t.Errorf("existing-header refusal doesn't name what exists:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root2, "jim", "homepage.go")); !os.IsNotExist(err) {
		t.Error("the refused scaffold still wrote its handler")
	}
}
