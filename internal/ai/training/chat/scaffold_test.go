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
