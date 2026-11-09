package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The webpage planner: one template into one .html file —
// created when missing, filled when empty, refused when the
// file already holds a page. The fixture folder jim/temp
// exists (a style.css stands in for its contents); the
// project's only Go is a root doc.go so it analyzes.

func webpageProject(t *testing.T, files map[string]string) (string, *Project) {
	t.Helper()
	base := map[string]string{
		"go.mod":            "module example.com/webpage\n\ngo 1.21\n",
		"doc.go":            "package webpage\n",
		"jim/temp/keep.css": "/* placeholder */\n",
	}
	for rel, src := range files {
		base[rel] = src
	}
	return serveProject(t, base)
}

func TestPlanWebpageCreatesNewFile(t *testing.T) {
	dir, proj := webpageProject(t, nil)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "responsive"})
	if refusal != "" || plan == nil {
		t.Fatalf("plan refused: %q", refusal)
	}
	if len(plan.Files) != 1 || plan.Files[0].Rel != "jim/temp/index.html" {
		t.Fatalf("plan files = %+v", plan.Files)
	}
	if len(plan.Files[0].Before) != 0 {
		t.Errorf("a created page carries before bytes: %q", plan.Files[0].Before)
	}
	if got, want := string(plan.Files[0].After), WebpageContent("responsive", "index"); got != want {
		t.Errorf("page content mismatch:\n got: %q\nwant: %q", got, want)
	}
	// LinkPath renders relative to the process cwd, so the
	// summary's path is long here; its head and tail are the
	// facts (the codebase's convention for planner summaries).
	if !strings.Contains(plan.Summary, "Create ") || !strings.Contains(plan.Summary, "jim/temp/index.html with the responsive webpage") {
		t.Errorf("summary does not own the creation: %q", plan.Summary)
	}
	applyServePlan(t, dir, plan)
}

func TestPlanWebpageFillsEmptyFile(t *testing.T) {
	dir, proj := webpageProject(t, map[string]string{"jim/temp/index.html": ""})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "basic"})
	if refusal != "" || plan == nil {
		t.Fatalf("plan refused: %q", refusal)
	}
	if !strings.Contains(plan.Summary, "the file is there but empty") {
		t.Errorf("summary does not own the fill: %q", plan.Summary)
	}
	applyServePlan(t, dir, plan)
	got, err := os.ReadFile(filepath.Join(dir, "jim/temp/index.html"))
	if err != nil || string(got) != WebpageContent("basic", "index") {
		t.Errorf("filled page = %q, %v", got, err)
	}
}

func TestPlanWebpageRefusesNonEmptyFile(t *testing.T) {
	_, proj := webpageProject(t, map[string]string{"jim/temp/index.html": "<html><body>mine</body></html>\n"})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "basic"})
	if plan != nil {
		t.Fatalf("a non-empty page was planned over: %+v", plan)
	}
	if !strings.Contains(refusal, "already has a page in it") || !strings.Contains(refusal, "never overwrite") {
		t.Errorf("refusal = %q", refusal)
	}
}

func TestPlanWebpageRefusals(t *testing.T) {
	_, proj := webpageProject(t, nil)
	for _, tc := range []struct {
		name string
		req  EditRequest
		want string
	}{
		{"not html", EditRequest{Kind: EditWebpage, File: "jim/temp/page.txt", WebpageTemplate: "basic"}, ".html file name"},
		{"no template", EditRequest{Kind: EditWebpage, File: "jim/temp/index.html"}, "templates: basic, responsive, with assets, two columns, three columns, grid, rows, or material"},
		{"unknown template", EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "fancy"}, "templates: basic, responsive, with assets, two columns, three columns, grid, rows, or material"},
		{"missing folder", EditRequest{Kind: EditWebpage, File: "nowhere/index.html", WebpageTemplate: "basic"}, "create a folder nowhere"},
	} {
		plan, refusal := proj.PlanEdit(tc.req)
		if plan != nil || !strings.Contains(refusal, tc.want) {
			t.Errorf("%s: plan=%v refusal=%q — want a refusal containing %q", tc.name, plan != nil, refusal, tc.want)
		}
	}
}

// The templates themselves: what each one is, in the page's
// own bytes — the title comes from the file's name, the
// responsive one adapts, the assets one points at /static/.
func TestWebpageContentTemplates(t *testing.T) {
	basic := WebpageContent("basic", "about")
	for _, want := range []string{"<title>about</title>", "<h1>about</h1>", "TODO:"} {
		if !strings.Contains(basic, want) {
			t.Errorf("basic page missing %q:\n%s", want, basic)
		}
	}
	if strings.Contains(basic, "viewport") || strings.Contains(basic, "/static/") {
		t.Errorf("basic page carries another template's furniture:\n%s", basic)
	}
	resp := WebpageContent("responsive", "index")
	for _, want := range []string{`<meta name="viewport" content="width=device-width, initial-scale=1">`, "<style>", "max-width"} {
		if !strings.Contains(resp, want) {
			t.Errorf("responsive page missing %q:\n%s", want, resp)
		}
	}
	assets := WebpageContent("assets", "index")
	for _, want := range []string{`href="/static/style.css"`, `src="/static/image.png"`, `src="/static/script.js"`} {
		if !strings.Contains(assets, want) {
			t.Errorf("assets page missing %q:\n%s", want, assets)
		}
	}
}

// The counted grid, in the page's own bytes: exactly the
// named columns in the CSS, exactly cols*rows numbered
// boxes, and the responsive collapse the plain grid has.
// The plain grid itself is untouched — still the flowing
// three-across skeleton with its six unnumbered cards.
func TestWebpageGridContentCounted(t *testing.T) {
	page := WebpageGridContent(2, 3, "index")
	for _, want := range []string{"<title>index</title>", "display: grid", "grid-template-columns: repeat(2, 1fr)",
		"card 1's words", "card 6's words", "2 columns by 3 rows, 6 boxes in all",
		"@media (max-width: 32em)", "grid-template-columns: 1fr"} {
		if !strings.Contains(page, want) {
			t.Errorf("2x3 grid page missing %q:\n%s", want, page)
		}
	}
	if got := strings.Count(page, `<div class="card">`); got != 6 {
		t.Errorf("2x3 grid holds %d boxes, want exactly 6:\n%s", got, page)
	}
	if strings.Contains(page, "card 7's words") {
		t.Errorf("2x3 grid numbers a seventh box:\n%s", page)
	}
	// Three across keeps the middling-screen step down to
	// two; one column has nothing to collapse.
	wide := WebpageGridContent(3, 2, "index")
	if !strings.Contains(wide, "grid-template-columns: repeat(3, 1fr)") ||
		!strings.Contains(wide, "@media (max-width: 50em) { .grid { grid-template-columns: repeat(2, 1fr); } }") {
		t.Errorf("3x2 grid lost its columns or its collapse:\n%s", wide)
	}
	if got := strings.Count(wide, `<div class="card">`); got != 6 {
		t.Errorf("3x2 grid holds %d boxes, want exactly 6", got)
	}
	one := WebpageGridContent(1, 1, "index")
	if got := strings.Count(one, `<div class="card">`); got != 1 {
		t.Errorf("1x1 grid holds %d boxes, want exactly 1", got)
	}
	if strings.Contains(one, "@media") {
		t.Errorf("1x1 grid carries a collapse it cannot use:\n%s", one)
	}
	plain := WebpageContent("grid", "index")
	if got := strings.Count(plain, `<div class="card">`); got != 6 {
		t.Errorf("plain grid holds %d cards, want its standing 6", got)
	}
	if strings.Contains(plain, "card 1's words") {
		t.Errorf("plain grid grew the counted grid's numbered cards:\n%s", plain)
	}
}

// The counted grid at the planner: the plan's content is
// the counted body, the summary states the counts and the
// box total, unnamed rows default to 2 — and the counts
// are range-checked here like everywhere (refuse, never
// clamp), belong to the grid alone, and need their columns.
func TestPlanWebpageCountedGrid(t *testing.T) {
	dir, proj := webpageProject(t, nil)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "grid", WebpageGridCols: 2, WebpageGridRows: 3})
	if refusal != "" || plan == nil {
		t.Fatalf("counted grid plan refused: %q", refusal)
	}
	if got, want := string(plan.Files[0].After), WebpageGridContent(2, 3, "index"); got != want {
		t.Errorf("planned content is not the counted grid:\n got: %q\nwant: %q", got, want)
	}
	for _, want := range []string{"grid webpage with 2 columns and 3 rows", "6 boxes"} {
		if !strings.Contains(plan.Summary, want) {
			t.Errorf("summary does not state %q: %q", want, plan.Summary)
		}
	}
	applyServePlan(t, dir, plan)

	// Columns only: the rows default to 2, in the content
	// and in the summary alike. (A fresh project — the one
	// above now holds the landed page.)
	_, proj = webpageProject(t, nil)
	plan, refusal = proj.PlanEdit(EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "grid", WebpageGridCols: 3})
	if refusal != "" || plan == nil {
		t.Fatalf("columns-only plan refused: %q", refusal)
	}
	if got, want := string(plan.Files[0].After), WebpageGridContent(3, 2, "index"); got != want {
		t.Errorf("columns-only content is not the 3x2 grid")
	}
	if !strings.Contains(plan.Summary, "grid webpage with 3 columns and 2 rows") {
		t.Errorf("columns-only summary does not state the defaulted rows: %q", plan.Summary)
	}

	for _, tc := range []struct {
		name string
		req  EditRequest
		want string
	}{
		{"too many columns", EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "grid", WebpageGridCols: 7, WebpageGridRows: 2}, "1 to 6 columns and 1 to 6 rows"},
		{"too many rows", EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "grid", WebpageGridCols: 2, WebpageGridRows: 9}, "1 to 6 columns and 1 to 6 rows"},
		{"rows without columns", EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "grid", WebpageGridRows: 3}, "names its columns"},
		{"counts on another template", EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "basic", WebpageGridCols: 2, WebpageGridRows: 3}, "belong to the grid template"},
	} {
		plan, refusal := proj.PlanEdit(tc.req)
		if plan != nil || !strings.Contains(refusal, tc.want) {
			t.Errorf("%s: plan=%v refusal=%q — want a refusal containing %q", tc.name, plan != nil, refusal, tc.want)
		}
	}
}

// The material template: Material Design 3 from scratch in
// the page's own bytes — the baseline color roles as
// variables, the dark scheme flipping them, every starter
// component present, the marker written, and not one line
// of JavaScript or anything loaded from elsewhere.
func TestWebpageMaterialTemplate(t *testing.T) {
	page := WebpageContent("material", "index")
	for _, want := range []string{
		"<title>index</title>", "gollemer-material",
		"--primary: #6750A4", "--on-primary: #FFFFFF",
		"--primary-container: #EADDFF", "--on-primary-container: #21005D",
		"--secondary: #625B71", "--surface: #FEF7FF", "--on-surface: #1D1B20",
		"--surface-variant: #E7E0EC", "--outline: #79747E", "--surface-container: #F3EDF7",
		"@media (prefers-color-scheme: dark)", "--primary: #D0BCFF", "--surface: #141218", "--on-surface: #E6E0E9",
		`class="app-bar"`, "btn btn-filled", "btn-tonal", "btn-outlined", "btn-text",
		`class="card"`, `class="fab"`, `class="chip chip-selected"`, `class="field"`, `class="switch"`,
		">Buttons<", ">Card<", ">Chips<", ">Text field<", ">Switch<",
	} {
		if !strings.Contains(page, want) {
			t.Errorf("material page missing %q", want)
		}
	}
	for _, never := range []string{"<script", "http://", "https://", "@import", "url("} {
		if strings.Contains(page, never) {
			t.Errorf("material page carries %q — the look is hand-written in the page, nothing loaded, no JavaScript:\n%s", never, page)
		}
	}
	_, proj := webpageProject(t, nil)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "material"})
	if refusal != "" || plan == nil {
		t.Fatalf("material plan refused: %q", refusal)
	}
	if got := string(plan.Files[0].After); got != page {
		t.Errorf("planned content is not the material page")
	}
	if !strings.Contains(plan.Summary, "material webpage") {
		t.Errorf("summary does not name the template: %q", plan.Summary)
	}
}

// The layout templates, in the page's own bytes and at the
// planner: each carries its structure in the style block,
// collapses the way its blurb says, and plans clean.
func TestWebpageLayoutTemplates(t *testing.T) {
	layouts := []struct {
		id    string
		wants []string
	}{
		{"two-columns", []string{`<meta name="viewport"`, ".columns { display: flex;", ".columns main { flex: 2; }", "@media (max-width: 40em)", "flex-direction: column", "<aside>", "<footer>"}},
		{"three-columns", []string{`<meta name="viewport"`, ".columns { display: flex;", ".columns main { flex: 2; }", "@media (max-width: 40em)", "flex-direction: column", "left sidebar", "right sidebar"}},
		{"grid", []string{`<meta name="viewport"`, "display: grid", "repeat(3, 1fr)", "repeat(2, 1fr)", "grid-template-columns: 1fr", `class="card"`}},
		{"rows", []string{`<meta name="viewport"`, "band band-top", "band band-content", "band band-bottom", "#dce9f7", "#b9cfe8"}},
	}
	_, proj := webpageProject(t, nil)
	for _, tc := range layouts {
		page := WebpageContent(tc.id, "index")
		for _, want := range []string{"<title>index</title>", "<h1>index</h1>", "TODO:"} {
			if !strings.Contains(page, want) {
				t.Errorf("%s page missing %q:\n%s", tc.id, want, page)
			}
		}
		for _, want := range tc.wants {
			if !strings.Contains(page, want) {
				t.Errorf("%s page missing %q:\n%s", tc.id, want, page)
			}
		}
		plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: tc.id})
		if refusal != "" || plan == nil {
			t.Errorf("%s: plan refused: %q", tc.id, refusal)
			continue
		}
		if got, want := string(plan.Files[0].After), page; got != want {
			t.Errorf("%s: planned content is not the template's page", tc.id)
		}
		if !strings.Contains(plan.Summary, WebpageTemplateWords(tc.id)) {
			t.Errorf("%s: summary does not name the template: %q", tc.id, plan.Summary)
		}
	}
}
