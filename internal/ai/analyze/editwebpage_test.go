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
		{"no template", EditRequest{Kind: EditWebpage, File: "jim/temp/index.html"}, "templates: basic, responsive, or with assets"},
		{"unknown template", EditRequest{Kind: EditWebpage, File: "jim/temp/index.html", WebpageTemplate: "fancy"}, "templates: basic, responsive, or with assets"},
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
