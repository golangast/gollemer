package analyze

import (
	"strings"
	"testing"
)

// componentProject scans a tempdir holding the given files,
// on the shared serve-fixture builder (a real Go project the
// scanner accepts).
func componentProject(t *testing.T, files map[string]string) *Project {
	t.Helper()
	base := map[string]string{
		"go.mod":            "module example.com/component\n\ngo 1.21\n",
		"doc.go":            "package component\n",
		"jim/temp/keep.css": "/* placeholder */\n",
	}
	for rel, src := range files {
		base[rel] = src
	}
	_, proj := serveProject(t, base)
	return proj
}

// planOnContent plans one component against a project whose
// page holds exactly content, returning the plan's After (or
// the refusal).
func planOnContent(t *testing.T, content, component string) (string, string) {
	t.Helper()
	proj := componentProject(t, map[string]string{"jim/temp/index.html": content})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, File: "jim/temp/index.html", Component: component})
	if refusal != "" || plan == nil {
		return "", refusal
	}
	return string(plan.Files[0].After), ""
}

// The style-carrying rule, at the planner: a button into a
// basic page carries the variables and the button chunk in
// exactly once; a second button adds neither again; a card
// into the same page adds only the card chunk; a button
// into a material page adds no CSS at all.
func TestPlanComponentStylesCarriedOnce(t *testing.T) {
	basic := WebpageContent("basic", "index")
	after, refusal := planOnContent(t, basic, "button")
	if refusal != "" {
		t.Fatalf("button into a basic page refused: %q", refusal)
	}
	for _, want := range []string{MaterialStyleMarker, "--primary: #6750A4", ".btn {", `class="btn btn-filled"`} {
		if !strings.Contains(after, want) {
			t.Errorf("after the first button, the page is missing %q", want)
		}
	}
	if strings.Count(after, "--primary: #6750A4") != 1 || strings.Count(after, ".btn {") != 1 {
		t.Error("the first button carried the styles more than once")
	}
	if !strings.Contains(after, "<style>") || strings.Index(after, "<style>") > strings.Index(after, "</head>") {
		t.Error("the styles did not land in a style element in the head")
	}
	if i, j := strings.Index(after, `class="btn btn-filled"`), strings.Index(after, "</body>"); i < 0 || i > j {
		t.Error("the button did not land just before </body>")
	}
	// The second button: snippet only.
	after2, refusal := planOnContent(t, after, "button")
	if refusal != "" {
		t.Fatalf("second button refused: %q", refusal)
	}
	if strings.Count(after2, "--primary: #6750A4") != 1 || strings.Count(after2, ".btn {") != 1 {
		t.Error("the second button carried the styles again")
	}
	if strings.Count(after2, `class="btn btn-filled"`) != 2 {
		t.Error("the second button did not land beside the first")
	}
	// A card into the same page: the card chunk only.
	after3, refusal := planOnContent(t, after, "card")
	if refusal != "" {
		t.Fatalf("card after button refused: %q", refusal)
	}
	if strings.Count(after3, "--primary: #6750A4") != 1 || strings.Count(after3, ".card {") != 1 {
		t.Error("the card did not add exactly its own chunk")
	}
	// A button into a material page: no CSS at all — the
	// after is the page with only the snippet inserted.
	material := WebpageContent("material", "index")
	after4, refusal := planOnContent(t, material, "button")
	if refusal != "" {
		t.Fatalf("button into a material page refused: %q", refusal)
	}
	snippet, _ := MaterialComponentSnippet("button", "")
	want := strings.Replace(material, "</body>", snippet+"</body>", 1)
	if after4 != want {
		t.Error("a button into a material page changed more than the snippet")
	}
}

// The navigation bar and footer: their snippets and style
// chunks land together, the nav keeps its pure-CSS toggle
// and breakpoint in the page, and a second footer carries
// no styles again.
func TestPlanComponentNavigationBarAndFooter(t *testing.T) {
	basic := WebpageContent("basic", "index")
	after, refusal := planOnContent(t, basic, "navigation-bar")
	if refusal != "" {
		t.Fatalf("navigation bar into a basic page refused: %q", refusal)
	}
	for _, want := range []string{
		MaterialNavigationBarSnippet,
		`class="site-nav-toggle"`, `type="checkbox"`,
		".site-nav-toggle:checked ~ .site-nav-links",
		"@media (max-width: 40em)",
	} {
		if !strings.Contains(after, want) {
			t.Errorf("page after the navigation bar is missing %q", want)
		}
	}
	if strings.Contains(after, "<script") {
		t.Error("the navigation bar brought JavaScript into the page")
	}
	after, refusal = planOnContent(t, after, "footer")
	if refusal != "" {
		t.Fatalf("footer after navigation bar refused: %q", refusal)
	}
	if !strings.Contains(after, MaterialFooterSnippet) || !strings.Contains(after, ".site-footer {") {
		t.Error("the footer snippet and styles did not land")
	}
	after2, refusal := planOnContent(t, after, "footer")
	if refusal != "" {
		t.Fatalf("second footer refused: %q", refusal)
	}
	if strings.Count(after2, ".site-footer {") != 1 || strings.Count(after2, `class="site-footer"`) != 2 {
		t.Error("the second footer did not add only its snippet")
	}
}

// The fragment mode, at the planner: a component aimed at
// a templated page's body file lands above the body's
// footer call, its styles joining the shared header
// exactly once — the footer file is not in the plan at
// all. A second component accumulates; the same one again
// is the already-present refusal; the navigation bar and
// the footer answer that the page already has them; and
// the shared header and footer as targets get the guidance
// pointing at the body.
func TestPlanComponentInScaffoldBody(t *testing.T) {
	scaffold := map[string]string{
		"jim/templates/header.html": templatedPageHeaderContent(),
		"jim/templates/footer.html": templatedPageFooterContent(),
		"jim/templates/home.html":   templatedPageBodyContent("home"),
	}
	proj := componentProject(t, scaffold)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, File: "jim/templates/home.html", Component: "card"})
	if refusal != "" || plan == nil {
		t.Fatalf("card into a scaffold body refused: %q", refusal)
	}
	if len(plan.Files) != 2 || plan.Files[0].Rel != "jim/templates/header.html" || plan.Files[1].Rel != "jim/templates/home.html" {
		t.Fatalf("the fragment plan touches %+v, want the header then the body", plan.Files)
	}
	header, body := string(plan.Files[0].After), string(plan.Files[1].After)
	snippet, _ := MaterialComponentSnippet("card", "")
	cardAt, footAt := strings.Index(body, snippet), strings.Index(body, "{{template \"footer\"")
	if cardAt < 0 || footAt < 0 || cardAt > footAt {
		t.Errorf("the card did not land above the footer call:\n%s", body)
	}
	for _, call := range []string{"{{template \"header\" .}}", "{{template \"footer\" .}}"} {
		if !strings.Contains(body, call) {
			t.Errorf("the body lost its %s call", call)
		}
	}
	if n := strings.Count(header, "/* gollemer-material: card */"); n != 1 {
		t.Errorf("the header holds the card styles %d times, want exactly 1", n)
	}
	if n := strings.Count(header, "--primary: #6750A4"); n != 1 {
		t.Errorf("the header's variables were duplicated: %d copies", n)
	}

	// From the landed state: a button accumulates in the
	// body and its chunk joins the header; the card chunk
	// is not carried again.
	landed := map[string]string{
		"jim/templates/header.html": header,
		"jim/templates/footer.html": templatedPageFooterContent(),
		"jim/templates/home.html":   body,
	}
	proj = componentProject(t, landed)
	plan, refusal = proj.PlanEdit(EditRequest{Kind: EditComponent, File: "jim/templates/home.html", Component: "button"})
	if refusal != "" || plan == nil {
		t.Fatalf("button into the grown body refused: %q", refusal)
	}
	if len(plan.Files) != 2 {
		t.Fatalf("the button plan touches %d files, want the header and the body", len(plan.Files))
	}
	header2, body2 := string(plan.Files[0].After), string(plan.Files[1].After)
	if strings.Count(header2, "/* gollemer-material: buttons */") != 1 || strings.Count(header2, "/* gollemer-material: card */") != 1 {
		t.Error("the header's chunks were not carried exactly once each")
	}
	btnAt, cardAt2 := strings.Index(body2, `class="btn btn-filled"`), strings.Index(body2, snippet)
	if btnAt < 0 || cardAt2 < 0 || cardAt2 > btnAt {
		t.Errorf("the button did not accumulate below the card:\n%s", body2)
	}

	// The same component again: the already-present
	// refusal, and no plan.
	proj = componentProject(t, map[string]string{
		"jim/templates/header.html": header2,
		"jim/templates/footer.html": templatedPageFooterContent(),
		"jim/templates/home.html":   body2,
	})
	plan, refusal = proj.PlanEdit(EditRequest{Kind: EditComponent, File: "jim/templates/home.html", Component: "card"})
	if plan != nil || !strings.Contains(refusal, "already has a card") || !strings.Contains(refusal, "nothing to add") {
		t.Errorf("the repeated card: plan=%v refusal=%q, want the already-present refusal", plan != nil, refusal)
	}
	// The navigation bar and the footer are already on the
	// page, in the shared files the body calls.
	plan, refusal = proj.PlanEdit(EditRequest{Kind: EditComponent, File: "jim/templates/home.html", Component: "navigation-bar"})
	if plan != nil || !strings.Contains(refusal, "already has its navigation bar") || !strings.Contains(refusal, "jim/templates/header.html") {
		t.Errorf("the navigation bar on a body: plan=%v refusal=%q", plan != nil, refusal)
	}
	plan, refusal = proj.PlanEdit(EditRequest{Kind: EditComponent, File: "jim/templates/home.html", Component: "footer"})
	if plan != nil || !strings.Contains(refusal, "already has its footer") || !strings.Contains(refusal, "jim/templates/footer.html") {
		t.Errorf("the footer on a body: plan=%v refusal=%q", plan != nil, refusal)
	}
	// The shared pieces as targets: guidance naming the
	// body sentence, never a landing.
	for _, target := range []string{"jim/templates/header.html", "jim/templates/footer.html"} {
		plan, refusal = proj.PlanEdit(EditRequest{Kind: EditComponent, File: target, Component: "card"})
		if plan != nil || !strings.Contains(refusal, "\"add a card to jim/templates/home.html\"") {
			t.Errorf("the card aimed at %s: plan=%v refusal=%q, want guidance naming the body sentence", target, plan != nil, refusal)
		}
	}
}

// The refusals: unknown component (answering with what
// exists, from the registry), a non-.html target, a missing
// file and an empty file (both two-step guidance at the
// webpage piece), and a page with no </body>.
func TestPlanComponentRefusals(t *testing.T) {
	rows := []struct {
		name  string
		files map[string]string
		req   EditRequest
		want  string
	}{
		{"unknown component", map[string]string{"jim/temp/index.html": WebpageContent("basic", "index")},
			EditRequest{Kind: EditComponent, File: "jim/temp/index.html", Component: "carousel"}, "material components: button, tonal button"},
		{"not html", map[string]string{"jim/main.go": "package main\n"},
			EditRequest{Kind: EditComponent, File: "jim/main.go", Component: "button"}, "isn't one"},
		{"missing file", map[string]string{},
			EditRequest{Kind: EditComponent, File: "jim/temp/index.html", Component: "button"}, "add a webpage to jim/temp/index.html"},
		{"empty file", map[string]string{"jim/temp/index.html": ""},
			EditRequest{Kind: EditComponent, File: "jim/temp/index.html", Component: "button"}, "is empty"},
		{"no body", map[string]string{"jim/temp/index.html": "<html>\n<head></head>\n<p>hi</p>\n</html>\n"},
			EditRequest{Kind: EditComponent, File: "jim/temp/index.html", Component: "button"}, "no </body>"},
	}
	for _, tc := range rows {
		proj := componentProject(t, tc.files)
		plan, refusal := proj.PlanEdit(tc.req)
		if plan != nil || !strings.Contains(refusal, tc.want) {
			t.Errorf("%s: plan=%v refusal=%q, want a refusal containing %q", tc.name, plan != nil, refusal, tc.want)
		}
	}
}

// The registry's own coherence: every row resolves by id
// and by every sentence form, carries CSS and a snippet,
// and its probe is present in its own chunk.
func TestMaterialRegistryCoherence(t *testing.T) {
	if len(MaterialComponents) != 12 {
		t.Fatalf("registry holds %d components, want the starter set of 12", len(MaterialComponents))
	}
	for _, c := range MaterialComponents {
		if _, ok := MaterialComponentByID(c.ID); !ok {
			t.Errorf("%s does not resolve by id", c.ID)
		}
		for _, form := range append([]string{c.Words}, c.Aliases...) {
			if got, ok := MaterialComponentByForm(form); !ok || got.ID != c.ID {
				t.Errorf("form %q resolves to %q/%v, want %s", form, got.ID, ok, c.ID)
			}
		}
		if !strings.Contains(c.CSS, c.CSSProbe) {
			t.Errorf("%s: its chunk does not contain its probe %q", c.ID, c.CSSProbe)
		}
		if snippet, ok := MaterialComponentSnippet(c.ID, "index"); !ok || snippet == "" {
			t.Errorf("%s carries no snippet", c.ID)
		}
	}
	// The material template carries every component's
	// chunk exactly once: the registry and the skeleton
	// are the same set.
	page := WebpageContent("material", "index")
	for _, c := range MaterialComponents {
		if !strings.Contains(page, c.CSSProbe) {
			t.Errorf("the material page is missing %s's styles (%q)", c.ID, c.CSSProbe)
		}
	}
	if !strings.Contains(page, MaterialStyleMarker) {
		t.Error("the material page does not write the styles marker")
	}
}
