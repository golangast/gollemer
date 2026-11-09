package analyze

// The component piece: one Material component (the registry
// in editmaterial.go) inserted into an existing .html page,
// just before its </body>. The styles travel with the
// component: a page that already carries the Material
// variables (the marker the material template writes, or a
// first component's insertion) gets the snippet alone; a
// page without them gets the variables and the component's
// own chunk inserted into its style block — creating the
// <style> element in <head> when the page has none — and a
// page that already carries a component's chunk never
// receives it twice, however many components are added.
// The whole ask is one file change, so the edit spine's
// transaction covers the snippet and the styles together.
//
// A templated page's files are routed by what they ARE
// (scaffoldPartOf), before the plain-page shape is read
// into them. The shared header and footer never take a
// component — the answer points at a page's body, with the
// sentence spelled against it. The body is a fragment: it
// has no </body> of its own (the closing tag lives in the
// footer it calls), so the component lands there in
// fragment mode (planComponentInBody) — the snippet above
// the body's footer call, the styles joining the shared
// header. And two components are already on every templated
// page — the navigation bar in the header, the footer bar
// in the footer file — so asking for either on a body is
// the already-present answer, never a second copy.
//
// The refusals are guidance, in the package's two-step
// voice: a missing file and an empty file both point at
// the webpage piece first (a template lays the page out),
// and a page with no </body> — or no <head> to hold new
// styles — names the structure it is missing.

import (
	"bytes"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

func (p *Project) planComponent(req EditRequest) (*EditPlan, string) {
	comp, ok := MaterialComponentByID(req.Component)
	if !ok {
		return nil, "that edit needs one of the material components: " + componentFormsText() + "."
	}
	if req.WasmFolder != "" {
		// The wasm mode (editwasm.go): the target is an app,
		// not a page — the component lands as a Go DOM
		// builder in the app's own files.
		return p.planComponentWasm(req, comp)
	}
	rel := filepath.ToSlash(strings.TrimSpace(req.File))
	if rel == "" || !strings.HasSuffix(strings.ToLower(rel), ".html") {
		return nil, "that edit adds a component to a .html page — " + rel + " isn't one."
	}
	if why, blocked := p.fileBlocked(rel); blocked {
		return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
	}
	abs := filepath.Join(p.Root, filepath.FromSlash(rel))
	before, err := os.ReadFile(abs)
	if err != nil {
		return nil, "there's no file " + rel + " yet — say \"add a webpage to " + rel + "\" and I'll fill it with a template first; then the " + comp.Words + " can go into the page it makes."
	}
	if len(bytes.TrimSpace(before)) == 0 {
		return nil, rel + " is empty, so it has no page body to hold the " + comp.Words + " yet — say \"add a webpage to " + rel + "\" first and the template lays the page out; then the " + comp.Words + " goes in."
	}
	content := string(before)
	// A templated page's part is routed by what it IS,
	// before the plain-page shape is read into it: the
	// footer file carries a </body> a component must never
	// land inside, and the body has none to land before.
	switch part := scaffoldPartOf(content); part {
	case "header", "footer":
		return nil, p.componentPartGuidance(rel, part, comp)
	case "body":
		return p.planComponentInBody(req, rel, comp, before, content)
	}
	if !strings.Contains(content, "</body>") {
		return nil, "there's no </body> in " + rel + " for the " + comp.Words + " to sit inside — the pages the templates make always close their body. Add the component to one of those, or close the body in the file first."
	}
	// The styles this page still needs: the variables when
	// the marker is absent, then the component's chunk when
	// its probe is.
	needCSS := ""
	if !strings.Contains(content, MaterialStyleMarker) {
		needCSS = MaterialCSSVariables()
	}
	if !strings.Contains(content, comp.CSSProbe) {
		needCSS += comp.CSS
	}
	after := content
	if needCSS != "" {
		if i := strings.Index(after, "</style>"); i >= 0 {
			after = after[:i] + needCSS + after[i:]
		} else if i := strings.Index(after, "</head>"); i >= 0 {
			after = after[:i] + "<style>\n" + needCSS + "</style>\n" + after[i:]
		} else {
			return nil, rel + " has no <head> to hold the " + comp.Words + "'s styles and no style block of its own — the pages the templates make have both. Add the component to one of those."
		}
	}
	snippet, _ := MaterialComponentSnippet(req.Component, pageTitleOf(content))
	j := strings.LastIndex(after, "</body>")
	after = after[:j] + snippet + after[j:]
	afterBytes := []byte(after)
	summary := "Put a " + comp.Words + " in " + p.LinkPath(rel) + ", just before </body>"
	verifyDesc := fmt.Sprintf("%s now has a %s just before </body>", p.LinkPath(rel), comp.Words)
	if needCSS != "" {
		summary += ", with the Material styles it needs"
		verifyDesc += ", and the Material styles it needs are in the page"
	} else {
		summary += " — its Material styles are already in the page"
	}
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       p.LinkPath(rel),
		Files:      []EditFile{finishFile(rel, before, afterBytes)},
		verify: func(after *Project) bool {
			b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(rel)))
			return err == nil && bytes.Equal(b, afterBytes)
		},
	}, ""
}

// scaffoldPartOf reads file content as a templated page's
// part, the analyze-side twin of the chat's recognition:
// "header" or "footer" when the file holds that shared
// definition, "body" when it calls both shared pieces —
// "" when it is none of them. Content is the recognition,
// as everywhere in this family: a file is routed by what
// it IS, never by its name.
func scaffoldPartOf(content string) string {
	switch {
	case strings.Contains(content, "{{define \"header\"}}"):
		return "header"
	case strings.Contains(content, "{{define \"footer\"}}"):
		return "footer"
	case strings.Contains(content, "{{template \"header\"") && strings.Contains(content, "{{template \"footer\""):
		return "body"
	}
	return ""
}

// scaffoldBodiesBeside lists the body files sharing a
// templates folder with the named file — the pages the
// shared header and footer serve — sorted, as
// project-relative paths.
func (p *Project) scaffoldBodiesBeside(rel string) []string {
	dir := path.Dir(rel)
	entries, err := os.ReadDir(filepath.Join(p.Root, filepath.FromSlash(dir)))
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".html") || name == "header.html" || name == "footer.html" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(dir), name))
		if err != nil {
			continue
		}
		if scaffoldPartOf(string(b)) == "body" {
			out = append(out, path.Join(dir, name))
		}
	}
	sort.Strings(out)
	return out
}

// componentPartGuidance answers a component ask aimed at
// a templated page's shared header or footer: a component
// never lands in one of the shared pieces — it goes in a
// page's body — so the answer says what the file is and
// names the body the ask can land in, with the sentence
// spelled against it (the template seam's explanation
// style). With no body beside it, it names the place
// without promising a sentence.
func (p *Project) componentPartGuidance(rel, part string, comp MaterialComponent) string {
	folder := path.Dir(path.Dir(rel))
	where := "in " + folder
	if folder == "." || folder == "" {
		where = "at the top of this project"
	}
	var what string
	if part == "header" {
		what = fmt.Sprintf("%s is the shared header of the templated pages %s — the file that holds each page's design, its Material styles — so a %s doesn't go in it.", p.LinkPath(rel), where, comp.Words)
	} else {
		what = fmt.Sprintf("%s is the shared footer of the templated pages %s — the closing piece at the foot of every page that calls it — so a %s doesn't go in it.", p.LinkPath(rel), where, comp.Words)
	}
	bodies := p.scaffoldBodiesBeside(rel)
	if len(bodies) == 0 {
		return what + " A component goes in a page's own body file instead — the file in the same templates folder that calls the header at its top and the footer at its foot."
	}
	body := bodies[0]
	page := strings.TrimSuffix(path.Base(body), ".html")
	return what + fmt.Sprintf(" A component goes in a page's body: \"add a %s to %s\" puts it in the %s page's body, with its styles joining the header.", comp.Words, body, page)
}

// planComponentInBody is the component piece's fragment
// mode: the target is a templated page's BODY file — a
// fragment with no </body> of its own, because the closing
// tag lives in the shared footer it calls. The snippet
// lands in the body just above its footer call, below
// everything the body already holds so components
// accumulate in ask order, and the styles it needs join
// the shared header's style block, each carried once —
// the scaffold section's own discipline
// (planScaffoldSection). The footer file is never touched,
// and the header is only touched when a style it needs is
// genuinely missing.
//
// Two components are already on every templated page and
// never land in a body: the navigation bar rides in the
// header the body calls, and the footer bar in the footer
// file, so asking for either answers that the page already
// has it. And a body, unlike a plain page, takes each
// component once — asking for one it already holds is the
// already-present answer, never a second copy.
func (p *Project) planComponentInBody(req EditRequest, rel string, comp MaterialComponent, before []byte, content string) (*EditPlan, string) {
	page := strings.TrimSuffix(path.Base(rel), ".html")
	dir := path.Dir(rel)
	headerRel := path.Join(dir, "header.html")
	footerRel := path.Join(dir, "footer.html")
	headerBefore, herr := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(headerRel)))
	if herr != nil {
		folder := path.Dir(dir)
		return nil, fmt.Sprintf("there's no shared header beside %s — the header whose style block a body component's styles join sits in the same folder and isn't there. Say \"add a templated page called <name> in %s\" first, then ask again.", p.LinkPath(rel), folder)
	}
	if why, blocked := p.fileBlocked(headerRel); blocked {
		return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(headerRel), why)
	}
	header := string(headerBefore)
	// The navigation bar already rides on this page: the
	// scaffold's header carries its snippet for every page
	// that calls it, so there is nothing to land. (A header
	// that has lost the snippet falls through — the bar
	// then lands in the body like any other component.)
	if comp.ID == "navigation-bar" && strings.Contains(header, MaterialNavigationBarSnippet) {
		return nil, fmt.Sprintf("the %s page already has its navigation bar — it lives in the header file, %s, that the body calls at its top, so it already shows on the page. Nothing to add.", page, p.LinkPath(headerRel))
	}
	// The footer likewise: the bar lives in the shared
	// footer the body calls at its foot, and a footer
	// snippet never goes into a body.
	if comp.ID == "footer" {
		if b, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(footerRel))); err == nil && strings.Contains(string(b), MaterialFooterSnippet) {
			return nil, fmt.Sprintf("the %s page already has its footer — the footer bar lives in the shared footer file, %s, that the body calls at its foot, so it already closes the page. Nothing to add.", page, p.LinkPath(footerRel))
		}
		return nil, fmt.Sprintf("a footer never goes in a page's body — the footer is the shared piece in %s that the body calls at its foot, and the footer bar belongs in that file.", p.LinkPath(footerRel))
	}
	styleEnd := strings.Index(header, "</style>")
	if styleEnd < 0 {
		return nil, fmt.Sprintf("%s has no style block for the %s's styles to join — it isn't the scaffold's header I recognise.", p.LinkPath(headerRel), comp.Words)
	}
	// The app bar speaks the page's title — in a templated
	// page the title lives in the header.
	snippet, _ := MaterialComponentSnippet(comp.ID, pageTitleOf(header))
	if strings.Contains(content, snippet) || (comp.ID == "app-bar" && strings.Contains(content, "<header class=\"app-bar\">")) {
		return nil, fmt.Sprintf("%s already has a %s in it — nothing to add.", p.LinkPath(rel), comp.Words)
	}
	// The styles this page still needs, read off the
	// header (where a templated page's styles live): the
	// variables when their marker is absent, then the
	// component's own chunk when the header does not carry
	// it whole. A chunk is carried whole or not at all, so
	// no other piece's styles can stand in for it.
	needCSS := ""
	if !strings.Contains(header, MaterialStyleMarker) {
		needCSS = MaterialCSSVariables()
	}
	if !strings.Contains(header, comp.CSS) {
		needCSS += comp.CSS
	}
	footerCall := strings.Index(content, "{{template \"footer\"")
	newBody := content[:footerCall] + snippet + content[footerCall:]
	files := []EditFile{}
	if needCSS != "" {
		files = append(files, finishFile(headerRel, headerBefore, []byte(header[:styleEnd]+needCSS+header[styleEnd:])))
	}
	files = append(files, finishFile(rel, before, []byte(newBody)))
	summary := fmt.Sprintf("Put a %s in the %s page's body in %s, just above its footer call", comp.Words, page, p.LinkPath(rel))
	verifyDesc := fmt.Sprintf("the %s page's body in %s now has a %s above its footer call", page, p.LinkPath(rel), comp.Words)
	if needCSS != "" {
		summary += ", with the Material styles it needs joining the header's style block in " + p.LinkPath(headerRel)
		verifyDesc += ", and the styles it needs are in the header"
	} else {
		summary += " — its Material styles are already in the header"
	}
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       p.LinkPath(rel),
		Files:      files,
		Notes: []string{
			"The " + comp.Words + " lands in the body file only — the shared footer is untouched, and the header changes only if a style it needs is missing.",
		},
		verify: func(after *Project) bool {
			b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(rel)))
			if err != nil || string(b) != newBody {
				return false
			}
			if strings.Count(string(b), snippet) != strings.Count(content, snippet)+1 {
				return false
			}
			h, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(headerRel)))
			if err != nil {
				return false
			}
			return strings.Contains(string(h), MaterialStyleMarker) && strings.Contains(string(h), comp.CSS)
		},
	}, ""
}

// componentFormsText renders the registry's canonical
// forms as prose ("button, tonal button, …, or switch")
// for refusals that answer with what exists.
func componentFormsText() string {
	words := []string{}
	for _, c := range MaterialComponents {
		words = append(words, c.Words)
	}
	if len(words) < 2 {
		return strings.Join(words, "")
	}
	return strings.Join(words[:len(words)-1], ", ") + ", or " + words[len(words)-1]
}

// pageTitleOf reads a page's <title> words, for the app
// bar's snippet ("" when the page names none).
func pageTitleOf(content string) string {
	i := strings.Index(content, "<title>")
	if i < 0 {
		return ""
	}
	rest := content[i+len("<title>"):]
	j := strings.Index(rest, "</title>")
	if j < 0 {
		return ""
	}
	return strings.TrimSpace(rest[:j])
}
