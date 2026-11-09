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
// The refusals are guidance, in the package's two-step
// voice: a missing file and an empty file both point at
// the webpage piece first (a template lays the page out),
// and a page with no </body> — or no <head> to hold new
// styles — names the structure it is missing.

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func (p *Project) planComponent(req EditRequest) (*EditPlan, string) {
	rel := filepath.ToSlash(strings.TrimSpace(req.File))
	comp, ok := MaterialComponentByID(req.Component)
	if !ok {
		return nil, "that edit needs one of the material components: " + componentFormsText() + "."
	}
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
