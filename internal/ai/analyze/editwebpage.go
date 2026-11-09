package analyze

// The webpage piece (chat/webpage.go): one plain webpage from
// a named template, written into ONE .html file. The page
// family's kinds (form/home/about/contact/list/details) build
// a page+handler pair; this piece is the simpler thing a
// beginner asks for first — "put a webpage in this file" —
// and it touches no Go code at all.
//
// The file rules are the family's honesty rule, widened one
// honest step: a missing file is created, an existing EMPTY
// file is filled (the live-session case: the file was created
// through the chat moments before, and holds nothing), and an
// existing file with anything in it is never overwritten —
// the refusal says so and names the way back.
//
// Templates are deterministic content, in the book's skeleton
// style (ServeHTMLStarterPage): a complete, valid starter
// page whose words are marked TODO. The "assets" template
// references the conventional /static/ route the serve-static
// piece registers (editservestatic.go) — the page assembles
// the same whether or not that wiring has run yet.

import (
	"bytes"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// WebpageTemplateWords renders a template id the way
// sentences name it: "basic webpage", "responsive webpage",
// "webpage with assets".
func WebpageTemplateWords(template string) string {
	switch template {
	case "responsive":
		return "responsive webpage"
	case "assets":
		return "webpage with assets"
	default:
		return "basic webpage"
	}
}

// webpageTemplateBlurb is what a template holds, in plain
// words for the plan summary.
func webpageTemplateBlurb(template string) string {
	switch template {
	case "responsive":
		return "a title, a heading, and a paragraph, plus a viewport line and a small style block, so the page adapts to the screen's width"
	case "assets":
		return "a title, a heading, and a paragraph, wired to load its style from /static/style.css, a picture from /static/image.png, and a script from /static/script.js — the files the static file serving answers for"
	default:
		return "a title, a heading, and a paragraph — the smallest complete page"
	}
}

// WebpageContent renders one template as a complete page.
// The title and heading come from the page's own file name
// (index.html reads "index"), so the starter never carries
// another page's words.
func WebpageContent(template, title string) string {
	switch template {
	case "responsive":
		return "<!DOCTYPE html>\n" +
			"<html lang=\"en\">\n" +
			"<head>\n" +
			"<meta charset=\"utf-8\">\n" +
			"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" +
			"<title>" + title + "</title>\n" +
			"<style>\n" +
			"body { font-family: sans-serif; line-height: 1.5; margin: 0 auto; max-width: 42em; padding: 1em; }\n" +
			"img { max-width: 100%; height: auto; }\n" +
			"</style>\n" +
			"</head>\n" +
			"<body>\n" +
			"<h1>" + title + "</h1>\n" +
			"<p>TODO: your words here — this page fits the screen it is shown on: the viewport line above tells a phone to use its real width, and the style block keeps the text in a readable column and any pictures inside it.</p>\n" +
			"</body>\n" +
			"</html>\n"
	case "assets":
		return "<!DOCTYPE html>\n" +
			"<html lang=\"en\">\n" +
			"<head>\n" +
			"<meta charset=\"utf-8\">\n" +
			"<title>" + title + "</title>\n" +
			"<link rel=\"stylesheet\" href=\"/static/style.css\">\n" +
			"</head>\n" +
			"<body>\n" +
			"<h1>" + title + "</h1>\n" +
			"<p>TODO: your words here — this page loads its style from /static/style.css, its picture from /static/image.png, and its script from /static/script.js, so the server has to serve the folder those files live in: if it doesn't yet, ask me to serve static files.</p>\n" +
			"<img src=\"/static/image.png\" alt=\"TODO: say what the picture shows\">\n" +
			"<script src=\"/static/script.js\"></script>\n" +
			"</body>\n" +
			"</html>\n"
	default: // basic
		return "<!DOCTYPE html>\n" +
			"<html lang=\"en\">\n" +
			"<head>\n" +
			"<meta charset=\"utf-8\">\n" +
			"<title>" + title + "</title>\n" +
			"</head>\n" +
			"<body>\n" +
			"<h1>" + title + "</h1>\n" +
			"<p>TODO: your words here — this is a starter page. Change the heading and this paragraph, save the file, and refresh your browser to see it.</p>\n" +
			"</body>\n" +
			"</html>\n"
	}
}

// planWebpage assembles the webpage edit: the template's
// content as the whole after-state of one .html file —
// created (no before bytes) or filled (the before bytes are
// the empty file's). A file with content refuses, by the
// family's never-overwrite rule.
func (p *Project) planWebpage(req EditRequest) (*EditPlan, string) {
	rel := filepath.ToSlash(strings.TrimSpace(req.File))
	template := strings.TrimSpace(req.WebpageTemplate)
	if rel == "" || !strings.HasSuffix(strings.ToLower(rel), ".html") {
		return nil, "that edit needs the webpage's .html file name."
	}
	if template != "basic" && template != "responsive" && template != "assets" {
		return nil, "that edit needs one of the webpage templates: basic, responsive, or with assets."
	}
	if why, blocked := p.fileBlocked(rel); blocked {
		return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
	}
	abs := filepath.Join(p.Root, filepath.FromSlash(rel))
	var before []byte
	exists := false
	if b, err := os.ReadFile(abs); err == nil {
		exists = true
		before = b
		if len(bytes.TrimSpace(b)) > 0 {
			return nil, fmt.Sprintf("%s already has a page in it — I never overwrite what's yours, so nothing was written. If you want a fresh template page there instead, empty the file yourself (or remove it) and ask again.", p.LinkPath(rel))
		}
	} else {
		if st, serr := os.Stat(filepath.Dir(abs)); serr != nil || !st.IsDir() {
			dirRel := path.Dir(rel)
			return nil, fmt.Sprintf("there's no folder %s in this project for the webpage to live in — say \"create a folder %s\" first, then ask again", dirRel, dirRel)
		}
	}

	base := path.Base(rel)
	title := base
	if len(base) > len(".html") {
		title = base[:len(base)-len(".html")]
	}
	content := WebpageContent(template, title)
	ef := finishFile(rel, before, []byte(content))
	words := WebpageTemplateWords(template)
	summary := fmt.Sprintf("Create %s with the %s: %s. The words are marked TODO — make them yours, save, and the page is ready to serve.",
		p.LinkPath(rel), words, webpageTemplateBlurb(template))
	if exists {
		summary = fmt.Sprintf("Put the %s into %s — the file is there but empty, so nothing of yours is overwritten: %s. The words are marked TODO — make them yours, save, and the page is ready to serve.",
			words, p.LinkPath(rel), webpageTemplateBlurb(template))
	}
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s now holds the %s", p.LinkPath(rel), words),
		Site:       p.LinkPath(rel),
		Files:      []EditFile{ef},
		verify: func(after *Project) bool {
			b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(rel)))
			return err == nil && bytes.Equal(b, []byte(content))
		},
	}, ""
}
