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
// the same whether or not that wiring has run yet. The layout
// templates (two columns, three columns, grid, rows) carry
// their structure in an embedded style block; the column and
// grid layouts collapse to fewer — finally one — stacked
// columns on a narrow screen, and the rows are full-width
// bands by construction. The material template is a
// Material Design 3 look written from scratch in plain CSS
// (editmaterial.go), where the component chunks live.

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
	case "two-columns":
		return "two-column webpage"
	case "three-columns":
		return "three-column webpage"
	case "grid":
		return "grid webpage"
	case "rows":
		return "rows webpage"
	case "material":
		return "material webpage"
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
	case "two-columns":
		return "a header, a sidebar about a third of the width beside a main area about two thirds wide, and a footer — on a narrow screen the two columns stack into one"
	case "three-columns":
		return "a header, a sidebar on each side of the main area, and a footer — on a narrow screen the three columns stack into one"
	case "grid":
		return "a heading and a grid of cards — three across on a wide screen, two on a middling one, one on a phone — each card waiting for your words"
	case "rows":
		return "full-width bands stacked down the page — a top band, a content band, and a bottom band — each in a different shade, so the page's parts are easy to see"
	case "material":
		return "Google's Material Design 3 look, written by hand in plain CSS in the page itself — the Material colors as variables, an app bar, buttons, a card, chips, a text field, and a switch; no framework, no library, nothing loaded from the web, and a dark scheme that follows the reader's own setting"
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
	case "two-columns":
		return "<!DOCTYPE html>\n" +
			"<html lang=\"en\">\n" +
			"<head>\n" +
			"<meta charset=\"utf-8\">\n" +
			"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" +
			"<title>" + title + "</title>\n" +
			"<style>\n" +
			"body { font-family: sans-serif; line-height: 1.5; margin: 0 auto; max-width: 64em; padding: 1em; }\n" +
			".columns { display: flex; gap: 1em; }\n" +
			".columns aside { flex: 1; background: #f2f2f2; padding: 1em; }\n" +
			".columns main { flex: 2; }\n" +
			"footer { margin-top: 1em; }\n" +
			"@media (max-width: 40em) { .columns { flex-direction: column; } }\n" +
			"</style>\n" +
			"</head>\n" +
			"<body>\n" +
			"<header>\n" +
			"<h1>" + title + "</h1>\n" +
			"</header>\n" +
			"<div class=\"columns\">\n" +
			"<aside>\n" +
			"<p>TODO: the sidebar — links, a menu, or a short note. It takes about a third of the width, beside the main area.</p>\n" +
			"</aside>\n" +
			"<main>\n" +
			"<p>TODO: the main area — your words here, in the wider two thirds of the page. On a narrow screen the sidebar stacks above this area instead of sitting beside it.</p>\n" +
			"</main>\n" +
			"</div>\n" +
			"<footer>\n" +
			"<p>TODO: the footer — the small print at the bottom of the page.</p>\n" +
			"</footer>\n" +
			"</body>\n" +
			"</html>\n"
	case "three-columns":
		return "<!DOCTYPE html>\n" +
			"<html lang=\"en\">\n" +
			"<head>\n" +
			"<meta charset=\"utf-8\">\n" +
			"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" +
			"<title>" + title + "</title>\n" +
			"<style>\n" +
			"body { font-family: sans-serif; line-height: 1.5; margin: 0 auto; max-width: 64em; padding: 1em; }\n" +
			".columns { display: flex; gap: 1em; }\n" +
			".columns aside { flex: 1; background: #f2f2f2; padding: 1em; }\n" +
			".columns main { flex: 2; }\n" +
			"footer { margin-top: 1em; }\n" +
			"@media (max-width: 40em) { .columns { flex-direction: column; } }\n" +
			"</style>\n" +
			"</head>\n" +
			"<body>\n" +
			"<header>\n" +
			"<h1>" + title + "</h1>\n" +
			"</header>\n" +
			"<div class=\"columns\">\n" +
			"<aside>\n" +
			"<p>TODO: the left sidebar — links or a menu, in the narrower column on the left.</p>\n" +
			"</aside>\n" +
			"<main>\n" +
			"<p>TODO: the main area — your words here, in the wider middle column. On a narrow screen all three columns stack into one, left sidebar first.</p>\n" +
			"</main>\n" +
			"<aside>\n" +
			"<p>TODO: the right sidebar — a short note or extra links, in the narrower column on the right.</p>\n" +
			"</aside>\n" +
			"</div>\n" +
			"<footer>\n" +
			"<p>TODO: the footer — the small print at the bottom of the page.</p>\n" +
			"</footer>\n" +
			"</body>\n" +
			"</html>\n"
	case "grid":
		return "<!DOCTYPE html>\n" +
			"<html lang=\"en\">\n" +
			"<head>\n" +
			"<meta charset=\"utf-8\">\n" +
			"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" +
			"<title>" + title + "</title>\n" +
			"<style>\n" +
			"body { font-family: sans-serif; line-height: 1.5; margin: 0 auto; max-width: 64em; padding: 1em; }\n" +
			".grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1em; }\n" +
			".card { border: 1px solid #ccc; padding: 1em; }\n" +
			"@media (max-width: 50em) { .grid { grid-template-columns: repeat(2, 1fr); } }\n" +
			"@media (max-width: 32em) { .grid { grid-template-columns: 1fr; } }\n" +
			"</style>\n" +
			"</head>\n" +
			"<body>\n" +
			"<h1>" + title + "</h1>\n" +
			"<p>TODO: a line about what these cards show — the grid fits three cards across on a wide screen, two on a middling one, and one on a phone.</p>\n" +
			"<div class=\"grid\">\n" +
			"<div class=\"card\">\n<p>TODO: this card's words — one card per thing the page shows off.</p>\n</div>\n" +
			"<div class=\"card\">\n<p>TODO: this card's words — one card per thing the page shows off.</p>\n</div>\n" +
			"<div class=\"card\">\n<p>TODO: this card's words — one card per thing the page shows off.</p>\n</div>\n" +
			"<div class=\"card\">\n<p>TODO: this card's words — one card per thing the page shows off.</p>\n</div>\n" +
			"<div class=\"card\">\n<p>TODO: this card's words — one card per thing the page shows off.</p>\n</div>\n" +
			"<div class=\"card\">\n<p>TODO: this card's words — one card per thing the page shows off.</p>\n</div>\n" +
			"</div>\n" +
			"</body>\n" +
			"</html>\n"
	case "rows":
		return "<!DOCTYPE html>\n" +
			"<html lang=\"en\">\n" +
			"<head>\n" +
			"<meta charset=\"utf-8\">\n" +
			"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" +
			"<title>" + title + "</title>\n" +
			"<style>\n" +
			"body { font-family: sans-serif; line-height: 1.5; margin: 0; }\n" +
			".band { padding: 2em 1em; }\n" +
			".band-top { background: #dce9f7; }\n" +
			".band-content { background: #ffffff; }\n" +
			".band-bottom { background: #b9cfe8; }\n" +
			"</style>\n" +
			"</head>\n" +
			"<body>\n" +
			"<header class=\"band band-top\">\n" +
			"<h1>" + title + "</h1>\n" +
			"<p>TODO: the top band — the page's name and its one-line promise sit here.</p>\n" +
			"</header>\n" +
			"<main class=\"band band-content\">\n" +
			"<p>TODO: the content band — your words here, in the plain middle band. Each band runs the full width of the screen, however wide or narrow it is.</p>\n" +
			"</main>\n" +
			"<footer class=\"band band-bottom\">\n" +
			"<p>TODO: the bottom band — the small print at the bottom of the page.</p>\n" +
			"</footer>\n" +
			"</body>\n" +
			"</html>\n"
	case "material":
		return "<!DOCTYPE html>\n" +
			"<html lang=\"en\">\n" +
			"<head>\n" +
			"<meta charset=\"utf-8\">\n" +
			"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" +
			"<title>" + title + "</title>\n" +
			"<style>\n" +
			materialCSSAll() +
			"</style>\n" +
			"</head>\n" +
			"<body>\n" +
			materialBody(title) +
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
	if template != "basic" && template != "responsive" && template != "assets" &&
		template != "two-columns" && template != "three-columns" && template != "grid" && template != "rows" &&
		template != "material" {
		return nil, "that edit needs one of the webpage templates: basic, responsive, with assets, two columns, three columns, grid, rows, or material."
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
