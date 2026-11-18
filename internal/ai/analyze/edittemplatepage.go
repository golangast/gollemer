package analyze

// The templated page scaffold: a page as THREE files, not
// one. The named folder gains a templates folder holding a
// shared header (the page's head, the Material styles, and
// the responsive Material navigation bar), a shared footer,
// and this page's own body file between them. The three are
// joined when the page is served, by Go's html/template:
// the body calls {{template "header" .}} at its top and
// {{template "footer" .}} at its foot, and the handler the
// plan adds in the folder's package parses the three files
// and executes the body.
//
// The split is the point (John, 2026-11-09): the header and
// footer are written once and shared by every page the
// folder grows, while each page's words live in a body file
// of their own. The files are never overwritten — a header
// that is already there refuses the whole plan, because a
// second scaffold in the same folder reuses the shared
// pieces that exist rather than replacing them (a body file
// that is already there refuses the same way: it is a page
// that exists).
//
// The handler parses its templates on each request, from
// project-root-relative paths the plan writes as literals —
// honest starter behaviour a beginner can follow, whose one
// rule the generated comment states: run the server from
// the project root. No route is wired: the project's
// existing server keeps its routes, and this handler waits
// for the wiring sentence like every other unwired handler.

import (
	"bytes"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// templatedPageNameRe is the shape a page name must have:
// one plain word a file name and a Go handler name can both
// be built from — a letter, then letters, digits, hyphens
// and underscores.
var templatedPageNameRe = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_-]*$`)

// TemplatedPageHandlerName derives the handler's name from
// the page's name: each hyphen/underscore segment is
// capitalized and the segments joined, with "Page" at the
// end — "home" serves through HomePage, "about-us" through
// AboutUsPage.
func TemplatedPageHandlerName(name string) string {
	var b strings.Builder
	for _, seg := range strings.FieldsFunc(name, func(r rune) bool { return r == '-' || r == '_' }) {
		runes := []rune(seg)
		if len(runes) == 0 {
			continue
		}
		runes[0] = unicode.ToUpper(runes[0])
		b.WriteString(string(runes))
	}
	return b.String() + "Page"
}

// templatedPageHandlerFile is the handler's file name: the
// page name in file-safe letters, with "page.go" at the end
// — "home" lands in homepage.go.
func templatedPageHandlerFile(name string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(name) {
		switch {
		case r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_':
			b.WriteRune(r)
		case r == '-':
			b.WriteRune('_')
		}
	}
	return b.String() + "page.go"
}

// TemplatedPageFiles names the scaffold's four files for a
// folder and page name, project-relative: the shared header
// and footer, the page's body, and the handler.
func TemplatedPageFiles(folder, name string) (header, footer, body, handler string) {
	templates := path.Join(folder, "templates")
	return path.Join(templates, "header.html"),
		path.Join(templates, "footer.html"),
		path.Join(templates, name+".html"),
		path.Join(folder, templatedPageHandlerFile(name))
}

// templatedPageHeaderContent is the shared header file: the
// "header" definition — doctype and head (with the styles:
// they live here because the header is on every page), the
// responsive Material navigation bar, and the opening body
// tag. The navigation markup is the component piece's own
// constant, so a scaffold page and a page that grew its nav
// by sentence can never drift apart.
func templatedPageHeaderContent() string {
	return "{{define \"header\"}}\n" +
		"<!DOCTYPE html>\n" +
		"<html lang=\"en\">\n" +
		"<head>\n" +
		"<meta charset=\"utf-8\">\n" +
		"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" +
		"<title>TODO: your site's name</title>\n" +
		"<style>\n" +
		"/* This page's styles live in the header because the header is part of every templated page. */\n" +
		MaterialCSSVariables() +
		MaterialCSSBase() +
		MaterialNavigationBarCSS() +
		MaterialFooterCSS() +
		"</style>\n" +
		"</head>\n" +
		"<body>\n" +
		MaterialNavigationBarSnippet +
		"{{end}}\n"
}

// templatedPageFooterContent is the shared footer file:
// the "footer" definition — the Material footer bar (the
// component piece's own constant) and the closing tags the
// header opened.
func templatedPageFooterContent() string {
	return "{{define \"footer\"}}\n" +
		MaterialFooterSnippet +
		"</body>\n" +
		"</html>\n" +
		"{{end}}\n"
}

// templatedPageBodyContent is one page's body file: the
// header call, a welcome section waiting for the user's
// words, and the footer call.
func templatedPageBodyContent(name string) string {
	return "{{template \"header\" .}}\n" +
		"<main>\n" +
		"<h1 class=\"headline\">Welcome to " + name + "</h1>\n" +
		"<p>TODO: this page's own words — this body is its own file (" + name + ".html, in the templates folder), between the shared header and footer it calls at its top and foot. The header and footer come from their own files, so changing one of them changes every page that calls it.</p>\n" +
		"</main>\n" +
		"{{template \"footer\" .}}\n"
}

// templatedPageHandlerContent renders the handler: it
// parses the body, header, and footer files — paths written
// as project-root-relative literals, in that order, so the
// returned template is the body — and executes the body by
// name. A parse or render failure is an honest HTTP error,
// never a panic.
func templatedPageHandlerContent(pkgName, folder, name string) string {
	header, footer, body, _ := TemplatedPageFiles(folder, name)
	handler := TemplatedPageHandlerName(name)
	var b strings.Builder
	b.WriteString("package " + pkgName + "\n\n")
	b.WriteString("import (\n\t\"html/template\"\n\t\"net/http\"\n)\n\n")
	fmt.Fprintf(&b, "// %s serves the %s page: the body file in the templates folder,\n", handler, name)
	b.WriteString("// joined to the shared header and footer by html/template when\n")
	b.WriteString("// the page is asked for. The three file names are written from\n")
	b.WriteString("// the project root, so run the server from the project root.\n")
	fmt.Fprintf(&b, "func %s(w http.ResponseWriter, r *http.Request) {\n", handler)
	fmt.Fprintf(&b, "\tpage, err := template.ParseFiles(%s, %s, %s)\n", strconv.Quote(body), strconv.Quote(header), strconv.Quote(footer))
	b.WriteString("\tif err != nil {\n")
	b.WriteString("\t\thttp.Error(w, \"This page's templates could not be read: \"+err.Error(), http.StatusInternalServerError)\n")
	b.WriteString("\t\treturn\n\t}\n")
	fmt.Fprintf(&b, "\tif err := page.ExecuteTemplate(w, %s, nil); err != nil {\n", strconv.Quote(path.Base(body)))
	b.WriteString("\t\thttp.Error(w, \"This page could not be shown: \"+err.Error(), http.StatusInternalServerError)\n")
	b.WriteString("\t\treturn\n\t}\n}\n")
	return b.String()
}

// ScaffoldSectionWords renders a section's counts the way
// sentences and plans name them: "2 columns and 3 rows".
func ScaffoldSectionWords(cols, rows int) string {
	return countPhrase(cols, "column", "columns") + " and " + countPhrase(rows, "row", "rows")
}

// scaffoldSectionContent is one columns-and-rows section
// for a body file: a grid of exactly cols*rows boxes, in
// the counted grid's card style, under a comment that says
// what it is and where its styles live.
func scaffoldSectionContent(cols, rows int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- A columns-and-rows section: %s across by %s down, %s. Its styles live in the header, with this page's other styles. -->\n",
		countPhrase(cols, "column", "columns"), countPhrase(rows, "row", "rows"), countPhrase(cols*rows, "box", "boxes"))
	fmt.Fprintf(&b, "<div class=\"body-grid body-grid-%d\">\n", cols)
	for n := 1; n <= cols*rows; n++ {
		fmt.Fprintf(&b, "<div class=\"card\">\n<p>TODO: box %d's words — one box per thing this section shows off.</p>\n</div>\n", n)
	}
	b.WriteString("</div>\n")
	return b.String()
}

// scaffoldSectionBaseCSS is the styles every body section
// shares — the grid flow, the gaps, the card look — carried
// into the header's style block once, under its marker.
const scaffoldSectionBaseCSS = "/* Columns-and-rows sections in the body: the grid and its cards. */\n" +
	".body-grid { display: grid; gap: 1em; margin: 1em 0; }\n" +
	".body-grid .card { border: 1px solid var(--outline); border-radius: 0.75em; padding: 1em; }\n"

const scaffoldSectionBaseProbe = "/* Columns-and-rows sections in the body"

// scaffoldSectionColsCSS is one column count's rule, also
// carried once per count: how many boxes sit across, and
// the narrow-screen collapse to one.
func scaffoldSectionColsCSS(cols int) string {
	s := fmt.Sprintf("/* A body section %s across. */\n.body-grid-%d { grid-template-columns: repeat(%d, 1fr); }\n",
		countPhrase(cols, "column", "columns"), cols, cols)
	if cols > 1 {
		s += fmt.Sprintf("@media (max-width: 32em) { .body-grid-%d { grid-template-columns: 1fr; } }\n", cols)
	}
	return s
}

func scaffoldSectionColsProbe(cols int) string {
	// The full rule, not just the selector: the narrow
	// screen's media query repeats the selector with a
	// different body, and only the rule itself proves the
	// count's styles are in.
	return fmt.Sprintf(".body-grid-%d { grid-template-columns: repeat(%d, 1fr); }", cols, cols)
}

// planScaffoldSection inserts one counted section into a
// scaffold body file, just before its footer call — below
// everything the body already holds, so sections
// accumulate in ask order — and carries the section's
// styles into the shared header's style block, each chunk
// once, the way component styles are carried. The footer
// file is never touched, and the header is only touched
// when a style chunk is genuinely missing.
func (p *Project) planScaffoldSection(req EditRequest) (*EditPlan, string) {
	folder := path.Clean(filepath.ToSlash(strings.TrimSpace(req.SiteFolder)))
	cols, rows := req.SiteCols, req.SiteRows
	if cols < 1 || cols > 6 || rows < 1 || rows > 6 {
		return nil, fmt.Sprintf("a columns-and-rows section can have 1 to 6 columns and 1 to 6 rows — %s is outside that. Ask again with numbers in that range.", ScaffoldSectionWords(cols, rows))
	}
	headerRel, _, _, _ := TemplatedPageFiles(folder, "x")
	bodyRel := path.Clean(filepath.ToSlash(strings.TrimSpace(req.SiteBody)))
	if bodyRel == headerRel || bodyRel == path.Join(folder, "templates", "footer.html") ||
		!strings.HasPrefix(bodyRel, path.Join(folder, "templates")+"/") || !strings.HasSuffix(bodyRel, ".html") {
		return nil, "that edit needs the templated page's body file — a page file in the folder's templates folder, not the shared header or footer."
	}
	headerBefore, herr := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(headerRel)))
	if herr != nil {
		return nil, fmt.Sprintf("there's no templated page in %s yet — the header that every body section's styles join isn't there. Say \"add a templated page called <name> in %s\" first, then ask again.", folder, folder)
	}
	bodyBefore, berr := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(bodyRel)))
	if berr != nil {
		return nil, fmt.Sprintf("%s isn't there — the body the section belongs in is missing. Say \"add a templated page called <name> in %s\" first, then ask again.", p.LinkPath(bodyRel), folder)
	}
	header := string(headerBefore)
	body := string(bodyBefore)
	footerCall := strings.Index(body, "{{template \"footer\"")
	if footerCall < 0 {
		return nil, fmt.Sprintf("%s doesn't call the shared footer at its foot, so it isn't a templated page's body I recognise — the section has nowhere honest to land.", p.LinkPath(bodyRel))
	}
	styleEnd := strings.Index(header, "</style>")
	if styleEnd < 0 {
		return nil, fmt.Sprintf("%s has no style block for the section's styles to join — it isn't the scaffold's header I recognise.", p.LinkPath(headerRel))
	}
	var css strings.Builder
	if !strings.Contains(header, scaffoldSectionBaseProbe) {
		css.WriteString(scaffoldSectionBaseCSS)
	}
	if !strings.Contains(header, scaffoldSectionColsProbe(cols)) {
		css.WriteString(scaffoldSectionColsCSS(cols))
	}
	section := scaffoldSectionContent(cols, rows)
	newBody := body[:footerCall] + section + body[footerCall:]
	pageName := strings.TrimSuffix(path.Base(bodyRel), ".html")
	files := []EditFile{}
	if css.Len() > 0 {
		files = append(files, finishFile(headerRel, headerBefore, []byte(header[:styleEnd]+css.String()+header[styleEnd:])))
	}
	files = append(files, finishFile(bodyRel, bodyBefore, []byte(newBody)))
	beforeSections := strings.Count(body, "class=\"body-grid ")
	summary := fmt.Sprintf("Add a section of %s (%s) to the %s page's body in %s, below what the body already holds and above its footer call%s.",
		ScaffoldSectionWords(cols, rows), countPhrase(cols*rows, "box", "boxes"), pageName, p.LinkPath(bodyRel),
		map[bool]string{true: ", with the section's styles joining the header's style block in " + p.LinkPath(headerRel), false: " — its styles are already in the header"}[css.Len() > 0])
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("the %s page's body in %s now has a section of %s", pageName, p.LinkPath(bodyRel), ScaffoldSectionWords(cols, rows)),
		Site:       p.LinkPath(bodyRel),
		Files:      files,
		Notes: []string{
			"The section lands in the body file only — the shared footer is untouched, and the header changes only if a style it needs is missing.",
		},
		verify: func(after *Project) bool {
			b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(bodyRel)))
			if err != nil || string(b) != newBody {
				return false
			}
			if strings.Count(string(b), "class=\"body-grid ") != beforeSections+1 {
				return false
			}
			h, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(headerRel)))
			if err != nil {
				return false
			}
			return strings.Contains(string(h), scaffoldSectionBaseProbe) &&
				strings.Contains(string(h), scaffoldSectionColsProbe(cols))
		},
	}, ""
}

// planTemplatedPage assembles the scaffold: four new files,
// each refused if it already exists. The folder must exist
// (a two-step, like the neighbours), its Go files must
// speak one package (which the handler joins), and the
// handler's name must be free in that package.
func (p *Project) planTemplatedPage(req EditRequest) (*EditPlan, string) {
	folder := path.Clean(filepath.ToSlash(strings.TrimSpace(req.SiteFolder)))
	name := strings.TrimSpace(req.SiteName)
	if folder == "" || folder == "/" {
		return nil, "that edit needs the folder the templated page belongs in."
	}
	if !templatedPageNameRe.MatchString(name) {
		return nil, fmt.Sprintf("%q isn't a plain page name — a templated page's name is one word of letters, digits, hyphens, and underscores, starting with a letter, like \"home\".", name)
	}
	if strings.EqualFold(name, "header") || strings.EqualFold(name, "footer") {
		return nil, fmt.Sprintf("%q names one of the shared pieces — every templated page has a header and a footer of its own kind. Call the page something else.", name)
	}
	folderAbs := filepath.Join(p.Root, filepath.FromSlash(folder))
	if st, err := os.Stat(folderAbs); err != nil || !st.IsDir() {
		if err == nil {
			return nil, fmt.Sprintf("%s isn't a folder — a templated page belongs to a folder, with its handler in the folder's package.", folder)
		}
		return nil, fmt.Sprintf("there's no folder %s in this project for the templated page to live in — say \"create a folder %s\" first, then ask again.", folder, folder)
	}
	// The handler joins the folder's package, read off its
	// own files the way the folder planners read it: one
	// package lends its name, no Go files yet takes the page
	// host default, and a folder mixing packages can't say
	// which package a new file belongs to.
	pkgDir := folder
	if folder == "." {
		pkgDir = ""
	}
	pkgs, _ := folderPackages(p.Root, folder)
	pkgName := ""
	switch {
	case len(pkgs) == 0:
		pkgName = p.pageHostPackage(pkgDir)
	case len(pkgs) == 1:
		for n := range pkgs {
			pkgName = n
		}
	default:
		names := make([]string, 0, len(pkgs))
		for n := range pkgs {
			names = append(names, n)
		}
		sort.Strings(names)
		var parts []string
		for _, n := range names {
			var files []string
			for _, f := range pkgs[n] {
				files = append(files, p.LinkPath(path.Join(folder, f)))
			}
			parts = append(parts, fmt.Sprintf("package %s (%s)", n, strings.Join(files, ", ")))
		}
		return nil, fmt.Sprintf("the %s folder mixes %s — Go allows one package per folder, so I can't tell which package the %s handler belongs to. Sort the folder out by hand, then ask again.", folder, strings.Join(parts, " and "), name)
	}
	handlerName := TemplatedPageHandlerName(name)
	headerRel, footerRel, bodyRel, handlerRel := TemplatedPageFiles(folder, name)
	if pkgName2, file, line, found := p.pageNameCollision(pkgDir, handlerName, handlerRel, nil, nil); found {
		return nil, fmt.Sprintf("package %s already has a %s in %s:%d — a second one can't join it. Call the page something else.", pkgName2, handlerName, p.LinkPath(file), line)
	}
	templatesAbs := filepath.Join(folderAbs, "templates")
	if st, err := os.Stat(templatesAbs); err == nil && !st.IsDir() {
		return nil, fmt.Sprintf("%s is a file, not a folder — the templated page's header, footer, and body live in a templates folder there.", p.LinkPath(path.Join(folder, "templates")))
	}
	for _, rel := range []string{headerRel, footerRel, bodyRel, handlerRel} {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
		}
		if _, err := os.Lstat(filepath.Join(p.Root, filepath.FromSlash(rel))); err == nil {
			return nil, fmt.Sprintf("%s is already there — I never overwrite what's yours, so nothing was written. A second templated page in %s reuses the header and footer that exist; this plan writes none of the four files.", p.LinkPath(rel), folder)
		}
	}
	header := templatedPageHeaderContent()
	footer := templatedPageFooterContent()
	body := templatedPageBodyContent(name)
	handler := templatedPageHandlerContent(pkgName, folder, name)
	files := []EditFile{
		finishFile(headerRel, nil, []byte(header)),
		finishFile(footerRel, nil, []byte(footer)),
		finishFile(bodyRel, nil, []byte(body)),
		finishFile(handlerRel, nil, []byte(handler)),
	}
	want := map[string]string{
		headerRel: header, footerRel: footer, bodyRel: body, handlerRel: handler,
	}
	summary := fmt.Sprintf("Create the %s templated page in %s, as separate files joined by html/template: %s (the head, the Material styles, and the responsive navigation bar), %s (the footer bar), and %s (this page's body, calling the header at its top and the footer at its foot) — plus the handler %s in %s (package %s), which parses the three files and renders the body. No route is wired, so %s answers nowhere yet.",
		name, folder, p.LinkPath(headerRel), p.LinkPath(footerRel), p.LinkPath(bodyRel), handlerName, p.LinkPath(handlerRel), pkgName, handlerName)
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("the %s templated page's body is in %s, joined to the header in %s and the footer in %s, and its handler %s is in %s", name, p.LinkPath(bodyRel), p.LinkPath(headerRel), p.LinkPath(footerRel), handlerName, p.LinkPath(handlerRel)),
		Site:       p.LinkPath(handlerRel),
		Files:      files,
		Notes: []string{
			"The handler reads its templates from the project root — run the server from there.",
			fmt.Sprintf("No route is wired by this plan — %s is placed, not registered.", handlerName),
		},
		verify: func(after *Project) bool {
			for rel, content := range want {
				b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(rel)))
				if err != nil || !bytes.Equal(b, []byte(content)) {
					return false
				}
			}
			for _, pkg := range after.Packages {
				if pkg.Dir != pkgDir {
					continue
				}
				for _, fn := range pkg.Funcs {
					if fn.Receiver == "" && fn.Name == handlerName {
						return true
					}
				}
			}
			return false
		},
	}, ""
}
