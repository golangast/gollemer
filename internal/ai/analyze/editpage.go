package analyze

// The page family: named page kinds as matched page+handler
// pairs. One ask — "add a form page", "add a homepage" —
// produces BOTH halves of a working page, plus its wiring
// when the project already has a server that can take it:
//
//   - the page itself (req.PageHTML, a project-relative .html
//     path): created with starter content, NEVER overwritten
//     when it already exists (that state is a refusal);
//   - the handler in req.File (the host .go file, created when
//     missing): a named func serving the page with
//     http.ServeFile. For the form kinds (form, contact) the
//     handler is the concept book's form-handler piece with
//     its GET branch adapted to serve the page, and the
//     starter page's form asks for exactly the fields that
//     piece reads — the two halves are built from ONE field
//     list, so they cannot drift apart;
//   - the registration, when a composable serving site (the
//     default mux, or a *http.ServeMux variable) lives in the
//     host file or is the ONLY composable site in the host
//     file's package: one HandleFunc line just before the
//     serve statement, assembled with the wire family's
//     machinery (editwire.go). One special incumbent is
//     replaced rather than refused: the webserver component's
//     own starter handler at "/" — the homepage ask is the
//     natural next step after "add a webserver", and the
//     starter exists to be replaced. Every other incumbent at
//     the route refuses, naming it.
//
// With no composable site the plan still lands page+handler
// and its note spells the finishing sentence ("connect the
// <Name> handler to the webserver") — placement first, wiring
// as its own gated step, exactly like the concept book's
// handlers.

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// formShowMarker is the exact comment line in the shipped
// form-handler template's non-POST branch — the place that
// shows the form. The page family's adaptation replaces that
// comment with the ServeFile call for the page; a template
// without it has drifted from the shape this recipe assembles
// from, and is refused rather than guessed at.
const formShowMarker = "// TODO: show the form here — a browser asks for it with GET before it can send anything"

// starterHelloMarker is the friendly line the webserver
// component's own starter handler answers with. A registration
// whose handler is an inline function containing it is the
// component's starter — the one incumbent the page family
// replaces (at "/", the homepage's route).
const starterHelloMarker = "hello — your webserver is running"

// kindWords is the display phrase for a page kind: "form
// page", "homepage", "about page", "contact page", "item list
// page", "details page". It rides the plan's summary, the
// page-exists refusal, and the name-collision refusal.
func kindWords(kind string) string {
	switch kind {
	case "form":
		return "form page"
	case "home":
		return "homepage"
	case "about":
		return "about page"
	case "contact":
		return "contact page"
	case "list":
		return "item list page"
	case "details":
		return "details page"
	}
	return "page"
}

// pageBaseForKind is the page file name a kind's starter page
// carries (its <title>): the chat layer derives the page path
// from the kind by this same table.
func pageBaseForKind(kind string) string {
	switch kind {
	case "form":
		return "form.html"
	case "home":
		return "index.html"
	case "about":
		return "about.html"
	case "contact":
		return "contact.html"
	case "list":
		return "items.html"
	case "details":
		return "details.html"
	}
	return "page.html"
}

// pageRouteForKind is the route a kind answers at, by the same
// table the chat layer derives PageRoute from. The planner
// only falls back to it when the request carries no route.
func pageRouteForKind(kind string) string {
	switch kind {
	case "form":
		return "/form"
	case "home":
		return "/"
	case "about":
		return "/about"
	case "contact":
		return "/contact"
	case "list":
		return "/items"
	case "details":
		return "/details"
	}
	return ""
}

// formValueRe finds a template's form-field reads:
// FormValue("email") names the field "email".
var formValueRe = regexp.MustCompile(`FormValue\("([^"]+)"\)`)

// formTemplateFields returns the template's FormValue
// arguments, in order, deduped — the ONE field list both the
// handler (as shipped in the template) and the starter page's
// form are built from.
func formTemplateFields(template string) []string {
	seen := map[string]bool{}
	var out []string
	for _, m := range formValueRe.FindAllStringSubmatch(template, -1) {
		if !seen[m[1]] {
			seen[m[1]] = true
			out = append(out, m[1])
		}
	}
	return out
}

// PageStarterPage is the starter page for one page kind: the
// serve-html recipe's shell (doctype, head, title = the page's
// file name) with a per-kind body. Form kinds (form, contact)
// carry the form generated from fields — the same list the
// handler reads. The bodies invent no facts: headings and
// items the user hasn't supplied are TODOs.
func PageStarterPage(kind, route string, fields []string) []byte {
	var body string
	switch kind {
	case "home":
		body = `<h1>TODO: your homepage heading</h1>
<p>TODO: write the words your homepage opens with here.</p>
`
	case "about":
		body = `<h1>About us</h1>
<p>TODO: write about yourselves here — who you are, and what this site is for.</p>
`
	case "contact":
		body = "<h1>Contact us</h1>\n" + pageFormMarkup(route, fields)
	case "form":
		body = "<h1>TODO: what this form is for</h1>\n" + pageFormMarkup(route, fields)
	case "list":
		body = `<h1>TODO: your items</h1>
<!-- TODO: replace these placeholder items with your real items, or loop over the data in the handler. -->
<ul>
<li>TODO: an item</li>
<li>TODO: an item</li>
<li>TODO: an item</li>
</ul>
`
	case "details":
		body = `<h1>TODO: the item's name</h1>
<dl>
<dt>Name</dt>
<dd>TODO: the item's name</dd>
<dt>Description</dt>
<dd>TODO: what the item is</dd>
<dt>Price</dt>
<dd>TODO: what the item costs</dd>
</dl>
`
	default:
		body = `<h1>TODO: your page</h1>
<p>TODO: put your own words here.</p>
`
	}
	return []byte(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>` + pageBaseForKind(kind) + `</title>
</head>
<body>
` + body + `</body>
</html>
`)
}

// pageFormMarkup renders the starter form for a form kind from
// the field list: one labelled control per field — an email
// input for "email", a textarea for "message", a text input
// for everything else — posting back to the page's own route.
func pageFormMarkup(route string, fields []string) string {
	var b strings.Builder
	b.WriteString("<form method=\"post\" action=\"" + route + "\">\n")
	for _, f := range fields {
		label := titleFirst(f)
		switch f {
		case "email":
			b.WriteString("<p><label>" + label + " <input type=\"email\" name=\"email\"></label></p>\n")
		case "message":
			b.WriteString("<p><label>" + label + " <textarea name=\"message\" rows=\"5\"></textarea></label></p>\n")
		default:
			b.WriteString("<p><label>" + label + " <input type=\"text\" name=\"" + f + "\"></label></p>\n")
		}
	}
	b.WriteString("<p><button type=\"submit\">Send</button></p>\n</form>\n")
	return b.String()
}

// pagePlainHandler renders a plain kind's handler: a named
// func answering with the page file, read from disk on every
// request (the serve-html convention: servePath is the page's
// project-root-relative path).
func pagePlainHandler(name, pageBase, servePath string) string {
	return "// " + name + " serves the " + pageBase + " page: the file is read from disk on\n" +
		"// every request, so edits to the page show on refresh.\n" +
		"func " + name + "(w http.ResponseWriter, r *http.Request) {\n" +
		"\thttp.ServeFile(w, r, " + strconv.Quote(servePath) + ")\n" +
		"}\n"
}

// replaceFormShowLine swaps the form template's show-the-form
// comment for the ServeFile call that shows THIS page, keeping
// the comment line's own indent. ok is false when the marker
// comment is absent — the template has drifted.
func replaceFormShowLine(rendered, servePath string) (string, bool) {
	lines := strings.Split(rendered, "\n")
	for i, ln := range lines {
		if strings.TrimSpace(ln) != formShowMarker {
			continue
		}
		indent := ln[:len(ln)-len(strings.TrimLeft(ln, " \t"))]
		lines[i] = indent + "http.ServeFile(w, r, " + strconv.Quote(servePath) + ")"
		return strings.Join(lines, "\n"), true
	}
	return "", false
}

// pageHostDir is the host file's package dir in the index's
// form ("" = the project root).
func pageHostDir(rel string) string {
	dir := path.Dir(rel)
	if dir == "." {
		return ""
	}
	return dir
}

// pageHostPackage resolves the package name a CREATED host
// file carries: the indexed package in that dir when there is
// one, else the package clause of the first non-test .go file
// on disk there, else "main" (which covers a new main.go and
// the project root).
func (p *Project) pageHostPackage(dir string) string {
	if pkg := p.pkgByDir(dir); pkg != nil {
		return pkg.Name
	}
	diskDir := p.Root
	if dir != "" {
		diskDir = filepath.Join(p.Root, filepath.FromSlash(dir))
	}
	entries, err := os.ReadDir(diskDir)
	if err == nil {
		for _, e := range entries {
			name := e.Name()
			if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
				continue
			}
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, filepath.Join(diskDir, name), nil, parser.PackageClauseOnly)
			if err == nil && f != nil && f.Name != nil {
				return f.Name.Name
			}
		}
	}
	return "main"
}

// pageNameCollision finds a top-level declaration of name in
// the host file's package: the index first (Funcs, Types and
// Vars of the package at hostDir), then the host file's own
// AST when it exists — which covers a host file the index
// never walked. It returns the package's name and the
// declaring file and line.
func (p *Project) pageNameCollision(hostDir, name, rel string, hostFset *token.FileSet, hostAST *ast.File) (string, string, int, bool) {
	for _, pkg := range p.Packages {
		if pkg.Dir != hostDir {
			continue
		}
		for _, fn := range pkg.Funcs {
			if fn.Receiver == "" && fn.Name == name {
				return pkg.Name, fn.File, fn.Line, true
			}
		}
		for _, t := range pkg.Types {
			if t.Name == name {
				return pkg.Name, t.File, t.Line, true
			}
		}
		for _, v := range pkg.Vars {
			if v.Name == name {
				return pkg.Name, v.File, v.Line, true
			}
		}
	}
	if hostAST != nil && hostFset != nil {
		for _, d := range hostAST.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				if d.Recv == nil && d.Name.Name == name {
					return hostAST.Name.Name, rel, lineOfPos(hostFset, d.Pos()), true
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if s.Name.Name == name {
							return hostAST.Name.Name, rel, lineOfPos(hostFset, s.Pos()), true
						}
					case *ast.ValueSpec:
						for _, n := range s.Names {
							if n.Name == name {
								return hostAST.Name.Name, rel, lineOfPos(hostFset, n.Pos()), true
							}
						}
					}
				}
			}
		}
	}
	return "", "", 0, false
}

// wireFileOf builds the wireFile for a parsed file the wire
// scan hasn't seen (an unindexed host file), mirroring
// wireScan's own construction.
func wireFileOf(rel, pkgDir string, src []byte, fset *token.FileSet, fileAST *ast.File) *wireFile {
	wf := &wireFile{rel: rel, src: src, fset: fset, ast: fileAST, pkgDir: pkgDir, pkgName: fileAST.Name.Name}
	for _, is := range fileAST.Imports {
		pth, _ := strconv.Unquote(is.Path.Value)
		if pth != "net/http" {
			continue
		}
		_, eff := importSpecOf(is)
		wf.httpName = eff
		if is.Name != nil && is.Name.Name == "." {
			wf.dotHTTP = true
			wf.httpName = ""
		}
	}
	return wf
}

// findStarterRegistration finds the registration at route
// whose handler is the webserver component's own starter: an
// inline function whose source contains the starter's
// friendly line. Only the host package's own program is
// read: a route belongs to ONE program (its package) —
// another folder's package-main program answering "/" with
// its own starter is a different webserver in a different
// binary, and neither replaces nor blocks this page. It
// returns the registration, the statement holding it (via
// enclosingStmt), its wire file, and the registration call's
// own receiver-and-method as written ("http.HandleFunc" in
// the component) — or nils when no registration at the route
// in this program is the starter.
func findStarterRegistration(w *wireWorld, route, hostDir string) (*wireReg, ast.Stmt, *wireFile, string) {
	for _, r := range w.regs {
		if r.route != route || r.handlerName != "" {
			continue
		}
		wf := w.files[r.file]
		if wf == nil || wf.pkgDir != hostDir {
			continue
		}
		var stmt ast.Stmt
		var recv string
		ast.Inspect(wf.ast, func(n ast.Node) bool {
			if stmt != nil {
				return false
			}
			call, ok := n.(*ast.CallExpr)
			if !ok || len(call.Args) < 2 {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || (sel.Sel.Name != "HandleFunc" && sel.Sel.Name != "Handle") {
				return true
			}
			if id, ok := sel.X.(*ast.Ident); ok {
				recv = id.Name + "." + sel.Sel.Name
			}
			lit, ok := call.Args[0].(*ast.BasicLit)
			if !ok {
				return true
			}
			got, err := strconv.Unquote(lit.Value)
			if err != nil || got != route {
				return true
			}
			fn, ok := call.Args[1].(*ast.FuncLit)
			if !ok {
				return true
			}
			start, end := offOf(wf.fset, fn.Pos()), offOf(wf.fset, fn.End())
			if start < 0 || end > len(wf.src) || start > end {
				return true
			}
			if !strings.Contains(string(wf.src[start:end]), starterHelloMarker) {
				return true
			}
			_, s := enclosingStmt(wf.ast, call)
			if s != nil {
				stmt = s
				return false
			}
			return true
		})
		if stmt != nil {
			return r, stmt, wf, recv
		}
	}
	return nil, nil, nil, ""
}

// pickPageSite chooses the composable serving site for a page
// whose handler lives in hostRel / hostDir: a composable site
// in the host file itself first, else the package's ONLY
// composable site. More than one candidate in the package, or
// none, composes nowhere — the plan lands unregistered with
// the follow-up note. Custom sites never compose.
func pickPageSite(w *wireWorld, hostRel, hostDir string) *wireSite {
	var samePkg []*wireSite
	for _, s := range w.sites {
		if s.kind == siteCustom {
			continue
		}
		if s.file == hostRel {
			return s
		}
		if s.pkgDir == hostDir {
			samePkg = append(samePkg, s)
		}
	}
	if len(samePkg) == 1 {
		return samePkg[0]
	}
	return nil
}

// pageRegReceiver is the receiver a registration in wf's file
// is written against: the file's mux variable when one of its
// sites serves through one, else the file's own http name
// ("http" when the file doesn't import net/http under a name
// this scan can see — the starter's file always does).
func pageRegReceiver(w *wireWorld, file string, wf *wireFile) string {
	for _, s := range w.sites {
		if s.file == file && s.kind == siteMux {
			return s.muxVar
		}
	}
	if wf != nil && wf.httpName != "" {
		return wf.httpName
	}
	return "http"
}

// planPage assembles the page family's plan. See the file
// header for the shape and the states.
func (p *Project) planPage(req EditRequest) (*EditPlan, string) {
	rel := filepath.ToSlash(strings.TrimSpace(req.File))
	htmlRel := filepath.ToSlash(strings.TrimSpace(req.PageHTML))
	kind := strings.TrimSpace(req.PageKind)
	name := strings.TrimSpace(req.PageName)
	route := strings.TrimSpace(req.PageRoute)
	if route == "" {
		route = pageRouteForKind(kind)
	}
	if rel == "" {
		return nil, "that edit needs a Go file to hold the page's handler."
	}
	if htmlRel == "" || !strings.HasSuffix(strings.ToLower(htmlRel), ".html") {
		return nil, "that edit needs the page's .html file name."
	}
	if why, blocked := p.fileBlocked(rel); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(rel), why)
	}
	if why, blocked := p.fileBlocked(htmlRel); blocked {
		return nil, fmt.Sprintf("I won't create %s — %s.", p.LinkPath(htmlRel), why)
	}
	if _, ok := routePathOf(rel, htmlRel); !ok {
		return nil, fmt.Sprintf("the page %s has to live in the same folder as %s (or a folder inside it) for this recipe", p.LinkPath(htmlRel), p.LinkPath(rel))
	}
	words := kindWords(kind)

	// The page is created only when absent, never overwritten.
	if _, err := os.Stat(filepath.Join(p.Root, htmlRel)); err == nil {
		return nil, fmt.Sprintf("%s is already there — I never overwrite what's yours, so the %s wasn't made. If you want this ask to create it, remove that file yourself first, then ask again.", p.LinkPath(htmlRel), words)
	}

	if !token.IsIdentifier(name) {
		return nil, fmt.Sprintf("%q isn't a name Go allows for a function.", name)
	}

	// The ServeFile argument: the page's project-root-relative
	// path, the serve-html convention.
	servePath := htmlRel
	pageBase := path.Base(htmlRel)
	hostBase := path.Base(rel)
	hostDir := pageHostDir(rel)

	// The handler source, and the imports it needs.
	var handlerText string
	var importPaths []string
	var fields []string
	isForm := req.Concept != nil
	if isForm {
		fields = formTemplateFields(req.Concept.Template)
		if len(fields) == 0 {
			return nil, "the concept book's form handler piece doesn't have the shape this recipe assembles from — it names no form fields — so I won't invent a handler. Nothing was written."
		}
		rendered := strings.ReplaceAll(req.Concept.Template, "{name}", name)
		rendered = indentConceptDecl(rendered)
		replaced, ok := replaceFormShowLine(rendered, servePath)
		if !ok {
			return nil, "the concept book's form handler piece doesn't have the shape this recipe assembles from — it has no place that shows the form — so I won't invent a handler. Nothing was written."
		}
		handlerText = replaced
		if !strings.HasSuffix(handlerText, "\n") {
			handlerText += "\n"
		}
		importPaths = append(importPaths, req.Concept.Imports...)
	} else {
		handlerText = pagePlainHandler(name, pageBase, servePath)
		importPaths = []string{"net/http"}
	}

	// The host file's state: an existing file is read now (its
	// AST serves the collision check and the append); a
	// missing one is created by the plan.
	hostExists := false
	var hostSrc []byte
	var hostFset *token.FileSet
	var hostAST *ast.File
	if _, err := os.Stat(filepath.Join(p.Root, rel)); err == nil {
		hostExists = true
		src, fset, fileAST, refusal := p.readProjectFile(rel)
		if refusal != "" {
			return nil, refusal
		}
		if fileBytesBlocked(src) {
			return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(rel))
		}
		hostSrc, hostFset, hostAST = src, fset, fileAST
	}

	// The handler's name must be free in its package.
	if pkgName, declFile, declLine, found := p.pageNameCollision(hostDir, name, rel, hostFset, hostAST); found {
		return nil, fmt.Sprintf("package %s already has a function named %s (%s:%d), so the %s handler can't take that name too — give the page's handler a different name with \"named <Name>\" and ask again.", pkgName, name, p.LinkPath(declFile), declLine, words)
	}

	// The wiring world. A host file the index never walked is
	// parsed into it, so its own sites and registrations are
	// seen like any indexed file's.
	w := p.wireScan()
	if hostExists {
		if _, ok := w.files[rel]; !ok {
			wf := wireFileOf(rel, hostDir, hostSrc, hostFset, hostAST)
			w.files[rel] = wf
			w.scanFile(p, wf)
		}
	}

	// Conflicts at the route, read across the whole project.
	for _, r := range w.regs {
		if r.route == route && r.handlerName != "" && r.handlerDir == hostDir && strings.EqualFold(r.handlerName, name) {
			return nil, fmt.Sprintf("%s is already registered at %s (%s:%d) — nothing to add.", name, route, p.LinkPath(r.file), r.line)
		}
	}
	starterReg, starterStmt, starterWF, starterRecv := findStarterRegistration(w, route, hostDir)
	if starterReg == nil {
		// A taken route refuses only when the incumbent
		// lives in THIS program (the host package): a route
		// is per-program, and another folder's program
		// answering the same path in its own binary neither
		// conflicts with nor blocks this page.
		for _, r := range w.regs {
			if r.route != route {
				continue
			}
			if wf := w.files[r.file]; wf == nil || wf.pkgDir != hostDir {
				continue
			}
			return nil, fmt.Sprintf("%s is already taken — %s:%d registers it for %s. I won't point one path at two handlers, so %s wasn't added. If that registration is yours to move, change it by hand and ask again.", route, p.LinkPath(r.file), r.line, r.target, name)
		}
	}

	// Composition: the starter replacement, or an insert at
	// the chosen composable site, or none.
	composed := false
	replacedStarter := false
	serveRel := ""
	regLine := ""
	var site *wireSite
	if starterReg != nil {
		composed, replacedStarter = true, true
		serveRel = starterReg.file
		// The replacement keeps the starter call's own
		// receiver and method — the registration being
		// replaced is the authority on how this file
		// writes one.
		recv := starterRecv
		if recv == "" {
			recv = pageRegReceiver(w, serveRel, starterWF) + ".HandleFunc"
		}
		regLine = recv + "(" + strconv.Quote(route) + ", " + name + ")"
	} else if site = pickPageSite(w, rel, hostDir); site != nil {
		composed = true
		serveRel = site.file
		recv := site.wf.httpName
		if recv == "" {
			recv = "http"
		}
		if site.kind == siteMux {
			recv = site.muxVar
		}
		regLine = recv + ".HandleFunc(" + strconv.Quote(route) + ", " + name + ")"
	}
	if composed {
		if why, blocked := p.fileBlocked(serveRel); blocked {
			return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(serveRel), why)
		}
		if serveRel != rel {
			if wf := w.files[serveRel]; wf != nil && fileBytesBlocked(wf.src) {
				return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(serveRel))
			}
		}
	}

	// The registration edit, computed from the serving file's
	// ORIGINAL bytes: an insert just before the serving
	// statement, or a span replacing the starter's statement
	// (keeping its line's indent).
	// Each statement's offsets are read through its OWN wire
	// file's fset and bytes (the serving file's originals).
	regEditFor := func() (textEdit, bool) {
		if replacedStarter {
			if starterWF == nil || starterStmt == nil {
				return textEdit{}, false
			}
			line := lineOfPos(starterWF.fset, starterStmt.Pos())
			indent := lineIndent(starterWF.src, line)
			return textEdit{start: lineStartOffset(starterWF.src, line), end: offOf(starterWF.fset, starterStmt.End()), text: indent + regLine}, true
		}
		if site == nil || site.wf == nil {
			return textEdit{}, false
		}
		line := lineOfPos(site.wf.fset, site.stmt.Pos())
		indent := lineIndent(site.wf.src, line)
		insertAt := lineStartOffset(site.wf.src, line)
		return textEdit{start: insertAt, end: insertAt, text: indent + regLine + "\n"}, true
	}

	var specs []string
	for _, ip := range importPaths {
		specs = append(specs, strconv.Quote(ip))
	}
	internalErr := func(bad string) (*EditPlan, string) {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}

	serveSameAsHost := composed && serveRel == rel
	var hostEF EditFile
	var serveEF *EditFile
	switch {
	case !hostExists:
		// Create the host file: package clause (the folder's
		// own package), imports, handler — assembled through
		// the family's one import inserter over a parsed
		// skeleton, like the concept family's creations.
		pkgName := p.pageHostPackage(hostDir)
		base := "package " + pkgName + "\n\n" + handlerText
		if !strings.HasSuffix(base, "\n") {
			base += "\n"
		}
		_, skelAST, err := parseGoSrc(rel, []byte(base))
		if err != nil || skelAST == nil {
			return internalErr("the assembled file doesn't parse")
		}
		after := []byte(base)
		if miss := missingImportSpecs(skelAST, specs); len(miss) > 0 {
			fset2, ast2, err := parseGoSrc(rel, after)
			if err != nil || ast2 == nil {
				return internalErr("the assembled file doesn't parse")
			}
			after = applyTextEdits(after, conceptImportEdits(ast2, fset2, after, miss))
		}
		if f, err := format.Source(after); err == nil {
			after = f
		}
		if bad := selfCheckConceptDeclaration(after, name); bad != "" {
			return internalErr(bad)
		}
		hostEF = EditFile{Rel: rel, Before: nil, After: after, Diff: unifiedDiff(rel, nil, after)}
	case serveSameAsHost:
		// Every edit — imports, the registration, the handler
		// append — composes over the ORIGINAL bytes in one
		// splice, so every offset comes from the original AST.
		miss := missingImportSpecs(hostAST, specs)
		edits := conceptImportEdits(hostAST, hostFset, hostSrc, miss)
		regEdit, ok := regEditFor()
		if !ok {
			return internalErr("the registration site could not be re-read in the serving file")
		}
		edits = append(edits, regEdit)
		appendText := "\n" + handlerText
		if len(hostSrc) == 0 || hostSrc[len(hostSrc)-1] != '\n' {
			appendText = "\n" + appendText
		}
		edits = append(edits, textEdit{start: len(hostSrc), end: len(hostSrc), text: appendText})
		after := applyTextEdits(hostSrc, edits)
		if bad := selfCheckConceptDeclaration(after, name); bad != "" {
			return internalErr(bad)
		}
		hostEF = finishFile(rel, hostSrc, after)
	default:
		// Append the handler at the end of the host file, with
		// its missing imports — the concept declaration's
		// append pattern.
		miss := missingImportSpecs(hostAST, specs)
		base := applyTextEdits(hostSrc, conceptImportEdits(hostAST, hostFset, hostSrc, miss))
		text := "\n" + handlerText
		if len(base) == 0 || base[len(base)-1] != '\n' {
			text = "\n" + text
		}
		after := append(append([]byte{}, base...), text...)
		if bad := selfCheckConceptDeclaration(after, name); bad != "" {
			return internalErr(bad)
		}
		hostEF = finishFile(rel, hostSrc, after)
	}
	if composed && !serveSameAsHost {
		wf := w.files[serveRel]
		if wf == nil {
			return internalErr("the registration site could not be re-read in the serving file")
		}
		regEdit, ok := regEditFor()
		if !ok {
			return internalErr("the registration site could not be re-read in the serving file")
		}
		sAfter := applyTextEdits(wf.src, []textEdit{regEdit})
		ef := finishFile(serveRel, wf.src, sAfter)
		serveEF = &ef
	}

	// Files: the page first (the headline of the ask), then
	// the host file, then the serving file when it differs.
	starter := PageStarterPage(kind, route, fields)
	files := []EditFile{
		{Rel: htmlRel, Before: nil, After: starter, Diff: unifiedDiff(htmlRel, nil, starter)},
		hostEF,
	}
	if serveEF != nil {
		files = append(files, *serveEF)
	}

	notes := []string{serveRunNote(rel, servePath)}
	switch {
	case replacedStarter:
		notes = append(notes, "The webserver's own starter handler at / — the one that answers \"hello — your webserver is running\" — is replaced by the registration above; nothing else in the server changes.")
	case composed:
		notes = append(notes, "Your existing server is untouched apart from the one registration above — no second ListenAndServe is added.")
	default:
		// The finishing sentence must be the one that, fed
		// back verbatim, registers the handler at THIS
		// page's route. The wiring grammar's own default is
		// "/" + the handler's lowercased name, which is the
		// kind's route for most kinds — but a homepage
		// answers at "/" (not "/home"), and a named handler
		// (Solo serving an about page) defaults to its name,
		// so when the two differ the sentence carries the
		// route with "at".
		sentence := fmt.Sprintf("connect the %s handler to the webserver", name)
		if route != "/"+strings.ToLower(name) {
			sentence += " at " + route
		}
		note := fmt.Sprintf("Nothing serves %s yet. The finishing sentence is \"%s\" — it registers %s at %s.", route, sentence, name, route)
		if len(w.sites) == 0 {
			note += " This project has no webserver yet, so that sentence will first tell you the add-a-webserver step."
		}
		notes = append(notes, note)
	}
	if isForm {
		notes = append(notes, fmt.Sprintf("The form in %s asks for exactly the fields %s reads — %s — so the page and the handler can't drift apart.", p.LinkPath(htmlRel), name, joinWords(fields)))
		if req.Concept.Note != "" {
			notes = append(notes, req.Concept.Note)
		}
	}

	var summary, verifyDesc string
	switch {
	case replacedStarter:
		summary = fmt.Sprintf("Add the %s: %s (a starter page), its handler %s in %s, and the starter registration at / gives way to %s, so %s answers http://localhost:8080%s.", words, htmlRel, name, p.LinkPath(rel), regLine, name, route)
		verifyDesc = fmt.Sprintf("%s serves %s at %s", name, pageBase, route)
	case composed:
		summary = fmt.Sprintf("Add the %s: %s (a starter page), its handler %s in %s, and one registration — %s — added in %s just before it serves, so %s answers http://localhost:8080%s.", words, htmlRel, name, p.LinkPath(rel), regLine, p.LinkPath(serveRel), name, route)
		verifyDesc = fmt.Sprintf("%s serves %s at %s", name, pageBase, route)
	default:
		summary = fmt.Sprintf("Add the %s: %s (a starter page) and its handler %s in %s. %s isn't registered with a server yet.", words, htmlRel, name, p.LinkPath(rel), name)
		verifyDesc = fmt.Sprintf("%s is declared in %s and %s exists", name, hostBase, pageBase)
	}

	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(hostEF.After, "func "+name+"(")),
		Files:      files,
		Notes:      notes,
		verify: func(after *Project) bool {
			if _, err := os.Stat(filepath.Join(after.Root, filepath.FromSlash(htmlRel))); err != nil {
				return false
			}
			w2 := after.wireScan()
			found := false
			for _, h := range w2.handlers {
				if h.pkgDir == hostDir && strings.EqualFold(h.name, name) {
					found = true
					break
				}
			}
			if !found {
				return false
			}
			if !composed {
				return true
			}
			for _, r := range w2.regs {
				if r.route == route && r.handlerDir == hostDir && strings.EqualFold(r.handlerName, name) {
					return true
				}
			}
			return false
		},
	}, ""
}
