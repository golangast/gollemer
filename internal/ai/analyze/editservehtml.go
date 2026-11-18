package analyze

// The serve-an-html-file recipe: a composite edit planned here
// beside the add-a-webserver shape it extends. One ask produces
// BOTH halves of serving a page:
//
//   - the page itself (req.ServeHTML, a project-relative .html
//     path): created with a starter page when it does not exist,
//     NEVER overwritten when it does;
//   - the wiring in req.File (a package-main .go file): a
//     handler answering one route with http.ServeFile, plus
//     http.ListenAndServe ONLY when the file has no server yet.
//
// The file's own state decides the wiring's shape, mirroring
// planAddServer's states:
//
//	(a) a missing file (or a bare package clause) becomes the
//	    whole program: handler + ListenAndServe;
//	(b) a main-less file grows a main carrying both;
//	(c) an empty main is filled with both;
//	(d) a file already serving on the DEFAULT mux composes: the
//	    registration is inserted just before the existing
//	    ListenAndServe statement — a second ListenAndServe is
//	    never emitted into a project that has one;
//	(e) a file already serving some other way (a custom handler
//	    or mux, a method-style serve) refuses, handing over the
//	    exact line to paste;
//	(f) a non-empty main with no server refuses the same way
//	    planAddServer does: the component's lines to paste;
//	(g) a library package refuses: serving is a program's job.
//
// Two more states are answers, not edits: the route already
// serves THIS page (nothing to add — unless the page itself is
// missing, in which case the plan is the page alone), and the
// route is taken by a different handler (a conflict the recipe
// will not paper over).
//
// The page lives in the Go file's folder (or below it): the
// chat layer resolves it there, and this planner refuses
// anything outside it. The ServeFile argument is the page's
// PROJECT-ROOT-relative path — the path the chat itself uses
// for the page — so the page is found when the program runs
// from the project root, which is how the chat's own runner
// runs it (and what the plan's notes say out loud). The route
// is the page's path within the program's folder; only the
// disk lookup is rooted. A program run by hand from inside
// its own folder inverts the old behavior: the root-relative
// path misses there — the project root is the contract.
//
// One page name bends the route rule: an index page answers
// at its folder's address ("/" for a folder-root index,
// "/docs/" for docs/index.html), never at its own name —
// net/http's ServeFile redirects every request ending in
// /index.html to the folder's address, so a registration at
// the page's name would hand the browser a redirect to an
// address nothing answers. See serveRouteOf.

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

// serveHTMLRegister renders the one registration statement the
// recipe writes: the route answered by a handler that serves
// the file from disk on every request. indent is the leading
// whitespace of the line it will sit on.
func serveHTMLRegister(route, servePath, indent string) string {
	return indent + "http.HandleFunc(" + strconv.Quote(route) + ", func(w http.ResponseWriter, r *http.Request) {\n" +
		indent + "\thttp.ServeFile(w, r, " + strconv.Quote(servePath) + ")\n" +
		indent + "})\n"
}

// serveHTMLMainBody is the statement block for a fresh main:
// the registration, the announcement, and the serve call —
// WebServerMainBody's shape, with the page's handler in place
// of the component's friendly-line handler.
func serveHTMLMainBody(route, servePath string) string {
	return serveHTMLRegister(route, servePath, "\t") +
		"\tfmt.Println(\"listening at http://localhost:8080 — press Ctrl+C to stop\")\n" +
		"\tif err := http.ListenAndServe(\":8080\", nil); err != nil {\n" +
		"\t\tfmt.Println(\"the server stopped:\", err)\n" +
		"\t}\n"
}

// serveHTMLSource is the whole program as a standalone file,
// written when the target file is missing or a bare package
// clause. goDir is the file's project-relative folder ("." for
// the project root) — the comment names the run command,
// because ServeFile finds the page at its root-relative path
// from the working directory: the program runs from the
// project root.
func serveHTMLSource(route, servePath, goDir string) string {
	runCmd := "go run ."
	if goDir != "." {
		runCmd = "go run ./" + goDir
	}
	return `package main

import (
	"fmt"
	"net/http"
)

// Serves the page ` + servePath + ` (its path from the project
// root). Run it from the project root: ` + runCmd + `
// Then open http://localhost:8080` + route + ` in a browser.
func main() {
` + serveHTMLMainBody(route, servePath) + `}
`
}

// ServeHTMLStarterPage is the page the recipe creates when the
// asked-for file does not exist: a valid minimal HTML page
// whose body is a TODO, with just enough visible text that
// opening it proves the serving works.
func ServeHTMLStarterPage(htmlBase, goBase string) []byte {
	return []byte(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>` + htmlBase + `</title>
</head>
<body>
<!-- TODO: replace this with your own page. ` + goBase + ` serves this file
     as-is: edit it, save, and refresh the browser — no restart needed. -->
<h1>TODO: your page</h1>
<p>Your Go program is serving this file. Put your own words here.</p>
</body>
</html>
`)
}

// routePathOf derives the route the page answers at (the
// page's path relative to the Go file's folder) from the two
// project-relative paths. ok is false when the page is not in
// the Go file's folder or below it. The ServeFile argument is
// NOT derived here: it is htmlRel itself, the page's
// project-root-relative path.
func routePathOf(goRel, htmlRel string) (string, bool) {
	dir := filepath.ToSlash(filepath.Dir(goRel))
	if dir == "." {
		return htmlRel, true
	}
	if strings.HasPrefix(htmlRel, dir+"/") {
		return strings.TrimPrefix(htmlRel, dir+"/"), true
	}
	return "", false
}

// serveRouteOf derives the route the page answers at from
// its path within the program's folder. A non-index page
// answers at its own path. An index page answers at its
// folder's address: net/http's ServeFile answers a request
// for …/index.html with a redirect to ./ — the folder's
// address — for ANY wiring, so a handler registered at the
// page's own name never serves the page; the browser lands
// on the folder route, where a standalone program has
// nothing (a mux 404) and a composed one has whatever else
// lives there (verified live: old wiring GET /index.html →
// 301 → 404, GET / → 404; directory wiring serves the page
// at both addresses). The directory route is the address
// that actually serves an index page.
func serveRouteOf(routePath string) string {
	if path.Base(routePath) == "index.html" {
		return "/" + strings.TrimSuffix(routePath, "index.html")
	}
	return "/" + routePath
}

// planServeHTML assembles the recipe's plan. See the file
// header for the state table.
func (p *Project) planServeHTML(req EditRequest) (*EditPlan, string) {
	rel := filepath.ToSlash(strings.TrimSpace(req.File))
	htmlRel := filepath.ToSlash(strings.TrimSpace(req.ServeHTML))
	if rel == "" {
		return nil, "that edit needs a Go file to serve the page from."
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
	routePath, ok := routePathOf(rel, htmlRel)
	if !ok {
		return nil, fmt.Sprintf("the page %s has to live in the same folder as %s (or a folder inside it) for this recipe", p.LinkPath(htmlRel), p.LinkPath(rel))
	}
	// The ServeFile argument: the page's project-root-relative
	// path, so the program finds the page when it runs from
	// the project root — the chat runner's working directory.
	servePath := htmlRel
	route := serveRouteOf(routePath)
	goBase := filepath.Base(rel)
	htmlBase := filepath.Base(htmlRel)

	// The page's half of the plan: created only when absent,
	// never overwritten. It rides FIRST in the file list — the
	// page is the headline of the ask, the wiring its support.
	htmlExists := false
	if _, err := os.Stat(filepath.Join(p.Root, htmlRel)); err == nil {
		htmlExists = true
	}
	var files []EditFile
	if !htmlExists {
		starter := ServeHTMLStarterPage(htmlBase, goBase)
		files = append(files, EditFile{
			Rel:    htmlRel,
			Before: nil,
			After:  starter,
			Diff:   unifiedDiff(htmlRel, nil, starter),
		})
	}
	pageNote := ""
	if htmlExists {
		pageNote = fmt.Sprintf("The page %s already exists — it is not changed; only the wiring is added.", p.LinkPath(htmlRel))
	}
	runNote := serveRunNote(rel, servePath)
	serveNote := fmt.Sprintf("After it runs, open http://localhost:8080%s — the handler reads %s from disk on every request, so edits to the page show on refresh.", route, servePath)
	if htmlBase == "index.html" {
		// The suggested address is the folder's, and the
		// plan says why in one clause: the address at the
		// page's own name only redirects there.
		serveNote += fmt.Sprintf(" It is an index page, so it answers at its folder's address: a request for /%s is redirected to http://localhost:8080%s by the server — net/http sends every index.html request to the folder — so %s is the address that serves the page.", routePath, route, route)
	}

	// The Go file's state. A missing file is not an error
	// here (planAddServer's chat layer refuses those; the
	// recipe CREATES its program the way it creates its page):
	// the plan writes the whole program.
	goAbs := filepath.Join(p.Root, rel)
	if _, err := os.Stat(goAbs); os.IsNotExist(err) {
		src := serveHTMLSource(route, servePath, filepath.ToSlash(filepath.Dir(rel)))
		ef := EditFile{Rel: rel, Before: nil, After: []byte(src), Diff: unifiedDiff(rel, nil, []byte(src))}
		files = append(files, ef)
		notes := []string{runNote, serveNote}
		if pageNote != "" {
			notes = append([]string{pageNote}, notes...)
		}
		return &EditPlan{
			Summary:    fmt.Sprintf("Serve %s: a new program in %s registers a handler that answers http://localhost:8080%s with the page, and starts the server. %s", htmlRel, p.LinkPath(rel), route, newPageWords(htmlExists, htmlRel)),
			VerifyDesc: fmt.Sprintf("%s serves %s via http.ServeFile", goBase, htmlBase),
			Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, "http.ServeFile")),
			Files:      files,
			Notes:      notes,
			verify:     func(after *Project) bool { return verifyServeHTMLPresent(after.Root, rel, servePath) },
		}, ""
	}

	src, fset, fileAST, refusal := p.readProjectFile(rel)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(rel))
	}
	if fileAST.Name.Name != "main" {
		return nil, p.serveNeedsMainRefusal(rel, htmlRel, fileAST.Name.Name)
	}

	serving := hasImportPath(fileAST, "net/http") && callsListenAndServe(fileAST)
	if serving {
		return p.planServeHTMLCompose(req, src, fset, fileAST, composeInput{
			rel: rel, htmlRel: htmlRel, servePath: servePath, route: route,
			goBase: goBase, htmlBase: htmlBase, htmlExists: htmlExists,
			files: files, pageNote: pageNote, runNote: runNote, serveNote: serveNote,
		})
	}

	var mainDecl *ast.FuncDecl
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == "main" {
			mainDecl = fd
			break
		}
	}

	// (a) A bare package clause: the file becomes the whole
	// program — the same move planAddServer makes.
	if len(fileAST.Decls) == 0 && strings.TrimSpace(string(src)) == "package main" {
		ef := finishFile(rel, src, []byte(serveHTMLSource(route, servePath, filepath.ToSlash(filepath.Dir(rel)))))
		files = append(files, ef)
		notes := []string{runNote, serveNote}
		if pageNote != "" {
			notes = append([]string{pageNote}, notes...)
		}
		return &EditPlan{
			Summary:    fmt.Sprintf("Serve %s: %s becomes a program that answers http://localhost:8080%s with the page and starts the server. %s", htmlRel, p.LinkPath(rel), route, newPageWords(htmlExists, htmlRel)),
			VerifyDesc: fmt.Sprintf("%s serves %s via http.ServeFile", goBase, htmlBase),
			Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, "http.ServeFile")),
			Files:      files,
			Notes:      notes,
			verify:     func(after *Project) bool { return verifyServeHTMLPresent(after.Root, rel, servePath) },
		}, ""
	}

	// (f) A main that already does things, and no server to
	// compose with: refuse, handing over the exact lines.
	if mainDecl != nil && mainDecl.Body != nil && len(mainDecl.Body.List) > 0 {
		return nil, serveByHandRefusal(p.LinkPath(rel), htmlRel, route, servePath)
	}

	// Imports the body needs that the file lacks.
	var missing []string
	for _, path := range []string{"fmt", "net/http"} {
		if !hasImportPath(fileAST, path) {
			missing = append(missing, strconv.Quote(path))
		}
	}
	var edits []textEdit
	if len(missing) > 0 {
		edits = append(edits, insertImportSpecs(fileAST, fset, src, missing)...)
	}

	var summary string
	if mainDecl == nil {
		// (b) No main yet: append one carrying the recipe.
		text := "\n// Serves the page " + servePath + " (its path from the project\n" +
			"// root — run the program from there). It listens at http://localhost:8080.\n" +
			"func main() {\n" + serveHTMLMainBody(route, servePath) + "}\n"
		if !strings.HasSuffix(string(src), "\n") {
			text = "\n" + text
		}
		edits = append(edits, textEdit{start: len(src), end: len(src), text: text})
		summary = fmt.Sprintf("Serve %s: a new main() in %s registers a handler that answers http://localhost:8080%s with the page, and starts the server. %s", htmlRel, p.LinkPath(rel), route, newPageWords(htmlExists, htmlRel))
	} else {
		// (c) An empty main: fill it with the recipe's body.
		edits = append(edits, textEdit{
			start: offOf(fset, mainDecl.Body.Lbrace) + 1,
			end:   offOf(fset, mainDecl.Body.Rbrace),
			text:  "\n" + serveHTMLMainBody(route, servePath),
		})
		summary = fmt.Sprintf("Serve %s: main() in %s registers a handler that answers http://localhost:8080%s with the page, and starts the server. %s", htmlRel, p.LinkPath(rel), route, newPageWords(htmlExists, htmlRel))
	}
	after := applyTextEdits(src, edits)
	ef := finishFile(rel, src, after)
	files = append(files, ef)
	notes := []string{runNote, serveNote}
	if pageNote != "" {
		notes = append([]string{pageNote}, notes...)
	}
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s serves %s via http.ServeFile", goBase, htmlBase),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, "http.ServeFile")),
		Files:      files,
		Notes:      notes,
		verify:     func(after *Project) bool { return verifyServeHTMLPresent(after.Root, rel, servePath) },
	}, ""
}

// newPageWords is the summary's page clause: created fresh, or
// already there and untouched.
func newPageWords(htmlExists bool, htmlRel string) string {
	if htmlExists {
		return fmt.Sprintf("The page %s is already there and stays exactly as it is.", htmlRel)
	}
	return fmt.Sprintf("The page %s is created fresh with a starter page (its body is a TODO).", htmlRel)
}

// serveRunNote is the plan's beginner-honesty line: ServeFile
// resolves the page from the program's working directory, and
// the wired path is the page's path from the project root, so
// the root is where the program has to run from — not the
// program's own folder.
func serveRunNote(goRel, servePath string) string {
	dir := filepath.ToSlash(filepath.Dir(goRel))
	if dir == "." {
		return fmt.Sprintf("Run it from the project root (go run .) — the program looks for %s in the folder it is run from.", servePath)
	}
	return fmt.Sprintf("Run it from the project root (go run ./%s) — the program looks for %s in the folder it is run from, so it has to start at the root, not in the %s folder.", dir, servePath, dir)
}

// composeInput bundles what planServeHTML hands its
// already-serving branch.
type composeInput struct {
	rel, htmlRel, servePath, route string
	goBase, htmlBase               string
	htmlExists                     bool
	files                          []EditFile
	pageNote, runNote, serveNote   string
}

// planServeHTMLCompose is state (d)/(e): the file already runs
// a server. The recipe composes ONLY onto the default mux — a
// ListenAndServe(addr, nil) call — by inserting the
// registration just before the statement that serves. Anything
// else refuses with the line to paste, because a second
// ListenAndServe or a registration on a mux this planner cannot
// see would be a lie about what runs.
func (p *Project) planServeHTMLCompose(req EditRequest, src []byte, fset *token.FileSet, fileAST *ast.File, in composeInput) (*EditPlan, string) {
	// The route may already be answered. Same page behind it:
	// the wiring half is done. A different handler: a conflict.
	for route, handler := range routeRegistrations(fileAST) {
		if route != in.route {
			continue
		}
		if sp, ok := findServeFilePath(handler); ok && sp == in.servePath {
			if in.htmlExists {
				return nil, fmt.Sprintf("%s already serves %s at http://localhost:8080%s — nothing to add.", p.LinkPath(in.rel), in.htmlRel, in.route)
			}
			// The wiring exists; only the page is missing.
			return &EditPlan{
				Summary:    fmt.Sprintf("%s already serves %s — the only thing missing is the page itself, created here with a starter page (its body is a TODO). No Go code changes.", p.LinkPath(in.rel), in.htmlRel),
				VerifyDesc: fmt.Sprintf("%s exists and %s serves it", in.htmlBase, in.goBase),
				Site:       fmt.Sprintf("%s:%d", p.LinkPath(in.rel), insertedLine(src, "http.ServeFile")),
				Files:      in.files,
				Notes:      []string{in.runNote, in.serveNote},
				verify:     func(after *Project) bool { return verifyServeHTMLPresent(after.Root, in.rel, in.servePath) },
			}, ""
		}
		return nil, fmt.Sprintf("%s already answers http://localhost:8080%s with a different handler, so serving %s there would mean replacing it — I won't. Serve the page under a different name, or change that handler by hand.", p.LinkPath(in.rel), in.route, in.htmlRel)
	}

	// Find the default-mux serve call this recipe may compose
	// with: http.ListenAndServe(addr, nil), first in source.
	var serveCall *ast.CallExpr
	var customArg string
	for _, call := range listenAndServeCalls(fileAST) {
		if !isHTTPCall(call, "ListenAndServe") || len(call.Args) != 2 {
			continue
		}
		if id, ok := call.Args[1].(*ast.Ident); ok && id.Name == "nil" {
			serveCall = call
			break
		}
		// The by-hand line is written against the handler only
		// when the file proves it is a *http.ServeMux (assigned
		// from http.NewServeMux): an arbitrary http.Handler
		// value has no HandleFunc method, and a refusal must
		// never hand over a line that does not compile.
		if customArg == "" {
			if id, ok := call.Args[1].(*ast.Ident); ok && isServeMuxVar(fileAST, id.Name) {
				customArg = id.Name
			}
		}
	}
	if serveCall == nil {
		return nil, serveCustomServerRefusal(p.LinkPath(in.rel), in.htmlRel, in.route, in.servePath, customArg)
	}
	list, stmt := enclosingStmt(fileAST, serveCall)
	if list == nil || stmt == nil {
		return nil, serveCustomServerRefusal(p.LinkPath(in.rel), in.htmlRel, in.route, in.servePath, customArg)
	}
	line := lineOfPos(fset, stmt.Pos())
	indent := lineIndent(src, line)
	insertAt := lineStartOffset(src, line)
	after := applyTextEdits(src, []textEdit{{
		start: insertAt, end: insertAt,
		text: serveHTMLRegister(in.route, in.servePath, indent),
	}})
	ef := finishFile(in.rel, src, after)
	files := append(in.files, ef)
	notes := []string{
		"Your existing server is untouched apart from the one registration above — no second ListenAndServe is added.",
		in.runNote, in.serveNote,
	}
	if in.pageNote != "" {
		notes = append([]string{in.pageNote}, notes...)
	}
	return &EditPlan{
		Summary:    fmt.Sprintf("Serve %s from the server already in %s: one registration, just before its ListenAndServe, answers http://localhost:8080%s with the page. %s", in.htmlRel, p.LinkPath(in.rel), in.route, newPageWords(in.htmlExists, in.htmlRel)),
		VerifyDesc: fmt.Sprintf("%s serves %s via http.ServeFile", in.goBase, in.htmlBase),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(in.rel), insertedLine(ef.After, "http.ServeFile")),
		Files:      files,
		Notes:      notes,
		verify:     func(after *Project) bool { return verifyServeHTMLPresent(after.Root, in.rel, in.servePath) },
	}, ""
}

// serveNeedsMainRefusal is state (g)'s answer, mirroring
// serverNeedsMainRefusal: why not, and the nearest real move —
// the project's own program when the analyzer knows one, the
// recipe itself aimed at a new main.go when it doesn't.
func (p *Project) serveNeedsMainRefusal(rel, htmlRel, pkgName string) string {
	base := fmt.Sprintf("%s is package %s — serving a page is a program's job, and programs live in package main.", p.LinkPath(rel), pkgName)
	if eps := p.EntryPoints(); len(eps) > 0 {
		mf := eps[0]
		return fmt.Sprintf("%s This project's program starts in %s — say \"serve %s to %s\" and I'll put it there.", base, p.LinkPath(mf.File), htmlRel, mf.File)
	}
	return fmt.Sprintf("%s Say \"serve %s to main.go\" and I'll write the program and the page together.", base, htmlRel)
}

// serveByHandRefusal is state (f)'s answer: main already does
// its own work, so the recipe's exact lines — the registration
// and the serve call, verbatim — are handed over to paste.
func serveByHandRefusal(linkPath, htmlRel, route, servePath string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "main() in %s already does its own work, so I won't splice a server into running code. To serve %s by hand, paste these lines inside main (the file also needs \"fmt\" and \"net/http\" imported):\n\n", linkPath, htmlRel)
	b.WriteString(strings.TrimRight(serveHTMLMainBody(route, servePath), "\n"))
	return b.String()
}

// serveCustomServerRefusal is state (e)'s answer: the file
// serves, but not on the default mux, so the registration this
// recipe writes would never be reached — and a second
// ListenAndServe is never the answer. The exact line to paste
// is handed over; when the handler is a plain identifier (a
// mux the user named), the line is written against it.
func serveCustomServerRefusal(linkPath, htmlRel, route, servePath, muxName string) string {
	if muxName != "" {
		return fmt.Sprintf("%s already runs a server with its own handler (%s), so a registration on the default mux would never be reached — and I won't add a second ListenAndServe. To serve %s there, add this line where %s is set up, before the serve call:\n\n%s.HandleFunc(%s, func(w http.ResponseWriter, r *http.Request) {\n\t\thttp.ServeFile(w, r, %s)\n\t})",
			linkPath, muxName, htmlRel, muxName, muxName, strconv.Quote(route), strconv.Quote(servePath))
	}
	return fmt.Sprintf("%s already runs a server in its own way, so I can't register the page on it from here — and I won't add a second ListenAndServe. To serve %s, add this line where the server is set up, before it starts serving (on your mux or handler instead of http, if that is what the file uses):\n\n%s",
		linkPath, htmlRel, strings.TrimRight(serveHTMLRegister(route, servePath, ""), "\n"))
}

// --- AST helpers for the compose state -------------------------

// listenAndServeCalls returns every call to a ListenAndServe
// selector in source order.
func listenAndServeCalls(f *ast.File) []*ast.CallExpr {
	var out []*ast.CallExpr
	ast.Inspect(f, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "ListenAndServe" {
				out = append(out, call)
			}
		}
		return true
	})
	return out
}

// isHTTPCall reports whether the call is http.<name>(...).
func isHTTPCall(call *ast.CallExpr, name string) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != name {
		return false
	}
	id, ok := sel.X.(*ast.Ident)
	return ok && id.Name == "http"
}

// isServeMuxVar reports whether name is assigned from
// http.NewServeMux anywhere in the file.
func isServeMuxVar(f *ast.File, name string) bool {
	found := false
	ast.Inspect(f, func(n ast.Node) bool {
		if found {
			return false
		}
		assign, ok := n.(*ast.AssignStmt)
		if !ok {
			return true
		}
		for i, lhs := range assign.Lhs {
			id, ok := lhs.(*ast.Ident)
			if !ok || id.Name != name || i >= len(assign.Rhs) {
				continue
			}
			if call, ok := assign.Rhs[i].(*ast.CallExpr); ok && isHTTPCall(call, "NewServeMux") {
				found = true
				return false
			}
		}
		return true
	})
	return found
}

// routeRegistrations maps each route string registered through
// HandleFunc/Handle (on any receiver — the default mux or a
// mux value) to its handler expression.
func routeRegistrations(f *ast.File) map[string]ast.Expr {
	out := map[string]ast.Expr{}
	ast.Inspect(f, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || (sel.Sel.Name != "HandleFunc" && sel.Sel.Name != "Handle") {
			return true
		}
		if len(call.Args) < 2 {
			return true
		}
		lit, ok := call.Args[0].(*ast.BasicLit)
		if !ok {
			return true
		}
		route, err := strconv.Unquote(lit.Value)
		if err != nil {
			return true
		}
		if _, seen := out[route]; !seen {
			out[route] = call.Args[1]
		}
		return true
	})
	return out
}

// findServeFilePath returns the path argument of the first
// http.ServeFile call inside node, when it is a string literal.
func findServeFilePath(node ast.Node) (string, bool) {
	found := ""
	ast.Inspect(node, func(n ast.Node) bool {
		if found != "" {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok || !isHTTPCall(call, "ServeFile") || len(call.Args) < 3 {
			return true
		}
		if lit, ok := call.Args[2].(*ast.BasicLit); ok {
			if v, err := strconv.Unquote(lit.Value); err == nil {
				found = v
				return false
			}
		}
		return true
	})
	return found, found != ""
}

// enclosingStmt finds the innermost statement list holding the
// target node and the direct element of that list containing
// it — the place a registration can be inserted so it runs
// just before the target does. Statement lists are block
// bodies and case/comm clause bodies.
func enclosingStmt(f *ast.File, target ast.Node) ([]ast.Stmt, ast.Stmt) {
	var lists [][]ast.Stmt
	ast.Inspect(f, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.BlockStmt:
			lists = append(lists, node.List)
		case *ast.CaseClause:
			lists = append(lists, node.Body)
		case *ast.CommClause:
			lists = append(lists, node.Body)
		}
		return true
	})
	var bestList []ast.Stmt
	var bestStmt ast.Stmt
	bestSpan := -1
	for _, list := range lists {
		for _, stmt := range list {
			if !nodeContains(stmt, target) {
				continue
			}
			span := int(stmt.End() - stmt.Pos())
			if bestSpan < 0 || span < bestSpan {
				bestSpan, bestList, bestStmt = span, list, stmt
			}
		}
	}
	return bestList, bestStmt
}

// nodeContains reports whether target sits inside root.
func nodeContains(root, target ast.Node) bool {
	found := false
	ast.Inspect(root, func(n ast.Node) bool {
		if n == target {
			found = true
			return false
		}
		return !found
	})
	return found
}

// verifyServeHTMLPresent re-reads the edited file and confirms
// the intended structure: package main, a ServeFile call for
// exactly our path, and a server — and the page on disk. The
// verifyServerPresent pattern, for this recipe's shape.
func verifyServeHTMLPresent(root, rel, servePath string) bool {
	_, _, fileAST, refusal := (&Project{Root: root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil || fileAST.Name.Name != "main" {
		return false
	}
	if !callsListenAndServe(fileAST) {
		return false
	}
	if !hasServeFilePath(fileAST, servePath) {
		return false
	}
	// servePath is the page's project-root-relative path: it
	// joins the root directly, not the Go file's folder.
	htmlAbs := filepath.Join(root, filepath.FromSlash(servePath))
	_, err := os.Stat(htmlAbs)
	return err == nil
}

// hasServeFilePath reports whether any http.ServeFile call in
// the file names exactly this path. A file serving several
// pages carries one ServeFile call per page, so verification
// must find THIS recipe's path among them — checking only the
// first call (as a single-page reading would) fails every
// second page composed into the same file.
func hasServeFilePath(node ast.Node, want string) bool {
	found := false
	ast.Inspect(node, func(n ast.Node) bool {
		if found {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok || !isHTTPCall(call, "ServeFile") || len(call.Args) < 3 {
			return true
		}
		if lit, ok := call.Args[2].(*ast.BasicLit); ok {
			if v, err := strconv.Unquote(lit.Value); err == nil && v == want {
				found = true
				return false
			}
		}
		return true
	})
	return found
}
