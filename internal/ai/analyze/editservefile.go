package analyze

// The serve-a-file piece (chat/servefile.go): one handler
// that answers one route with a file the user already has —
// a logo, a stylesheet, a data file, a page — read from disk
// on every request by http.ServeFile:
//
//	func LogoHandler(w http.ResponseWriter, r *http.Request) {
//		http.ServeFile(w, r, "jim/logo.png")
//	}
//
// The handler lands in req.File (the host — created when
// missing, in its folder's package, the page family's rule),
// and the registration composes under the wiring machinery's
// rules when the host package holds exactly one composable
// serving site (pickPageSite — the page family's chooser):
// one HandleFunc line just before the serve statement. With
// nowhere to register, the plan still lands the handler and
// its note spells the wiring sentence that finishes it —
// placement first, wiring as its own gated step, exactly
// like the concept book's handlers.
//
// The states that are answers, not edits:
//
//   - the route already serves THIS file (an inline
//     ServeFile handler, or a named one): nothing to add,
//     naming where;
//   - a handler with the asked name already serves this
//     file: it is named, with its state — registered (its
//     route), or the finishing sentence that wires it;
//   - any other declaration with the asked name in the
//     host package: the collision refusal, with the
//     "named <Name>" way out;
//   - the route is taken in the host program by another
//     handler: a refusal naming both, the wiring family's
//     rule (a route is per-program: another folder's
//     program answering the same path in its own binary
//     neither conflicts nor blocks);
//   - the served file is missing: refused — a handler can
//     only serve a file that's really there, and this
//     piece never invents a file's contents.
//
// ServeFile picks the answer's content type from the file's
// extension — .png goes as an image, .css as a stylesheet,
// .json as data — and the plan says so: the one mechanism
// covers every file as what it is. The ServeFile argument
// is the file's project-root-relative path (the serve-html
// convention: the program finds it when run from the
// project root, which is how the chat's runner runs it).
// The route, when the sentence names none, is the file's
// path within the host program's folder — an index.html
// answers at its folder's address, never at its own name
// (serveRouteOf, net/http's redirect rule) — or, for a
// file outside that folder, its path from the project root.

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

// serveFileHandler renders the handler: a named func
// answering with the file, read from disk on every request
// (the serve-html convention: servePath is the file's
// project-root-relative path). The page family's plain
// handler shape, spoken for a file.
func serveFileHandler(name, servePath string) string {
	return "// " + name + " serves the " + path.Base(servePath) + " file: it is read from disk on\n" +
		"// every request, so a new copy of the file shows on refresh.\n" +
		"func " + name + "(w http.ResponseWriter, r *http.Request) {\n" +
		"\thttp.ServeFile(w, r, " + strconv.Quote(servePath) + ")\n" +
		"}\n"
}

// handlerServingFile finds a plain function named name in
// the host package whose body answers with http.ServeFile
// over exactly servePath — the "that handler is already
// this ask" fact. It returns the handler's wire info and
// its declaration.
func handlerServingFile(w *wireWorld, hostDir, name, servePath string) (*wireHandlerInfo, *ast.FuncDecl) {
	for _, h := range w.handlers {
		if h.pkgDir != hostDir || !strings.EqualFold(h.name, name) {
			continue
		}
		wf := w.files[h.file]
		if wf == nil {
			continue
		}
		for _, d := range wf.ast.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Recv != nil || fd.Name.Name != h.name {
				continue
			}
			if hasServeFilePath(fd, servePath) {
				return h, fd
			}
		}
	}
	return nil, nil
}

// inlineServingAtRoute finds a registration at route in the
// host program whose handler is an inline function serving
// exactly servePath — the other "already serving this file"
// shape. It returns the registration and its file.
func inlineServingAtRoute(p *Project, w *wireWorld, hostDir, route, servePath string) (*wireReg, bool) {
	for _, r := range w.regs {
		if r.route != route || r.handlerName != "" {
			continue
		}
		wf := w.files[r.file]
		if wf == nil || wf.pkgDir != hostDir {
			continue
		}
		if expr, ok := routeRegistrations(wf.ast)[route]; ok {
			if sp, found := findServeFilePath(expr); found && sp == servePath {
				return r, true
			}
		}
	}
	return nil, false
}

// planServeFile assembles the serve-a-file plan. See the
// file header for the shape and the states.
func (p *Project) planServeFile(req EditRequest) (*EditPlan, string) {
	rel := filepath.ToSlash(strings.TrimSpace(req.File))
	serveRel := filepath.ToSlash(strings.TrimSpace(req.ServeFile))
	name := strings.TrimSpace(req.ServeFileName)
	route := strings.TrimSpace(req.ServeFileRoute)
	if rel == "" {
		return nil, "that edit needs a Go file to hold the handler."
	}
	if serveRel == "" {
		return nil, "that edit needs the file the handler should serve."
	}
	if !token.IsIdentifier(name) {
		return nil, fmt.Sprintf("%q isn't a name Go allows for a function — name the handler with \"named <Name>\".", name)
	}
	if route != "" && !strings.HasPrefix(route, "/") {
		return nil, fmt.Sprintf("the path %q needs to start with a slash — say it like \"at /logo.png\".", req.ServeFileRoute)
	}
	if why, blocked := p.fileBlocked(rel); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(rel), why)
	}
	if why, blocked := p.fileBlocked(serveRel); blocked {
		return nil, fmt.Sprintf("I won't serve %s — %s.", p.LinkPath(serveRel), why)
	}
	if st, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(serveRel))); err != nil || st.IsDir() {
		return nil, fmt.Sprintf("there's no file %s in this project for a handler to serve — put the file there first, then ask again. A handler can only serve a file that's really there.", p.LinkPath(serveRel))
	}

	// The ServeFile argument: the file's project-root-relative
	// path, the serve-html convention.
	servePath := serveRel
	serveBase := path.Base(serveRel)
	hostBase := path.Base(rel)
	hostDir := pageHostDir(rel)

	// The route, when the sentence names none: the file's
	// path within the host file's folder (an index page at
	// its folder's address), or its root-relative path when
	// it lives outside that folder.
	if route == "" {
		if rp, ok := routePathOf(rel, serveRel); ok {
			route = serveRouteOf(rp)
		} else {
			route = serveRouteOf(serveRel)
		}
	}

	// The host file's state: an existing file is read now
	// (its AST serves the checks below and the append); a
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

	// The wiring world. A host file the index never walked is
	// parsed into it, so its own sites and registrations are
	// seen like any indexed file's (the page family's move).
	w := p.wireScan()
	if hostExists {
		if _, ok := w.files[rel]; !ok {
			wf := wireFileOf(rel, hostDir, hostSrc, hostFset, hostAST)
			w.files[rel] = wf
			w.scanFile(p, wf)
		}
	}

	// The ask may already be done, in whole: a handler with
	// this name in this package already serves this file.
	if h, _ := handlerServingFile(w, hostDir, name, servePath); h != nil {
		if reg := w.registrationOf(h); reg != nil {
			return nil, fmt.Sprintf("%s already serves %s — it answers at %s (%s:%d). Nothing to add.", h.name, serveRel, reg.route, p.LinkPath(reg.file), reg.line)
		}
		sentence := fmt.Sprintf("connect the %s handler to the webserver", h.name)
		if route != "/"+strings.ToLower(h.name) {
			sentence += " at " + route
		}
		return nil, fmt.Sprintf("%s already serves %s (%s:%d) — it just isn't registered with a server yet. The finishing sentence is \"%s\": it wires the handler you already have at %s.", h.name, serveRel, p.LinkPath(h.file), h.line, sentence, route)
	}
	// Or at the route, by an inline handler serving this file.
	if r, found := inlineServingAtRoute(p, w, hostDir, route, servePath); found {
		return nil, fmt.Sprintf("the server in %s already serves %s at http://localhost:8080%s (%s:%d) — nothing to add.", p.LinkPath(r.file), serveRel, route, p.LinkPath(r.file), r.line)
	}

	// The handler's name must be free in its package.
	if pkgName, declFile, declLine, found := p.pageNameCollision(hostDir, name, rel, hostFset, hostAST); found {
		return nil, fmt.Sprintf("package %s already has a declaration named %s (%s:%d), so the new handler can't take that name too — give the handler a different name with \"named <Name>\" and ask again.", pkgName, name, p.LinkPath(declFile), declLine)
	}

	// Conflicts at the route, read across the host program:
	// a route answers for one thing, never two.
	for _, r := range w.regs {
		if r.route != route {
			continue
		}
		if wf := w.files[r.file]; wf == nil || wf.pkgDir != hostDir {
			continue
		}
		return nil, fmt.Sprintf("%s is already taken — %s:%d registers it for %s. I won't point one path at two handlers, so %s wasn't added. Pick a different path and name it with \"at\", like \"add a handler named %s that serves %s in %s at /%s\".", route, p.LinkPath(r.file), r.line, r.target, name, name, serveBase, folderWords(hostDir), strings.TrimPrefix(route, "/"))
	}

	// Composition: an insert at the host package's one
	// composable site, or none.
	var site *wireSite
	composed := false
	siteRel := ""
	regLine := ""
	if site = pickPageSite(w, rel, hostDir); site != nil {
		composed = true
		siteRel = site.file
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
		if why, blocked := p.fileBlocked(siteRel); blocked {
			return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(siteRel), why)
		}
		if siteRel != rel {
			if wf := w.files[siteRel]; wf != nil && fileBytesBlocked(wf.src) {
				return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(siteRel))
			}
		}
	}

	handlerText := serveFileHandler(name, servePath)
	specs := []string{strconv.Quote("net/http")}
	internalErr := func(bad string) (*EditPlan, string) {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}

	// The registration edit, computed from the serving file's
	// ORIGINAL bytes: an insert just before the serving
	// statement (the page family's regEditFor, insert half).
	regEditFor := func() (textEdit, bool) {
		if site == nil || site.wf == nil {
			return textEdit{}, false
		}
		line := lineOfPos(site.wf.fset, site.stmt.Pos())
		indent := lineIndent(site.wf.src, line)
		insertAt := lineStartOffset(site.wf.src, line)
		return textEdit{start: insertAt, end: insertAt, text: indent + regLine + "\n"}, true
	}

	serveSameAsHost := composed && siteRel == rel
	var hostEF EditFile
	var serveEF *EditFile
	switch {
	case !hostExists:
		// Create the host file: package clause (the folder's
		// own package), imports, handler — assembled through
		// the family's one import inserter over a parsed
		// skeleton, like the page family's creations.
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
		wf := w.files[siteRel]
		if wf == nil {
			return internalErr("the registration site could not be re-read in the serving file")
		}
		regEdit, ok := regEditFor()
		if !ok {
			return internalErr("the registration site could not be re-read in the serving file")
		}
		sAfter := applyTextEdits(wf.src, []textEdit{regEdit})
		ef := finishFile(siteRel, wf.src, sAfter)
		serveEF = &ef
	}

	files := []EditFile{hostEF}
	if serveEF != nil {
		files = append(files, *serveEF)
	}

	notes := []string{
		fmt.Sprintf("http.ServeFile picks the answer's content type from the file's extension — a .png goes as an image, a .css as a stylesheet, a .json as data — so this one handler serves %s as what it is. The file itself isn't changed.", serveBase),
		serveRunNote(rel, servePath),
	}
	switch {
	case composed:
		notes = append(notes, "Your existing server is untouched apart from the one registration above — no second ListenAndServe is added.")
	default:
		// The finishing sentence must be the one that, fed
		// back verbatim, registers the handler at THIS route
		// (the page family's rule, route carried with "at"
		// whenever it differs from the wiring default).
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

	var summary, verifyDesc string
	switch {
	case composed:
		summary = fmt.Sprintf("Add a handler that serves %s: %s in %s, and one registration — %s — added in %s just before it serves, so http://localhost:8080%s is answered with the %s file. The file itself isn't changed.", serveRel, name, p.LinkPath(rel), regLine, p.LinkPath(siteRel), route, serveBase)
		verifyDesc = fmt.Sprintf("%s serves %s at %s", name, serveBase, route)
	default:
		summary = fmt.Sprintf("Add a handler that serves %s: %s in %s. %s isn't registered with a server yet — the plan's last note is the sentence that wires it.", serveRel, name, p.LinkPath(rel), name)
		verifyDesc = fmt.Sprintf("%s is declared in %s and serves %s", name, hostBase, serveBase)
	}

	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(hostEF.After, "func "+name+"(")),
		Files:      files,
		Notes:      notes,
		verify: func(after *Project) bool {
			if _, err := os.Stat(filepath.Join(after.Root, filepath.FromSlash(serveRel))); err != nil {
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
