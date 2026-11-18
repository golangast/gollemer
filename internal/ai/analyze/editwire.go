package analyze

// Wire a handler to the webserver: the registration recipe.
// The concept book PLACES handlers — a form handler named Jake
// lands in jim/main.go as a complete, compiling function — but
// placement is not wiring: nothing registered it, so the
// server never answered for it. This planner closes that gap
// for the sentence "connect the Jake handler to the
// webserver": ONE HandleFunc line, inserted just before the
// serve call in the function that serves, on the default mux
// or on the file's own *http.ServeMux variable. Nothing is
// created and no second server is ever added.
//
// Everything is read off the project's own code, AST-first:
// the handler must be a plain function shaped exactly like
// http.HandleFunc's argument (w http.ResponseWriter,
// r *http.Request, no results); the serving site is the
// function calling http.ListenAndServe or http.Serve; the
// conflicts are the project's existing registrations, read
// across every file (a route on the default mux is global,
// and a handler already answering at one path is not silently
// registered at a second). The states that cannot be wired
// honestly are answers, not edits:
//
//   - the named thing missing, or present but not
//     handler-shaped (its real signature is quoted back);
//   - the bare ask with zero registerable unregistered
//     handlers (none exist, all are registered — named — or
//     the unregistered ones live where no server can reach
//     them), or several (named, with their sentences);
//   - no server anywhere (the refusal spells the working
//     "add a webserver to <file>" sentence);
//   - servers exist but none can register this handler: it
//     sits in another program (package main elsewhere) or is
//     unexported in another package — the facts, and the
//     by-hand line where that helps;
//   - more than one server could take it and the sentence
//     cannot say which;
//   - the path is taken (the existing registration is named,
//     with its file and line), or this handler already
//     answers at another path;
//   - the server is custom (its own handler value, a server
//     struct, TLS): a registration on the default mux would
//     never be reached, so the exact line is handed over.
//
// The handler reference follows the uses family's rule across
// packages: the serving file's existing import of the
// handler's package when it has one, its declared package
// name otherwise, with the import added by the plan. An
// unexported handler in another package can never be
// registered from outside it, and says so.

import (
	"fmt"
	"go/ast"
	"go/token"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"
)

// --- the world scan ------------------------------------------------

// wireFile is one parsed project file, with the facts the
// wiring questions ask of it.
type wireFile struct {
	rel      string
	src      []byte
	fset     *token.FileSet
	ast      *ast.File
	pkgDir   string
	pkgName  string
	httpName string // the file's local name for the net/http import ("" = not imported)
	dotHTTP  bool   // net/http is dot-imported
}

// wireHandlerInfo is one handler-shaped plain function.
type wireHandlerInfo struct {
	name     string // the declared name (case as written)
	file     string
	line     int
	pkgDir   string
	pkgName  string
	exported bool
}

// wireReg is one existing HandleFunc/Handle registration.
type wireReg struct {
	route       string
	file        string
	line        int
	target      string // the handler expression in words: "Jake", "handlers.Login", "an inline handler"
	handlerDir  string // the referenced handler's package dir ("" = not a plain project function)
	handlerName string // the referenced handler's name ("" = not a plain function reference)
}

// siteKind classifies a serving call by what a registration
// can be written against.
type siteKind int

const (
	siteDefault siteKind = iota // ListenAndServe(addr, nil) / Serve(l, nil)
	siteMux                     // the handler argument is a *http.ServeMux variable
	siteCustom                  // anything else: never wired into, answered by hand
)

// wireSite is one function that serves.
type wireSite struct {
	file     string
	pkgDir   string
	pkgName  string
	funcName string
	kind     siteKind
	muxVar   string // siteMux: the mux variable's name
	how      string // siteCustom: how it serves, in words for the refusal
	stmt     ast.Stmt
	wf       *wireFile
}

// wireWorld is the scan: every handler, registration, and
// serving site in the project, in package/file order.
type wireWorld struct {
	files    map[string]*wireFile
	order    []string // parsed files, in scan order
	handlers []*wireHandlerInfo
	funcSigs map[string]string // "dir\x00name" -> rendered signature, for not-a-handler answers
	regs     []*wireReg
	sites    []*wireSite
}

// wireScan reads every non-test, non-vendor .go file in the
// project once and collects the wiring facts.
func (p *Project) wireScan() *wireWorld {
	w := &wireWorld{files: map[string]*wireFile{}, funcSigs: map[string]string{}}
	for _, pkg := range p.Packages {
		for _, rel := range pkg.Files {
			if strings.HasSuffix(rel, "_test.go") || rel == "vendor" || strings.HasPrefix(rel, "vendor/") {
				continue
			}
			src, fset, fileAST, refusal := p.readProjectFile(rel)
			if refusal != "" || fileAST == nil {
				continue // an unreadable file holds no facts this planner may invent from
			}
			wf := &wireFile{rel: rel, src: src, fset: fset, ast: fileAST, pkgDir: pkg.Dir, pkgName: fileAST.Name.Name}
			for _, is := range fileAST.Imports {
				path, _ := strconv.Unquote(is.Path.Value)
				if path != "net/http" {
					continue
				}
				_, eff := importSpecOf(is)
				wf.httpName = eff
				if is.Name != nil && is.Name.Name == "." {
					wf.dotHTTP = true
					wf.httpName = ""
				}
			}
			w.files[rel] = wf
			w.order = append(w.order, rel)
		}
	}
	for _, rel := range w.order {
		wf := w.files[rel]
		w.scanFile(p, wf)
	}
	return w
}

// scanFile collects one file's handlers, function signatures,
// registrations, and serving sites.
func (w *wireWorld) scanFile(p *Project, wf *wireFile) {
	for _, d := range wf.ast.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if fd.Recv == nil {
			if sig := renderFuncSig(wf.fset, fd); sig != "" {
				w.funcSigs[wf.pkgDir+"\x00"+fd.Name.Name] = sig
			}
			if isHandlerShape(wf, fd) {
				w.handlers = append(w.handlers, &wireHandlerInfo{
					name:     fd.Name.Name,
					file:     wf.rel,
					line:     lineOfPos(wf.fset, fd.Pos()),
					pkgDir:   wf.pkgDir,
					pkgName:  wf.pkgName,
					exported: isExportedName(fd.Name.Name),
				})
			}
		}
		if fd.Body != nil {
			if site := w.findSite(p, wf, fd); site != nil {
				w.sites = append(w.sites, site)
			}
		}
	}
	ast.Inspect(wf.ast, func(n ast.Node) bool {
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
		reg := &wireReg{route: route, file: wf.rel, line: lineOfPos(wf.fset, call.Pos())}
		reg.target, reg.handlerDir, reg.handlerName = w.handlerRefOf(p, wf, call.Args[1])
		w.regs = append(w.regs, reg)
		return true
	})
}

// renderFuncSig renders a function declaration's signature
// ("func Jake(w http.ResponseWriter, r *http.Request)") for
// the not-a-handler answer, which quotes the real thing. The
// whole FuncType is rendered at once: a bare FieldList does
// not format standalone (an empty one renders as nothing),
// while the type carries its parentheses and results.
func renderFuncSig(fset *token.FileSet, fd *ast.FuncDecl) string {
	if fd.Type == nil {
		return "func " + fd.Name.Name + "()"
	}
	rendered := strings.TrimPrefix(stmtText(fset, fd.Type), "func")
	return "func " + fd.Name.Name + rendered
}

// isExportedName reports whether a declared name is exported
// (its first letter is an uppercase letter).
func isExportedName(name string) bool {
	for _, r := range name {
		return unicode.IsUpper(r)
	}
	return false
}

// isHandlerShape reports whether fd is a plain function
// http.HandleFunc could take: exactly the parameters
// (http.ResponseWriter, *http.Request) and no results.
func isHandlerShape(wf *wireFile, fd *ast.FuncDecl) bool {
	if fd.Type == nil || fd.Type.Params == nil {
		return false
	}
	if fd.Type.Results != nil && len(fd.Type.Results.List) > 0 {
		return false
	}
	var types []ast.Expr
	for _, field := range fd.Type.Params.List {
		n := len(field.Names)
		if n == 0 {
			n = 1
		}
		for k := 0; k < n; k++ {
			types = append(types, field.Type)
		}
	}
	if len(types) != 2 {
		return false
	}
	return isHTTPType(wf, types[0], "ResponseWriter") && isHTTPStarType(wf, types[1], "Request")
}

// isHTTPType reports whether e is the net/http type name as
// this file can write it: <http>.<Name> through the file's
// own import name, or the bare name under a dot import.
func isHTTPType(wf *wireFile, e ast.Expr, name string) bool {
	switch t := e.(type) {
	case *ast.SelectorExpr:
		id, ok := t.X.(*ast.Ident)
		return ok && wf.httpName != "" && id.Name == wf.httpName && t.Sel.Name == name
	case *ast.Ident:
		return wf.dotHTTP && t.Name == name
	}
	return false
}

// isHTTPStarType is isHTTPType for the pointer form
// (*http.Request).
func isHTTPStarType(wf *wireFile, e ast.Expr, name string) bool {
	star, ok := e.(*ast.StarExpr)
	return ok && isHTTPType(wf, star.X, name)
}

// handlerRefOf reads a registration's handler argument into
// (words for messages, package dir, name). The dir/name pair
// is set only when the argument plainly names a project
// function: a bare identifier (this file's package), a
// selector on an imported package, or either wrapped in
// http.HandlerFunc(...).
func (w *wireWorld) handlerRefOf(p *Project, wf *wireFile, e ast.Expr) (target, dir, name string) {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name, wf.pkgDir, t.Name
	case *ast.SelectorExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			if path := importPathForQualifier(wf.ast, id.Name); path != "" {
				target = id.Name + "." + t.Sel.Name
				if pkg := p.pkgByPath(path); pkg != nil {
					return target, pkg.Dir, t.Sel.Name
				}
				return target, path, t.Sel.Name // another module's package: a dir no project handler has
			}
			// A selector on a local value: a method value —
			// a registration, but not of a plain function.
			return stmtText(wf.fset, e), "", ""
		}
		return stmtText(wf.fset, e), "", ""
	case *ast.CallExpr:
		// http.HandlerFunc(Jake) — the wrap http.Handle takes.
		if sel, ok := t.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "HandlerFunc" && len(t.Args) == 1 {
			if id, ok := sel.X.(*ast.Ident); ok && wf.httpName != "" && id.Name == wf.httpName {
				return w.handlerRefOf(p, wf, t.Args[0])
			}
		}
		if id, ok := t.Fun.(*ast.Ident); ok && id.Name == "HandlerFunc" && wf.dotHTTP && len(t.Args) == 1 {
			return w.handlerRefOf(p, wf, t.Args[0])
		}
		return stmtText(wf.fset, e), "", ""
	case *ast.FuncLit:
		return "an inline handler", "", ""
	}
	return stmtText(wf.fset, e), "", ""
}

// findSite finds a function's serving call — its FIRST serve
// call in source order — and classifies it.
func (w *wireWorld) findSite(p *Project, wf *wireFile, fd *ast.FuncDecl) *wireSite {
	var found *wireSite
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		if found != nil {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		site := w.classifyServeCall(p, wf, fd, call)
		if site != nil {
			found = site
			return false
		}
		return true
	})
	return found
}

// classifyServeCall reads one call as a serving call, or
// returns nil. http.ListenAndServe and http.Serve classify by
// their handler argument; the TLS variants and the
// server-value method calls are custom sites (real servers
// this recipe hands the line over for, never a "no server"
// answer).
func (w *wireWorld) classifyServeCall(p *Project, wf *wireFile, fd *ast.FuncDecl, call *ast.CallExpr) *wireSite {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return nil
	}
	base := &wireSite{
		file: wf.rel, pkgDir: wf.pkgDir, pkgName: wf.pkgName,
		funcName: fd.Name.Name, wf: wf,
	}
	xIdent, _ := sel.X.(*ast.Ident)
	isHTTP := wf.httpName != "" && xIdent != nil && xIdent.Name == wf.httpName
	switch {
	case isHTTP && sel.Sel.Name == "ListenAndServe" && len(call.Args) == 2:
		return w.classifyHandlerArg(base, wf, fd, call, call.Args[1])
	case isHTTP && sel.Sel.Name == "Serve" && len(call.Args) == 2:
		return w.classifyHandlerArg(base, wf, fd, call, call.Args[1])
	case isHTTP && (sel.Sel.Name == "ListenAndServeTLS" || sel.Sel.Name == "ServeTLS"):
		base.kind = siteCustom
		base.how = "it serves over TLS (" + sel.Sel.Name + "), whose registrations go where that call's handler is set up"
	case !isHTTP && (sel.Sel.Name == "ListenAndServe" || sel.Sel.Name == "ListenAndServeTLS") && xIdent != nil:
		// A method call on a server value: srv.ListenAndServe().
		base.kind = siteCustom
		base.how = "it serves through a server value (" + xIdent.Name + "." + sel.Sel.Name + "()), whose Handler is set where the value is built"
	default:
		return nil
	}
	_, stmt := enclosingStmt(wf.ast, call)
	if stmt == nil {
		return nil
	}
	base.stmt = stmt
	return base
}

// classifyHandlerArg classifies a serve call by its handler
// argument: nil is the default mux, a *http.ServeMux variable
// is the mux shape, anything else is custom.
func (w *wireWorld) classifyHandlerArg(base *wireSite, wf *wireFile, fd *ast.FuncDecl, call *ast.CallExpr, arg ast.Expr) *wireSite {
	if id, ok := arg.(*ast.Ident); ok && id.Name == "nil" {
		base.kind = siteDefault
	} else if id, ok := arg.(*ast.Ident); ok && isServeMuxVar(wf.ast, id.Name) {
		base.kind = siteMux
		base.muxVar = id.Name
	} else {
		base.kind = siteCustom
		if id, ok := arg.(*ast.Ident); ok {
			base.how = "it hands ListenAndServe its own handler (" + id.Name + "), not the default mux"
		} else {
			base.how = "it hands ListenAndServe its own handler, not the default mux"
		}
	}
	_, stmt := enclosingStmt(wf.ast, call)
	if stmt == nil {
		return nil
	}
	base.stmt = stmt
	return base
}

// registrationOf returns the handler's existing registration
// (at any path), or nil when it is not registered anywhere.
func (w *wireWorld) registrationOf(h *wireHandlerInfo) *wireReg {
	for _, r := range w.regs {
		if r.handlerName != "" && r.handlerDir == h.pkgDir && strings.EqualFold(r.handlerName, h.name) {
			return r
		}
	}
	return nil
}

// --- resolution ----------------------------------------------------

// resolveNamed finds the handler a name asks for, or the
// refusal that tells the truth about the name: no such
// function, a function that isn't a handler (its real
// signature quoted), or several handlers sharing the name
// with none of them beside a server.
func (p *Project) resolveNamedHandler(w *wireWorld, name string) (*wireHandlerInfo, string) {
	var matches []*wireHandlerInfo
	for _, h := range w.handlers {
		if strings.EqualFold(h.name, name) {
			matches = append(matches, h)
		}
	}
	switch {
	case len(matches) == 1:
		return matches[0], ""
	case len(matches) > 1:
		// Several handlers share the name: the one beside a
		// server (same package as a serving site) is the one
		// the sentence can mean; still several — named below.
		var beside []*wireHandlerInfo
		for _, h := range matches {
			for _, s := range w.sites {
				if s.pkgDir == h.pkgDir {
					beside = append(beside, h)
					break
				}
			}
		}
		if len(beside) == 1 {
			return beside[0], ""
		}
		var b strings.Builder
		fmt.Fprintf(&b, "%q is more than one handler in this project:\n", name)
		for _, h := range matches {
			fmt.Fprintf(&b, "  - %s in package %s (%s:%d)\n", h.name, h.pkgName, p.LinkPath(h.file), h.line)
		}
		b.WriteString("The sentence can't say which package's one you mean, and I won't guess. Wire the one you mean by hand: add its HandleFunc line in the server that lives in its own package.")
		return nil, b.String()
	}
	// No handler by that name. A plain function by that name is
	// a different refusal: it exists, and it isn't a handler.
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest || fn.Receiver != "" || !strings.EqualFold(fn.Name, name) {
				continue
			}
			sig := w.funcSigs[fn.PkgDir+"\x00"+fn.Name]
			if sig == "" {
				sig = fn.Sig
			}
			return nil, fmt.Sprintf("%s exists (%s:%d) but it isn't a handler — a handler is a function shaped like func %s(w http.ResponseWriter, r *http.Request), with no results, and %s is %s. I can only register handlers with a webserver.", fn.Name, p.LinkPath(fn.File), fn.Line, fn.Name, fn.Name, sig)
		}
	}
	sug := "main.go"
	if eps := p.EntryPoints(); len(eps) > 0 {
		sug = eps[0].File
	}
	return nil, fmt.Sprintf("I couldn't find a handler named %q in this project — there's no function with that name. If it doesn't exist yet, add it first: say \"add a form handler named %s to %s\", then ask me to connect it.", name, name, sug)
}

// canRegister reports whether site's package could register
// h: same package always; another package only when the
// handler is exported and its package is importable (not a
// program's package main).
func canRegister(site *wireSite, h *wireHandlerInfo) bool {
	if site.pkgDir == h.pkgDir {
		return true
	}
	return h.exported && h.pkgName != "main"
}

// pickSite chooses the serving site for h: its own file
// first, then its own package, then the only candidate. The
// refusal covers the two unreachable shapes (another
// program's handler, an unexported one) and the several-
// servers ambiguity. Generated/vendor serving files are
// skipped while a clean candidate exists, so the blocked
// refusal fires only when the blocked file is the only
// server there is.
func (p *Project) pickSite(w *wireWorld, h *wireHandlerInfo) (*wireSite, string) {
	var reg []*wireSite
	for _, s := range w.sites {
		if canRegister(s, h) {
			reg = append(reg, s)
		}
	}
	if len(reg) == 0 {
		if len(w.sites) == 0 {
			return nil, p.wireNoServerRefusal(h)
		}
		if h.pkgName == "main" {
			return nil, fmt.Sprintf("%s lives in %s — package main in the %s folder is a program of its own, and the webserver in %s can't reach into another program to register it. Wire it in its own program instead: if that file can take a webserver, say \"add a webserver to %s\" and ask again; otherwise add the line http.HandleFunc(\"/%s\", %s) by hand in the main that should serve it, just before its ListenAndServe.",
				h.name, p.LinkPath(h.file), folderWords(h.pkgDir), p.LinkPath(w.sites[0].file), p.LinkPath(h.file), strings.ToLower(h.name), h.name)
		}
		return nil, fmt.Sprintf("%s is declared in package %s (%s:%d) with a lowercase name, so no other package can see it — the webserver in %s can't register it from outside its package. Rename it to start with a capital letter and ask again.",
			h.name, h.pkgName, p.LinkPath(h.file), h.line, p.LinkPath(w.sites[0].file))
	}
	var clean []*wireSite
	for _, s := range reg {
		if _, blocked := p.fileBlocked(s.file); !blocked {
			clean = append(clean, s)
		}
	}
	if len(clean) > 0 {
		reg = clean
	}
	for _, s := range reg {
		if s.file == h.file {
			return s, ""
		}
	}
	for _, s := range reg {
		if s.pkgDir == h.pkgDir {
			return s, ""
		}
	}
	if len(reg) == 1 {
		return reg[0], ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "this project has more than one webserver that could take %s — ", h.name)
	for i, s := range reg {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "the one in %s (%s)", p.LinkPath(s.file), s.funcName)
	}
	fmt.Fprintf(&b, " — and the sentence doesn't say which one should answer for it, so I won't guess. The line is http.HandleFunc(\"/%s\", %s): add it by hand in the main you mean, just before its ListenAndServe.", strings.ToLower(h.name), h.name)
	return nil, b.String()
}

// folderWords renders a package dir for the own-program
// refusal ("the jim folder"; "" is the project root).
func folderWords(dir string) string {
	if dir == "" {
		return "project root"
	}
	return dir
}

// wireNoServerRefusal is the no-server-anywhere answer: the
// working "add a webserver" sentence, aimed at the handler's
// own file when it is a program file, else at the project's
// program, else at a new main.go.
func (p *Project) wireNoServerRefusal(h *wireHandlerInfo) string {
	target := ""
	switch {
	case h != nil && h.pkgName == "main":
		target = h.file
	default:
		if eps := p.EntryPoints(); len(eps) > 0 {
			target = eps[0].File
		} else {
			target = "main.go"
		}
	}
	who := "it"
	if h != nil {
		who = h.name
	}
	return fmt.Sprintf("there's no webserver in this project yet — nothing calls ListenAndServe, so there's nothing to connect %s to. Add one first: say \"add a webserver to %s\", then ask me again.", who, target)
}

// wireCustomRefusal is the custom-server answer: the server
// runs its own way, so a default-mux registration would never
// be reached — the exact line is handed over instead.
func (p *Project) wireCustomRefusal(site *wireSite, h *wireHandlerInfo, route, ref string) string {
	return fmt.Sprintf("%s already runs a server in its own way — %s — so a registration on the default mux would never be reached, and I won't add a second server. To wire %s in there, add this line where the server is set up, before it starts serving (on your mux instead of http, if that is what the file uses):\n\n%s.HandleFunc(%s, %s)",
		p.LinkPath(site.file), site.how, h.name, receiverName(site), strconv.Quote(route), ref)
}

// receiverName is the registration receiver a by-hand line
// should use: the file's own http name (custom sites serve in
// their own way, so the line is a starting point).
func receiverName(site *wireSite) string {
	if site.wf != nil && site.wf.httpName != "" {
		return site.wf.httpName
	}
	return "http"
}

// --- the planner ---------------------------------------------------

// planWireHandler assembles the registration plan. See the
// file header for the state table.
func (p *Project) planWireHandler(req EditRequest) (*EditPlan, string) {
	route := strings.TrimSpace(req.WireRoute)
	if route != "" && !strings.HasPrefix(route, "/") {
		return nil, fmt.Sprintf("the path %q needs to start with a slash — say it like \"at /contact\".", req.WireRoute)
	}
	w := p.wireScan()

	var h *wireHandlerInfo
	if name := strings.TrimSpace(req.WireName); name != "" {
		var refusal string
		h, refusal = p.resolveNamedHandler(w, name)
		if refusal != "" {
			return nil, refusal
		}
	} else {
		var refusal string
		h, refusal = p.resolveBareHandler(w)
		if refusal != "" {
			return nil, refusal
		}
	}
	if route == "" {
		route = "/" + strings.ToLower(h.name)
	}

	site, refusal := p.pickSite(w, h)
	if refusal != "" {
		return nil, refusal
	}
	if why, blocked := p.fileBlocked(site.file); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(site.file), why)
	}
	if fileBytesBlocked(site.wf.src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(site.file))
	}

	// The handler reference as the serving file must write it,
	// and the import that needs (the uses family's rule).
	ref := h.name
	var importSpecs []string
	qualNote := ""
	if site.pkgDir != h.pkgDir {
		path := p.pkgPathOf(h.pkgDir)
		if p.Module == "" || path == "" || strings.HasPrefix(path, "/") {
			return nil, "I couldn't work out the handler's package import path (the project has no go.mod module line I can read), so I can't write the import honestly."
		}
		qual := ""
		for _, is := range site.wf.ast.Imports {
			if ip, _ := strconv.Unquote(is.Path.Value); ip == path {
				_, qual = importSpecOf(is)
			}
		}
		if qual == "" {
			qual = h.pkgName
			if other := importPathForQualifier(site.wf.ast, qual); other != "" && other != path {
				return nil, fmt.Sprintf("%s already uses the name %s for %s, so I can't also write the handler's package (%s) under it — rename that import by hand and ask again.", p.LinkPath(site.file), qual, other, path)
			}
			if sp := p.pkgByDir(site.pkgDir); sp != nil && p.pkgDeclares(sp, qual) {
				return nil, fmt.Sprintf("the package in %s already declares a %s of its own, so the handler's package (%s) can't be written under that name there — rename one of them by hand and ask again.", p.LinkPath(site.file), qual, path)
			}
			importSpecs = append(importSpecs, strconv.Quote(path))
			qualNote = fmt.Sprintf("%s also gains the import %s — the registration names the handler as %s.", p.LinkPath(site.file), strconv.Quote(path), qual+"."+h.name)
		}
		ref = qual + "." + h.name
	}

	// Conflicts, read across the whole project: a route on
	// any mux belongs to whoever holds it, and this handler
	// answering at one path is never quietly given a second.
	for _, r := range w.regs {
		if r.route != route {
			continue
		}
		if r.handlerName != "" && r.handlerDir == h.pkgDir && strings.EqualFold(r.handlerName, h.name) {
			return nil, fmt.Sprintf("%s is already registered at %s (%s:%d) — nothing to add.", h.name, route, p.LinkPath(r.file), r.line)
		}
		return nil, fmt.Sprintf("%s is already taken — %s:%d registers it for %s. I won't point one path at two handlers; pick a different path and name it with \"at\", like \"connect the %s handler to the webserver at /contact\".", route, p.LinkPath(r.file), r.line, r.target, h.name)
	}
	if existing := w.registrationOf(h); existing != nil {
		return nil, fmt.Sprintf("%s is already registered at %s (%s:%d) — it answers there now, and I won't register it at %s as well. To move it, change that line by hand.", h.name, existing.route, p.LinkPath(existing.file), existing.line, route)
	}

	if site.kind == siteCustom {
		return nil, p.wireCustomRefusal(site, h, route, ref)
	}

	// Assemble: one line, immediately before the statement
	// that serves, at that statement's indent.
	recv := site.wf.httpName
	if recv == "" {
		recv = "http"
	}
	if site.kind == siteMux {
		recv = site.muxVar
	}
	regLine := recv + ".HandleFunc(" + strconv.Quote(route) + ", " + ref + ")"
	line := lineOfPos(site.wf.fset, site.stmt.Pos())
	indent := lineIndent(site.wf.src, line)
	insertAt := lineStartOffset(site.wf.src, line)
	edits := []textEdit{{start: insertAt, end: insertAt, text: indent + regLine + "\n"}}
	if len(importSpecs) > 0 {
		edits = append(edits, insertImportSpecs(site.wf.ast, site.wf.fset, site.wf.src, importSpecs)...)
	}
	after := applyTextEdits(site.wf.src, edits)
	ef := finishFile(site.file, site.wf.src, after)

	notes := []string{"Your existing server is untouched apart from the one registration above — no second ListenAndServe is added."}
	if qualNote != "" {
		notes = append([]string{qualNote}, notes...)
	}
	summary := fmt.Sprintf("Wire %s into the webserver in %s: one line — %s — added in %s just before it serves, so http://localhost:8080%s is answered by %s (declared at %s:%d).",
		h.name, p.LinkPath(site.file), regLine, site.funcName, route, h.name, p.LinkPath(h.file), h.line)
	if site.kind == siteMux {
		summary = fmt.Sprintf("Wire %s into the webserver in %s: one line — %s — added in %s alongside its other registrations, just before it serves, so http://localhost:8080%s is answered by %s (declared at %s:%d).",
			h.name, p.LinkPath(site.file), regLine, site.funcName, route, h.name, p.LinkPath(h.file), h.line)
	}
	siteRel, hDir, hName := site.file, h.pkgDir, h.name
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s registers %s at %s", filepath.Base(site.file), h.name, route),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(site.file), insertedLine(ef.After, regLine)),
		Files:      []EditFile{ef},
		Notes:      notes,
		verify: func(after *Project) bool {
			// The intended structure, read off the fresh
			// analysis: a registration in the serving file at
			// the route, naming this handler.
			for _, r := range after.wireScan().regs {
				if r.file == siteRel && r.route == route && r.handlerName != "" &&
					r.handlerDir == hDir && strings.EqualFold(r.handlerName, hName) {
					return true
				}
			}
			return false
		},
	}, ""
}

// WireCandidate is one handler a bare wiring ask could
// mean: an unregistered handler some server in the project
// can actually register, with where it is declared.
type WireCandidate struct {
	Name string
	File string // project-relative, slash form (LinkPath form)
	Line int
}

// BareWireCandidates returns the candidates for the bare
// "the handler" when there are several — the same set
// resolveBareHandler names in its refusal — or nil when the
// count is anything else (zero registerable, exactly one,
// no handlers, no server): those states have their own
// answers and are not a choice. The chat layer uses it to
// ask which handler instead of only naming them; the
// planner's refusal remains the backstop, computed from
// the same helper so the two can never disagree.
func (p *Project) BareWireCandidates() []WireCandidate {
	w := p.wireScan()
	if len(w.handlers) == 0 || len(w.sites) == 0 {
		return nil
	}
	cands, _ := w.bareCandidates()
	if len(cands) < 2 {
		return nil
	}
	out := make([]WireCandidate, len(cands))
	for i, h := range cands {
		out[i] = WireCandidate{Name: h.name, File: p.LinkPath(h.file), Line: h.line}
	}
	return out
}

// WireRegistration is one registration (a HandleFunc /
// Handle-style call) found in the project's code, in plain
// facts: the handler it points at, the route, and the file
// holding the call.
type WireRegistration struct {
	Name       string // the handler function's name; "" when the registration's argument is not a plain project function reference
	Route      string
	File       string // the file holding the registration (project-relative, slash form)
	HandlerDir string // the referenced handler's package dir ("" = not a plain project function)
}

// WireRegistrations returns every registration in the
// project, in scan order. The chat layer diffs the lists
// from before and after an edit to name the registration
// the edit added — from the project's own code, never from
// the sentence that asked for it.
func (p *Project) WireRegistrations() []WireRegistration {
	if p == nil {
		return nil
	}
	w := p.wireScan()
	out := make([]WireRegistration, 0, len(w.regs))
	for _, r := range w.regs {
		out = append(out, WireRegistration{
			Name:       r.handlerName,
			Route:      r.route,
			File:       r.file,
			HandlerDir: r.handlerDir,
		})
	}
	return out
}

// HandlerFacts is the wiring picture of one handler-shaped
// function, read from the project's own code: whether the
// name is a handler at all, where it is declared, and where
// it is registered when it is.
type HandlerFacts struct {
	IsHandler  bool
	File       string // the handler's own file (project-relative, slash form)
	Registered bool
	Route      string // the route it answers at, when Registered
	RegFile    string // the file holding the registration, when Registered
}

// HandlerFacts reports the wiring facts for the handler
// with the given name. dir, when non-empty, is the
// slash-relative folder the handler must live in (""
// for the project root) — it chooses between same-named
// handlers in different packages. A name held by several
// handlers with no dir to choose between them is
// ambiguous and reports no facts at all: a caller must
// never state wiring facts from a guess.
func (p *Project) HandlerFacts(name, dir string) HandlerFacts {
	var out HandlerFacts
	if p == nil || name == "" {
		return out
	}
	w := p.wireScan()
	var h *wireHandlerInfo
	for _, cand := range w.handlers {
		if cand.name != name {
			continue
		}
		if dir != "" && cand.pkgDir != dir {
			continue
		}
		if h != nil {
			return HandlerFacts{} // two handlers, one name: no facts
		}
		h = cand
	}
	if h == nil {
		return out
	}
	out.IsHandler = true
	out.File = h.file
	if reg := w.registrationOf(h); reg != nil {
		out.Registered = true
		out.Route = reg.route
		out.RegFile = reg.file
	}
	return out
}

// bareCandidates is the bare ask's candidate set: the
// unregistered handlers at least one serving site can
// register.
func (w *wireWorld) bareCandidates() (candidates, unreachable []*wireHandlerInfo) {
	for _, h := range w.handlers {
		if w.registrationOf(h) != nil {
			continue
		}
		reachable := false
		for _, s := range w.sites {
			if canRegister(s, h) {
				reachable = true
				break
			}
		}
		if reachable {
			candidates = append(candidates, h)
		} else {
			unreachable = append(unreachable, h)
		}
	}
	return candidates, unreachable
}

// resolveBareHandler settles the bare "the handler": the
// project's one unregistered handler that some server can
// actually register. Every other count is an answer naming
// the facts.
func (p *Project) resolveBareHandler(w *wireWorld) (*wireHandlerInfo, string) {
	if len(w.handlers) == 0 {
		sug := "main.go"
		if eps := p.EntryPoints(); len(eps) > 0 {
			sug = eps[0].File
		}
		return nil, fmt.Sprintf("I don't see a handler in this project yet — a handler is a function shaped like func Name(w http.ResponseWriter, r *http.Request). Add one first: say \"add a form handler named Jake to %s\", then \"connect the Jake handler to the webserver\" will wire it in.", sug)
	}
	if len(w.sites) == 0 {
		// Aim the add-a-webserver sentence the way the named
		// flow does: at the one handler's own program file
		// when there is exactly one handler, else generically.
		if len(w.handlers) == 1 {
			return nil, p.wireNoServerRefusal(w.handlers[0])
		}
		return nil, p.wireNoServerRefusal(nil)
	}
	candidates, unreachable := w.bareCandidates()
	switch {
	case len(candidates) == 1:
		return candidates[0], ""
	case len(candidates) > 1:
		var b strings.Builder
		b.WriteString("More than one handler isn't registered yet — say which one you mean:\n")
		for _, h := range candidates {
			fmt.Fprintf(&b, "  - \"connect the %s handler to the webserver\" (%s, %s:%d)\n", h.name, h.name, p.LinkPath(h.file), h.line)
		}
		b.WriteString("Nothing was written.")
		return nil, b.String()
	}
	// No registerable unregistered handler. Either everything
	// is registered already (name the answers), or the
	// unregistered ones live where no server can reach them.
	var registered []*wireHandlerInfo
	for _, h := range w.handlers {
		if w.registrationOf(h) != nil {
			registered = append(registered, h)
		}
	}
	if len(unreachable) == 0 && len(registered) > 0 {
		var parts []string
		for _, h := range registered {
			r := w.registrationOf(h)
			parts = append(parts, fmt.Sprintf("%s answers at %s (%s:%d)", h.name, r.route, p.LinkPath(r.file), r.line))
		}
		return nil, fmt.Sprintf("every handler I can reach is already registered — %s. Nothing to add.", strings.Join(parts, "; "))
	}
	var b strings.Builder
	b.WriteString("the handlers that aren't registered yet can't be reached by this project's webserver:\n")
	for _, h := range unreachable {
		fmt.Fprintf(&b, "  - %s (%s:%d) — ", h.name, p.LinkPath(h.file), h.line)
		if h.pkgName == "main" {
			b.WriteString("it lives in its own program (package main); wire it in that program instead\n")
		} else {
			b.WriteString("its name starts with a lowercase letter, so it stays inside package " + h.pkgName + "\n")
		}
	}
	b.WriteString("Name one of them after fixing that, like \"connect the <Name> handler to the webserver\". Nothing was written.")
	return nil, b.String()
}
