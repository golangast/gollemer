package analyze

// Cross-file delegate: the seventh assembled shape, and the first
// whose two ends live in different files. "Add a function in
// app/main.go that calls store.Load" assembles a wrapper in the
// target file whose signature is the source's REAL signature —
// read from the source's own declaration in the index, never from
// the sentence — and whose body is the one call, so the source's
// results (an error included) reach the wrapper's caller exactly
// as the source produced them.
//
// The proposer is the assembler over the analyzer index: no model
// emits patch text anywhere in this path. Two forms of the
// wrapper are assembled up front — the plain form (the source
// imported under its declared name) and the aliased retry form —
// and the plan names both, so the chat's one bounded retry never
// writes anything the gate didn't cover. Before a plan exists at
// all, the generated function is walked by selfCheckDelegate: a
// failure there is our bug, reported as one, with no plan shown.

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"sort"
	"strconv"
	"strings"
)

// planDelegate assembles a cross-file (or same-package) delegate.
func (p *Project) planDelegate(req EditRequest) (*EditPlan, string) {
	rel := strings.TrimSpace(req.DelegateFile)
	if rel == "" {
		return nil, "that edit needs a file to add the function to."
	}
	if strings.HasSuffix(rel, "_test.go") {
		return nil, fmt.Sprintf("%s is a test file — delegates don't go there. (The test skeleton this shape writes is the only test-file change it makes.)", p.LinkPath(rel))
	}
	if why, blocked := p.fileBlocked(rel); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(rel), why)
	}
	tgtPkg := p.pkgOfFile(rel)
	if tgtPkg == nil {
		return nil, fmt.Sprintf("%s isn't part of the project I analyzed — it may sit in a folder the analyzer skips, or under its own go.mod. I can only write into files the index knows.", p.LinkPath(rel))
	}
	srcBytes, tgtFset, tgtAST, refusal := p.readProjectFile(rel)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(srcBytes) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(rel))
	}

	// Resolve the source: a plain function that exists in the
	// index. Every miss is an honest line, never a fall-through.
	src, refusal := p.resolveDelegateSource(req.SourceName)
	if refusal != "" {
		return nil, refusal
	}
	cross := src.PkgDir != tgtPkg.Dir

	if cross {
		if srcPkg := p.pkgByDir(src.PkgDir); srcPkg != nil && srcPkg.Name == "main" {
			return nil, fmt.Sprintf("%s lives in package main — a program, and programs can't be imported, so nothing in another package can call it.", src.Display())
		}
		if !src.Exported {
			return nil, fmt.Sprintf("%s isn't exported (its name starts lowercase), and %s is a different package — Go only lets other packages call exported names. Export it (rename it to %s), or put the delegate in %s instead.", src.Name, tgtPkg.Name, titleFirst(src.Name), src.Pkg)
		}
	}

	// Wrapper naming: the user's name, else the source's name
	// when the target package doesn't declare it, else the one
	// fallback (the source's name + From + its package).
	wrapper := strings.TrimSpace(req.WrapperName)
	userNamed := wrapper != ""
	if wrapper == "" {
		wrapper = src.Name
	}
	if !token.IsIdentifier(wrapper) {
		return nil, fmt.Sprintf("%q isn't a name Go allows for a function.", wrapper)
	}
	if !cross && !userNamed {
		// Same package: the source's own name is taken by the
		// source itself, and the From-fallback would name the
		// package it already lives in. A same-package wrapper
		// needs a name of its own — ask for one, with the exact
		// sentence that works.
		return nil, fmt.Sprintf("%s already lives in package %s, so a wrapper next to it needs its own name — say \"add a function My%s in %s that calls %s\".", src.Name, tgtPkg.Name, src.Name, rel, req.SourceName)
	}
	if p.pkgDeclares(tgtPkg, wrapper) {
		if userNamed {
			return nil, fmt.Sprintf("%s already exists in package %s (%s) — pick a different name for the wrapper.", wrapper, tgtPkg.Name, p.LinkPath(rel))
		}
		srcPkgName := src.Pkg
		fallback := src.Name + "From" + titleFirst(srcPkgName)
		if p.pkgDeclares(tgtPkg, fallback) || !token.IsIdentifier(fallback) {
			return nil, fmt.Sprintf("package %s already declares both %s and %s — give the wrapper a name yourself, like \"add a function My%s in %s that calls %s\".", tgtPkg.Name, src.Name, fallback, src.Name, rel, req.SourceName)
		}
		wrapper = fallback
	}

	// Re-read the source's declaration: the wrapper's signature
	// is built from these AST nodes, not from the index's prose.
	_, srcFset, srcAST, refusal := p.readProjectFile(src.File)
	if refusal != "" {
		return nil, refusal
	}
	var srcDecl *ast.FuncDecl
	for _, d := range srcAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && lineOfPos(srcFset, fd.Pos()) == src.Line {
			srcDecl = fd
			break
		}
	}
	if srcDecl == nil || srcDecl.Type == nil {
		return nil, fmt.Sprintf("I found %s in the index but couldn't re-read its declaration in %s.", src.Display(), p.LinkPath(src.File))
	}

	g := &delegateGen{
		p:          p,
		src:        src,
		srcDecl:    srcDecl,
		srcFset:    srcFset,
		srcAST:     srcAST,
		tgtAST:     tgtAST,
		tgtFset:    tgtFset,
		tgtPkg:     tgtPkg,
		cross:      cross,
		wrapper:    wrapper,
		targetRel:  rel,
		targetSrc:  srcBytes,
		typeParams: map[string]bool{},
		srcTop:     map[string]bool{},
		srcImports: map[string]string{},
		tgtImports: map[string]string{},
	}
	if cross {
		g.srcPath = p.Module
		if src.PkgDir != "" {
			g.srcPath = p.Module + "/" + src.PkgDir
		}
		if g.srcPath == "" || strings.HasPrefix(g.srcPath, "/") {
			return nil, "I couldn't work out the source package's import path (the project has no go.mod module line I can read), so I can't write the import honestly."
		}
	}
	if srcPkg := p.pkgByDir(src.PkgDir); srcPkg != nil {
		for _, fn := range srcPkg.Funcs {
			if fn.Receiver == "" {
				g.srcTop[fn.Name] = true
			}
		}
		for _, t := range srcPkg.Types {
			g.srcTop[t.Name] = true
		}
		for _, v := range srcPkg.Vars {
			g.srcTop[v.Name] = true
		}
	}
	if srcDecl.Type.TypeParams != nil {
		for _, f := range srcDecl.Type.TypeParams.List {
			for _, n := range f.Names {
				g.typeParams[n.Name] = true
			}
		}
	}
	for _, is := range srcAST.Imports {
		_, eff := importSpecOf(is)
		path, _ := strconv.Unquote(is.Path.Value)
		g.srcImports[eff] = path
	}
	for _, is := range tgtAST.Imports {
		_, eff := importSpecOf(is)
		path, _ := strconv.Unquote(is.Path.Value)
		g.tgtImports[path] = eff
	}
	if cross {
		if eff, ok := g.tgtImports[g.srcPath]; ok && (eff == "." || eff == "_") {
			return nil, fmt.Sprintf("%s imports %s with a %s import, which gives the package no name I can call through — give that import its plain name first, then ask again.", p.LinkPath(rel), g.srcPath, map[string]string{".": "dot", "_": "blank"}[eff])
		}
	}

	primary, alt, refusal := g.forms()
	if refusal != "" {
		return nil, refusal
	}

	// The test skeleton rides in the same plan, as a second file.
	stubEF, stubNote := p.delegateStub(tgtPkg, rel, wrapper)

	files := []EditFile{*primary}
	if stubEF != nil {
		files = append(files, *stubEF)
	}
	var altFiles []EditFile
	altDesc := ""
	if alt != nil {
		altFiles = []EditFile{*alt}
		if stubEF != nil {
			altFiles = append(altFiles, *stubEF)
		}
		altDesc = fmt.Sprintf("the retry form: the same delegate, with the source imported under the alias %s (%s.%s) instead of the plain name — used only if the plain form fails to build over the import or the qualification.", g.altQual(), g.altQual(), src.Name)
	}

	notes := []string{}
	if fm := src.AnalyzeFailures(p); fm.CanFail || len(fm.Panics) > 0 {
		notes = append(notes, fmt.Sprintf("One thing to know: %s — the wrapper passes %s's results (and its failures) straight through to whoever calls it.", src.FailurePhrase(p), src.Name))
	}
	if stubNote != "" {
		notes = append(notes, stubNote)
	}
	if altDesc != "" {
		notes = append(notes, "If the plain form doesn't build: "+altDesc)
	} else if cross {
		notes = append(notes, "No alternate form is available for this one — if it doesn't build, I'll restore your file and show you the compiler's words.")
	} else {
		notes = append(notes, "Same package, so there's no import to retry with a different form — if it doesn't build, I'll restore your file and show you the compiler's words.")
	}

	summary := fmt.Sprintf("Add %s to %s: a delegate that calls %s (%s:%d) with its own arguments and returns the results unchanged.",
		wrapper, p.LinkPath(rel), src.Display(), p.LinkPath(src.File), src.Line)
	wName, sName := wrapper, src.Name
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s now exists in %s and calls %s", wrapper, p.LinkPath(rel), src.Name),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(primary.After, "func "+wrapper)),
		Files:      files,
		Provenance: fmt.Sprintf("built from %s:%d — %s", p.LinkPath(src.File), src.Line, src.Sig),
		Notes:      notes,
		AltFiles:   altFiles,
		AltDesc:    altDesc,
		verify: func(after *Project) bool {
			return verifyDelegateCall(after.Root, rel, wName, sName)
		},
	}, ""
}

// resolveDelegateSource settles the typed source reference to one
// plain function in the index, or an honest refusal: methods,
// main/init, a type's name, ambiguity, and absence each get their
// own line — and absence names near matches when the project's
// near-name machinery finds them.
func (p *Project) resolveDelegateSource(typed string) (*Func, string) {
	typed = strings.TrimSpace(typed)
	matches := p.findProjectFuncs(typed)
	var plain, methods []*Func
	for _, fn := range matches {
		if fn.Receiver == "" {
			plain = append(plain, fn)
		} else {
			methods = append(methods, fn)
		}
	}
	// findProjectFuncs matches case-insensitively (a Q&A
	// tolerance), so "Load" also finds the unexported "load".
	// A delegate writes code against ONE function: when an
	// exact-case match exists, it is the one meant.
	lastSeg := typed
	if i := strings.LastIndex(typed, "."); i >= 0 {
		lastSeg = typed[i+1:]
	}
	var exact []*Func
	for _, fn := range plain {
		if fn.Name == lastSeg {
			exact = append(exact, fn)
		}
	}
	if len(exact) > 0 {
		plain = exact
	}
	if len(plain) == 0 && len(methods) > 0 {
		m := methods[0]
		return nil, fmt.Sprintf("%s is a method on %s (%s:%d) — this shape wraps plain functions only; a method keeps its receiver, and that shape isn't built yet.", typed, strings.Trim(m.Receiver, "*"), p.LinkPath(m.File), m.Line)
	}
	if len(plain) > 1 {
		return nil, p.funcCandidates("name its package", typed, plain)
	}
	if len(plain) == 1 {
		fn := plain[0]
		if fn.IsMain {
			return nil, "main is the program's entry point — other code can't call it, so there's nothing to delegate to."
		}
		if fn.IsInit {
			return nil, "init runs by itself before main — it can't be called, so there's nothing to delegate to."
		}
		return fn, ""
	}
	// Not a function. A type of that name gets the v1 refusal for
	// type-shaped asks, not the generic not-found line.
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if strings.EqualFold(t.Name, lastSeg) {
				return nil, fmt.Sprintf("%q is a %s (in %s, %s:%d), not a function — this shape delegates functions only.", typed, t.Kind, t.Pkg, p.LinkPath(t.File), t.Line)
			}
		}
	}
	msg := fmt.Sprintf("I can't find a function called %q in this project — I can only build from code that exists.", typed)
	type nearHit struct {
		name, pkg, file string
		line            int
	}
	var hits []nearHit
	seen := map[string]bool{}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.Receiver != "" || fn.IsTest || seen[fn.Name] {
				continue
			}
			if nearName(lastSeg, fn.Name) {
				seen[fn.Name] = true
				hits = append(hits, nearHit{fn.Name, fn.Pkg, fn.File, fn.Line})
			}
		}
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].name < hits[j].name })
	if len(hits) > 3 {
		hits = hits[:3]
	}
	if len(hits) > 0 {
		var parts []string
		for _, h := range hits {
			parts = append(parts, fmt.Sprintf("%q (in %s, %s:%d)", h.name, h.pkg, p.LinkPath(h.file), h.line))
		}
		msg += " Did you mean " + strings.Join(parts, " or ") + "?"
	}
	return nil, msg
}

// pkgByDir returns the indexed package at dir ("" = root), or nil.
func (p *Project) pkgByDir(dir string) *Package {
	for _, pkg := range p.Packages {
		if pkg.Dir == dir {
			return pkg
		}
	}
	return nil
}

// pkgDeclares reports whether pkg already declares a top-level
// name that a new plain function would collide with: a plain
// function, a type, or a package-level variable/constant.
func (p *Project) pkgDeclares(pkg *Package, name string) bool {
	for _, fn := range pkg.Funcs {
		if fn.Receiver == "" && fn.Name == name {
			return true
		}
	}
	for _, t := range pkg.Types {
		if t.Name == name {
			return true
		}
	}
	for _, v := range pkg.Vars {
		if v.Name == name {
			return true
		}
	}
	return false
}

// --- the generator -----------------------------------------------------------

// delegateGen assembles the wrapper text for one source/target
// pair. It renders types from the source declaration's AST with
// every reference re-qualified for the target file: the source
// package's own names gain the form's qualifier, and types from
// packages the source imported keep working because the plan
// adds those imports (under the source's own qualifier, or the
// target file's existing one).
type delegateGen struct {
	p       *Project
	src     *Func
	srcDecl *ast.FuncDecl
	srcFset *token.FileSet
	srcAST  *ast.File
	tgtAST  *ast.File
	tgtFset *token.FileSet
	tgtPkg  *Package
	cross   bool
	wrapper string

	targetRel string
	targetSrc []byte
	srcPath   string // import path of the source package (cross only)

	typeParams map[string]bool
	srcTop     map[string]bool   // top-level names of the source package
	srcImports map[string]string // qualifier -> path, in the source file
	tgtImports map[string]string // path -> effective qualifier, in the target file

	needed map[string]string // path -> qualifier the generated code uses (per form)
}

// altQual is the retry form's qualifier for the source package.
func (g *delegateGen) altQual() string { return g.src.Pkg + "pkg" }

// formQual returns the source package's qualifier for a form:
// primary uses the declared name — or the target file's existing
// qualifier when it already imports the source — and the retry
// uses the deterministic alias.
func (g *delegateGen) formQual(alt bool) string {
	if alt {
		return g.altQual()
	}
	if existing, ok := g.tgtImports[g.srcPath]; ok && g.cross {
		return existing
	}
	return g.src.Pkg
}

// altAvailable reports whether the retry form exists: cross-
// package only, and the alias must not already name a different
// import in the target file.
func (g *delegateGen) altAvailable() bool {
	if !g.cross {
		return false
	}
	for path, eff := range g.tgtImports {
		if eff == g.altQual() && path != g.srcPath {
			return false
		}
	}
	return true
}

// forms assembles the primary wrapper file and, when available,
// the retry form's. A self-check failure on the primary is a
// refusal (our bug, stated as one); on the retry it just drops
// the retry.
func (g *delegateGen) forms() (primary *EditFile, alt *EditFile, refusal string) {
	ef, added, why := g.formFile(false)
	if why != "" {
		return nil, nil, why
	}
	if bad := selfCheckDelegate(ef.After, g.expect(g.formQual(false), added)); bad != "" {
		return nil, nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}
	primary = ef
	if g.altAvailable() {
		aef, aadded, awhy := g.formFile(true)
		if awhy == "" && selfCheckDelegate(aef.After, g.expect(g.formQual(true), aadded)) == "" {
			alt = aef
		}
	}
	return primary, alt, ""
}

// expect builds the self-check expectations for one form.
func (g *delegateGen) expect(qual string, added []string) delegateExpect {
	return delegateExpect{
		wrapper:   g.wrapper,
		srcName:   g.src.Name,
		srcQual:   map[bool]string{true: qual, false: ""}[g.cross],
		results:   g.resultCount(),
		args:      len(g.wrapperParams(qual)),
		variadic:  g.isVariadic(),
		addedImps: added,
	}
}

// formFile assembles the target file's After bytes for one form:
// the imports the wrapper needs, plus the wrapper appended at the
// end of the file. Returns the file and the import paths this
// form newly declares.
func (g *delegateGen) formFile(alt bool) (*EditFile, []string, string) {
	g.needed = map[string]string{}
	qual := g.formQual(alt)
	text := g.wrapperText(qual)

	// Import edits. The source package's own spec (or its
	// replacement, on the retry when the file already imports the
	// path under the plain name) plus one spec per signature-
	// driven package the render collected.
	var edits []textEdit
	var added []string
	var specs []string
	if g.cross {
		existing, already := g.tgtImports[g.srcPath]
		switch {
		case alt && already && existing != qual:
			// The retry adds an aliased import of the source
			// ALONGSIDE the file's existing one (Go permits the
			// same path under two names) — re-pointing the
			// existing spec would break any other code in the
			// file that uses the plain qualifier.
			specs = append(specs, delegateImportSpec(g.srcPath, qual, g.src.Pkg))
			added = append(added, g.srcPath)
		case !already:
			specs = append(specs, delegateImportSpec(g.srcPath, qual, g.src.Pkg))
			added = append(added, g.srcPath)
		}
		// When the file already imports the source and this is
		// the primary form, the existing qualifier is used as-is.
	}
	paths := make([]string, 0, len(g.needed))
	for path := range g.needed {
		if path != g.srcPath {
			paths = append(paths, path)
		}
	}
	sort.Strings(paths)
	for _, path := range paths {
		if _, already := g.tgtImports[path]; already {
			continue
		}
		specs = append(specs, delegateImportSpec(path, g.needed[path], ""))
		added = append(added, path)
	}
	if len(specs) > 0 {
		edits = append(edits, insertImportSpecs(g.tgtAST, g.tgtFset, g.targetSrc, specs)...)
	}

	// Append the wrapper at the end of the file.
	appendText := "\n" + text
	if len(g.targetSrc) == 0 || g.targetSrc[len(g.targetSrc)-1] != '\n' {
		appendText = "\n" + appendText
	}
	edits = append(edits, textEdit{start: len(g.targetSrc), end: len(g.targetSrc), text: appendText})

	after := applyTextEdits(g.targetSrc, edits)
	ef := finishDelegateFile(g.targetRel, g.targetSrc, after)
	return &ef, added, ""
}

// delegateImportSpec renders one import spec for a form: a bare
// quoted path when the qualifier is the package's own name (or
// unknown — the path came from the source file's plain import,
// so its qualifier IS the declared name), an explicit name
// otherwise.
func delegateImportSpec(path, qual, declared string) string {
	if declared != "" && qual == declared {
		return strconv.Quote(path)
	}
	if declared == "" {
		// A signature-driven import copied from the source file:
		// if the source used an alias, qual carries it and the
		// spec must repeat it; a plain source import means qual
		// is the declared name. The caller can't distinguish
		// here, so the spec naming it explicitly is always
		// correct either way.
		return qual + " " + strconv.Quote(path)
	}
	return qual + " " + strconv.Quote(path)
}

// wrapperText renders the delegate function for one form.
func (g *delegateGen) wrapperText(qual string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "// %s calls %s — assembled by gollemer from %s:%d.\n", g.wrapper, g.src.Name, g.targetRelSource(), g.src.Line)
	fmt.Fprintf(&b, "// It passes its arguments through and returns %s's results unchanged.\n", g.src.Name)
	b.WriteString("func " + g.wrapper)
	if tp := g.renderTypeParams(qual); tp != "" {
		b.WriteString(tp)
	}
	params := g.wrapperParams(qual)
	var ptxt []string
	for _, prm := range params {
		ptxt = append(ptxt, prm.name+" "+prm.typ)
	}
	b.WriteString("(" + strings.Join(ptxt, ", ") + ")")
	b.WriteString(g.renderFieldList(g.srcDecl.Type.Results, qual, true))
	b.WriteString(" {\n")
	call := g.src.Name
	if g.cross {
		call = qual + "." + call
	}
	if len(g.typeParamNames()) > 0 {
		call += "[" + strings.Join(g.typeParamNames(), ", ") + "]"
	}
	var args []string
	for i, prm := range params {
		a := prm.name
		if prm.variadic && i == len(params)-1 {
			a += "..."
		}
		args = append(args, a)
	}
	call += "(" + strings.Join(args, ", ") + ")"
	if g.resultCount() > 0 {
		b.WriteString("\treturn " + call + "\n")
	} else {
		b.WriteString("\t" + call + "\n")
	}
	b.WriteString("}\n")
	return b.String()
}

// targetRelSource is the source file's project-relative path, for
// the wrapper's doc comment.
func (g *delegateGen) targetRelSource() string { return g.src.File }

func (g *delegateGen) typeParamNames() []string {
	var out []string
	if g.srcDecl.Type.TypeParams == nil {
		return nil
	}
	for _, f := range g.srcDecl.Type.TypeParams.List {
		for _, n := range f.Names {
			out = append(out, n.Name)
		}
	}
	return out
}

func (g *delegateGen) renderTypeParams(qual string) string {
	if g.srcDecl.Type.TypeParams == nil || len(g.srcDecl.Type.TypeParams.List) == 0 {
		return ""
	}
	var fields []string
	for _, f := range g.srcDecl.Type.TypeParams.List {
		var names []string
		for _, n := range f.Names {
			names = append(names, n.Name)
		}
		fields = append(fields, strings.Join(names, ", ")+" "+g.renderType(f.Type, qual))
	}
	return "[" + strings.Join(fields, ", ") + "]"
}

type delegateParam struct {
	name     string
	typ      string
	variadic bool
}

// wrapperParams renders the wrapper's parameters from the source
// declaration: declared names kept (a blank "_" or a missing name
// becomes a deterministic argN placeholder), types re-qualified
// for the target file.
func (g *delegateGen) wrapperParams(qual string) []delegateParam {
	fl := g.srcDecl.Type.Params
	if fl == nil {
		return nil
	}
	used := map[string]bool{}
	for _, f := range fl.List {
		for _, n := range f.Names {
			used[n.Name] = true
		}
	}
	var out []delegateParam
	idx := 0
	for fi, f := range fl.List {
		typ := g.renderType(f.Type, qual)
		_, isEllipsis := f.Type.(*ast.Ellipsis)
		count := len(f.Names)
		if count == 0 {
			count = 1
		}
		for i := 0; i < count; i++ {
			name := ""
			if len(f.Names) > 0 {
				name = f.Names[i].Name
			}
			if name == "" || name == "_" {
				name = fmt.Sprintf("arg%d", idx)
				for used[name] {
					name += "_"
				}
				used[name] = true
			}
			out = append(out, delegateParam{
				name:     name,
				typ:      typ,
				variadic: isEllipsis && fi == len(fl.List)-1,
			})
			idx++
		}
	}
	return out
}

func (g *delegateGen) isVariadic() bool {
	fl := g.srcDecl.Type.Params
	if fl == nil || len(fl.List) == 0 {
		return false
	}
	_, ok := fl.List[len(fl.List)-1].Type.(*ast.Ellipsis)
	return ok
}

// resultCount counts the source's result values.
func (g *delegateGen) resultCount() int {
	fl := g.srcDecl.Type.Results
	if fl == nil {
		return 0
	}
	n := 0
	for _, f := range fl.List {
		if len(f.Names) == 0 {
			n++
		} else {
			n += len(f.Names)
		}
	}
	return n
}

// renderFieldList renders a parameter/result field list for a
// signature (with the leading space a result list needs). When
// leadSpace is set and the list is a single unnamed field, it
// renders bare (" Item"); otherwise parenthesized.
func (g *delegateGen) renderFieldList(fl *ast.FieldList, qual string, leadSpace bool) string {
	if fl == nil || len(fl.List) == 0 {
		return ""
	}
	var fields []string
	for _, f := range fl.List {
		fields = append(fields, g.renderField(f, qual))
	}
	if len(fl.List) == 1 && len(fl.List[0].Names) == 0 {
		if leadSpace {
			return " " + fields[0]
		}
		return fields[0]
	}
	return "(" + strings.Join(fields, ", ") + ")"
}

// renderField renders one parameter/result field with its type
// re-qualified.
func (g *delegateGen) renderField(f *ast.Field, qual string) string {
	typ := g.renderType(f.Type, qual)
	if len(f.Names) == 0 {
		return typ
	}
	var names []string
	for _, n := range f.Names {
		names = append(names, n.Name)
	}
	return strings.Join(names, ", ") + " " + typ
}

// renderFuncTail renders "(params) results" for a func type
// appearing inside a signature.
func (g *delegateGen) renderFuncTail(ft *ast.FuncType, qual string) string {
	params := "()"
	if ft.Params != nil && len(ft.Params.List) > 0 {
		var fields []string
		for _, f := range ft.Params.List {
			fields = append(fields, g.renderField(f, qual))
		}
		params = "(" + strings.Join(fields, ", ") + ")"
	}
	return params + g.renderFieldList(ft.Results, qual, true)
}

// renderType renders one type expression from the source
// declaration, re-qualified for the target file and registering
// every package it leans on in g.needed.
func (g *delegateGen) renderType(e ast.Expr, qual string) string {
	switch t := e.(type) {
	case *ast.Ident:
		if g.typeParams[t.Name] {
			return t.Name
		}
		if g.cross && g.srcTop[t.Name] {
			return qual + "." + t.Name
		}
		return t.Name
	case *ast.SelectorExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			if path, isImport := g.srcImports[id.Name]; isImport {
				return g.qualFor(path, qual) + "." + t.Sel.Name
			}
		}
		return exprText(g.srcFset, e)
	case *ast.StarExpr:
		return "*" + g.renderType(t.X, qual)
	case *ast.ArrayType:
		ln := ""
		if t.Len != nil {
			ln = g.renderType(t.Len, qual)
		}
		return "[" + ln + "]" + g.renderType(t.Elt, qual)
	case *ast.Ellipsis:
		return "..." + g.renderType(t.Elt, qual)
	case *ast.MapType:
		return "map[" + g.renderType(t.Key, qual) + "]" + g.renderType(t.Value, qual)
	case *ast.ChanType:
		switch t.Dir {
		case ast.SEND:
			return "chan<- " + g.renderType(t.Value, qual)
		case ast.RECV:
			return "<-chan " + g.renderType(t.Value, qual)
		}
		return "chan " + g.renderType(t.Value, qual)
	case *ast.FuncType:
		return "func" + g.renderFuncTail(t, qual)
	case *ast.InterfaceType:
		if t.Methods == nil || len(t.Methods.List) == 0 {
			return "interface{}"
		}
		var parts []string
		for _, m := range t.Methods.List {
			if len(m.Names) == 0 {
				parts = append(parts, g.renderType(m.Type, qual))
				continue
			}
			if ft, ok := m.Type.(*ast.FuncType); ok {
				parts = append(parts, m.Names[0].Name+g.renderFuncTail(ft, qual))
			}
		}
		return "interface{ " + strings.Join(parts, "; ") + " }"
	case *ast.StructType:
		if t.Fields == nil || len(t.Fields.List) == 0 {
			return "struct{}"
		}
		var parts []string
		for _, f := range t.Fields.List {
			s := ""
			if len(f.Names) > 0 {
				var names []string
				for _, n := range f.Names {
					names = append(names, n.Name)
				}
				s = strings.Join(names, ", ") + " "
			}
			s += g.renderType(f.Type, qual)
			if f.Tag != nil {
				s += " " + f.Tag.Value
			}
			parts = append(parts, s)
		}
		return "struct{ " + strings.Join(parts, "; ") + " }"
	case *ast.ParenExpr:
		return "(" + g.renderType(t.X, qual) + ")"
	case *ast.IndexExpr:
		return g.renderType(t.X, qual) + "[" + g.renderType(t.Index, qual) + "]"
	case *ast.IndexListExpr:
		var idx []string
		for _, ix := range t.Indices {
			idx = append(idx, g.renderType(ix, qual))
		}
		return g.renderType(t.X, qual) + "[" + strings.Join(idx, ", ") + "]"
	case *ast.BasicLit:
		return t.Value
	}
	return exprText(g.srcFset, e)
}

// qualFor returns the qualifier the generated code uses for an
// imported package path, registering the import as needed when
// the target file doesn't already import it.
func (g *delegateGen) qualFor(path, srcQual string) string {
	if existing, ok := g.tgtImports[path]; ok {
		return existing
	}
	desired := srcQual
	if path != g.srcPath {
		// The qualifier the source file itself used for this
		// path (an alias, or the declared name it wrote).
		for q, pth := range g.srcImports {
			if pth == path {
				desired = q
				break
			}
		}
	}
	g.needed[path] = desired
	return desired
}

// finishDelegateFile is finishFile plus an unconditional format
// pass: assembled bytes are formatted before any write, so a
// delegate never lands unformatted even in a file whose original
// wasn't gofmt-clean.
func finishDelegateFile(rel string, before, after []byte) EditFile {
	ef := finishFile(rel, before, after)
	if f, err := format.Source(ef.After); err == nil && !bytes.Equal(f, ef.After) {
		ef.After = f
		ef.Diff = unifiedDiff(rel, before, f)
	}
	return ef
}

// --- the test skeleton -------------------------------------------------------

// delegateStub builds the plan's second file: a table-driven test
// skeleton for the wrapper, appended to the package's natural
// _test.go (the target's own base name) or created there. It is
// labeled a skeleton in the plan and is never verification
// evidence. Returns nil (with a note) when the test already
// exists.
func (p *Project) delegateStub(tgtPkg *Package, targetRel, wrapper string) (*EditFile, string) {
	dir := tgtPkg.Dir
	base := targetRel
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	testRel := strings.TrimSuffix(base, ".go") + "_test.go"
	if dir != "" {
		testRel = dir + "/" + testRel
	}
	stubName := "Test" + titleFirst(wrapper)
	stub := delegateStubText(stubName, wrapper)
	note := fmt.Sprintf("Plus a test skeleton (%s in %s) — a table with TODO rows for you to fill in. It checks nothing until you add cases; it is not part of how the edit is verified.", stubName, testRel)

	existing, _, testAST, _ := p.readProjectFile(testRel)
	if testAST != nil {
		for _, d := range testAST.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == stubName {
				return nil, fmt.Sprintf("No test skeleton added: %s already has a %s.", testRel, stubName)
			}
		}
		// Append to the existing test file, adding the testing
		// import when it lacks one.
		_, tfset, _, _ := p.readProjectFile(testRel)
		var edits []textEdit
		if !hasImportPath(testAST, "testing") {
			edits = append(edits, insertImportSpecs(testAST, tfset, existing, []string{strconv.Quote("testing")})...)
		}
		appendText := "\n" + stub
		if len(existing) == 0 || existing[len(existing)-1] != '\n' {
			appendText = "\n" + appendText
		}
		edits = append(edits, textEdit{start: len(existing), end: len(existing), text: appendText})
		after := applyTextEdits(existing, edits)
		ef := finishDelegateFile(testRel, existing, after)
		return &ef, note
	}
	// Create it: the target package's own clause, the testing
	// import, the skeleton.
	after := []byte("package " + tgtPkg.Name + "\n\nimport \"testing\"\n\n" + stub)
	ef := finishDelegateFile(testRel, nil, after)
	return &ef, note
}

// delegateStubText renders the skeleton body.
func delegateStubText(stubName, wrapper string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "// %s is a skeleton assembled by gollemer next to the %s delegate.\n", stubName, wrapper)
	b.WriteString("// It has no real cases yet — fill in the TODO rows with your inputs\n")
	b.WriteString("// and the results you expect. Until then it checks nothing.\n")
	fmt.Fprintf(&b, "func %s(t *testing.T) {\n", stubName)
	b.WriteString("\tcases := []struct {\n")
	b.WriteString("\t\tname string\n")
	fmt.Fprintf(&b, "\t\t// TODO: one field per input of %s, plus the result you want.\n", wrapper)
	b.WriteString("\t}{\n")
	b.WriteString("\t\t// TODO: add a case, e.g. {name: \"first case\"},\n")
	b.WriteString("\t}\n")
	b.WriteString("\tfor _, tc := range cases {\n")
	b.WriteString("\t\tt.Run(tc.name, func(t *testing.T) {\n")
	fmt.Fprintf(&b, "\t\t\t// TODO: call %s with tc's fields and compare with what you expect.\n", wrapper)
	b.WriteString("\t\t\t_ = tc\n")
	b.WriteString("\t\t})\n")
	b.WriteString("\t}\n")
	b.WriteString("}\n")
	return b.String()
}

// --- the self-check ----------------------------------------------------------

// delegateExpect is what selfCheckDelegate holds the generated
// wrapper to: its name, the source it must call, the exact call
// shape, and the imports the plan added on its behalf.
type delegateExpect struct {
	wrapper   string
	srcName   string
	srcQual   string // "" for a same-package source
	results   int
	args      int
	variadic  bool
	addedImps []string
}

// selfCheckDelegate AST-walks the GENERATED function in the
// assembled file and asserts the delegate's invariants: every
// error-typed result of the source is propagated (the body is
// the single direct call, returned when the source returns),
// the generated code performs no map writes and no nil-risky
// initialization, and every import the body references is
// declared while every import the plan added is used. It
// returns "" when the generated code is exactly the delegate it
// claims to be, else a description of our bug.
func selfCheckDelegate(after []byte, exp delegateExpect) string {
	fset, fileAST, err := parseGoSrc("delegate.go", after)
	if err != nil || fileAST == nil {
		return "the assembled file doesn't parse"
	}
	var decl *ast.FuncDecl
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == exp.wrapper && fd.Recv == nil {
			decl = fd
			break
		}
	}
	if decl == nil || decl.Body == nil {
		return fmt.Sprintf("the assembled file has no function named %s", exp.wrapper)
	}
	if len(decl.Body.List) != 1 {
		return fmt.Sprintf("%s's body isn't the single call it should be (%d statements)", exp.wrapper, len(decl.Body.List))
	}
	var call *ast.CallExpr
	switch stmt := decl.Body.List[0].(type) {
	case *ast.ReturnStmt:
		if exp.results == 0 {
			return fmt.Sprintf("%s returns a value but its source returns nothing", exp.wrapper)
		}
		if len(stmt.Results) != 1 {
			return fmt.Sprintf("%s doesn't return the call directly", exp.wrapper)
		}
		c, ok := stmt.Results[0].(*ast.CallExpr)
		if !ok {
			return fmt.Sprintf("%s's return isn't the source call", exp.wrapper)
		}
		call = c
	case *ast.ExprStmt:
		if exp.results != 0 {
			return fmt.Sprintf("%s drops its source's %d result(s) instead of returning them — an error result would be lost", exp.wrapper, exp.results)
		}
		c, ok := stmt.X.(*ast.CallExpr)
		if !ok {
			return fmt.Sprintf("%s's body isn't the source call", exp.wrapper)
		}
		call = c
	default:
		return fmt.Sprintf("%s's body is a %T, not the single source call", exp.wrapper, decl.Body.List[0])
	}
	// The call must be exactly the source: bare for same-package,
	// qualified by the form's qualifier across packages. A
	// generic source is instantiated explicitly, so the call
	// target may sit under an index expression — unwrap it.
	callFun := call.Fun
	switch idx := callFun.(type) {
	case *ast.IndexExpr:
		callFun = idx.X
	case *ast.IndexListExpr:
		callFun = idx.X
	}
	switch fun := callFun.(type) {
	case *ast.Ident:
		if exp.srcQual != "" {
			return fmt.Sprintf("%s calls %s unqualified, but the source is in another package", exp.wrapper, fun.Name)
		}
		if fun.Name != exp.srcName {
			return fmt.Sprintf("%s calls %s, not %s", exp.wrapper, fun.Name, exp.srcName)
		}
	case *ast.SelectorExpr:
		id, ok := fun.X.(*ast.Ident)
		if !ok || id.Name != exp.srcQual || fun.Sel.Name != exp.srcName {
			return fmt.Sprintf("%s calls the wrong function (not %s.%s)", exp.wrapper, exp.srcQual, exp.srcName)
		}
	default:
		return fmt.Sprintf("%s's call target has an unexpected shape", exp.wrapper)
	}
	if len(call.Args) != exp.args {
		return fmt.Sprintf("%s passes %d arguments but %s takes %d", exp.wrapper, len(call.Args), exp.srcName, exp.args)
	}
	if exp.variadic && !call.Ellipsis.IsValid() {
		return fmt.Sprintf("%s doesn't forward its variadic argument with ...", exp.wrapper)
	}
	if !exp.variadic && call.Ellipsis.IsValid() {
		return fmt.Sprintf("%s forwards with ... but %s isn't variadic", exp.wrapper, exp.srcName)
	}
	// No map writes, no nil-risky initialization, anywhere in
	// the generated function: an index assignment, a map
	// composite literal, or a make() of a map would all be the
	// assembler doing something a delegate never does.
	bad := ""
	ast.Inspect(decl, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			for _, lhs := range x.Lhs {
				if _, isIndex := lhs.(*ast.IndexExpr); isIndex {
					bad = fmt.Sprintf("%s assigns through an index — a delegate never writes a map or slice", exp.wrapper)
					return false
				}
			}
		case *ast.CompositeLit:
			if strings.HasPrefix(exprText(fset, x.Type), "map[") {
				bad = fmt.Sprintf("%s builds a map literal — a delegate initializes nothing", exp.wrapper)
				return false
			}
		case *ast.CallExpr:
			if id, ok := x.Fun.(*ast.Ident); ok && id.Name == "make" && len(x.Args) > 0 {
				if strings.HasPrefix(exprText(fset, x.Args[0]), "map[") {
					bad = fmt.Sprintf("%s makes a map — a delegate initializes nothing", exp.wrapper)
					return false
				}
			}
		}
		return bad == ""
	})
	if bad != "" {
		return bad
	}
	// Imports: every qualifier the generated signature and body
	// lean on must be declared in the file, and every import the
	// plan added must be leaned on.
	declared := map[string]bool{}
	pathByQual := map[string]string{}
	for _, is := range fileAST.Imports {
		_, eff := importSpecOf(is)
		declared[eff] = true
		path, _ := strconv.Unquote(is.Path.Value)
		pathByQual[eff] = path
	}
	used := map[string]bool{}
	ast.Inspect(decl, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if id, ok := sel.X.(*ast.Ident); ok {
				used[id.Name] = true
			}
		}
		return true
	})
	for q := range used {
		if !declared[q] {
			return fmt.Sprintf("%s uses %s but the file doesn't import it", exp.wrapper, q)
		}
	}
	for _, path := range exp.addedImps {
		found := false
		for q, pth := range pathByQual {
			if pth == path && used[q] {
				found = true
			}
		}
		if !found {
			return fmt.Sprintf("the plan imports %s but %s never uses it", path, exp.wrapper)
		}
	}
	return ""
}

// verifyDelegateCall re-reads the target file after an edit and
// confirms the wrapper exists and its body calls the source —
// the verifyCaseExists pattern, over the re-analyzed project.
func verifyDelegateCall(root, rel, wrapper, srcName string) bool {
	_, fset, fileAST, refusal := (&Project{Root: root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil {
		return false
	}
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != wrapper || fd.Body == nil {
			continue
		}
		found := false
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				txt := exprText(fset, call.Fun)
				// A generic call renders with its instantiation
				// (store.First[T]); compare on the bare target.
				if i := strings.Index(txt, "["); i >= 0 {
					txt = txt[:i]
				}
				if txt == srcName || strings.HasSuffix(txt, "."+srcName) {
					found = true
				}
			}
			return !found
		})
		return found
	}
	return false
}
