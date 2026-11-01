package analyze

// Cross-file use: the assembled shape beside the delegate.
// "In web.go add a function that uses LoadUser from store.go"
// writes a NEW function into web.go whose body is a first use
// of a symbol the project already has — assembled from that
// symbol's REAL signature (a function) or REAL definition (a
// type), read from the analyzer index, never from the sentence.
//
// How it differs from its neighbors, deliberately:
//
//   - The delegate's new function IS the source, re-homed: its
//     signature mirrors the source's and its body is the one
//     call returned. A use's new function has its OWN empty
//     signature; the call happens inside its body, with
//     starting values the assembler derives for the source's
//     real parameters, and the source's real results are
//     consumed there — an error result is checked with a guard
//     and an early return, value results are bound and kept.
//   - Instantiation puts a literal of an existing type inside
//     an EXISTING function. A use of a type opens a NEW
//     function in a NAMED file with that value.
//   - Clone copies a function. A use writes a caller, not a
//     copy.
//
// The honesty rules are the settled ones: a starting value is
// written only when the scaffold's zero table can derive it —
// anything underivable refuses the WHOLE plan, naming the
// parameter. The generated body is walked by selfCheckUses
// before a plan exists (an AST check over the generated code
// only). Imports come from the index, files are formatted
// before any write, and the plan names ONE alternate form —
// the source imported under a deterministic alias — for the
// chat's single bounded retry in the import/qualification
// failure class. No model emits patch text anywhere here.

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"
	"strconv"
	"strings"
)

// planUse dispatches a use request to the function or type
// form once the sentence's symbol is resolved.
func (p *Project) planUse(req EditRequest) (*EditPlan, string) {
	rel := strings.TrimSpace(req.UseFile)
	if rel == "" {
		return nil, "that edit needs a file to add the function to."
	}
	if strings.HasSuffix(rel, "_test.go") {
		return nil, fmt.Sprintf("%s is a test file — new functions don't go there. (The test scaffold this shape writes is the only test-file change it makes.)", p.LinkPath(rel))
	}
	if why, blocked := p.fileBlocked(rel); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(rel), why)
	}
	tgtPkg := p.pkgOfFile(rel)
	if tgtPkg == nil {
		return nil, fmt.Sprintf("%s isn't part of the project I analyzed — it may sit in a folder the analyzer skips, or under its own go.mod. I can only write into files the index knows.", p.LinkPath(rel))
	}
	tgtBytes, tgtFset, tgtAST, refusal := p.readProjectFile(rel)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(tgtBytes) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(rel))
	}

	sym, refusal := p.resolveUseSymbol(req)
	if refusal != "" {
		return nil, refusal
	}
	if sym.fn != nil {
		return p.planUseFunc(req, rel, tgtPkg, tgtBytes, tgtFset, tgtAST, sym.fn)
	}
	return p.planUseType(req, rel, tgtPkg, tgtBytes, tgtFset, tgtAST, sym.ty, sym.tyPkg)
}

// useSymbol is one resolved reference: exactly one of fn / ty
// is set.
type useSymbol struct {
	fn    *Func
	ty    *Type
	tyPkg *Package
}

// resolveUseSymbol settles the typed symbol to one plain
// function or one type in the index — or an honest refusal.
// The "from <file>" clause, when present, restricts candidates
// to symbols declared in that file. A name that is both a
// function and a type (in different packages) refuses with
// both listed and the "type" designator named as the way to
// force the type reading.
func (p *Project) resolveUseSymbol(req EditRequest) (useSymbol, string) {
	typed := strings.TrimSpace(req.UseSymbol)
	if typed == "" {
		return useSymbol{}, "that edit names no function or type to use."
	}
	from := strings.TrimSpace(req.UseFromFile)
	if from != "" {
		found := false
		for _, pkg := range p.Packages {
			for _, f := range pkg.Files {
				if f == from || baseOfPath(f) == baseOfPath(from) {
					found = true
				}
			}
		}
		if !found {
			return useSymbol{}, fmt.Sprintf("I can't find a file named %q in this project, so I can't look for %s in it.", from, typed)
		}
	}
	inFrom := func(file string) bool {
		if from == "" {
			return true
		}
		return file == from || baseOfPath(file) == baseOfPath(from)
	}

	// Functions: the delegate resolver's discipline — plain
	// functions only, exact case preferred over the index's
	// case-insensitive forgiveness.
	var plain, methods []*Func
	if !req.UseType {
		lastSeg := typed
		if i := strings.LastIndex(typed, "."); i >= 0 {
			lastSeg = typed[i+1:]
		}
		var exact []*Func
		for _, fn := range p.findProjectFuncs(typed) {
			if !inFrom(fn.File) {
				continue
			}
			if fn.Receiver == "" {
				plain = append(plain, fn)
				if fn.Name == lastSeg {
					exact = append(exact, fn)
				}
			} else {
				methods = append(methods, fn)
			}
		}
		if len(exact) > 0 {
			plain = exact
		}
	}

	// Types: every kind, with the instantiate resolver's
	// exact-case-then-folded discipline and package qualifier.
	var types []*Type
	var typePkgs []*Package
	{
		qual, base := "", typed
		if i := strings.LastIndex(typed, "."); i > 0 {
			qual, base = typed[:i], typed[i+1:]
		}
		type cand struct {
			ty  *Type
			pkg *Package
		}
		var exactT, foldedT []cand
		for _, pk := range p.Packages {
			if qual != "" && !strings.EqualFold(pk.Name, qual) {
				continue
			}
			for _, t := range pk.Types {
				if !inFrom(t.File) {
					continue
				}
				switch {
				case t.Name == base:
					exactT = append(exactT, cand{t, pk})
				case strings.EqualFold(t.Name, base):
					foldedT = append(foldedT, cand{t, pk})
				}
			}
		}
		pool := exactT
		if len(pool) == 0 {
			pool = foldedT
		}
		for _, c := range pool {
			types = append(types, c.ty)
			typePkgs = append(typePkgs, c.pkg)
		}
	}

	if from != "" && len(plain) == 0 && len(types) == 0 && len(methods) == 0 {
		// The symbol may well exist — somewhere else. Say so,
		// naming where, instead of a bare not-found.
		anyFn := p.findProjectFuncs(typed)
		for _, fn := range anyFn {
			if fn.Receiver == "" {
				return useSymbol{}, fmt.Sprintf("%s doesn't declare %s — %s lives in %s instead. Say \"from %s\", or drop the from-clause.", from, typed, fn.Name, p.LinkPath(fn.File), fn.File)
			}
		}
		for _, pk := range p.Packages {
			for _, t := range pk.Types {
				if strings.EqualFold(t.Name, typed) || strings.EqualFold(pk.Name+"."+t.Name, typed) {
					return useSymbol{}, fmt.Sprintf("%s doesn't declare %s — the type %s lives in %s instead. Say \"from %s\", or drop the from-clause.", from, typed, t.Name, p.LinkPath(t.File), t.File)
				}
			}
		}
	}

	switch {
	case len(plain) == 1 && len(types) == 0:
		return useSymbol{fn: plain[0]}, ""
	case len(plain) > 1:
		return useSymbol{}, p.funcCandidates("name its package", typed, plain)
	case len(plain) == 1 && len(types) > 0:
		var b strings.Builder
		fmt.Fprintf(&b, "%q is both a function and a type in this project:\n  - the function %s (%s:%d)\n", typed, plain[0].Display(), p.LinkPath(plain[0].File), plain[0].Line)
		for i, t := range types {
			fmt.Fprintf(&b, "  - the %s %s in package %s (%s:%d)\n", t.Kind, t.Name, typePkgs[i].Name, p.LinkPath(t.File), t.Line)
		}
		b.WriteString("Say which one — write \"the type\" before the name (\"that uses the type ")
		b.WriteString(typed)
		b.WriteString("\") for the type, or name the function's package for the function.")
		return useSymbol{}, b.String()
	case len(plain) == 0 && len(types) == 1:
		return useSymbol{ty: types[0], tyPkg: typePkgs[0]}, ""
	case len(types) > 1:
		var where []string
		for i, t := range types {
			where = append(where, fmt.Sprintf("%s in package %s (%s)", t.Name, typePkgs[i].Name, p.LinkPath(t.File)))
		}
		return useSymbol{}, fmt.Sprintf("the name %s is declared in more than one package: %s. Say which one, like %q.", typed, strings.Join(where, "; "), typePkgs[0].Name+"."+types[0].Name)
	case len(methods) > 0:
		m := methods[0]
		return useSymbol{}, fmt.Sprintf("%s is a method on %s (%s:%d) — this shape uses plain functions and types only; a method needs a receiver value to call through, and that shape isn't built yet.", typed, strings.Trim(m.Receiver, "*"), p.LinkPath(m.File), m.Line)
	}

	// Nothing matched. Near names get named, the way the
	// delegate's not-found does.
	msg := fmt.Sprintf("I can't find a function or type called %q in this project — I can only build from code that exists.", typed)
	lastSeg := typed
	if i := strings.LastIndex(typed, "."); i >= 0 {
		lastSeg = typed[i+1:]
	}
	type nearHit struct {
		name, pkg, file string
		line            int
	}
	var hits []nearHit
	seen := map[string]bool{}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.Receiver != "" || fn.IsTest || seen["f:"+fn.Name] {
				continue
			}
			if nearName(lastSeg, fn.Name) {
				seen["f:"+fn.Name] = true
				hits = append(hits, nearHit{fn.Name, fn.Pkg, fn.File, fn.Line})
			}
		}
		for _, t := range pkg.Types {
			if seen["t:"+t.Name] {
				continue
			}
			if nearName(lastSeg, t.Name) {
				seen["t:"+t.Name] = true
				hits = append(hits, nearHit{t.Name, t.Pkg, t.File, t.Line})
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
	return useSymbol{}, msg
}

// baseOfPath is the final element of a slash path.
func baseOfPath(p string) string {
	if i := strings.LastIndex(p, "/"); i >= 0 {
		return p[i+1:]
	}
	return p
}

// useFuncName settles the new function's name: the sentence's,
// or "Use" + the symbol's name — with the delegate's collision
// discipline (a user-given collision refuses; a derived one
// falls back to + "From" + the symbol's package, then refuses
// with the sentence that would work).
func (p *Project) useFuncName(tgtPkg *Package, rel, symbolName, symbolPkg, userName, sentenceHint string) (string, string) {
	name := strings.TrimSpace(userName)
	userNamed := name != ""
	if name == "" {
		name = "Use" + symbolName
	}
	if !token.IsIdentifier(name) || token.Lookup(name).IsKeyword() {
		return "", fmt.Sprintf("%q isn't a name Go allows for a function.", name)
	}
	if p.pkgDeclares(tgtPkg, name) {
		if userNamed {
			return "", fmt.Sprintf("%s already exists in package %s (%s) — pick a different name for the new function.", name, tgtPkg.Name, p.LinkPath(rel))
		}
		fallback := name + "From" + titleFirst(symbolPkg)
		if p.pkgDeclares(tgtPkg, fallback) || !token.IsIdentifier(fallback) {
			return "", fmt.Sprintf("package %s already declares both %s and %s — give the new function a name yourself, like %s.", tgtPkg.Name, name, fallback, sentenceHint)
		}
		name = fallback
	}
	return name, ""
}

// --- the generator -----------------------------------------------------------

// usesGen assembles one use plan over the delegate's rendering
// context: the same type rendering, qualification, and import
// machinery, with the body built by the shape's own builder
// (a call with derived starting values, or a value of a type).
// pkgName is the referenced symbol's package name — the
// qualifier source (the gen's src is the referenced FUNCTION
// for the call form and nil for the type form, so the name is
// carried explicitly).
type usesGen struct {
	g       *delegateGen
	pkgName string

	// bodyFor renders the new function's full text for one
	// form (its qualifier), plus the self-check expectations
	// for that form. A refusal is a derivation refusal — the
	// same in every form, probed once before forms() runs.
	bodyFor func(qual string) (string, usesExpect, string)

	// literalNotes collects the instantiate literal's honesty
	// notes during a body build (the probe call's are kept).
	literalNotes []string
}

// altQual is the retry form's qualifier for the source package.
func (u *usesGen) altQual() string { return u.pkgName + "pkg" }

// formQual mirrors the delegate's rule: the retry uses the
// alias; the primary uses the target file's existing qualifier
// when it already imports the source, else the declared name.
func (u *usesGen) formQual(alt bool) string {
	if alt {
		return u.altQual()
	}
	if existing, ok := u.g.tgtImports[u.g.srcPath]; ok && u.g.cross {
		return existing
	}
	return u.pkgName
}

// altAvailable mirrors the delegate's: cross-package only, and
// the alias must not already name a different import.
func (u *usesGen) altAvailable() bool {
	if !u.g.cross {
		return false
	}
	for path, eff := range u.g.tgtImports {
		if eff == u.altQual() && path != u.g.srcPath {
			return false
		}
	}
	return true
}

// forms assembles the primary file and, when available, the
// retry form's. A derivation refusal surfaces as-is (the probe
// normally catches it first); a self-check failure is our bug,
// stated as one, for the primary — and just drops the retry.
func (u *usesGen) forms() (primary *EditFile, alt *EditFile, refusal string) {
	ef, ref := u.formFile(false)
	if ref != "" {
		return nil, nil, ref
	}
	primary = ef
	if u.altAvailable() {
		aef, aref := u.formFile(true)
		if aref == "" {
			alt = aef
		}
	}
	return primary, alt, ""
}

// formFile assembles the target file's After bytes for one
// form: the imports the body needs, plus the new function
// appended at the end of the file. Mirrors the delegate's
// formFile import discipline exactly (the retry adds the
// aliased import ALONGSIDE an existing plain one, never
// re-pointing it).
func (u *usesGen) formFile(alt bool) (*EditFile, string) {
	g := u.g
	g.needed = map[string]string{}
	qual := u.formQual(alt)
	text, exp, refusal := u.bodyFor(qual)
	if refusal != "" {
		return nil, refusal
	}
	var edits []textEdit
	var added []string
	var specs []string
	if g.cross {
		existing, already := g.tgtImports[g.srcPath]
		switch {
		case alt && already && existing != qual:
			specs = append(specs, delegateImportSpec(g.srcPath, qual, u.pkgName))
			added = append(added, g.srcPath)
		case !already:
			specs = append(specs, delegateImportSpec(g.srcPath, qual, u.pkgName))
			added = append(added, g.srcPath)
		}
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
	appendText := "\n" + text
	if len(g.targetSrc) == 0 || g.targetSrc[len(g.targetSrc)-1] != '\n' {
		appendText = "\n" + appendText
	}
	// A package-clause-only target anchors the import block at
	// EOF too — the same offset as this append. No local merge
	// is needed: applyTextEdits composes same-offset inserts
	// imports-first (see composeSameOffsetInserts).
	edits = append(edits, textEdit{start: len(g.targetSrc), end: len(g.targetSrc), text: appendText})
	after := applyTextEdits(g.targetSrc, edits)
	ef := finishDelegateFile(g.targetRel, g.targetSrc, after)
	exp.addedImps = added
	if bad := selfCheckUses(ef.After, exp); bad != "" {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}
	return &ef, ""
}

// useArg is one derived call argument: its rendered value and
// the label ("id int") the plan's note names it by.
type useArg struct {
	label string
	text  string
}

// deriveArgs derives the call's argument values for one form:
// one honest starting value per parameter of the source, from
// the scaffold's zero table ( + the instantiate literal for
// struct parameters). The variadic tail is omitted — zero
// values for a variadic parameter is no values at all. Any
// parameter with no derivable value refuses the whole use.
func (u *usesGen) deriveArgs(qual string) (args []useArg, variadicName string, refusal string) {
	g := u.g
	decl := g.srcDecl
	if decl.Type.Params == nil {
		return nil, "", ""
	}
	b := &scaffoldBuilder{g: g}
	b.qual = qual
	ctx := scaffoldCtx{imports: g.srcImports, pkgDir: g.src.PkgDir}
	idx := 0
	fields := decl.Type.Params.List
	for fi, f := range fields {
		_, isEllipsis := f.Type.(*ast.Ellipsis)
		rendered := g.renderType(f.Type, qual)
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
			}
			label := name + " " + rendered
			if isEllipsis && fi == len(fields)-1 {
				variadicName = name
				idx++
				continue
			}
			text, ref := u.deriveArgValue(b, ctx, f.Type, qual, label)
			if ref != "" {
				return nil, "", ref
			}
			args = append(args, useArg{label: label, text: text})
			idx++
		}
	}
	return args, variadicName, ""
}

// deriveArgValue derives ONE argument's starting value. A
// parameter whose type is a project struct gets the
// instantiate literal (field-named, from the type's real
// definition); everything else goes through the scaffold's
// zero table; a generic type refuses by name.
func (u *usesGen) deriveArgValue(b *scaffoldBuilder, ctx scaffoldCtx, expr ast.Expr, qual, label string) (string, string) {
	g := u.g
	if ty, generic := projectTypeOfExpr(b, expr); generic {
		return "", fmt.Sprintf("I can't assemble that use: %s's parameter %s is the generic type %s — its type arguments are a choice, and I can't make it for you. I won't guess values, so the whole plan is off.", g.src.Name, label, ty.Name)
	} else if ty != nil {
		if spec, specFset := u.g.p.instantiateTypeSpec(ty); spec != nil {
			if st, ok := spec.Type.(*ast.StructType); ok {
				typeRef := g.renderType(expr, qual)
				lit, notes := u.g.p.instantiateStructLiteral(ty, st, specFset, typeRef, ty.PkgDir != g.tgtPkg.Dir, g.tgtPkg.Name)
				u.literalNotes = append(u.literalNotes, notes...)
				return lit, ""
			}
		}
	}
	v, ok, reason := b.zeroValue(expr, ctx, 0)
	if !ok {
		return "", fmt.Sprintf("I can't assemble that use: %s's parameter %s has no starting value I can derive — %s — and I won't invent one, so the whole plan is off. (A delegate passes values through instead: \"add a function in %s that calls %s\".)", g.src.Name, label, reason, g.targetRel, g.src.Name)
	}
	return v, ""
}

// projectTypeOfExpr resolves a parameter's type expression to
// a project type when it names one (bare, qualified, or a
// generic instantiation of one), reporting whether that type
// is generic. Anything else returns (nil, false).
func projectTypeOfExpr(b *scaffoldBuilder, expr ast.Expr) (*Type, bool) {
	switch e := expr.(type) {
	case *ast.IndexExpr:
		if ty, _ := projectTypeOfExpr(b, e.X); ty != nil {
			return ty, true
		}
		return nil, false
	case *ast.IndexListExpr:
		if ty, _ := projectTypeOfExpr(b, e.X); ty != nil {
			return ty, true
		}
		return nil, false
	case *ast.Ident:
		ty := b.findProjectType(b.g.src.PkgDir, e.Name)
		if ty == nil {
			return nil, false
		}
		return ty, typeIsGeneric(b, ty)
	case *ast.SelectorExpr:
		id, ok := e.X.(*ast.Ident)
		if !ok {
			return nil, false
		}
		path, isImport := b.g.srcImports[id.Name]
		if !isImport {
			return nil, false
		}
		dir, isProj := b.projectDirForPath(path)
		if !isProj {
			return nil, false
		}
		ty := b.findProjectType(dir, e.Sel.Name)
		if ty == nil {
			return nil, false
		}
		return ty, typeIsGeneric(b, ty)
	}
	return nil, false
}

// typeIsGeneric reports whether a project type's declaration
// carries type parameters.
func typeIsGeneric(b *scaffoldBuilder, ty *Type) bool {
	spec, _ := b.typeSpec(ty)
	return spec != nil && spec.TypeParams != nil && len(spec.TypeParams.List) > 0
}

// useResults is the result plan for a use body: the variables
// the call's results are bound to, in the signature's own
// order (lhs), the value results among them (valVars), and
// the error variable when the source returns an error.
type useResults struct {
	lhs      []string
	valVars  []string
	errVar   string
	hasError bool
}

// analyzeUseResults reads the source's result list once (the
// names are the same in every form): a declared result name is
// kept; an unnamed value result is named from its type; the
// error result is "err" unless the declaration names it. More
// than one error-typed result refuses — the checks for that
// can't be written honestly by assembly.
func analyzeUseResults(g *delegateGen, decl *ast.FuncDecl) (useResults, string) {
	var out useResults
	if decl.Type.Results == nil {
		return out, ""
	}
	type slot struct {
		declared, rendered string
		isErr              bool
	}
	var slots []slot
	for _, f := range decl.Type.Results.List {
		rendered := g.renderType(f.Type, g.formQual(false))
		n := len(f.Names)
		if n == 0 {
			n = 1
		}
		for i := 0; i < n; i++ {
			d := ""
			if len(f.Names) > 0 {
				d = f.Names[i].Name
			}
			slots = append(slots, slot{d, rendered, rendered == "error"})
		}
	}
	errs := 0
	for _, s := range slots {
		if s.isErr {
			errs++
		}
	}
	if errs > 1 {
		return out, fmt.Sprintf("%s returns more than one error — I can't write honest checks for that by assembly, so this shape won't build the use.", g.src.Name)
	}
	used := map[string]bool{}
	uniq := func(base string) string {
		if base == "" {
			base = "value"
		}
		name := base
		for i := 2; used[name]; i++ {
			name = fmt.Sprintf("%s%d", base, i)
		}
		used[name] = true
		return name
	}
	for _, s := range slots {
		if s.isErr {
			out.hasError = true
			nm := s.declared
			if nm == "" || nm == "_" {
				nm = "err"
			}
			out.errVar = uniq(nm)
			out.lhs = append(out.lhs, out.errVar)
			continue
		}
		nm := s.declared
		if nm == "" || nm == "_" {
			nm = deriveUseVarName(s.rendered)
		}
		if qual := g.formQual(false); qual != "" && nm == qual {
			nm = "value"
		}
		nm = uniq(nm)
		out.valVars = append(out.valVars, nm)
		out.lhs = append(out.lhs, nm)
	}
	return out, ""
}

// useLowerName lowers a type name into a variable name: the
// usual first-letter lowering, except an all-caps name (IDs,
// URL) lowercases whole — iDs is nobody's variable name.
func useLowerName(name string) string {
	if len(name) > 1 {
		allCaps := true
		for _, r := range name {
			if r < 'A' || r > 'Z' {
				allCaps = false
				break
			}
		}
		if allCaps {
			return strings.ToLower(name)
		}
	}
	return lowerFirstName(name)
}

// deriveUseVarName names an unnamed result from its rendered
// type: the type's base name, lowered — or "value" when the
// base is a builtin or a keyword.
func deriveUseVarName(rendered string) string {
	last, cur := "", ""
	flush := func() {
		if cur != "" {
			last, cur = cur, ""
		}
	}
	for _, r := range rendered {
		if r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			cur += string(r)
		} else {
			flush()
		}
	}
	flush()
	switch last {
	case "", "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
		"float32", "float64", "complex64", "complex128",
		"string", "bool", "byte", "rune", "error", "any",
		"true", "false", "nil":
		return "value"
	}
	name := useLowerName(last)
	if name == "time" {
		// A time.Time result binds to t, not to a variable
		// named after the package it came through.
		return "t"
	}
	if token.Lookup(name).IsKeyword() {
		return "value"
	}
	return name
}

// --- the function form -------------------------------------------------------

// planUseFunc assembles a new function whose body calls src.
func (p *Project) planUseFunc(req EditRequest, rel string, tgtPkg *Package, tgtBytes []byte, tgtFset *token.FileSet, tgtAST *ast.File, src *Func) (*EditPlan, string) {
	cross := src.PkgDir != tgtPkg.Dir
	if cross {
		if srcPkg := p.pkgByDir(src.PkgDir); srcPkg != nil && srcPkg.Name == "main" {
			return nil, fmt.Sprintf("%s lives in package main — a program, and programs can't be imported, so nothing in another package can call it.", src.Display())
		}
		if !src.Exported {
			return nil, fmt.Sprintf("%s isn't exported (its name starts lowercase), and %s is a different package — Go only lets other packages call exported names. Export it (rename it to %s), or put the new function in %s instead.", src.Name, tgtPkg.Name, titleFirst(src.Name), src.Pkg)
		}
	}
	if src.IsMain {
		return nil, "main is the program's entry point — it can't be called, so there's nothing to use."
	}
	if src.IsInit {
		return nil, "init runs by itself before main — it can't be called, so there's nothing to use."
	}

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
	if srcDecl.Type.TypeParams != nil && len(srcDecl.Type.TypeParams.List) > 0 {
		return nil, fmt.Sprintf("%s is generic — using it would mean picking its type arguments, and I can't choose those for you. A delegate mirrors them instead: \"add a function in %s that calls %s\".", src.Name, rel, req.UseSymbol)
	}

	hint := fmt.Sprintf("\"add a function My%s in %s that uses %s\"", src.Name, rel, req.UseSymbol)
	newName, refusal := p.useFuncName(tgtPkg, rel, src.Name, src.Pkg, req.UseName, hint)
	if refusal != "" {
		return nil, refusal
	}
	g, refusal := p.newDelegateGen(src, srcDecl, srcFset, srcAST, tgtAST, tgtFset, tgtPkg, cross, newName, rel, tgtBytes)
	if refusal != "" {
		return nil, refusal
	}
	g.needed = map[string]string{}
	res, refusal := analyzeUseResults(g, srcDecl)
	if refusal != "" {
		return nil, refusal
	}

	u := &usesGen{g: g, pkgName: src.Pkg}
	var probeArgs []useArg
	var probeVariadic string
	u.bodyFor = func(qual string) (string, usesExpect, string) {
		text, exp, args, variadic, ref := u.funcBody(qual, src, res)
		if ref == "" {
			probeArgs, probeVariadic = args, variadic
		}
		return text, exp, ref
	}
	// The derivation probe: its refusal is form-independent and
	// user-facing, so it surfaces bare, before any form is built.
	if _, _, ref := u.bodyFor(u.formQual(false)); ref != "" {
		return nil, ref
	}
	literalNotes := append([]string{}, u.literalNotes...)
	u.literalNotes = nil

	primary, alt, refusal := u.forms()
	if refusal != "" {
		return nil, refusal
	}
	plan, refusal := p.finishUsePlan(u, req, rel, tgtPkg, primary, alt, newName, literalNotes,
		fmt.Sprintf("Add %s to %s: a new function whose body calls %s (%s:%d) with starting values for its parameters and handles what it returns.",
			newName, p.LinkPath(rel), src.Display(), p.LinkPath(src.File), src.Line),
		fmt.Sprintf("built from %s:%d — %s", p.LinkPath(src.File), src.Line, src.Sig),
		fmt.Sprintf("%s now exists in %s and calls %s", newName, p.LinkPath(rel), src.Name),
		func(after *Project) bool { return verifyDelegateCall(after.Root, rel, newName, src.Name) })
	if refusal != "" {
		return nil, refusal
	}
	// Use-specific notes, ahead of the scaffold/alt notes
	// finishUsePlan appends.
	var notes []string
	if len(probeArgs) > 0 {
		var pairs []string
		for _, a := range probeArgs {
			pairs = append(pairs, a.label)
		}
		notes = append(notes, fmt.Sprintf("The call starts %s off with placeholder values for its parameters (%s) — they're the types' zero values, there to be replaced, and the \"_ =\" lines keep the results used until you write the real work.", src.Name, strings.Join(pairs, ", ")))
	} else if probeVariadic == "" {
		notes = append(notes, fmt.Sprintf("The call passes %s no arguments — it takes none — and the \"_ =\" lines keep its results used until you write the real work.", src.Name))
	}
	if probeVariadic != "" {
		notes = append(notes, fmt.Sprintf("%s is variadic in %s — the call passes it none of those; add the ones you need in the body.", probeVariadic, src.Name))
	}
	if fm := src.AnalyzeFailures(p); fm.CanFail || len(fm.Panics) > 0 {
		notes = append(notes, fmt.Sprintf("One thing to know: %s — the new function checks %s's error and returns early; what should happen on failure is yours to write in the TODO.", src.FailurePhrase(p), src.Name))
	}
	plan.Notes = append(notes, plan.Notes...)
	return plan, ""
}

// funcBody renders the new function for one form: the doc
// comment, the empty signature, the call with derived starting
// values, the error guard its signature demands, and the
// keep-used lines for its value results.
func (u *usesGen) funcBody(qual string, src *Func, res useResults) (string, usesExpect, []useArg, string, string) {
	args, variadic, refusal := u.deriveArgs(qual)
	if refusal != "" {
		return "", usesExpect{}, nil, "", refusal
	}
	callName := src.Name
	if u.g.cross {
		callName = qual + "." + src.Name
	}
	var argTexts []string
	for _, a := range args {
		argTexts = append(argTexts, a.text)
	}
	call := callName + "(" + strings.Join(argTexts, ", ") + ")"
	var lines []string
	if len(args) > 0 {
		lines = append(lines, "// Starting values for "+src.Name+"'s parameters — replace them with real ones.")
	}
	switch {
	case res.hasError && len(res.valVars) > 0:
		lines = append(lines, strings.Join(res.lhs, ", ")+" := "+call)
		lines = append(lines, "if "+res.errVar+" != nil {")
		lines = append(lines, "\t// TODO: decide what to do when "+src.Name+" fails")
		lines = append(lines, "\treturn")
		lines = append(lines, "}")
	case res.hasError:
		lines = append(lines, "if "+res.errVar+" := "+call+"; "+res.errVar+" != nil {")
		lines = append(lines, "\t// TODO: decide what to do when "+src.Name+" fails")
		lines = append(lines, "\treturn")
		lines = append(lines, "}")
	case len(res.valVars) > 0:
		lines = append(lines, strings.Join(res.valVars, ", ")+" := "+call)
	default:
		lines = append(lines, call)
	}
	for _, v := range res.valVars {
		lines = append(lines, "_ = "+v+" // TODO: use "+v+" here")
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s calls %s — assembled by gollemer from %s:%d.\n", u.g.wrapper, src.Name, src.File, src.Line)
	b.WriteString("// It starts the call with placeholder values and handles the results; the real work is yours to write.\n")
	b.WriteString("func " + u.g.wrapper + "() {\n")
	for _, ln := range lines {
		b.WriteString("\t" + ln + "\n")
	}
	b.WriteString("}\n")
	exp := usesExpect{
		name:     u.g.wrapper,
		srcName:  src.Name,
		args:     len(argTexts),
		hasError: res.hasError,
		errVar:   res.errVar,
		valVars:  res.valVars,
	}
	if u.g.cross {
		exp.srcQual = qual
	}
	return b.String(), exp, args, variadic, ""
}

// --- the type form -----------------------------------------------------------

// planUseType assembles a new function whose body opens with
// a value of an existing type, from the type's real definition.
func (p *Project) planUseType(req EditRequest, rel string, tgtPkg *Package, tgtBytes []byte, tgtFset *token.FileSet, tgtAST *ast.File, ty *Type, tyPkg *Package) (*EditPlan, string) {
	cross := ty.PkgDir != tgtPkg.Dir
	if cross && !ty.Exported {
		return nil, fmt.Sprintf("%s isn't exported (its name starts lowercase), and %s is a different package — Go only lets other packages name exported types. Export it (rename it to %s), or put the new function in %s instead.", ty.Name, tgtPkg.Name, titleFirst(ty.Name), ty.Pkg)
	}
	spec, specFset := p.instantiateTypeSpec(ty)
	if spec == nil {
		return nil, fmt.Sprintf("I couldn't re-read %s's declaration in %s, so I can't build its value honestly.", ty.Name, p.LinkPath(ty.File))
	}
	if spec.TypeParams != nil && len(spec.TypeParams.List) > 0 {
		return nil, fmt.Sprintf("%s is a generic type — it needs its type arguments in brackets (like %s[int]) before anything can be made of it, and I can't pick those for you.", ty.Name, ty.Name)
	}
	if _, isIface := spec.Type.(*ast.InterfaceType); isIface {
		return nil, fmt.Sprintf("%s is an interface — you can't make one directly, only a concrete type that satisfies it. Name one of your own structs (or another concrete type) instead.", ty.Name)
	}

	hint := fmt.Sprintf("\"add a function My%s in %s that uses the type %s\"", ty.Name, rel, req.UseSymbol)
	newName, refusal := p.useFuncName(tgtPkg, rel, ty.Name, ty.Pkg, req.UseName, hint)
	if refusal != "" {
		return nil, refusal
	}

	// The rendering context, with the type's own file as the
	// signature source (the gen's src/srcDecl stay nil: nothing
	// here renders a function signature).
	_, tyFset, tyAST, refusal := p.readProjectFile(ty.File)
	if refusal != "" {
		return nil, refusal
	}
	g := &delegateGen{
		p:          p,
		srcFset:    tyFset,
		srcAST:     tyAST,
		tgtAST:     tgtAST,
		tgtFset:    tgtFset,
		tgtPkg:     tgtPkg,
		cross:      cross,
		wrapper:    newName,
		targetRel:  rel,
		targetSrc:  tgtBytes,
		typeParams: map[string]bool{},
		srcTop:     map[string]bool{},
		srcImports: map[string]string{},
		tgtImports: map[string]string{},
	}
	if cross {
		g.srcPath = p.Module
		if ty.PkgDir != "" {
			g.srcPath = p.Module + "/" + ty.PkgDir
		}
		if g.srcPath == "" || strings.HasPrefix(g.srcPath, "/") {
			return nil, "I couldn't work out the type's package import path (the project has no go.mod module line I can read), so I can't write the import honestly."
		}
	}
	for _, fn := range tyPkg.Funcs {
		if fn.Receiver == "" {
			g.srcTop[fn.Name] = true
		}
	}
	for _, t := range tyPkg.Types {
		g.srcTop[t.Name] = true
	}
	for _, v := range tyPkg.Vars {
		g.srcTop[v.Name] = true
	}
	for _, is := range tyAST.Imports {
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
			return nil, fmt.Sprintf("%s imports %s with a %s import, which gives the package no name I can refer to the type through — give that import its plain name first, then ask again.", p.LinkPath(rel), g.srcPath, map[string]string{".": "dot", "_": "blank"}[eff])
		}
	}

	u := &usesGen{g: g, pkgName: ty.Pkg}
	varName := useLowerName(ty.Name)
	if !token.IsIdentifier(varName) || token.Lookup(varName).IsKeyword() {
		varName += "Value"
	}
	u.bodyFor = func(qual string) (string, usesExpect, string) {
		return u.typeBody(qual, ty, spec, specFset, varName)
	}
	if _, _, ref := u.bodyFor(u.formQual(false)); ref != "" {
		return nil, ref
	}
	literalNotes := append([]string{}, u.literalNotes...)
	u.literalNotes = nil

	primary, alt, refusal := u.forms()
	if refusal != "" {
		return nil, refusal
	}
	plan, refusal := p.finishUsePlan(u, req, rel, tgtPkg, primary, alt, newName, literalNotes,
		fmt.Sprintf("Add %s to %s: a new function whose body makes a %s, from its definition in %s:%d.",
			newName, p.LinkPath(rel), ty.Name, p.LinkPath(ty.File), ty.Line),
		fmt.Sprintf("built from %s:%d — type %s (%s)", p.LinkPath(ty.File), ty.Line, ty.Name, ty.Kind),
		fmt.Sprintf("%s now exists in %s and makes a %s", newName, p.LinkPath(rel), ty.Name),
		func(after *Project) bool { return verifyUsesType(after.Root, rel, newName, varName) })
	if refusal != "" {
		return nil, refusal
	}
	plan.Notes = append([]string{
		fmt.Sprintf("The %s value is a starting point — its fields are the zero values from %s's own definition, and the \"_ =\" line keeps it used until you write the real work.", ty.Name, ty.Name),
	}, plan.Notes...)
	return plan, ""
}

// typeBody renders the type form's new function for one form:
// a value of the type (the instantiate literal for a struct,
// the honest zero form for the other literal-able kinds), and
// the keep-used line.
func (u *usesGen) typeBody(qual string, ty *Type, spec *ast.TypeSpec, specFset *token.FileSet, varName string) (string, usesExpect, string) {
	g := u.g
	typeRef := g.renderType(&ast.Ident{Name: ty.Name}, qual)
	var stmt string
	switch e := spec.Type.(type) {
	case *ast.StructType:
		lit, notes := u.g.p.instantiateStructLiteral(ty, e, specFset, typeRef, ty.PkgDir != g.tgtPkg.Dir, g.tgtPkg.Name)
		u.literalNotes = append(u.literalNotes, notes...)
		stmt = varName + " := " + lit
	case *ast.ArrayType, *ast.MapType:
		stmt = varName + " := " + typeRef + "{}"
	default:
		b := &scaffoldBuilder{g: g}
		zero, ok, reason := b.namedZero(ty, typeRef, 0)
		if !ok {
			return "", usesExpect{}, fmt.Sprintf("I can't assemble that use: a starting value of %s can't be derived honestly — %s — and I won't invent one, so the whole plan is off.", ty.Name, reason)
		}
		if zero == "nil" {
			stmt = "var " + varName + " " + typeRef
		} else {
			stmt = varName + " := " + zero
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s makes a %s — assembled by gollemer from %s:%d.\n", g.wrapper, ty.Name, ty.File, ty.Line)
	b.WriteString("// The value is a starting point; the real work is yours to write.\n")
	b.WriteString("func " + g.wrapper + "() {\n")
	b.WriteString("\t" + stmt + "\n")
	b.WriteString("\t_ = " + varName + " // TODO: use " + varName + " here\n")
	b.WriteString("}\n")
	exp := usesExpect{
		name:    g.wrapper,
		typeRef: typeRef,
		valVar:  varName,
	}
	return b.String(), exp, ""
}

// finishUsePlan composes a use plan's files (primary + the
// courtesy scaffold), its alternate set, and the shared notes
// (scaffold, then the retry's), around the caller's summary,
// provenance, and verify.
func (p *Project) finishUsePlan(u *usesGen, req EditRequest, rel string, tgtPkg *Package, primary, alt *EditFile, newName string, literalNotes []string, summary, provenance, verifyDesc string, verify func(after *Project) bool) (*EditPlan, string) {
	// The courtesy scaffold for the new function, parsed back
	// out of the assembled primary — the same bargain as the
	// delegate's: present, or honestly omitted with the reason
	// in the notes, and never part of verification.
	var stubEF *EditFile
	var stubNote string
	if reason := entryReason(newName); reason != "" {
		stubNote = fmt.Sprintf("No test scaffold added: %s — a test can't call it like an ordinary function, so there's nothing honest for a scaffold to aim at.", reason)
	} else if fset, fileAST, err := parseGoSrc(rel, primary.After); err == nil && fileAST != nil {
		var decl *ast.FuncDecl
		for _, d := range fileAST.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == newName && fd.Recv == nil {
				decl = fd
				break
			}
		}
		if decl != nil {
			// The scaffold target is the NEW function, parsed
			// back out of the assembled bytes — it isn't in the
			// index yet, so its index entry is fabricated from
			// the plan's own facts, the way its decl was.
			fn := &Func{Name: newName, Pkg: tgtPkg.Name, PkgDir: tgtPkg.Dir, File: rel, Line: insertedLine(primary.After, "func "+newName)}
			tgt := scaffoldTarget{
				fn: fn, decl: decl, fset: fset, fileAST: fileAST,
				name: newName, display: newName,
			}
			pres := scaffoldPresentation{
				display:    newName,
				verifyNoun: "this edit",
				intro: "%s is a test scaffold assembled by gollemer for " + newName + ", the new\n" +
					"function this edit adds. The cases call it as it stands — with the\n" +
					"starting-point body this edit writes — and nothing more.",
				accompany: "It scaffolds " + newName + " as this edit leaves it — the real work inside it is still yours to write.",
			}
			stubEF, stubNote = p.scaffoldGenFor(tgt).planScaffold(tgtPkg, rel, pres)
		}
	}
	if stubEF == nil && stubNote == "" {
		stubNote = "No test scaffold added: the assembled function couldn't be re-read to aim one."
	}

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
		altDesc = fmt.Sprintf("the retry form: the same function, with %s imported under the alias %s (%s) instead of the plain name — used only if the plain form fails to build over the import or the qualification.", u.pkgName, u.altQual(), u.altQual())
	}

	notes := append([]string{}, literalNotes...)
	if stubNote != "" {
		notes = append(notes, stubNote)
	}
	if altDesc != "" {
		notes = append(notes, "If the plain form doesn't build: "+altDesc)
	} else if u.g.cross {
		notes = append(notes, "No alternate form is available for this one — if it doesn't build, I'll restore your file and show you the compiler's words.")
	} else {
		notes = append(notes, "Same package, so there's no import to retry with a different form — if it doesn't build, I'll restore your file and show you the compiler's words.")
	}

	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(primary.After, "func "+newName)),
		Files:      files,
		Provenance: provenance,
		Notes:      notes,
		AltFiles:   altFiles,
		AltDesc:    altDesc,
		verify:     verify,
	}, ""
}

// --- the self-check ----------------------------------------------------------

// usesExpect is what selfCheckUses holds a generated body to:
// the new function's name, and then the form's facts — for a
// call: the source it must call, with how many arguments, and
// how its results must be handled; for a value: the type ref
// it must build and the variable it must bind.
type usesExpect struct {
	name    string
	srcName string
	srcQual string // "" for a same-package source
	args    int

	hasError bool
	errVar   string
	valVars  []string

	typeRef string // type form only
	valVar  string // type form only

	addedImps []string
}

// selfCheckUses AST-walks the GENERATED function in the
// assembled file — never the user's existing code — and
// asserts the use's invariants: the new function takes no
// parameters and returns nothing; a call form calls the
// source exactly once with the derived argument count, binds
// every value result and keeps it used, and checks an error
// result with a guard and an early return; a value form
// builds the type it names; and every import the body leans
// on is declared while every import the plan added is used.
// It returns "" when the generated code is exactly the use it
// claims to be, else a description of our bug.
func selfCheckUses(after []byte, exp usesExpect) string {
	fset, fileAST, err := parseGoSrc("use.go", after)
	if err != nil || fileAST == nil {
		return "the assembled file doesn't parse"
	}
	var decl *ast.FuncDecl
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == exp.name && fd.Recv == nil {
			decl = fd
			break
		}
	}
	if decl == nil || decl.Body == nil {
		return fmt.Sprintf("the assembled file has no function named %s", exp.name)
	}
	if decl.Type.Params != nil {
		for _, f := range decl.Type.Params.List {
			if len(f.Names) > 0 {
				return fmt.Sprintf("%s takes parameters — a use function starts empty and derives its call's values itself", exp.name)
			}
		}
	}
	if decl.Type.Results != nil && len(decl.Type.Results.List) > 0 {
		return fmt.Sprintf("%s returns results — a use function handles its results inside its own body", exp.name)
	}

	if exp.typeRef != "" {
		found := false
		for _, stmt := range decl.Body.List {
			switch s := stmt.(type) {
			case *ast.AssignStmt:
				if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
					if id, ok := s.Lhs[0].(*ast.Ident); ok && id.Name == exp.valVar {
						switch rhs := s.Rhs[0].(type) {
						case *ast.CompositeLit:
							if exprText(fset, rhs.Type) == exp.typeRef {
								found = true
							}
						case *ast.CallExpr:
							// A defined basic kind's value is a
							// conversion: T(zero).
							if exprText(fset, rhs.Fun) == exp.typeRef {
								found = true
							}
						}
					}
				}
			case *ast.DeclStmt:
				if gd, ok := s.Decl.(*ast.GenDecl); ok && gd.Tok == token.VAR {
					for _, spec := range gd.Specs {
						if vs, ok := spec.(*ast.ValueSpec); ok && vs.Type != nil && exprText(fset, vs.Type) == exp.typeRef {
							for _, n := range vs.Names {
								if n.Name == exp.valVar {
									found = true
								}
							}
						}
					}
				}
			}
		}
		if !found {
			return fmt.Sprintf("%s doesn't build a %s as its first statement", exp.name, exp.typeRef)
		}
	} else {
		// The call: exactly one, to the source, with the
		// derived argument count.
		var call *ast.CallExpr
		calls := 0
		ast.Inspect(decl.Body, func(n ast.Node) bool {
			ce, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			switch fun := ce.Fun.(type) {
			case *ast.Ident:
				if exp.srcQual == "" && fun.Name == exp.srcName {
					calls++
					call = ce
				}
			case *ast.SelectorExpr:
				if id, ok := fun.X.(*ast.Ident); ok && id.Name == exp.srcQual && fun.Sel.Name == exp.srcName {
					calls++
					call = ce
				}
			}
			return true
		})
		if calls != 1 {
			return fmt.Sprintf("%s calls %s %d times, not once", exp.name, exp.srcName, calls)
		}
		if len(call.Args) != exp.args {
			return fmt.Sprintf("%s passes %d arguments but the plan derived %d", exp.name, len(call.Args), exp.args)
		}
		if exp.hasError && !useErrorGuarded(decl, call, exp.errVar) {
			return fmt.Sprintf("%s doesn't check %s's error with a guard and an early return", exp.name, exp.srcName)
		}
	}
	// Every value result (or the type form's value) is kept
	// used with its "_ =" line.
	keepUsed := exp.valVars
	if exp.typeRef != "" {
		keepUsed = []string{exp.valVar}
	}
	for _, v := range keepUsed {
		found := false
		for _, stmt := range decl.Body.List {
			if as, ok := stmt.(*ast.AssignStmt); ok && len(as.Lhs) == 1 && len(as.Rhs) == 1 {
				if id, ok := as.Lhs[0].(*ast.Ident); ok && id.Name == "_" {
					if rv, ok := as.Rhs[0].(*ast.Ident); ok && rv.Name == v {
						found = true
					}
				}
			}
		}
		if !found {
			return fmt.Sprintf("%s binds %s but never keeps it used", exp.name, v)
		}
	}
	// No map writes or map initialization in the generated
	// body — a use writes neither (the delegate's guard).
	bad := ""
	ast.Inspect(decl, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			for _, lhs := range x.Lhs {
				if _, isIndex := lhs.(*ast.IndexExpr); isIndex {
					bad = fmt.Sprintf("%s assigns through an index — a use never writes a map or slice", exp.name)
					return false
				}
			}
		case *ast.CompositeLit:
			if strings.HasPrefix(exprText(fset, x.Type), "map[") {
				bad = fmt.Sprintf("%s builds a map literal — its only composite is the value it was asked to make", exp.name)
				return false
			}
		case *ast.CallExpr:
			if id, ok := x.Fun.(*ast.Ident); ok && id.Name == "make" && len(x.Args) > 0 {
				if strings.HasPrefix(exprText(fset, x.Args[0]), "map[") {
					bad = fmt.Sprintf("%s makes a map — a use initializes no map", exp.name)
					return false
				}
			}
		}
		return bad == ""
	})
	if bad != "" {
		return bad
	}
	// Imports: every qualifier the generated function leans on
	// is declared, and every import the plan added is leaned on.
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
			return fmt.Sprintf("%s uses %s but the file doesn't import it", exp.name, q)
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
			return fmt.Sprintf("the plan imports %s but %s never uses it", path, exp.name)
		}
	}
	return ""
}

// useErrorGuarded reports whether the generated body checks
// the call's error the way the signature demands: the error
// variable is bound from the call (as an if-init or a plain
// assignment), a guard tests it against nil, and the guard's
// body returns.
func useErrorGuarded(decl *ast.FuncDecl, call *ast.CallExpr, errVar string) bool {
	binds := func(stmt ast.Stmt) bool {
		as, ok := stmt.(*ast.AssignStmt)
		if !ok {
			return false
		}
		// A multi-result call is ONE right-hand side feeding
		// every left-hand name; the error variable is among
		// them.
		for _, rhs := range as.Rhs {
			if rhs == call {
				for _, lhs := range as.Lhs {
					if id, ok := lhs.(*ast.Ident); ok && id.Name == errVar {
						return true
					}
				}
			}
		}
		return false
	}
	guarded := func(ifStmt *ast.IfStmt) bool {
		cond, ok := ifStmt.Cond.(*ast.BinaryExpr)
		if !ok || cond.Op != token.NEQ {
			return false
		}
		x, okX := cond.X.(*ast.Ident)
		y, okY := cond.Y.(*ast.Ident)
		if !okX || !okY || x.Name != errVar || y.Name != "nil" {
			return false
		}
		if ifStmt.Body == nil {
			return false
		}
		for _, s := range ifStmt.Body.List {
			if _, ok := s.(*ast.ReturnStmt); ok {
				return true
			}
		}
		return false
	}
	for _, stmt := range decl.Body.List {
		if ifStmt, ok := stmt.(*ast.IfStmt); ok {
			if ifStmt.Init != nil && binds(ifStmt.Init) && guarded(ifStmt) {
				return true
			}
			continue
		}
		if binds(stmt) {
			// The guard must be the NEXT statement.
			seen := false
			for _, s2 := range decl.Body.List {
				if s2 == stmt {
					seen = true
					continue
				}
				if !seen {
					continue
				}
				if ifStmt, ok := s2.(*ast.IfStmt); ok && ifStmt.Init == nil {
					return guarded(ifStmt)
				}
				return false
			}
		}
	}
	return false
}

// verifyUsesType re-reads the target file after an edit and
// confirms the new function exists and opens with a value of
// the type: a composite literal bound to the variable, or a
// var declaration of the variable.
func verifyUsesType(root, rel, fnName, varName string) bool {
	_, fset, fileAST, refusal := (&Project{Root: root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil {
		return false
	}
	_ = fset
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != fnName || fd.Body == nil {
			continue
		}
		found := false
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.AssignStmt:
				for i, rhs := range x.Rhs {
					_, isLit := rhs.(*ast.CompositeLit)
					_, isConv := rhs.(*ast.CallExpr)
					if (isLit || isConv) && i < len(x.Lhs) {
						if id, ok := x.Lhs[i].(*ast.Ident); ok && id.Name == varName {
							found = true
						}
					}
				}
			case *ast.ValueSpec:
				for _, nm := range x.Names {
					if nm.Name == varName {
						found = true
					}
				}
			}
			return !found
		})
		return found
	}
	return false
}
