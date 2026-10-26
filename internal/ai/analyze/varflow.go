package analyze

// Variable data flow: where a named value's data comes from and where
// it changes, answered at site granularity (file:line), not just
// function granularity like the field read/write indexes. It powers
// the beginner questions "where does Items get its data", "where does
// Items come from", "where does Items change", "what changes Items".
//
// Three kinds of values are tracked:
//   - package-level variables and constants (project-wide),
//   - struct fields (project-wide, like the value-flow index),
//   - locals, tracked only inside their declaring function — the
//     answer says so when that's the case.
//
// Resolution is heuristic in the same spirit as the rest of the
// package: same-package names, parameter/local types from composite
// literals, and import qualifiers through the package alias cache.
// Generated files are skipped (vendor/ already is, by the walk).

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"regexp"
	"sort"
	"strings"
)

// Var is one package-level var or const declaration.
type Var struct {
	Name     string
	Pkg      string // package name
	PkgDir   string
	Const    bool
	Exported bool
	Type     string // rendered declared (or initializer-inferred) type; "" if unknown
	Init     string // rendered initializer expression; "" if none
	File     string // relative to Root
	Line     int

	initNode ast.Expr       // initializer AST, kept until the BuildGraph sweep
	initFset *token.FileSet // FileSet for initNode positions
}

// key is the event-index key for the variable: "pkg.Name".
func (v *Var) key() string { return v.Pkg + "." + v.Name }

// fieldKey is the event-index key for a struct field: "pkg.Type.Field"
// (the same key shape the value-flow index uses).
func fieldKey(pkg, typ, field string) string { return pkg + "." + typ + "." + field }

// localKey is the event-index key for a local: fn ID + "#" + name.
func localKey(fn *Func, name string) string { return fn.ID + "#" + name }

// varEvent is one site where a tracked value gains data or changes.
type varEvent struct {
	Kind string // assign | append | index | delete | clear | copy | incdec | handoff | returned
	File string // relative to Root
	Line int
	In   string // enclosing function Display(); "" at package level
	Text string // the site in plain words
}

// localVar is one local variable (or parameter) in one function.
type localVar struct {
	Fn       *Func
	Name     string
	TypeText string // declared/inferred type text; "" if unknown
	File     string
	Line     int  // declaration line
	Param    bool // a parameter (or receiver): its data comes from callers
	InitDesc string // plain words for its creation value; "" if none
}

// varIndex holds the name lookups the dataflow pass and answers share.
type varIndex struct {
	varsByKey   map[string]*Var
	varsByName  map[string][]*Var
	typeByKey   map[string]*Type
	typesByName map[string][]*Type
	fieldsByKey map[string]*Type // "pkg.Type.Field" -> owning type
	fieldsByNam map[string][]string
}

func (p *Project) buildVarIndex() {
	if p.vidx != nil {
		return
	}
	ix := &varIndex{
		varsByKey:   map[string]*Var{},
		varsByName:  map[string][]*Var{},
		typeByKey:   map[string]*Type{},
		typesByName: map[string][]*Type{},
		fieldsByKey: map[string]*Type{},
		fieldsByNam: map[string][]string{},
	}
	for _, pkg := range p.Packages {
		for _, v := range pkg.Vars {
			ix.varsByKey[v.key()] = v
			ix.varsByName[v.Name] = append(ix.varsByName[v.Name], v)
		}
		for _, t := range pkg.Types {
			ix.typeByKey[t.Pkg+"."+t.Name] = t
			ix.typesByName[t.Name] = append(ix.typesByName[t.Name], t)
			if t.Kind == "struct" {
				for _, f := range t.Fields {
					k := fieldKey(t.Pkg, t.Name, f)
					ix.fieldsByKey[k] = t
					ix.fieldsByNam[f] = append(ix.fieldsByNam[f], k)
				}
			}
		}
	}
	p.vidx = ix
}

// varIn returns the package-level variable of pkg with the given name
// (exact, then case-insensitive), or nil.
func varIn(pkg *Package, name string) *Var {
	for _, v := range pkg.Vars {
		if v.Name == name {
			return v
		}
	}
	for _, v := range pkg.Vars {
		if strings.EqualFold(v.Name, name) {
			return v
		}
	}
	return nil
}

// lookupType resolves a raw type text ("Config", "*Config",
// "config.Config") written in fromPkg to a project type, or nil.
func (p *Project) lookupType(base string, fromPkg *Package) *Type {
	p.buildVarIndex()
	base = strings.TrimSpace(base)
	for strings.HasPrefix(base, "*") {
		base = strings.TrimSpace(base[1:])
	}
	if i := strings.LastIndex(base, "."); i >= 0 {
		qual, tn := base[:i], base[i+1:]
		if fromPkg != nil {
			if q, ok := fromPkg.aliasCache[qual]; ok {
				if tt := p.vidx.typeByKey[q.Name+"."+tn]; tt != nil {
					return tt
				}
			}
		}
		if tt := p.vidx.typeByKey[qual+"."+tn]; tt != nil {
			return tt
		}
		if ts := p.vidx.typesByName[tn]; len(ts) == 1 {
			return ts[0]
		}
		return nil
	}
	if fromPkg != nil {
		if tt := p.vidx.typeByKey[fromPkg.Name+"."+base]; tt != nil {
			return tt
		}
	}
	if ts := p.vidx.typesByName[base]; len(ts) == 1 {
		return ts[0]
	}
	return nil
}

// generatedMarker is the standard "// Code generated ... DO NOT EDIT."
// line (the regex from the Go code-generation convention).
var generatedMarker = regexp.MustCompile(`^// Code generated .* DO NOT EDIT\.$`)

// isGeneratedFile reports whether f carries the generated-code marker.
func isGeneratedFile(f *ast.File) bool {
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			for _, line := range strings.Split(c.Text, "\n") {
				if generatedMarker.MatchString(strings.TrimSpace(line)) {
					return true
				}
			}
		}
	}
	return false
}

// exprText renders an expression with the given FileSet, whitespace
// collapsed to one line; "" for nil or unrenderable nodes.
func exprText(fset *token.FileSet, e ast.Expr) string {
	if e == nil || fset == nil {
		return ""
	}
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, e); err != nil {
		return ""
	}
	return strings.Join(strings.Fields(buf.String()), " ")
}

// capText truncates s to n bytes with an ASCII ellipsis, the style the
// rest of the package uses for excerpts.
func capText(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}

// refLikeType reports whether a value of this type shares its contents
// when passed around (slice, map, pointer, channel) — the honest basis
// for "that function can change its contents".
func refLikeType(t string) bool {
	t = strings.TrimSpace(t)
	return strings.HasPrefix(t, "[]") || strings.HasPrefix(t, "map[") ||
		strings.HasPrefix(t, "*") || strings.HasPrefix(t, "chan ")
}

// inferInitType renders the type of an initializer expression where it
// is written plainly: []string{...}, Config{...}, make([]T, ...).
func inferInitType(e ast.Expr, fset *token.FileSet) string {
	switch e := e.(type) {
	case *ast.CompositeLit:
		return exprText(fset, e.Type)
	case *ast.UnaryExpr:
		if e.Op == token.AND {
			return inferInitType(e.X, fset)
		}
	case *ast.CallExpr:
		if id, ok := e.Fun.(*ast.Ident); ok && id.Name == "make" && len(e.Args) > 0 {
			return exprText(fset, e.Args[0])
		}
	}
	return ""
}

// addVarDecl records one package-level var/const spec into pkg.Vars.
func (p *Project) addVarDecl(pkg *Package, dir, rel string, isConst bool, vs *ast.ValueSpec, fset *token.FileSet) {
	for i, name := range vs.Names {
		v := &Var{
			Name:     name.Name,
			Pkg:      pkg.Name,
			PkgDir:   dir,
			Const:    isConst,
			Exported: ast.IsExported(name.Name),
			File:     rel,
			Line:     fset.Position(name.Pos()).Line,
		}
		if vs.Type != nil {
			v.Type = exprText(fset, vs.Type)
		}
		if i < len(vs.Values) {
			v.Init = exprText(fset, vs.Values[i])
			v.initNode = vs.Values[i]
			v.initFset = fset
			if v.Type == "" {
				v.Type = inferInitType(vs.Values[i], fset)
			}
		}
		pkg.Vars = append(pkg.Vars, v)
	}
}

// describeRHS renders where an assigned value comes from, in plain
// words a beginner can follow: "the return value of loadConfig()",
// "the value of cfg.Items", "a new []string built right there: ...".
func describeRHS(fset *token.FileSet, e ast.Expr) string {
	switch e := e.(type) {
	case *ast.CallExpr:
		if id, ok := e.Fun.(*ast.Ident); ok {
			switch id.Name {
			case "make":
				return "a fresh empty value from " + capText(exprText(fset, e), 56)
			case "append":
				return "the result of append(...) — the old items plus new ones"
			case "new":
				return "a fresh value from " + capText(exprText(fset, e), 48)
			}
		}
		return "the return value of " + exprText(fset, e.Fun) + "()"
	case *ast.CompositeLit:
		t := exprText(fset, e.Type)
		if t == "" {
			t = "value"
		}
		return "a new " + t + " built right there: " + capText(exprText(fset, e), 56)
	case *ast.Ident:
		switch e.Name {
		case "nil":
			return "nil — emptied"
		case "true", "false":
			return "the value " + e.Name
		}
		return "the value of " + e.Name
	case *ast.SelectorExpr:
		return "the value of " + exprText(fset, e)
	case *ast.BasicLit:
		return "the literal " + capText(e.Value, 40)
	case *ast.BinaryExpr:
		return "a computed value (" + capText(exprText(fset, e), 52) + ")"
	case *ast.IndexExpr:
		return "an element of " + exprText(fset, e.X)
	case *ast.UnaryExpr:
		if e.Op == token.AND {
			return "the address of " + exprText(fset, e.X)
		}
		return "the value " + capText(exprText(fset, e), 52)
	case *ast.StarExpr:
		return "the value " + exprText(fset, e.X) + " points to"
	case *ast.FuncLit:
		return "a function written right there"
	case *ast.SliceExpr:
		return "a slice of " + exprText(fset, e.X)
	case *ast.TypeAssertExpr:
		return "the value of " + exprText(fset, e.X) + ", taken as a specific type"
	}
	return "the value " + capText(exprText(fset, e), 56)
}

// --- site collection ------------------------------------------------------

// vTarget is a resolved tracked value inside one function.
type vTarget struct {
	key  string
	name string
	ref  bool // contents are shared when the value is handed over
}

// varCollector walks one function body recording varflow events.
type varCollector struct {
	p         *Project
	pkg       *Package
	fn        *Func
	ctx       *resolveCtx
	locals    map[string]*localVar
	localType map[string]string
}

func (p *Project) addVarEvent(key string, ev varEvent) {
	for _, old := range p.varEvents[key] {
		if old.Kind == ev.Kind && old.File == ev.File && old.Line == ev.Line && old.Text == ev.Text {
			return // identical site already recorded
		}
	}
	p.varEvents[key] = append(p.varEvents[key], ev)
}

// collectVarEvents records the dataflow sites in one function body.
// Called from BuildGraph right after analyzeEffects, while decls and
// the effects pass's type maps are still available.
func (p *Project) collectVarEvents(pkg *Package, fn *Func) {
	if fn.decl == nil || fn.decl.Body == nil || p.genFiles[fn.File] {
		return
	}
	p.buildVarIndex()
	if p.varEvents == nil {
		p.varEvents = map[string][]varEvent{}
	}
	if p.varLocals == nil {
		p.varLocals = map[string][]*localVar{}
	}
	ctx := fn.ctx
	if ctx == nil {
		ctx = p.newResolveCtx(pkg, fn)
	}
	vc := &varCollector{
		p:         p,
		pkg:       pkg,
		fn:        fn,
		ctx:       ctx,
		locals:    map[string]*localVar{},
		localType: map[string]string{},
	}
	vc.registerParams()
	ast.Inspect(fn.decl.Body, vc.visit)
}

// registerParams records the receiver and parameters as locals whose
// data comes from callers.
func (vc *varCollector) registerParams() {
	if vc.fn.decl.Type == nil {
		return
	}
	add := func(fl *ast.FieldList) {
		if fl == nil {
			return
		}
		for _, f := range fl.List {
			tt := exprText(vc.fn.fset, f.Type)
			for _, n := range f.Names {
				vc.registerLocal(n.Name, tt, n, "", true)
			}
		}
	}
	if vc.fn.decl.Recv != nil {
		add(vc.fn.decl.Recv)
	}
	add(vc.fn.decl.Type.Params)
}

// registerLocal records a local name once (first declaration wins —
// a closure parameter that shadows an outer name is attributed to the
// outer variable; an accepted heuristic gap, like the rest of the
// package's resolution).
func (vc *varCollector) registerLocal(name, typeText string, pos ast.Node, initDesc string, param bool) *localVar {
	if lv, ok := vc.locals[name]; ok {
		return lv
	}
	lv := &localVar{
		Fn:       vc.fn,
		Name:     name,
		TypeText: typeText,
		File:     vc.fn.File,
		Line:     vc.p.lineOf(vc.fn, pos),
		Param:    param,
		InitDesc: initDesc,
	}
	vc.locals[name] = lv
	if typeText != "" {
		vc.localType[name] = typeText
	}
	vc.p.varLocals[name] = append(vc.p.varLocals[name], lv)
	return lv
}

// namedValueType returns the type text of a bare name in this
// function: a local's declared type, a package variable's type, or a
// parameter's named type from the effects pass.
func (vc *varCollector) namedValueType(name string) string {
	if t := vc.localType[name]; t != "" {
		return t
	}
	if v := varIn(vc.pkg, name); v != nil {
		return v.Type
	}
	return vc.fn.ParamTypes[name]
}

// targetOf resolves an expression used as a value (assignment LHS,
// call argument, return value) to a tracked value, or nil.
func (vc *varCollector) targetOf(e ast.Expr) *vTarget {
	switch e := e.(type) {
	case *ast.Ident:
		if _, ok := vc.locals[e.Name]; ok {
			return &vTarget{key: localKey(vc.fn, e.Name), name: e.Name, ref: refLikeType(vc.localType[e.Name])}
		}
		if v := varIn(vc.pkg, e.Name); v != nil {
			return &vTarget{key: v.key(), name: e.Name, ref: refLikeType(v.Type)}
		}
	case *ast.SelectorExpr:
		return vc.selectorTarget(e)
	}
	return nil
}

// selectorTarget resolves x.Field: a package variable through an
// import qualifier, a field on a typed parameter/local, or a field on
// a struct-typed variable.
func (vc *varCollector) selectorTarget(sel *ast.SelectorExpr) *vTarget {
	x, ok := sel.X.(*ast.Ident)
	if !ok {
		return nil
	}
	// (a) pkg.Var through the file's imports.
	if q, isAlias := vc.pkg.aliasCache[x.Name]; isAlias {
		if v := varIn(q, sel.Sel.Name); v != nil {
			return &vTarget{key: v.key(), name: sel.Sel.Name, ref: refLikeType(v.Type)}
		}
	}
	fieldOf := func(tt *Type) *vTarget {
		if tt == nil || tt.Kind != "struct" {
			return nil
		}
		if _, isField := tt.FieldTypes[sel.Sel.Name]; isField {
			return &vTarget{
				key:  fieldKey(tt.Pkg, tt.Name, sel.Sel.Name),
				name: sel.Sel.Name,
				ref:  refLikeType(tt.FieldTypes[sel.Sel.Name]),
			}
		}
		return nil
	}
	// (b) cfg.Field where cfg's named type came from the effects pass.
	if t := valueTypeOf(vc.fn, x.Name); t != "" {
		if tgt := fieldOf(vc.p.vidx.typeByKey[t]); tgt != nil {
			return tgt
		}
	}
	// (c) cfg.Field where cfg is a struct-typed variable.
	if base := vc.namedValueType(x.Name); base != "" {
		if tgt := fieldOf(vc.p.lookupType(base, vc.pkg)); tgt != nil {
			return tgt
		}
	}
	return nil
}

// visit is the ast.Inspect callback for the collector.
func (vc *varCollector) visit(n ast.Node) bool {
	switch s := n.(type) {
	case *ast.DeclStmt:
		vc.declStmt(s)
	case *ast.AssignStmt:
		vc.assign(s)
	case *ast.IncDecStmt:
		word := "increased by one (++)"
		if s.Tok == token.DEC {
			word = "decreased by one (--)"
		}
		if idx, ok := s.X.(*ast.IndexExpr); ok {
			// x[i]++ changes the container's contents, like x[i] = ...
			if t := vc.targetOf(idx.X); t != nil {
				vc.event(t, "incdec", s, fmt.Sprintf("element [%s] is %s", exprText(vc.fn.fset, idx.Index), word))
			}
		} else if t := vc.targetOf(s.X); t != nil {
			vc.event(t, "incdec", s, word)
		}
	case *ast.CallExpr:
		vc.callExpr(s)
	case *ast.CompositeLit:
		vc.compositeLit(vc.pkg, vc.fn, s)
	case *ast.ReturnStmt:
		for _, r := range s.Results {
			if t := vc.targetOf(r); t != nil {
				vc.event(t, "returned", s, "handed back to the caller of "+vc.fn.Display())
			}
		}
	case *ast.SendStmt:
		if t := vc.targetOf(s.Value); t != nil {
			vc.event(t, "handoff", s, "sent over a channel to whoever receives it")
		}
	case *ast.RangeStmt:
		// Loop variables are fresh locals of this function.
		for _, e := range []ast.Expr{s.Key, s.Value} {
			if id, ok := e.(*ast.Ident); ok && id.Name != "_" {
				vc.registerLocal(id.Name, "", id, "each value the loop goes over, one at a time", false)
			}
		}
	}
	return true
}

func (vc *varCollector) event(t *vTarget, kind string, node ast.Node, text string) {
	vc.p.addVarEvent(t.key, varEvent{
		Kind: kind,
		File: vc.fn.File,
		Line: vc.p.lineOf(vc.fn, node),
		In:   vc.fn.Display(),
		Text: text,
	})
}

// declStmt handles `var x ...` inside a function body.
func (vc *varCollector) declStmt(s *ast.DeclStmt) {
	gd, ok := s.Decl.(*ast.GenDecl)
	if !ok || gd.Tok != token.VAR {
		return
	}
	for _, spec := range gd.Specs {
		vs, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		for i, name := range vs.Names {
			tt := ""
			if vs.Type != nil {
				tt = exprText(vc.fn.fset, vs.Type)
			}
			init := ""
			if i < len(vs.Values) {
				init = describeRHS(vc.fn.fset, vs.Values[i])
				if tt == "" {
					tt = inferInitType(vs.Values[i], vc.fn.fset)
				}
			}
			vc.registerLocal(name.Name, tt, name, init, false)
		}
	}
}

// assign records assignment-shaped sites: plain assignments with
// their RHS told in plain words, and element writes.
func (vc *varCollector) assign(s *ast.AssignStmt) {
	fresh := map[string]bool{}
	if s.Tok == token.DEFINE {
		// Register new locals first so their own LHS resolves to them;
		// their creation is the InitDesc, not an "assign" event.
		for i, lhs := range s.Lhs {
			id, ok := lhs.(*ast.Ident)
			if !ok || id.Name == "_" {
				continue
			}
			if _, known := vc.locals[id.Name]; known {
				continue
			}
			tt, init := "", ""
			if i < len(s.Rhs) {
				tt = inferInitType(s.Rhs[i], vc.fn.fset)
				init = describeRHS(vc.fn.fset, s.Rhs[i])
			}
			vc.registerLocal(id.Name, tt, id, init, false)
			fresh[localKey(vc.fn, id.Name)] = true
		}
	}
	for i, lhs := range s.Lhs {
		var rhs ast.Expr
		if i < len(s.Rhs) {
			rhs = s.Rhs[i]
		} else if len(s.Rhs) > 0 {
			rhs = s.Rhs[len(s.Rhs)-1]
		}
		if idx, ok := lhs.(*ast.IndexExpr); ok {
			if t := vc.targetOf(idx.X); t != nil {
				text := "an element is replaced"
				if rhs != nil {
					text = fmt.Sprintf("element [%s] is set to %s", exprText(vc.fn.fset, idx.Index), describeRHS(vc.fn.fset, rhs))
				}
				vc.event(t, "index", s, text)
			}
			continue
		}
		t := vc.targetOf(lhs)
		if t == nil || fresh[t.key] {
			continue
		}
		if rhs != nil && isAppendTo(rhs, t, vc) {
			continue // the CallExpr visit records the append itself
		}
		text := "assigned a new value"
		if rhs != nil {
			text = "set from " + describeRHS(vc.fn.fset, rhs)
		}
		vc.event(t, "assign", s, text)
	}
}

// isAppendTo reports whether rhs is append(<the same target>, ...).
func isAppendTo(rhs ast.Expr, t *vTarget, vc *varCollector) bool {
	call, ok := rhs.(*ast.CallExpr)
	if !ok || len(call.Args) == 0 {
		return false
	}
	id, ok := call.Fun.(*ast.Ident)
	if !ok || id.Name != "append" {
		return false
	}
	at := vc.targetOf(call.Args[0])
	return at != nil && at.key == t.key
}

// callExpr records append/delete/clear/copy sites and hand-offs of
// tracked values into calls.
func (vc *varCollector) callExpr(s *ast.CallExpr) {
	if id, ok := s.Fun.(*ast.Ident); ok {
		switch id.Name {
		case "append", "delete", "clear":
			if len(s.Args) > 0 {
				if t := vc.targetOf(s.Args[0]); t != nil {
					text := map[string]string{
						"append": "new items are added with " + capText(exprText(vc.fn.fset, s), 64),
						"delete": "an entry is deleted with " + capText(exprText(vc.fn.fset, s), 64),
						"clear":  "emptied out with " + capText(exprText(vc.fn.fset, s), 64),
					}[id.Name]
					vc.event(t, id.Name, s, text)
				}
			}
			return
		case "copy":
			if len(s.Args) > 0 {
				if t := vc.targetOf(s.Args[0]); t != nil {
					from := ""
					if len(s.Args) > 1 {
						from = " from " + exprText(vc.fn.fset, s.Args[1])
					}
					vc.event(t, "copy", s, "filled with a copy of another value's items"+from)
				}
			}
			return
		}
	}
	// A method call on a tracked value: its receiver is handed over.
	if sel, ok := s.Fun.(*ast.SelectorExpr); ok {
		if vc.selectorTarget(sel) == nil {
			if x, isID := sel.X.(*ast.Ident); isID {
				if t := vc.targetOf(x); t != nil {
					vc.event(t, "handoff", s, "its method "+sel.Sel.Name+"() is called — a method with a pointer receiver can change it")
				}
			}
		}
	}
	callee := vc.calleeName(s)
	for _, arg := range s.Args {
		byAddr := false
		a := arg
		if u, ok := a.(*ast.UnaryExpr); ok && u.Op == token.AND {
			byAddr = true
			a = u.X
		}
		t := vc.targetOf(a)
		if t == nil {
			continue
		}
		var text string
		switch {
		case byAddr:
			text = "passed by address (&" + t.name + ") to " + callee + " — " + callee + " can change it there"
		case t.ref:
			text = "passed to " + callee + " — that function can change its contents"
		default:
			text = "passed to " + callee + " as a copy — the original cannot change there"
		}
		vc.event(t, "handoff", s, text)
	}
}

// calleeName renders the called function for hand-off sentences: the
// project function's name when resolved, else the written callee.
func (vc *varCollector) calleeName(s *ast.CallExpr) string {
	if target := vc.p.resolveCall(vc.pkg, vc.ctx, vc.fn, s); target != nil {
		return target.Display()
	}
	if txt := exprText(vc.fn.fset, s.Fun); txt != "" {
		return txt
	}
	return "another function"
}

// compositeLit records field values set through a T{Field: ...}
// literal — how struct values usually get their data.
func (vc *varCollector) compositeLit(pkg *Package, fn *Func, lit *ast.CompositeLit) {
	tn := compositeTypeName(pkg, lit)
	if tn == "" {
		return
	}
	tt := vc.p.vidx.typeByKey[tn]
	if tt == nil || tt.Kind != "struct" {
		return
	}
	fset, file, in := fn.fset, fn.File, fn.Display()
	line := vc.p.lineOf(fn, lit)
	for _, elt := range lit.Elts {
		kv, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			continue
		}
		kid, ok := kv.Key.(*ast.Ident)
		if !ok {
			continue
		}
		if _, isField := tt.FieldTypes[kid.Name]; !isField {
			continue
		}
		vc.p.addVarEvent(fieldKey(tt.Pkg, tt.Name, kid.Name), varEvent{
			Kind: "assign",
			File: file,
			Line: line,
			In:   in,
			Text: fmt.Sprintf("set when a %s value is built: %s: %s", tt.Name, kid.Name, describeRHS(fset, kv.Value)),
		})
	}
}

// sweepVarInits records the sites hidden in package-level
// initializers (struct literals that set fields, appends onto other
// variables), then drops the kept initializer ASTs. Called from
// BuildGraph after the alias caches exist and before finalize.
func (p *Project) sweepVarInits() {
	if p.varSwept {
		return
	}
	p.varSwept = true
	p.buildVarIndex()
	if p.varEvents == nil {
		p.varEvents = map[string][]varEvent{}
	}
	for _, pkg := range p.Packages {
		for _, v := range pkg.Vars {
			if v.initNode == nil {
				continue
			}
			node, fset := v.initNode, v.initFset
			v.initNode, v.initFset = nil, nil
			ast.Inspect(node, func(n ast.Node) bool {
				switch s := n.(type) {
				case *ast.CompositeLit:
					tn := compositeTypeName(pkg, s)
					if tt := p.vidx.typeByKey[tn]; tt != nil && tt.Kind == "struct" {
						line := 0
						if fset != nil {
							line = fset.Position(s.Pos()).Line
						}
						for _, elt := range s.Elts {
							kv, ok := elt.(*ast.KeyValueExpr)
							if !ok {
								continue
							}
							kid, ok := kv.Key.(*ast.Ident)
							if !ok {
								continue
							}
							if _, isField := tt.FieldTypes[kid.Name]; !isField {
								continue
							}
							p.addVarEvent(fieldKey(tt.Pkg, tt.Name, kid.Name), varEvent{
								Kind: "assign",
								File: v.File,
								Line: line,
								Text: fmt.Sprintf("set when a %s value is built (in the declaration of %s): %s: %s",
									tt.Name, v.Name, kid.Name, describeRHS(fset, kv.Value)),
							})
						}
					}
				case *ast.CallExpr:
					if id, ok := s.Fun.(*ast.Ident); ok && (id.Name == "append" || id.Name == "delete") && len(s.Args) > 0 {
						if id2, ok := s.Args[0].(*ast.Ident); ok {
							if tgt := varIn(pkg, id2.Name); tgt != nil {
								line := 0
								if fset != nil {
									line = fset.Position(s.Pos()).Line
								}
								text := "new items are added with " + capText(exprText(fset, s), 64)
								if id.Name == "delete" {
									text = "an entry is deleted with " + capText(exprText(fset, s), 64)
								}
								p.addVarEvent(tgt.key(), varEvent{
									Kind: id.Name,
									File: v.File,
									Line: line,
									Text: text,
								})
							}
						}
					}
				}
				return true
			})
		}
	}
}

// --- the answer -----------------------------------------------------------

// cleanVarName strips question dressing from a captured name: "the
// Items slice" -> "Items", "this array" -> "array" (which then simply
// won't resolve). Mirrors cleanSymbol's never-invent discipline.
func cleanVarName(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, `"'().,?!`)
	for _, pre := range []string{"the ", "this ", "that ", "my "} {
		s = strings.TrimPrefix(s, pre)
	}
	for {
		trimmed := s
		for _, suf := range []string{" variable", " slice", " array", " list", " map", " field", " value", " thing"} {
			trimmed = strings.TrimSuffix(trimmed, suf)
		}
		if trimmed == s {
			break
		}
		s = trimmed
	}
	return strings.TrimSpace(strings.Trim(s, `"'().,?!`))
}

// dataSubject is one resolved tracked value, ready to render.
type dataSubject struct {
	kind     string // "var" | "const" | "field"
	name     string
	typeText string
	declNote string // "a package-level variable in package store"
	file     string
	line     int
	initDesc string // plain words for the starting value; "" if none
	key      string // event index key
}

// resolveDataSubjects maps a (possibly qualified) name to the tracked
// values it could mean: package variables, constants, struct fields.
// "Items", "Config.Items", "store.Items", "store.Config.Items" all
// work. Locals are resolved separately (they need their function).
func (p *Project) resolveDataSubjects(name string) []dataSubject {
	p.buildVarIndex()
	var out []dataSubject
	mkVar := func(v *Var) dataSubject {
		kind := "var"
		if v.Const {
			kind = "const"
		}
		s := dataSubject{
			kind:     kind,
			name:     v.Name,
			typeText: v.Type,
			declNote: "a package-level variable in package " + v.Pkg,
			file:     v.File,
			line:     v.Line,
			key:      v.key(),
		}
		if v.Const {
			s.declNote = "a constant in package " + v.Pkg
		}
		if v.Init != "" {
			s.initDesc = "starts as " + v.Init
		}
		return s
	}
	mkField := func(key string) dataSubject {
		tt := p.vidx.fieldsByKey[key]
		return dataSubject{
			kind:     "field",
			name:     key[strings.LastIndex(key, ".")+1:],
			typeText: tt.FieldTypes[key[strings.LastIndex(key, ".")+1:]],
			declNote: "a field of the " + tt.Name + " struct (package " + tt.Pkg + ")",
			file:     tt.File,
			line:     tt.FieldLines[key[strings.LastIndex(key, ".")+1:]],
			key:      key,
		}
	}
	parts := strings.Split(name, ".")
	base := parts[len(parts)-1]
	quals := parts[:len(parts)-1]
	matchName := func(have, want string) bool {
		return have == want || strings.EqualFold(have, want)
	}
	// Package variables.
	for _, v := range p.allVars() {
		if !matchName(v.Name, base) {
			continue
		}
		if len(quals) > 0 {
			q := quals[len(quals)-1]
			if !matchName(v.Pkg, q) && !matchName(shortDir(v.PkgDir), q) {
				continue
			}
		}
		out = append(out, mkVar(v))
	}
	// Struct fields.
	for key, tt := range p.vidx.fieldsByKey {
		fname := key[strings.LastIndex(key, ".")+1:]
		if !matchName(fname, base) {
			continue
		}
		if len(quals) > 0 {
			q := quals[len(quals)-1]
			if !matchName(tt.Name, q) && !matchName(tt.Pkg, q) &&
				!(len(quals) > 1 && matchName(tt.Pkg, quals[len(quals)-2])) {
				continue
			}
		}
		out = append(out, mkField(key))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].kind != out[j].kind {
			return out[i].kind < out[j].kind
		}
		return out[i].key < out[j].key
	})
	return out
}

func (p *Project) allVars() []*Var {
	var out []*Var
	for _, pkg := range p.Packages {
		out = append(out, pkg.Vars...)
	}
	return out
}

// HasData reports whether the project tracks a package variable,
// struct field, or function-local under this name — the chat layer
// uses it to pull the name out of a free-form question before
// calling DataflowText.
func (p *Project) HasData(name string) bool {
	name = cleanVarName(name)
	if name == "" {
		return false
	}
	return len(p.resolveDataSubjects(name)) > 0 || len(p.localsNamed(name)) > 0
}

// DataflowText answers "where does <name> get its data / come from /
// change": declaration, sources, and change sites for a package-level
// variable, constant, struct field, or local. It returns ok=false
// only when the name matches nothing and no near miss exists, so the
// chat can fall through to other brains instead of guessing.
func (p *Project) DataflowText(name string) (string, bool) {
	name = cleanVarName(name)
	if name == "" {
		return "", false
	}
	base := name
	if i := strings.LastIndex(name, "."); i >= 0 {
		base = name[i+1:]
	}
	if subs := p.resolveDataSubjects(name); len(subs) == 1 {
		return p.renderDataSubject(subs[0]), true
	} else if len(subs) > 1 {
		return p.renderAmbiguousSubjects(name, subs), true
	}
	if locals := p.localsNamed(base); len(locals) > 0 {
		return p.renderLocals(base, locals), true
	}
	if near := p.nearVarNames(base); len(near) > 0 {
		var b strings.Builder
		fmt.Fprintf(&b, "I couldn't find a variable or field named %q in this project. Did you mean one of these?\n", base)
		for _, n := range near {
			fmt.Fprintf(&b, "  - %s\n", n)
		}
		b.WriteString("Ask again with one of those names and I'll trace where its data comes from and where it changes.\n")
		return b.String(), true
	}
	return "", false
}

// localsNamed returns the locals (across functions) with this name.
func (p *Project) localsNamed(name string) []*localVar {
	var out []*localVar
	for n, lvs := range p.varLocals {
		if n == name || strings.EqualFold(n, name) {
			out = append(out, lvs...)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Fn.ID < out[j].Fn.ID })
	return out
}

// typePhrase says what a value is in a beginner's words.
func typePhrase(t string) string {
	if t == "" {
		return ""
	}
	switch {
	case strings.HasPrefix(t, "[]"):
		return "a slice — a growable list — of type " + t
	case strings.HasPrefix(t, "map["):
		return "a map — a lookup table — of type " + t
	case strings.HasPrefix(t, "*"):
		return "a pointer of type " + t + " (it points at a value kept elsewhere)"
	default:
		return "of type " + t
	}
}

func (p *Project) sortedEvents(key string) []varEvent {
	evs := append([]varEvent{}, p.varEvents[key]...)
	sort.Slice(evs, func(i, j int) bool {
		if evs[i].File != evs[j].File {
			return evs[i].File < evs[j].File
		}
		return evs[i].Line < evs[j].Line
	})
	return evs
}

// siteLine renders one event with its clickable location.
func (p *Project) siteLine(ev varEvent) string {
	loc := fmt.Sprintf("%s:%d", p.LinkPath(ev.File), ev.Line)
	if ev.In != "" {
		return fmt.Sprintf("  - %s, in %s — %s\n", loc, ev.In, ev.Text)
	}
	return fmt.Sprintf("  - %s — %s\n", loc, ev.Text)
}

// renderDataSubject renders the full dataflow report for one
// package-level variable, constant, or struct field.
func (p *Project) renderDataSubject(s dataSubject) string {
	var b strings.Builder
	what := typePhrase(s.typeText)
	if what != "" {
		fmt.Fprintf(&b, "`%s` — %s; %s.\n", s.name, what, s.declNote)
	} else {
		fmt.Fprintf(&b, "`%s` — %s.\n", s.name, s.declNote)
	}
	fmt.Fprintf(&b, "Declared at %s:%d.", p.LinkPath(s.file), s.line)
	if s.initDesc != "" {
		fmt.Fprintf(&b, " It %s.", s.initDesc)
	} else if s.kind == "field" {
		b.WriteString(" A field has no starting value of its own — each new " +
			"struct value either sets it (below) or leaves it empty.")
	} else if s.kind == "var" {
		b.WriteString(" Nothing is written at the declaration, so it starts as the " +
			"type's zero value — empty for slices and maps, zero for numbers.")
	}
	b.WriteString("\n")

	evs := p.sortedEvents(s.key)
	var sources, changes []varEvent
	for _, ev := range evs {
		if ev.Kind == "assign" {
			sources = append(sources, ev)
		} else {
			changes = append(changes, ev)
		}
	}
	b.WriteString("\nWhere its data comes from:\n")
	if len(sources) == 0 {
		if s.initDesc != "" {
			b.WriteString("  - only its declaration (above) — nothing else in this project ever sets it.\n")
		} else {
			b.WriteString("  - nowhere in this project — it is only ever read, or set from outside the code analyzed.\n")
		}
	}
	for i, ev := range sources {
		if i >= 12 {
			fmt.Fprintf(&b, "  ... and %d more\n", len(sources)-12)
			break
		}
		b.WriteString(p.siteLine(ev))
	}
	b.WriteString("\nWhere it changes:\n")
	if s.kind == "const" {
		b.WriteString("  - never. It is a constant — Go itself forbids changing it.\n")
	} else if len(changes) == 0 && len(sources) == 0 {
		b.WriteString("  - nowhere in this project.\n")
	} else if len(changes) == 0 {
		b.WriteString("  - its contents are never edited in place; it only ever gets a whole new value (the sources above).\n")
	}
	for i, ev := range changes {
		if i >= 12 {
			fmt.Fprintf(&b, "  ... and %d more\n", len(changes)-12)
			break
		}
		b.WriteString(p.siteLine(ev))
	}
	if s.kind != "const" && len(sources) > 0 && len(changes) > 0 {
		fmt.Fprintf(&b, "It is also replaced wholesale at %d site(s) — see the lines under \"Where its data comes from\" above.\n", len(sources))
	}
	return b.String()
}

// renderAmbiguousSubjects lists the candidates when one name means
// several tracked values, in the did-you-mean style.
func (p *Project) renderAmbiguousSubjects(name string, subs []dataSubject) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%q could mean more than one thing here:\n", name)
	for _, s := range subs {
		fmt.Fprintf(&b, "  - `%s` — %s, declared at %s\n", s.name, s.declNote, pickLoc(s.file, s.line))
	}
	b.WriteString("Ask with the fuller name — for a field, name its struct; for a variable, its package — and I'll trace just that one.\n")
	return b.String()
}

// renderLocals answers for local variables: one function gets the
// full report with a scope note; several get a compact per-function
// listing, because each is a separate variable that merely shares a
// name.
func (p *Project) renderLocals(name string, locals []*localVar) string {
	if len(locals) == 1 {
		lv := locals[0]
		var b strings.Builder
		what := typePhrase(lv.TypeText)
		if what != "" {
			fmt.Fprintf(&b, "`%s` — %s; a local variable inside %s.\n", lv.Name, what, lv.Fn.Display())
		} else {
			fmt.Fprintf(&b, "`%s` — a local variable inside %s.\n", lv.Name, lv.Fn.Display())
		}
		fmt.Fprintf(&b, "It only lives while %s runs — nothing outside that function can touch it directly.\n", lv.Fn.Name)
		if lv.Param {
			b.WriteString("It is a parameter: its data comes from whoever calls the function, each time it is called.\n")
		} else {
			fmt.Fprintf(&b, "Declared at %s:%d.", p.LinkPath(lv.File), lv.Line)
			if lv.InitDesc != "" {
				fmt.Fprintf(&b, " It starts as %s.", lv.InitDesc)
			}
			b.WriteString("\n")
		}
		evs := p.sortedEvents(localKey(lv.Fn, lv.Name))
		if len(evs) == 0 {
			b.WriteString("Inside that function it is never changed after that.\n")
			return b.String()
		}
		b.WriteString("Inside the function:\n")
		for i, ev := range evs {
			if i >= 12 {
				fmt.Fprintf(&b, "  ... and %d more\n", len(evs)-12)
				break
			}
			// The enclosing function is the subject itself here.
			ev.In = ""
			b.WriteString(p.siteLine(ev))
		}
		return b.String()
	}
	var b strings.Builder
	fmt.Fprintf(&b, "`%s` is a local variable name — it exists separately inside %d functions, and each one is a different variable that only lives while its function runs:\n", name, len(locals))
	for i, lv := range locals {
		if i >= 6 {
			fmt.Fprintf(&b, "  ... and %d more functions with a local named %s\n", len(locals)-6, name)
			break
		}
		evs := p.sortedEvents(localKey(lv.Fn, lv.Name))
		note := "never changed after it is created"
		if lv.Param {
			note = "a parameter — its data comes from the caller"
		}
		if len(evs) > 0 {
			note = fmt.Sprintf("touched at %d site(s) inside", len(evs))
		}
		fmt.Fprintf(&b, "  - in %s (%s:%d) — %s\n", lv.Fn.Display(), p.LinkPath(lv.File), lv.Line, note)
	}
	b.WriteString("Ask about one function's local and I'll walk just that one — or name a package variable or a struct field for the project-wide picture.\n")
	return b.String()
}

// nearVarNames finds did-you-mean candidates among variable, field,
// and local names: prefix, substring, or small edit distance.
func (p *Project) nearVarNames(name string) []string {
	p.buildVarIndex()
	type cand struct {
		display string
		score   int
	}
	var cands []cand
	seen := map[string]bool{}
	add := func(display, bare string, score int) {
		if seen[display] {
			return
		}
		seen[display] = true
		cands = append(cands, cand{display, score})
	}
	lower := strings.ToLower(name)
	score := func(bare string) int {
		bl := strings.ToLower(bare)
		// Prefix/substring matches need a real stem: without the
		// floor, a one-letter name like a receiver `b` is "contained"
		// in nearly any query and gets suggested for everything.
		short := len(bl)
		if len(lower) < short {
			short = len(lower)
		}
		switch {
		case short >= 3 && (strings.HasPrefix(bl, lower) || strings.HasPrefix(lower, bl)):
			return 3
		case short >= 3 && (strings.Contains(bl, lower) || strings.Contains(lower, bl)):
			return 2
		case len(name) >= 4 && levenshtein(lower, bl) <= 2:
			return 1
		}
		return 0
	}
	for _, v := range p.allVars() {
		if s := score(v.Name); s > 0 {
			add(fmt.Sprintf("`%s` — a package-level variable in %s (%s:%d)", v.Name, v.Pkg, p.LinkPath(v.File), v.Line), v.Name, s)
		}
	}
	for key := range p.vidx.fieldsByKey {
		parts := strings.Split(key, ".")
		if len(parts) != 3 {
			continue
		}
		if s := score(parts[2]); s > 0 {
			tt := p.vidx.fieldsByKey[key]
			add(fmt.Sprintf("`%s` — a field of the %s struct (%s:%d)", parts[2], parts[1], p.LinkPath(tt.File), tt.FieldLines[parts[2]]), parts[2], s)
		}
	}
	for n := range p.varLocals {
		if s := score(n); s > 0 {
			add(fmt.Sprintf("`%s` — a local variable in %s", n, p.varLocals[n][0].Fn.Display()), n, s)
		}
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].score != cands[j].score {
			return cands[i].score > cands[j].score
		}
		return cands[i].display < cands[j].display
	})
	var out []string
	for i, c := range cands {
		if i >= 5 {
			break
		}
		out = append(out, c.display)
	}
	return out
}

// levenshtein is the classic edit distance, over at most a few dozen
// runes for identifier-shaped inputs.
func levenshtein(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	prev := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur := make([]int, len(rb)+1)
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			cur[j] = min3(cur[j-1]+1, prev[j]+1, prev[j-1]+cost)
		}
		prev = cur
	}
	return prev[len(rb)]
}

func min3(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}

// --- Structured dataflow facts --------------------------------------
// DataflowText renders the varflow index as prose for one value at a
// time. Some callers — the chat's learn mode, which quizzes the user
// from analyzer ground truth — need the same facts as data: which
// values exist, and which functions give each one its data.

// DataFact is one tracked package-level value and where its data
// comes from, in structured form.
type DataFact struct {
	Name    string       // value name, e.g. "Items"
	Display string       // package-qualified, e.g. "store.Items"
	Kind    string       // "var" | "const"
	Type    string       // declared type text; "" if unknown
	Init    string       // starting value as written; "" if none
	Setters []DataSetter // functions that assign it, first-site order
}

// DataSetter is one function that gives a tracked value its data.
type DataSetter struct {
	Func  string // enclosing function, Display() form, e.g. "store.LoadAll"
	Short string // its short name, e.g. "LoadAll"
	File  string // first assignment site
	Line  int
}

// VarFacts returns the dataflow facts for every package-level
// variable and constant in the project, in name order — the same
// site-level index DataflowText renders, as data.
func (p *Project) VarFacts() []DataFact {
	p.buildVarIndex()
	vars := p.allVars()
	sort.Slice(vars, func(i, j int) bool {
		if vars[i].Name != vars[j].Name {
			return vars[i].Name < vars[j].Name
		}
		return vars[i].Pkg < vars[j].Pkg
	})
	out := make([]DataFact, 0, len(vars))
	for _, v := range vars {
		kind := "var"
		if v.Const {
			kind = "const"
		}
		f := DataFact{
			Name:    v.Name,
			Display: v.Pkg + "." + v.Name,
			Kind:    kind,
			Type:    v.Type,
			Init:    v.Init,
		}
		seen := map[string]bool{}
		for _, ev := range p.sortedEvents(v.key()) {
			if ev.Kind != "assign" || ev.In == "" || seen[ev.In] {
				continue
			}
			seen[ev.In] = true
			short := ev.In
			if i := strings.LastIndex(ev.In, "."); i >= 0 {
				short = ev.In[i+1:]
			}
			f.Setters = append(f.Setters, DataSetter{
				Func: ev.In, Short: short, File: ev.File, Line: ev.Line,
			})
		}
		out = append(out, f)
	}
	return out
}
