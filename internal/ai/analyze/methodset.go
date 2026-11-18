package analyze

// Method sets, constructor mapping, and precise receiver resolution
// for the heuristic call graph.
//
// The heuristic pass in graph.go historically resolved x.M() by
// method NAME within a package: the first same-package method named
// M won, whatever x actually held. This file gives the pass the
// facts it needs to do better without type info:
//
//   - a constructor table: New / New<Type> functions mapped to the
//     struct type their return values instantiate, read from the
//     declarations themselves;
//   - Go method sets computed structurally over the index: a type's
//     own methods (value- and pointer-receiver kept apart) plus the
//     methods promoted through embedded fields, with Go's depth,
//     shadowing, and ambiguity rules;
//   - implementer sets for interfaces, so an interface-typed value
//     whose only project implementation is constructor-built can
//     dispatch to that one implementation.
//
// Everything here is best-effort and silent on weird code, like the
// rest of the heuristics: an unresolvable type simply means the
// caller falls back to the name-based chain in graph.go. When the
// compiler refinement (typed.go) runs it still has the last word —
// these edges are the floor it starts from, never a rival truth.

import (
	"go/ast"
	"strings"
	"unicode"
)

// ownMethod is one method declared directly on a type.
type ownMethod struct {
	fn  *Func
	ptr bool // declared with a pointer receiver
}

// setKey identifies one memoized method-set computation.
type setKey struct {
	typ *Type
	ptr bool // the pointer method set (*T) rather than the value set (T)
}

// setHit is one name in a computed method set: the function it
// resolves to and the promotion depth at which it was found (0 =
// declared on the type itself). fn is nil when the name is known to
// be in the set but has no single concrete target — a method
// promoted from an embedded interface, or an ambiguous promotion
// (two paths at the same minimal depth). A nil hit still shadows
// deeper candidates, exactly as in Go.
type setHit struct {
	fn    *Func
	depth int
}

// buildCallIndex fills the constructor table, the type index, and
// the per-type own-method lists. It runs at the top of BuildGraph,
// after import alias caches exist and while Func declarations are
// still available.
func (p *Project) buildCallIndex() {
	p.typeByKey = map[typeKey]*Type{}
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			p.typeByKey[typeKey{t.PkgDir, t.Name}] = t
		}
	}
	p.ownMethods = map[typeKey][]ownMethod{}
	for _, fn := range p.byID {
		if fn.Receiver == "" || fn.decl == nil || fn.decl.Recv == nil {
			continue
		}
		ptr := false
		if len(fn.decl.Recv.List) > 0 {
			_, ptr = fn.decl.Recv.List[0].Type.(*ast.StarExpr)
		}
		key := typeKey{fn.PkgDir, fn.Receiver}
		p.ownMethods[key] = append(p.ownMethods[key], ownMethod{fn: fn, ptr: ptr})
	}
	p.ctorTable = map[string]*Type{}
	p.ctorBuilds = map[*Type]bool{}
	for _, fn := range p.byID {
		if fn.Receiver != "" || fn.decl == nil || fn.decl.Type == nil {
			continue
		}
		if !isCtorName(fn.Name) || fn.decl.Type.Results == nil {
			continue
		}
		for _, res := range fn.decl.Type.Results.List {
			rt := p.resolveTypeExpr(fn.pkg, res.Type)
			if rt == nil || rt.Kind != "struct" {
				continue
			}
			// The first struct-typed result is the product (the
			// (T, error) convention puts it first). The name only
			// nominates candidates; the return type is the fact —
			// NewWithErr builds whatever it returns, name or not.
			p.ctorTable[fn.ID] = rt
			p.ctorBuilds[rt] = true
			break
		}
	}
	p.setMemo = map[setKey]map[string]setHit{}
	p.setBusy = map[setKey]bool{}
}

// isCtorName reports whether a plain function name follows the
// constructor convention: New, or New followed by a capitalized
// suffix (NewStore, NewStoreWithOptions).
func isCtorName(name string) bool {
	if name == "New" {
		return true
	}
	rest, ok := strings.CutPrefix(name, "New")
	return ok && rest != "" && unicode.IsUpper([]rune(rest)[0])
}

// typeNamed looks up a named type declared in pkg (nil-safe).
func (p *Project) typeNamed(pkg *Package, name string) *Type {
	if pkg == nil {
		return nil
	}
	return p.typeByKey[typeKey{pkg.Dir, name}]
}

// pkgForQualifier resolves an import qualifier as written in pkg's
// files to a project package: the alias cache first, then any
// package whose clause name matches (the cache keys on the import
// path's tail, which misses when the package's real name differs).
func (p *Project) pkgForQualifier(pkg *Package, qual string) *Package {
	if pkg == nil {
		return nil
	}
	if q, ok := pkg.aliasCache[qual]; ok {
		return q
	}
	if qual == pkg.Name {
		return pkg
	}
	for _, q := range p.Packages {
		if q.Name == qual {
			return q
		}
	}
	return nil
}

// resolveTypeExpr maps a type expression in pkg's source to the
// project type it names, or nil for anything else (builtins,
// external types, composite shapes). Pointer and generic wrappers
// unwrap to the named type.
func (p *Project) resolveTypeExpr(pkg *Package, e ast.Expr) *Type {
	switch t := e.(type) {
	case *ast.StarExpr:
		return p.resolveTypeExpr(pkg, t.X)
	case *ast.ParenExpr:
		return p.resolveTypeExpr(pkg, t.X)
	case *ast.IndexExpr:
		return p.resolveTypeExpr(pkg, t.X)
	case *ast.IndexListExpr:
		return p.resolveTypeExpr(pkg, t.X)
	case *ast.Ident:
		return p.typeNamed(pkg, t.Name)
	case *ast.SelectorExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return p.typeNamed(p.pkgForQualifier(pkg, id.Name), t.Sel.Name)
		}
	}
	return nil
}

// resolveTypeText maps rendered type text ("store.Base", "*Base",
// "Base[T]") as recorded in Type.FieldTypes and Type.Embeds to the
// project type it names, resolved in the context of the package that
// declares the containing type. Composite shapes (slices, maps,
// funcs) do not resolve.
func (p *Project) resolveTypeText(pkg *Package, text string) *Type {
	text = strings.TrimSpace(text)
	text = strings.TrimPrefix(text, "*")
	if i := strings.IndexByte(text, '['); i >= 0 {
		text = text[:i]
	}
	if i := strings.LastIndexByte(text, '.'); i >= 0 {
		return p.typeNamed(p.pkgForQualifier(pkg, text[:i]), text[i+1:])
	}
	return p.typeNamed(pkg, text)
}

// methodSet computes the method set of t following Go's promotion
// rules: own methods first (depth 0), then methods promoted through
// embedded fields one level per hop, with shallower declarations
// shadowing deeper ones and equal-depth collisions — even of the
// same underlying method via a diamond — recorded as ambiguous
// (a nil-fn hit) rather than guessed. ptr selects the pointer
// method set (*T); the recursion follows the spec's method-set
// composition: an embedded value field contributes its value set
// to T and its pointer set to *T, while an embedded pointer field
// contributes its pointer set to both.
func (p *Project) methodSet(t *Type, ptr bool) map[string]setHit {
	if t == nil {
		return nil
	}
	key := setKey{t, ptr}
	if s, ok := p.setMemo[key]; ok {
		return s
	}
	if p.setBusy[key] {
		return nil // embedding cycle: only possible in code Go rejects
	}
	p.setBusy[key] = true
	defer delete(p.setBusy, key)

	out := map[string]setHit{}
	if t.Kind == "interface" {
		// The interface's own method names have no concrete target;
		// they occupy the set so promotions through them shadow
		// deeper candidates correctly.
		embedded := map[string]bool{}
		for _, emb := range t.Embeds {
			embedded[emb.Name] = true
		}
		for _, name := range t.Methods {
			if !embedded[name] {
				out[name] = setHit{fn: nil, depth: 0}
			}
		}
	} else {
		for _, m := range p.ownMethods[typeKey{t.PkgDir, t.Name}] {
			if m.ptr && !ptr {
				continue
			}
			out[m.fn.Name] = setHit{fn: m.fn, depth: 0}
		}
	}
	ctxPkg := p.byPkgDir[t.PkgDir]
	for _, emb := range t.Embeds {
		et := p.resolveTypeText(ctxPkg, emb.TypeText)
		if et == nil || et == t {
			continue
		}
		sub := p.methodSet(et, ptr || emb.Ptr)
		for name, hit := range sub {
			cand := setHit{fn: hit.fn, depth: hit.depth + 1}
			cur, exists := out[name]
			switch {
			case !exists || cand.depth < cur.depth:
				out[name] = cand
			case cand.depth == cur.depth:
				// Two paths at the same minimal depth: ambiguous
				// in Go even when both reach the same method.
				out[name] = setHit{fn: nil, depth: cand.depth}
			}
		}
	}
	p.setMemo[key] = out
	return out
}

// lookupMethod resolves a method name on a concrete type through
// its (pointer) method set. It returns nil when the name is absent,
// ambiguous, or only interface-promoted — never a guess.
func (p *Project) lookupMethod(t *Type, name string) *Func {
	if hit, ok := p.methodSet(t, true)[name]; ok {
		return hit.fn
	}
	return nil
}

// interfaceMethods returns the full method-name set an interface
// requires, including names promoted from embedded interfaces.
func (p *Project) interfaceMethods(t *Type) map[string]bool {
	out := map[string]bool{}
	for name := range p.methodSet(t, false) {
		out[name] = true
	}
	return out
}

// implementersOf returns every project type whose pointer method
// set covers the interface's required methods with concrete,
// unambiguous targets.
func (p *Project) implementersOf(iface *Type) []*Type {
	req := p.interfaceMethods(iface)
	if len(req) == 0 {
		return nil
	}
	var out []*Type
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Kind == "interface" || t == iface {
				continue
			}
			set := p.methodSet(t, true)
			ok := true
			for name := range req {
				if hit, has := set[name]; !has || hit.fn == nil {
					ok = false
					break
				}
			}
			if ok {
				out = append(out, t)
			}
		}
	}
	return out
}

// soleCtorImpl returns the interface's single project implementer
// when there is exactly one and the constructor table builds it —
// the one case where an interface-typed value's dynamic type is
// provable from the index alone. Otherwise nil.
func (p *Project) soleCtorImpl(iface *Type) *Type {
	impls := p.implementersOf(iface)
	if len(impls) != 1 || !p.ctorBuilds[impls[0]] {
		return nil
	}
	return impls[0]
}

// fieldTypeOf resolves the declared type of a field of struct type
// t: the field itself when declared on t, the embedded type when
// the field IS an embedded field, or a promoted field found through
// embedding (shallowest declaration wins; an equal-depth tie is
// ambiguous and resolves to nothing).
func (p *Project) fieldTypeOf(t *Type, field string) *Type {
	if t == nil || t.Kind != "struct" {
		return nil
	}
	ctxPkg := p.byPkgDir[t.PkgDir]
	if text, ok := t.FieldTypes[field]; ok {
		return p.resolveTypeText(ctxPkg, text)
	}
	for _, emb := range t.Embeds {
		if emb.Name == field {
			return p.resolveTypeText(ctxPkg, emb.TypeText)
		}
	}
	found, _, _ := p.promotedField(t, field, map[*Type]bool{t: true})
	return found
}

// promotedField searches t's embedded structs for a declaration of
// field, returning the field's type, its promotion depth, and
// whether the shallowest declaration is ambiguous.
func (p *Project) promotedField(t *Type, field string, seen map[*Type]bool) (*Type, int, bool) {
	ctxPkg := p.byPkgDir[t.PkgDir]
	var best *Type
	bestDepth := -1
	ambiguous := false
	for _, emb := range t.Embeds {
		et := p.resolveTypeText(ctxPkg, emb.TypeText)
		if et == nil || et.Kind != "struct" || seen[et] {
			continue
		}
		seen[et] = true
		var cand *Type
		depth := 1
		if text, ok := et.FieldTypes[field]; ok {
			cand = p.resolveTypeText(p.byPkgDir[et.PkgDir], text)
		} else if sub, d, amb := p.promotedField(et, field, seen); sub != nil || amb {
			cand, depth = sub, d+1
			if amb {
				cand = nil
				// propagate ambiguity at this depth via best tracking below
				if bestDepth == -1 || depth < bestDepth {
					best, bestDepth, ambiguous = nil, depth, true
				} else if depth == bestDepth {
					ambiguous = true
				}
				continue
			}
		}
		if cand == nil {
			continue
		}
		switch {
		case bestDepth == -1 || depth < bestDepth:
			best, bestDepth, ambiguous = cand, depth, false
		case depth == bestDepth:
			ambiguous = true
		}
	}
	if ambiguous {
		return nil, bestDepth, true
	}
	return best, bestDepth, false
}

// resolveOnType dispatches a selector call on a value of known type
// t: a concrete type through its promoted method set, an interface
// through its sole constructor-built implementer. The second return
// reports whether the type proof is CONCLUSIVE — a target found, or
// the name provably uncallable on t (an ambiguous promotion, or an
// interface dispatch the model cannot represent). A conclusive
// answer forbids the name-based fallback in resolveCall: when the
// index knows what x holds, guessing by name would only add the
// wrong edges this machinery exists to remove. Not conclusive means
// the type says nothing about the name — the fallback may still
// know a pattern the types don't (and the compiler refinement, when
// it runs, outranks both).
func (p *Project) resolveOnType(t *Type, name string) (*Func, bool) {
	if t == nil {
		return nil, false
	}
	if t.Kind == "interface" {
		impl := p.soleCtorImpl(t)
		if impl == nil {
			return nil, false
		}
		return p.lookupMethod(impl, name), true
	}
	hit, ok := p.methodSet(t, true)[name]
	if !ok {
		return nil, false
	}
	return hit.fn, true // hit.fn may be nil: ambiguous — conclusively no target
}

// resolveCtx carries everything resolveCall needs about one function
// being scanned: the historical package-level shortcuts (varPkg,
// paramPkgs) plus the receiver types the sharper heuristics prove —
// concrete types for variables whose values flow from constructors,
// composite literals, or typed fields, and declared types for
// parameters, the receiver, and var declarations.
type resolveCtx struct {
	varPkg    map[string]string
	paramPkgs map[string]*Package
	concrete  map[string]*Type
	declared  map[string]*Type
}

// newResolveCtx builds the resolution context for fn in pkg. The
// body scan is flow-insensitive (last assignment wins), in keeping
// with the other heuristics; it folds the historical constructorVars
// detection into the same single pass.
func (p *Project) newResolveCtx(pkg *Package, fn *Func) *resolveCtx {
	ctx := &resolveCtx{
		varPkg:    map[string]string{},
		paramPkgs: map[string]*Package{},
		concrete:  map[string]*Type{},
		declared:  map[string]*Type{},
	}
	if fn.decl == nil || fn.decl.Body == nil {
		return ctx
	}
	ctx.paramPkgs = paramPackages(fn, pkg)
	if fn.decl.Recv != nil {
		if rt := p.typeByKey[typeKey{fn.PkgDir, fn.Receiver}]; rt != nil {
			for _, field := range fn.decl.Recv.List {
				for _, name := range field.Names {
					ctx.declared[name.Name] = rt
				}
			}
		}
	}
	if fn.decl.Type != nil && fn.decl.Type.Params != nil {
		for _, field := range fn.decl.Type.Params.List {
			if t := p.resolveTypeExpr(pkg, field.Type); t != nil {
				for _, name := range field.Names {
					ctx.declared[name.Name] = t
				}
			}
		}
	}
	ast.Inspect(fn.decl.Body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.AssignStmt:
			// The historical varPkg shortcut: s := store.New()
			// marks s as holding "something from store" even when
			// the constructor table cannot name the type.
			if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
				if lhs, ok := s.Lhs[0].(*ast.Ident); ok {
					if call, ok := s.Rhs[0].(*ast.CallExpr); ok {
						if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
							if id, ok := sel.X.(*ast.Ident); ok {
								if _, isAlias := pkg.aliasCache[id.Name]; isAlias {
									ctx.varPkg[lhs.Name] = id.Name
								}
							}
						}
					}
				}
			}
			p.scanAssign(pkg, ctx, s)
		case *ast.DeclStmt:
			if gd, ok := s.Decl.(*ast.GenDecl); ok {
				for _, spec := range gd.Specs {
					if vs, ok := spec.(*ast.ValueSpec); ok {
						p.scanVarSpec(pkg, ctx, vs)
					}
				}
			}
		}
		return true
	})
	return ctx
}

// typeOf reports the best known type of a local name: the concrete
// type it provably holds, else its declared type.
func (ctx *resolveCtx) typeOf(name string) *Type {
	if t := ctx.concrete[name]; t != nil {
		return t
	}
	return ctx.declared[name]
}

// scanAssign records the types an assignment statement proves:
// constructor calls and composite literals pin a concrete type,
// field reads pin the field's declared type, and plain identifier
// copies propagate what the source name is known to hold. The
// (T, error) constructor form assigns the value to the first LHS.
func (p *Project) scanAssign(pkg *Package, ctx *resolveCtx, s *ast.AssignStmt) {
	// Single call producing all LHS values: v, err := pkg.New().
	if len(s.Rhs) == 1 && len(s.Lhs) >= 1 {
		switch rhs := s.Rhs[0].(type) {
		case *ast.CallExpr:
			if ct := p.ctorForCall(pkg, rhs); ct != nil {
				if id, ok := s.Lhs[0].(*ast.Ident); ok {
					ctx.concrete[id.Name] = ct
				}
				return
			}
		case *ast.CompositeLit:
			if id, ok := s.Lhs[0].(*ast.Ident); ok {
				if t := p.resolveTypeExpr(pkg, rhs.Type); t != nil {
					ctx.concrete[id.Name] = t
				}
				return
			}
		}
	}
	if len(s.Lhs) != len(s.Rhs) {
		return
	}
	for i, lhs := range s.Lhs {
		id, ok := lhs.(*ast.Ident)
		if !ok {
			continue
		}
		p.scanValue(pkg, ctx, id.Name, s.Rhs[i])
	}
}

// scanValue records what one assigned expression proves about the
// name it lands in.
func (p *Project) scanValue(pkg *Package, ctx *resolveCtx, name string, rhs ast.Expr) {
	switch e := rhs.(type) {
	case *ast.CallExpr:
		if ct := p.ctorForCall(pkg, e); ct != nil {
			ctx.concrete[name] = ct
		}
	case *ast.CompositeLit:
		if t := p.resolveTypeExpr(pkg, e.Type); t != nil {
			ctx.concrete[name] = t
		}
	case *ast.UnaryExpr: // &T{...}
		if cl, ok := e.X.(*ast.CompositeLit); ok {
			if t := p.resolveTypeExpr(pkg, cl.Type); t != nil {
				ctx.concrete[name] = t
			}
		}
	case *ast.SelectorExpr:
		if t := p.selectorType(pkg, ctx, e); t != nil {
			if t.Kind == "interface" {
				ctx.declared[name] = t
			} else {
				ctx.concrete[name] = t
			}
		}
	case *ast.Ident:
		if t := ctx.typeOf(e.Name); t != nil {
			if ctx.concrete[e.Name] != nil {
				ctx.concrete[name] = t
			} else {
				ctx.declared[name] = t
			}
		}
	}
}

// scanVarSpec records var declarations: the declared type, refined
// to a concrete type when the initializer is a constructor call or
// composite literal (var s Saver = store.New() holds a *Store).
func (p *Project) scanVarSpec(pkg *Package, ctx *resolveCtx, vs *ast.ValueSpec) {
	var declared *Type
	if vs.Type != nil {
		declared = p.resolveTypeExpr(pkg, vs.Type)
	}
	for i, name := range vs.Names {
		if declared != nil {
			ctx.declared[name.Name] = declared
		}
		if i < len(vs.Values) {
			p.scanValue(pkg, ctx, name.Name, vs.Values[i])
		}
	}
	// var a, b = f() form: one call, several names — the (T, error)
	// constructor shape lands its value in the first name.
	if len(vs.Values) == 1 && len(vs.Names) > 1 {
		if call, ok := vs.Values[0].(*ast.CallExpr); ok {
			if ct := p.ctorForCall(pkg, call); ct != nil {
				ctx.concrete[vs.Names[0].Name] = ct
			}
		}
	}
}

// ctorForCall returns the struct type a call provably constructs
// when its callee is in the constructor table, else nil.
func (p *Project) ctorForCall(pkg *Package, call *ast.CallExpr) *Type {
	var callee *Func
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		callee = p.findFunc(pkg, "", fun.Name)
	case *ast.SelectorExpr:
		if id, ok := fun.X.(*ast.Ident); ok {
			if q, ok := pkg.aliasCache[id.Name]; ok {
				callee = p.findFunc(q, "", fun.Sel.Name)
			}
		}
	}
	if callee == nil {
		return nil
	}
	return p.ctorTable[callee.ID]
}

// selectorType resolves the type of a selector expression a.Field
// (or a longer chain rooted at an identifier) through the context's
// known types and the index's field tables.
func (p *Project) selectorType(pkg *Package, ctx *resolveCtx, sel *ast.SelectorExpr) *Type {
	var base *Type
	switch x := sel.X.(type) {
	case *ast.Ident:
		base = ctx.typeOf(x.Name)
	case *ast.SelectorExpr:
		base = p.selectorType(pkg, ctx, x)
	}
	if base == nil {
		return nil
	}
	return p.fieldTypeOf(base, sel.Sel.Name)
}
