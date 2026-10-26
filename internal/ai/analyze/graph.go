package analyze

import (
	"go/ast"
	"sort"
	"strings"
)

// BuildGraph resolves call edges between the project's functions and fills
// in each Func's Callers list. Resolution is heuristic: selector calls
// dispatch through the receiver types the index can prove (constructor
// products, declared types, promoted method sets — methodset.go) with a
// name-based fallback chain, plain calls resolve by name within the
// package, and qualified pkg.Func() calls go through the file's imports
// when the target package is part of the project. Calls into external
// dependencies are ignored — they don't help map the project's own
// structure.
func (p *Project) BuildGraph() {
	// importAlias maps a qualifier used in selectors (package name or
	// explicit import alias) to the project package it refers to.
	for _, pkg := range p.Packages {
		alias := map[string]*Package{}
		for _, imp := range pkg.Imports {
			if target := p.importTarget(imp); target != nil {
				alias[importBase(imp)] = target
			}
		}
		// Also honor explicit aliases: import foo "example.com/bar".
		pkg.aliasCache = alias
	}
	// Constructor table, type index, and own-method lists for the
	// sharper receiver resolution (methodset.go).
	p.buildCallIndex()
	// Variable data flow: the name index and the sites hidden in
	// package-level initializers, now that alias caches and all
	// declarations exist.
	p.buildVarIndex()
	p.sweepVarInits()
	seen := map[[2]string]bool{}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.decl == nil || fn.decl.Body == nil {
				continue
			}
			ctx := p.newResolveCtx(pkg, fn)
			fn.ctx = ctx // shared with analyzeEffects below; dropped in finalize
			loops := loopRanges(fn.decl.Body)
			ast.Inspect(fn.decl.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				target := p.resolveCall(pkg, ctx, fn, call)
				if target != nil {
					fn.Calls = append(fn.Calls, target.ID)
					if inLoopRange(call, loops) {
						if fn.LoopCalls == nil {
							fn.LoopCalls = map[string]bool{}
						}
						fn.LoopCalls[target.ID] = true
					}
					key := [2]string{fn.ID, target.ID}
					if !seen[key] {
						seen[key] = true
						target.Callers = append(target.Callers, fn.ID)
					}
				}
				return true
			})
			p.analyzeEffects(pkg, fn)
			p.collectVarEvents(pkg, fn)
		}
	}
	// Compiler-grade pass: resolve selector calls through real method
	// sets before effects propagate, so refined edges carry effects.
	if !p.noRefine {
		p.refineTypes()
	}
	p.propagateEffects()
	p.buildFieldReaders()
	p.buildFieldWriters()
	for _, fn := range p.byID {
		sort.Strings(fn.Calls)
		sort.Strings(fn.Callers)
	}
	p.finalize()
}

// resolveCall maps a call expression to a project function, or nil when
// it targets an external package or can't be resolved. Shared by the
// call-graph pass and the effect/guard collection so both see the same
// edges.
//
// Selector calls first try the proven receiver paths from
// methodset.go: when the index knows what x holds — a constructor's
// product, a composite literal, a declared parameter/receiver/field
// type — the call dispatches through that type's promoted method
// set (or, for an interface, its sole constructor-built
// implementer). Only when those prove nothing does the historical
// name-based chain run: a same-package method named Foo; then the
// constructor-package pattern (s := store.New(); s.Foo() resolves
// Foo in store); then a parameter typed by a project package (fm
// filemanager.FileManager lets fm.DeleteFile() resolve); then
// pkg.Foo() through imports.
func (p *Project) resolveCall(pkg *Package, ctx *resolveCtx, fn *Func, call *ast.CallExpr) *Func {
	var target *Func
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		// Plain foo(): same-package function.
		target = p.findFunc(pkg, "", fun.Name)
	case *ast.SelectorExpr:
		if id, ok := fun.X.(*ast.Ident); ok {
			conclusive := false
			if t := ctx.typeOf(id.Name); t != nil {
				target, conclusive = p.resolveOnType(t, fun.Sel.Name)
			}
			if !conclusive && target == nil {
				target = p.findMethod(pkg, fun.Sel.Name)
				if target == nil {
					if alias, ok := ctx.varPkg[id.Name]; ok {
						if q, ok := pkg.aliasCache[alias]; ok {
							target = p.findMethod(q, fun.Sel.Name)
						}
					} else if q, ok := ctx.paramPkgs[id.Name]; ok {
						target = p.findMethod(q, fun.Sel.Name)
					} else if q, ok := pkg.aliasCache[id.Name]; ok {
						target = p.findFunc(q, "", fun.Sel.Name)
					}
				}
			}
		} else if sel, ok := fun.X.(*ast.SelectorExpr); ok {
			// a.Field.Foo(): dispatch on the field's declared type.
			if t := p.selectorType(pkg, ctx, sel); t != nil {
				target, _ = p.resolveOnType(t, fun.Sel.Name)
			}
		}
	}
	if target == fn {
		return nil
	}
	return target
}

// importTarget returns the project package for an import path, or nil for
// external dependencies. Matching is by module-relative directory: an
// import of <module>/a/b resolves to the package in dir a/b.
func (p *Project) importTarget(imp string) *Package {
	if p.Module != "" {
		if rest, ok := strings.CutPrefix(imp, p.Module+"/"); ok {
			if pkg, ok := p.byPkgDir[rest]; ok {
				return pkg
			}
		}
		if imp == p.Module {
			if pkg, ok := p.byPkgDir[""]; ok {
				return pkg
			}
		}
	}
	// No go.mod (or non-module import): match by directory tail.
	for _, pkg := range p.Packages {
		dir := pkg.Dir
		if dir == "" {
			continue
		}
		if strings.HasSuffix(imp, "/"+dir) || imp == dir {
			return pkg
		}
	}
	return nil
}

// importBase is the qualifier a selector would use: the last path element.
func importBase(imp string) string {
	if i := strings.LastIndex(imp, "/"); i >= 0 {
		return imp[i+1:]
	}
	return imp
}

// findFunc looks up a plain function by name in pkg.
func (p *Project) findFunc(pkg *Package, recv, name string) *Func {
	for _, fn := range pkg.Funcs {
		if fn.Receiver == recv && fn.Name == name {
			return fn
		}
	}
	return nil
}

// loopRanges returns the [pos, end) offsets of for/range loop bodies,
// so calls made per-iteration (the per-item hooks) can be told apart
// from calls evaluated once like the range expression itself.
func loopRanges(body *ast.BlockStmt) [][2]int {
	var ranges [][2]int
	ast.Inspect(body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.ForStmt:
			ranges = append(ranges, [2]int{int(x.Body.Pos()), int(x.Body.End())})
		case *ast.RangeStmt:
			ranges = append(ranges, [2]int{int(x.Body.Pos()), int(x.Body.End())})
		}
		return true
	})
	return ranges
}

func inLoopRange(call *ast.CallExpr, loops [][2]int) bool {
	pos, end := int(call.Pos()), int(call.End())
	for _, r := range loops {
		if pos >= r[0] && end <= r[1] {
			return true
		}
	}
	return false
}

// paramPackages maps parameter names to project packages via their
// declared types: RunCLI(fm filemanager.FileManager, ...) lets
// fm.DeleteFile() resolve into the filemanager package. Unwraps pointer
// types; only parameters typed as pkg.Type qualify.
func paramPackages(fn *Func, pkg *Package) map[string]*Package {
	out := map[string]*Package{}
	if fn.decl == nil || fn.decl.Type == nil || fn.decl.Type.Params == nil {
		return out
	}
	for _, field := range fn.decl.Type.Params.List {
		typ := field.Type
		if star, ok := typ.(*ast.StarExpr); ok {
			typ = star.X
		}
		sel, ok := typ.(*ast.SelectorExpr)
		if !ok {
			continue
		}
		id, ok := sel.X.(*ast.Ident)
		if !ok {
			continue
		}
		if q, ok := pkg.aliasCache[id.Name]; ok {
			for _, name := range field.Names {
				out[name.Name] = q
			}
		}
	}
	return out
}

// findMethod looks up a method by name in pkg, preferring an exact
// receiver-type match is impossible without type info, so the first
// same-package method with that name wins. Method names are usually
// unique enough within a package for mapping purposes.
func (p *Project) findMethod(pkg *Package, name string) *Func {
	for _, m := range p.byMethod[name] {
		if m.pkg == pkg {
			return m
		}
	}
	return nil
}

// Rank scores every function by structural importance: how many distinct
// project functions call it, with bonuses for being exported, an entry
// point, or part of package main, and a penalty for test helpers.
func (p *Project) Rank() {
	for _, fn := range p.byID {
		s := 2 * len(fn.Callers)
		if fn.Exported {
			s += 3
		}
		if fn.IsMain {
			s += 10
		}
		if fn.IsInit {
			s += 1
		}
		if fn.Pkg == "main" {
			s += 2
		}
		if fn.Receiver != "" && fn.Exported {
			s += 1
		}
		if fn.IsTest {
			s -= 5
		}
		fn.Score = s
	}
	// Interfaces: count heuristic implementers (types in the project that
	// declare every method of the interface).
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Kind != "interface" || len(t.Methods) == 0 {
				continue
			}
			need := map[string]bool{}
			for _, m := range t.Methods {
				need[m] = true
			}
			for _, other := range pkg.Types {
				if other == t || other.Kind != "struct" {
					continue
				}
				have := map[string]bool{}
				for _, m := range p.methodsOf(other) {
					have[m] = true
				}
				ok := true
				for m := range need {
					if !have[m] {
						ok = false
						break
					}
				}
				if ok {
					t.ImplCount++
				}
			}
		}
	}
}

// methodsOf returns the method names declared on a type.
func (p *Project) methodsOf(t *Type) []string {
	var out []string
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.Receiver == t.Name {
				out = append(out, fn.Name)
			}
		}
	}
	return out
}

// TopFuncs returns the n highest-scoring non-test functions.
func (p *Project) TopFuncs(n int) []*Func {
	var all []*Func
	for _, fn := range p.byID {
		if !fn.IsTest {
			all = append(all, fn)
		}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].Score != all[j].Score {
			return all[i].Score > all[j].Score
		}
		return all[i].ID < all[j].ID
	})
	if len(all) > n {
		all = all[:n]
	}
	return all
}

// EntryPoints returns all func main declarations, ordered by package dir.
func (p *Project) EntryPoints() []*Func {
	var out []*Func
	for _, fn := range p.byID {
		if fn.IsMain {
			out = append(out, fn)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].File < out[j].File })
	return out
}

// KeyTypes returns exported struct/interface types, interfaces with
// implementers first, then by method count.
func (p *Project) KeyTypes(n int) []*Type {
	var out []*Type
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Exported && (t.Kind == "struct" || t.Kind == "interface") {
				out = append(out, t)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ImplCount != out[j].ImplCount {
			return out[i].ImplCount > out[j].ImplCount
		}
		if len(out[i].Methods) != len(out[j].Methods) {
			return len(out[i].Methods) > len(out[j].Methods)
		}
		return out[i].Name < out[j].Name
	})
	if len(out) > n {
		out = out[:n]
	}
	return out
}
