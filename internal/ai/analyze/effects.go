package analyze

import (
	"bytes"
	"go/ast"
	"go/printer"
	"go/token"
	"sort"
	"strings"
)

// mutatingCalls lists external package-qualified calls that change the
// outside world (the filesystem). Curated for what CLI tools actually
// use; the heuristic over-approximates on purpose (os.OpenFile may open
// read-only, but a dry-run flag should still know about it).
var mutatingCalls = map[string]bool{
	"os.Remove":     true,
	"os.RemoveAll":  true,
	"os.WriteFile":  true,
	"os.Create":     true,
	"os.CreateTemp": true,
	"os.OpenFile":   true,
	"os.Rename":     true,
	"os.Mkdir":      true,
	"os.MkdirAll":   true,
	"os.Chmod":      true,
	"os.Chown":      true,
	"os.Lchown":     true,
	"os.Chtimes":    true,
	"os.Truncate":   true,
	"os.Symlink":    true,
	"os.Link":       true,

	"io/ioutil.WriteFile": true,
	"io/ioutil.TempFile":  true,
	"io/ioutil.TempDir":   true,
}

// Guard is one if-statement and the calls its body makes. It answers
// "what does this condition gate" — the shape of feature flags,
// dry-run modes, and verbose switches.
type Guard struct {
	Cond     string   // short rendered condition, e.g. "cfg.DryRun"
	Line     int      // line of the if statement
	Calls    []string // project callee IDs in the guarded body
	ExtCalls []string // external calls in the guarded body, e.g. "os.Remove"
	FlagLike bool     // the condition reads like an option/flag check
}

// mutation is one project function that directly mutates the filesystem
// and the external calls it mutates through.
type mutation struct {
	FuncName string   // e.g. "DeleteFile"
	Via      []string // e.g. ["os.Remove"]
	File     string   // relative to Root
	Line     int
}

// extCallKey maps a call like os.Remove(x) to "os.Remove" when the
// qualifier is an imported external package. Returns "" for project
// packages, unqualified calls, and unknown qualifiers. Dot-imports and
// explicit aliases ("o \"os\"") are not resolved — accepted heuristic gap.
func extCallKey(pkg *Package, call *ast.CallExpr) string {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	id, ok := sel.X.(*ast.Ident)
	if !ok {
		return ""
	}
	if _, isProj := pkg.aliasCache[id.Name]; isProj {
		return ""
	}
	for _, imp := range pkg.Imports {
		if importBase(imp) == id.Name {
			return id.Name + "." + sel.Sel.Name
		}
	}
	return ""
}

// analyzeEffects records a function's external calls, whether it mutates
// the filesystem, and the if-statement guards in its body. Called from
// BuildGraph while decls are still available.
func (p *Project) analyzeEffects(pkg *Package, fn *Func) {
	if fn.decl == nil || fn.decl.Body == nil {
		return
	}
	varPkg := constructorVars(fn.decl.Body, pkg)
	paramPkgs := paramPackages(fn, pkg)

	extSet := map[string]bool{}
	ast.Inspect(fn.decl.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if p.resolveCall(pkg, varPkg, paramPkgs, fn, call) != nil {
			return true
		}
		if key := extCallKey(pkg, call); key != "" {
			extSet[key] = true
		}
		return true
	})
	fn.ExtCalls = sortedKeys(extSet)
	for _, e := range fn.ExtCalls {
		if mutatingCalls[e] {
			fn.Mutates = true
			break
		}
	}
	p.collectGuards(pkg, fn, varPkg, paramPkgs)
}

// propagateEffects fills MutatesAll: a function transitively mutates when
// it or any of its callees directly mutates. Fixpoint over the call graph.
func (p *Project) propagateEffects() {
	for changed := true; changed; {
		changed = false
		for _, fn := range p.byID {
			if fn.MutatesAll {
				continue
			}
			if fn.Mutates {
				fn.MutatesAll = true
				changed = true
				continue
			}
			for _, id := range fn.Calls {
				if callee := p.byID[id]; callee != nil && callee.MutatesAll {
					fn.MutatesAll = true
					changed = true
					break
				}
			}
		}
	}
}

// collectGuards walks a function's if-statements, recording each
// condition and the project/external calls its body makes. A guard
// clause (if with no else whose body returns) also gates every
// statement after it in the block: `if cfg.DryRun { return }` guards
// the os.Remove below it.
func (p *Project) collectGuards(pkg *Package, fn *Func, varPkg map[string]string, paramPkgs map[string]*Package) {
	paramNames := map[string]bool{}
	if fn.decl.Type != nil && fn.decl.Type.Params != nil {
		for _, field := range fn.decl.Type.Params.List {
			for _, name := range field.Names {
				paramNames[name.Name] = true
			}
		}
	}
	addCalls := func(node ast.Node, callSet, extSet map[string]bool) {
		ast.Inspect(node, func(m ast.Node) bool {
			call, ok := m.(*ast.CallExpr)
			if !ok {
				return true
			}
			if target := p.resolveCall(pkg, varPkg, paramPkgs, fn, call); target != nil {
				callSet[target.ID] = true
			} else if key := extCallKey(pkg, call); key != "" {
				extSet[key] = true
			}
			return true
		})
	}
	var walkStmts func(stmts []ast.Stmt)
	walkStmts = func(stmts []ast.Stmt) {
		for i, stmt := range stmts {
			ifStmt, ok := stmt.(*ast.IfStmt)
			if !ok {
				for _, sub := range childStmtLists(stmt) {
					walkStmts(sub)
				}
				continue
			}
			g := Guard{Line: p.lineOf(fn, ifStmt), FlagLike: flagLikeCond(ifStmt.Cond, paramNames, paramPkgs)}
			g.Cond = shortCond(ifStmt.Cond)
			callSet, extSet := map[string]bool{}, map[string]bool{}
			addCalls(ifStmt.Body, callSet, extSet)
			if ifStmt.Else != nil {
				addCalls(ifStmt.Else, callSet, extSet)
			} else if blockReturns(ifStmt.Body) {
				for _, rest := range stmts[i+1:] {
					addCalls(rest, callSet, extSet)
				}
			}
			// Subtract calls made by the condition itself.
			condSet, condExt := map[string]bool{}, map[string]bool{}
			addCalls(ifStmt.Cond, condSet, condExt)
			for id := range condSet {
				delete(callSet, id)
			}
			for e := range condExt {
				delete(extSet, e)
			}
			g.Calls = sortedKeys(callSet)
			g.ExtCalls = sortedKeys(extSet)
			fn.Guards = append(fn.Guards, g)
			// Recurse for nested ifs.
			walkStmts(ifStmt.Body.List)
			if ifStmt.Else != nil {
				if blk, ok := ifStmt.Else.(*ast.BlockStmt); ok {
					walkStmts(blk.List)
				} else {
					walkStmts([]ast.Stmt{ifStmt.Else})
				}
			}
		}
	}
	walkStmts(fn.decl.Body.List)
}

// blockReturns reports whether a block ends in a return statement.
func blockReturns(b *ast.BlockStmt) bool {
	if b == nil || len(b.List) == 0 {
		return false
	}
	_, ok := b.List[len(b.List)-1].(*ast.ReturnStmt)
	return ok
}

// childStmtLists returns nested statement lists for recursion into
// loops, switches, and selects.
func childStmtLists(stmt ast.Stmt) [][]ast.Stmt {
	var out [][]ast.Stmt
	switch s := stmt.(type) {
	case *ast.ForStmt:
		out = append(out, s.Body.List)
	case *ast.RangeStmt:
		out = append(out, s.Body.List)
	case *ast.SwitchStmt:
		for _, c := range s.Body.List {
			if cc, ok := c.(*ast.CaseClause); ok {
				out = append(out, cc.Body)
			}
		}
	case *ast.TypeSwitchStmt:
		for _, c := range s.Body.List {
			if cc, ok := c.(*ast.CaseClause); ok {
				out = append(out, cc.Body)
			}
		}
	case *ast.SelectStmt:
		for _, c := range s.Body.List {
			if cc, ok := c.(*ast.CommClause); ok {
				out = append(out, cc.Body)
			}
		}
	}
	return out
}

// flagLikeCond reports whether a condition reads like an option/flag
// check: a bare bool parameter (if dryRun) or a selector on a parameter
// typed by a project package (if cfg.DryRun). err != nil and friends
// don't match either shape.
func flagLikeCond(cond ast.Expr, paramNames map[string]bool, paramPkgs map[string]*Package) bool {
	for {
		switch c := cond.(type) {
		case *ast.ParenExpr:
			cond = c.X
		case *ast.UnaryExpr:
			cond = c.X
		default:
			goto done
		}
	}
done:
	switch c := cond.(type) {
	case *ast.Ident:
		return paramNames[c.Name]
	case *ast.SelectorExpr:
		if id, ok := c.X.(*ast.Ident); ok {
			_, isParam := paramPkgs[id.Name]
			return isParam || paramNames[id.Name]
		}
	}
	return false
}

// shortCond renders a condition as one short line for display.
func shortCond(cond ast.Expr) string {
	var buf bytes.Buffer
	fset := token.NewFileSet()
	if err := printer.Fprint(&buf, fset, cond); err != nil {
		return "?"
	}
	s := strings.Join(strings.Fields(buf.String()), " ")
	if len(s) > 48 {
		s = s[:45] + "..."
	}
	return s
}

// flowMutations lists the functions reachable from flow that directly
// mutate the filesystem, with the external calls they mutate through.
// Capped so the plan stays readable.
func (p *Project) flowMutations(flow *Func) []mutation {
	seen := map[string]bool{}
	var out []mutation
	var visit func(fn *Func)
	visit = func(fn *Func) {
		if fn == nil || seen[fn.ID] {
			return
		}
		seen[fn.ID] = true
		if fn.Mutates {
			var via []string
			for _, e := range fn.ExtCalls {
				if mutatingCalls[e] {
					via = append(via, e)
				}
			}
			sort.Strings(via)
			out = append(out, mutation{FuncName: fn.Name, Via: via, File: fn.File, Line: fn.Line})
		}
		for _, id := range fn.Calls {
			visit(p.byID[id])
		}
	}
	for _, id := range flow.Calls {
		visit(p.byID[id])
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].FuncName != out[j].FuncName {
			return out[i].FuncName < out[j].FuncName
		}
		return out[i].File < out[j].File
	})
	if len(out) > 4 {
		out = out[:4]
	}
	return out
}

func sortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// lineOf returns the 1-based line of a node using the FileSet kept from
// the scan phase (one shared FileSet per analyzed project).
func (p *Project) lineOf(fn *Func, n ast.Node) int {
	if fn.fset == nil {
		return 0
	}
	if pos := fn.fset.Position(n.Pos()); pos.IsValid() {
		return pos.Line
	}
	return 0
}
