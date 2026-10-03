package analyze

import (
	"bytes"
	"go/ast"
	"go/printer"
	"go/token"
	"sort"
	"strings"
)

// Effect categories. callEffects is declarative knowledge about the
// stdlib (finite, stable): which external calls have which effects. All
// reasoning over it — transitive propagation, guard detection — is
// fully general and feature-agnostic.
const (
	fxWrite = "fs-write" // changes files: os.Remove, os.WriteFile, ...
	fxRead  = "fs-read"  // reads files: os.ReadFile, os.Stat, ...
	fxNet   = "net"      // network I/O: http.Get, net.Dial, ...
)

var callEffects = map[string][]string{
	"os.Remove": {fxWrite}, "os.RemoveAll": {fxWrite},
	"os.WriteFile": {fxWrite}, "os.Create": {fxWrite},
	"os.CreateTemp": {fxWrite}, "os.OpenFile": {fxWrite},
	"os.Rename": {fxWrite}, "os.Mkdir": {fxWrite},
	"os.MkdirAll": {fxWrite}, "os.Chmod": {fxWrite},
	"os.Chown": {fxWrite}, "os.Lchown": {fxWrite},
	"os.Chtimes": {fxWrite}, "os.Truncate": {fxWrite},
	"os.Symlink": {fxWrite}, "os.Link": {fxWrite},
	"ioutil.WriteFile": {fxWrite}, "ioutil.TempFile": {fxWrite},
	"ioutil.TempDir": {fxWrite},

	"os.ReadFile": {fxRead}, "os.Open": {fxRead},
	"os.Stat": {fxRead}, "os.Lstat": {fxRead},
	"os.ReadDir": {fxRead}, "ioutil.ReadFile": {fxRead},

	"net.Dial": {fxNet}, "net.DialTimeout": {fxNet},
	"http.Get": {fxNet}, "http.Post": {fxNet}, "http.Head": {fxNet},
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

// CallArg is one call site: the resolved target ("" when external or
// unresolved) and the identifier arguments passed.
type CallArg struct {
	Target string
	Args   []string
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

// paramTypeName renders a parameter type as "pkg.Type".
func paramTypeName(pkg *Package, e ast.Expr) string {
	for {
		if s, ok := e.(*ast.StarExpr); ok {
			e = s.X
			continue
		}
		break
	}
	switch t := e.(type) {
	case *ast.SelectorExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name + "." + t.Sel.Name
		}
	case *ast.Ident:
		return pkg.Name + "." + t.Name
	}
	return ""
}

// compositeTypeName renders the type of a composite literal expression:
// &config.Config{...} or config.Config{...} -> "config.Config".
func compositeTypeName(pkg *Package, e ast.Expr) string {
	if u, ok := e.(*ast.UnaryExpr); ok && u.Op == token.AND {
		e = u.X
	}
	if cl, ok := e.(*ast.CompositeLit); ok {
		return paramTypeName(pkg, cl.Type)
	}
	return ""
}

// rhsValue resolves the value of an assignment RHS to a "pkg.Type.Field"
// when it's a field read (cfg.DryRun) or a chained alias (x := dry).
// Anything else clears the alias.
func rhsValue(e ast.Expr, typeName func(string) string, aliasOf map[string]string) string {
	switch e := e.(type) {
	case *ast.SelectorExpr:
		if id, ok := e.X.(*ast.Ident); ok {
			if t := typeName(id.Name); t != "" {
				return t + "." + e.Sel.Name
			}
		}
	case *ast.Ident:
		return aliasOf[e.Name]
	}
	return ""
}

// analyzeEffects records a function's external calls, call-site
// arguments, value reads, and if-statement guards. Called from
// BuildGraph while decls are still available. Everything recorded here
// is feature-agnostic: effects, data flow, and guards describe the code
// itself, not any particular issue.
func (p *Project) analyzeEffects(pkg *Package, fn *Func) {
	if fn.decl == nil || fn.decl.Body == nil {
		return
	}
	varPkg := constructorVars(fn.decl.Body, pkg)
	paramPkgs := paramPackages(fn, pkg)

	// Parameter types ("config.Config") and names, receiver included.
	paramTypes := map[string]string{}
	paramNames := map[string]bool{}
	addParam := func(name string, typ ast.Expr) {
		paramNames[name] = true
		if tn := paramTypeName(pkg, typ); tn != "" {
			paramTypes[name] = tn
		}
	}
	if fn.decl.Type != nil {
		if fn.decl.Type.Params != nil {
			for _, field := range fn.decl.Type.Params.List {
				for _, name := range field.Names {
					addParam(name.Name, field.Type)
				}
			}
		}
		if fn.decl.Recv != nil {
			for _, field := range fn.decl.Recv.List {
				for _, name := range field.Names {
					addParam(name.Name, field.Type)
				}
			}
		}
	}
	fn.ParamTypes = paramTypes

	// Local variable types: cfg := &config.Config{...} / var cfg config.Config.
	// Without these, value flow stops at main — most values start as locals.
	localTypes := map[string]string{}
	ast.Inspect(fn.decl.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.AssignStmt:
			if n.Tok != token.DEFINE || len(n.Lhs) != 1 || len(n.Rhs) != 1 {
				return true
			}
			if id, ok := n.Lhs[0].(*ast.Ident); ok {
				if tn := compositeTypeName(pkg, n.Rhs[0]); tn != "" {
					localTypes[id.Name] = tn
				}
			}
		case *ast.DeclStmt:
			if gd, ok := n.Decl.(*ast.GenDecl); ok && gd.Tok == token.VAR {
				for _, spec := range gd.Specs {
					if vs, ok := spec.(*ast.ValueSpec); ok && vs.Type != nil {
						for _, name := range vs.Names {
							if tn := paramTypeName(pkg, vs.Type); tn != "" {
								localTypes[name.Name] = tn
							}
						}
					}
				}
			}
		}
		return true
	})
	fn.localTypes = localTypes

	// Aliases: dry := cfg.DryRun makes dry an alias of
	// "config.Config.DryRun" until reassigned. Guards on aliases are
	// flag checks too — the model must see through the local.
	aliasOf := map[string]string{}
	typeName := func(name string) string {
		if t := paramTypes[name]; t != "" {
			return t
		}
		return localTypes[name]
	}
	ast.Inspect(fn.decl.Body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.AssignStmt:
			if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
				if id, ok := s.Lhs[0].(*ast.Ident); ok {
					if v := rhsValue(s.Rhs[0], typeName, aliasOf); v != "" {
						aliasOf[id.Name] = v
					} else {
						delete(aliasOf, id.Name)
					}
					break
				}
			}
			for _, lhs := range s.Lhs {
				if id, ok := lhs.(*ast.Ident); ok {
					delete(aliasOf, id.Name)
				}
			}
		case *ast.IncDecStmt:
			if id, ok := s.X.(*ast.Ident); ok {
				delete(aliasOf, id.Name)
			}
		case *ast.RangeStmt:
			// for k, v := range — the loop variables are fresh aliases of nothing.
			if id, ok := s.Key.(*ast.Ident); ok {
				delete(aliasOf, id.Name)
			}
			if id, ok := s.Value.(*ast.Ident); ok {
				delete(aliasOf, id.Name)
			}
		}
		return true
	})

	// Call targets that are bare selectors (x.Foo()) so the read pass
	// below doesn't mistake method calls for field reads.
	callFuns := map[ast.Node]bool{}

	// Selectors on the LHS of a pure `=` assignment are writes, not
	// reads: `config.ShowProgress = *progress` must not make GetFlags a
	// "reader" of ShowProgress.
	writeSel := map[ast.Node]bool{}
	ast.Inspect(fn.decl.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || as.Tok != token.ASSIGN {
			return true
		}
		for _, lhs := range as.Lhs {
			if sel, ok := lhs.(*ast.SelectorExpr); ok {
				writeSel[sel] = true
			}
		}
		return true
	})
	ast.Inspect(fn.decl.Body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			callFuns[call.Fun] = true
		}
		return true
	})

	extSet := map[string]bool{}
	readSet := map[string]bool{}
	writeSet := map[string]bool{}
	ast.Inspect(fn.decl.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.CallExpr:
			var args []string
			for _, a := range n.Args {
				if id, ok := a.(*ast.Ident); ok {
					args = append(args, id.Name)
				}
			}
			target := p.resolveCall(pkg, varPkg, paramPkgs, fn, n)
			tid := ""
			if target != nil {
				tid = target.ID
			} else if key := extCallKey(pkg, n); key != "" {
				extSet[key] = true
			}
			fn.CallArgs = append(fn.CallArgs, CallArg{Target: tid, Args: args})
		case *ast.SelectorExpr:
			// x.Y where x is a parameter or local: a read of "pkg.Type.Field".
			// Skip method calls (x.Foo()) — those are calls, not reads —
			// and pure-`=` writes (x.Y = ...), which go to the writes index.
			if callFuns[n] {
				return true
			}
			if id, ok := n.X.(*ast.Ident); ok {
				if t := valueTypeOf(fn, id.Name); t != "" {
					if writeSel[n] {
						writeSet[t+"."+n.Sel.Name] = true
					} else {
						readSet[t+"."+n.Sel.Name] = true
					}
				}
			}
		}
		return true
	})
	fn.ExtCalls = sortedKeys(extSet)
	fn.Reads = sortedKeys(readSet)
	fn.Writes = sortedKeys(writeSet)
	p.collectGuards(pkg, fn, varPkg, paramPkgs, paramNames, aliasOf)
}

// propagateEffects fills EffectsAll (transitive effect categories) by
// fixpoint over the call graph, then derives Mutates/MutatesAll from the
// fs-write category.
func (p *Project) propagateEffects() {
	for _, fn := range p.byID {
		set := map[string]bool{}
		for _, e := range fn.ExtCalls {
			for _, c := range callEffects[e] {
				set[c] = true
			}
		}
		fn.Effects = sortedKeys(set)
	}
	for changed := true; changed; {
		changed = false
		for _, fn := range p.byID {
			set := map[string]bool{}
			for _, c := range fn.EffectsAll {
				set[c] = true
			}
			for _, c := range fn.Effects {
				if !set[c] {
					set[c] = true
					changed = true
				}
			}
			for _, id := range fn.Calls {
				if g := p.byID[id]; g != nil {
					for _, c := range g.EffectsAll {
						if !set[c] {
							set[c] = true
							changed = true
						}
					}
				}
			}
			fn.EffectsAll = sortedKeys(set)
		}
	}
	for _, fn := range p.byID {
		fn.Mutates = hasEffect(fn.Effects, fxWrite)
		fn.MutatesAll = hasEffect(fn.EffectsAll, fxWrite)
	}
}

func hasEffect(cats []string, want string) bool {
	for _, c := range cats {
		if c == want {
			return true
		}
	}
	return false
}

// collectGuards walks a function's if-statements, recording each
// condition and the project/external calls its body makes. A guard
// clause (if with no else whose body returns) also gates every
// statement after it in the block: `if cfg.DryRun { return }` guards
// the os.Remove below it.
func (p *Project) collectGuards(pkg *Package, fn *Func, varPkg map[string]string, paramPkgs map[string]*Package, paramNames map[string]bool, aliasOf map[string]string) {
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
			g := Guard{Line: p.lineOf(fn, ifStmt), FlagLike: flagLikeCond(ifStmt.Cond, paramNames, paramPkgs, aliasOf)}
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
// check: a bare bool parameter (if dryRun), a selector on a parameter
// typed by a project package (if cfg.DryRun), or a local alias of such
// a value (dry := cfg.DryRun; if dry). err != nil and friends don't
// match any shape.
func flagLikeCond(cond ast.Expr, paramNames map[string]bool, paramPkgs map[string]*Package, aliasOf map[string]string) bool {
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
		return paramNames[c.Name] || aliasOf[c.Name] != ""
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
				if hasEffect(callEffects[e], fxWrite) {
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

// flowFlagSites returns the flag-like guard sites in the flow function
// itself, deduplicated by condition. These are the local pattern a new
// flag should imitate: same shape, same neighborhood.
func (p *Project) flowFlagSites(flow *Func) []flagSite {
	seen := map[string]bool{}
	var out []flagSite
	for _, g := range flow.Guards {
		if !g.FlagLike || seen[g.Cond] {
			continue
		}
		seen[g.Cond] = true
		out = append(out, flagSite{Cond: g.Cond, File: flow.File, Line: g.Line})
		if len(out) >= 3 {
			break
		}
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
