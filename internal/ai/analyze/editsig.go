package analyze

// Signature changes: the Composer's first multi-file shape whose
// plan is an impact manifest. Renaming a parameter, reordering
// parameters, and adding a parameter all invalidate every call
// site of the target function across the workspace, so the sites
// are never guessed from the heuristic call graph alone: like
// rename, this planner loads the whole module with go/packages +
// go/types and works from object identity — every call site is a
// use of the target's *types.Func, every parameter use is a use of
// its *types.Var. The graph's caller list is then used as a
// cross-check: a caller the index believes in but the typed load
// cannot locate exactly is a refusal, never a silent miss.
//
// Every change is a position-based text edit on the original
// bytes — argument and parameter spans are permuted or spliced,
// never re-printed from an AST — so untouched regions stay
// byte-identical and the diff shown at the gate is the diff that
// lands. The boundaries are drawn tight on purpose:
//
//   - A method whose signature an interface dictates is refused:
//     changing one implementer alone breaks the contract. (A
//     parameter RENAME is exempt — names are not part of a type,
//     so no interface can dictate them.)
//   - Any use of the function as a value (assigned, passed,
//     returned) refuses the whole plan: its type is consumed
//     there, and no call-site edit can prove that safe.
//   - Adding a parameter needs a derivable argument at EVERY
//     call site — the caller's own in-scope context, or the
//     user's expression verbatim where it provably resolves.
//     One failing site refuses the whole plan, sites named.
//
// Planning only, like edits.go: nothing here writes to disk.

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/go/packages"
)

// loadForSignature loads the whole module with type info, test
// files included — the same load loadForRename performs, with
// signature wording. Any type error in a module package is a
// refusal: sites are only provably complete when the type
// checker saw everything.
func (p *Project) loadForSignature() ([]*packages.Package, string) {
	if p.Module == "" {
		return nil, "this project has no go.mod, so I can't load it with the compiler's type information — changing a signature safely means finding every call site exactly, and that takes type info."
	}
	cfg := &packages.Config{
		Mode:  packages.NeedName | packages.NeedFiles | packages.NeedSyntax | packages.NeedTypes | packages.NeedTypesInfo | packages.NeedImports | packages.NeedDeps,
		Dir:   p.Root,
		Tests: true,
	}
	pkgs, err := packages.Load(cfg, "./...")
	if err != nil {
		return nil, fmt.Sprintf("I couldn't load the project with type information (%v), so I can't change a signature safely.", err)
	}
	for _, pkg := range pkgs {
		if !p.inModulePath(pkg.PkgPath) {
			continue
		}
		for _, e := range pkg.Errors {
			return nil, fmt.Sprintf("The project doesn't type-check right now — %s. Fix that first; changing a signature needs a clean type-check so every call site can be found exactly.", e.Msg)
		}
	}
	return pkgs, ""
}

// sigParam is one parameter (or named result) of the target, in
// source order, with the offsets of its name in the declaration
// file so edits stay textual.
type sigParam struct {
	name      string
	typ       string // the type's source text
	tuple     *types.Var
	nameStart int // byte offsets in the declaration file; -1 when unnamed
	nameEnd   int
}

// sigTarget is the resolved target function: its index entry, its
// typed object, its declaration, and its expanded parameter list.
type sigTarget struct {
	fn       *Func
	obj      *types.Func
	key      objKey
	sig      *types.Signature
	declPkg  *packages.Package
	decl     *ast.FuncDecl
	declAST  *ast.File // the declaration's file node in the load
	declFile string    // == fn.File
	declLine int
	params   []sigParam
	results  []sigParam
	recvName string
	recvBase string
}

func sigDriftRefusal(fn *Func) string {
	return fmt.Sprintf("I found %s in my index but not in the type-checked project — the code may have changed since I read it. Analyze the project again and retry.", fn.Display())
}

// findSigTarget resolves the index entry to its typed object and
// declaration inside the loaded module, and expands the parameter
// list (grouped declarations become one entry per name).
func (p *Project) findSigTarget(pkgs []*packages.Package, fn *Func, src []byte) (*sigTarget, string) {
	wantPath := p.pkgPathOf(fn.PkgDir)
	var instances []*packages.Package
	for _, pkg := range pkgs {
		if pkg.PkgPath == wantPath && pkg.Types != nil {
			instances = append(instances, pkg)
		}
	}
	var obj *types.Func
	for _, pkg := range instances {
		scope := pkg.Types.Scope()
		if fn.Receiver == "" {
			if f, ok := scope.Lookup(fn.Name).(*types.Func); ok {
				obj = f
				break
			}
			continue
		}
		tn, ok := scope.Lookup(recvBaseName(fn.Receiver)).(*types.TypeName)
		if !ok {
			continue
		}
		named, ok := tn.Type().(*types.Named)
		if !ok {
			continue
		}
		for i := 0; i < named.NumMethods(); i++ {
			if m := named.Method(i); m.Name() == fn.Name {
				obj = m
				break
			}
		}
		if obj != nil {
			break
		}
	}
	if obj == nil {
		return nil, sigDriftRefusal(fn)
	}
	key, ok := keyOfTypesObj(obj)
	if !ok {
		return nil, sigDriftRefusal(fn)
	}
	var decl *ast.FuncDecl
	var declPkg *packages.Package
	for _, pkg := range instances {
		if pkg.TypesInfo == nil {
			continue
		}
		for _, file := range pkg.Syntax {
			for _, d := range file.Decls {
				if fd, ok := d.(*ast.FuncDecl); ok && pkg.TypesInfo.Defs[fd.Name] == obj {
					decl, declPkg = fd, pkg
				}
			}
		}
		if decl != nil {
			break
		}
	}
	if decl == nil {
		return nil, sigDriftRefusal(fn)
	}
	tf := declPkg.Fset.File(decl.Pos())
	if tf == nil {
		return nil, sigDriftRefusal(fn)
	}
	if rel, err := filepath.Rel(p.Root, tf.Name()); err != nil || filepath.ToSlash(rel) != fn.File {
		return nil, sigDriftRefusal(fn)
	}
	tgt := &sigTarget{
		fn: fn, obj: obj, key: key, sig: obj.Signature(),
		declPkg: declPkg, decl: decl, declFile: fn.File,
		declLine: tf.Line(decl.Pos()),
	}
	for _, file := range declPkg.Syntax {
		if file.Pos() <= decl.Pos() && decl.Pos() < file.End() {
			tgt.declAST = file
			break
		}
	}
	fset := declPkg.Fset
	if decl.Recv != nil && len(decl.Recv.List) > 0 {
		rf := decl.Recv.List[0]
		if len(rf.Names) > 0 {
			tgt.recvName = rf.Names[0].Name
		}
		tgt.recvBase = recvBaseName(exprText(fset, rf.Type))
	}
	expand := func(fl *ast.FieldList, tuple *types.Tuple) []sigParam {
		var out []sigParam
		if fl == nil {
			return out
		}
		ti := 0
		for _, field := range fl.List {
			typ := string(src[offOf(fset, field.Type.Pos()):offOf(fset, field.Type.End())])
			if len(field.Names) == 0 {
				sp := sigParam{typ: typ, nameStart: -1, nameEnd: -1}
				if tuple != nil && ti < tuple.Len() {
					sp.tuple = tuple.At(ti)
				}
				out = append(out, sp)
				ti++
				continue
			}
			for _, nm := range field.Names {
				sp := sigParam{name: nm.Name, typ: typ, nameStart: offOf(fset, nm.Pos()), nameEnd: offOf(fset, nm.End())}
				if tuple != nil && ti < tuple.Len() {
					sp.tuple = tuple.At(ti)
				}
				out = append(out, sp)
				ti++
			}
		}
		return out
	}
	tgt.params = expand(decl.Type.Params, tgt.sig.Params())
	tgt.results = expand(decl.Type.Results, tgt.sig.Results())
	if len(tgt.params) != tgt.sig.Params().Len() {
		return nil, sigDriftRefusal(fn)
	}
	for _, prm := range tgt.params {
		if prm.name != "" && string(src[prm.nameStart:prm.nameEnd]) != prm.name {
			return nil, sigDriftRefusal(fn)
		}
	}
	return tgt, ""
}

// paramIndex finds a parameter by name, -1 when absent.
func (t *sigTarget) paramIndex(name string) int {
	for i, prm := range t.params {
		if prm.name == name && name != "" {
			return i
		}
	}
	return -1
}

// paramNameList renders the target's named parameters for
// refusal messages.
func (t *sigTarget) paramNameList() string {
	var names []string
	for _, prm := range t.params {
		if prm.name != "" {
			names = append(names, prm.name)
		}
	}
	if len(names) == 0 {
		return "it has no named parameters"
	}
	return "its parameters are: " + strings.Join(names, ", ")
}

// isGeneric reports whether the target is a generic function or
// a method on a generic type — shapes whose instantiation sites
// a permutation cannot be proven against in v1.
func (t *sigTarget) isGeneric() bool {
	if t.sig.TypeParams() != nil && t.sig.TypeParams().Len() > 0 {
		return true
	}
	if t.sig.Recv() != nil {
		rt := t.sig.Recv().Type()
		for {
			if pt, ok := rt.(*types.Pointer); ok {
				rt = pt.Elem()
				continue
			}
			break
		}
		if named, ok := rt.(*types.Named); ok && named.TypeParams() != nil && named.TypeParams().Len() > 0 {
			return true
		}
	}
	return false
}

// loadedFileSet is the set of root-relative files the typed load
// covered (rename's "covered" guard, shared shape).
func (p *Project) loadedFileSet(pkgs []*packages.Package) map[string]bool {
	covered := map[string]bool{}
	for _, pkg := range pkgs {
		for _, f := range pkg.GoFiles {
			if rel, err := filepath.Rel(p.Root, f); err == nil && !strings.HasPrefix(rel, "..") {
				covered[filepath.ToSlash(rel)] = true
			}
		}
	}
	return covered
}

// allLoadedPkgs flattens the load result plus every transitively
// imported package — the universe the interface guard must see,
// since a dispatch can happen through any interface the project
// can name, the standard library's included.
func allLoadedPkgs(pkgs []*packages.Package) []*packages.Package {
	seen := map[*packages.Package]bool{}
	var out []*packages.Package
	var walk func(pkg *packages.Package)
	walk = func(pkg *packages.Package) {
		if pkg == nil || seen[pkg] {
			return
		}
		seen[pkg] = true
		out = append(out, pkg)
		for _, imp := range pkg.Imports {
			walk(imp)
		}
	}
	for _, pkg := range pkgs {
		walk(pkg)
	}
	return out
}

// sigInterfaceGuard refuses a signature change to a method whose
// signature an interface dictates: if the receiver type (in value
// or pointer form) implements any interface — from the project
// or from anything it imports — that declares a method with this
// name and signature, changing this one implementer alone would
// break the contract wherever the interface is used.
func (p *Project) sigInterfaceGuard(pkgs []*packages.Package, tgt *sigTarget) string {
	if tgt.fn.Receiver == "" {
		return ""
	}
	sig := tgt.sig
	rt := sig.Recv().Type()
	for {
		if pt, ok := rt.(*types.Pointer); ok {
			rt = pt.Elem()
			continue
		}
		break
	}
	named, ok := rt.(*types.Named)
	if !ok {
		return ""
	}
	var hits []string
	hitSeen := map[string]bool{}
	for _, pkg := range allLoadedPkgs(pkgs) {
		if pkg.Types == nil {
			continue
		}
		scope := pkg.Types.Scope()
		for _, name := range scope.Names() {
			tn, ok := scope.Lookup(name).(*types.TypeName)
			if !ok {
				continue
			}
			iface, ok := tn.Type().Underlying().(*types.Interface)
			if !ok {
				continue
			}
			declares := false
			for i := 0; i < iface.NumMethods(); i++ {
				m := iface.Method(i)
				if m.Name() != tgt.fn.Name {
					continue
				}
				ms := m.Signature()
				if types.Identical(sig.Params(), ms.Params()) &&
					types.Identical(sig.Results(), ms.Results()) &&
					sig.Variadic() == ms.Variadic() {
					declares = true
					break
				}
			}
			if !declares {
				continue
			}
			if types.Implements(named, iface) || types.Implements(types.NewPointer(named), iface) {
				disp := tn.Pkg().Name() + "." + tn.Name()
				if !hitSeen[disp] {
					hitSeen[disp] = true
					hits = append(hits, disp)
				}
			}
		}
	}
	if len(hits) == 0 {
		return ""
	}
	sort.Strings(hits)
	return fmt.Sprintf("%s satisfies the %s interface through this method — changing just this one method's signature would break that contract everywhere the interface is used. Changing a signature an interface dictates isn't something I do yet.", tgt.fn.Display(), hits[0])
}

// sigUncoveredGuard is renameUncoveredGuard's sibling for
// signature changes: a project file the typed load did not cover
// (other platforms' build tags, nested modules) that could
// reference the target — same package directory, or importing the
// target's package — and mentions the name is a call site nobody
// can locate exactly, so the change refuses.
func (p *Project) sigUncoveredGuard(fn *Func, covered map[string]bool) string {
	wantPath := p.pkgPathOf(fn.PkgDir)
	errFound := ""
	_ = filepath.WalkDir(p.Root, func(path string, d os.DirEntry, err error) error {
		if err != nil || errFound != "" {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if name == "vendor" || name == ".git" || (strings.HasPrefix(name, ".") && path != p.Root) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		rel, rerr := filepath.Rel(p.Root, path)
		if rerr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if covered[rel] {
			return nil
		}
		sameDir := filepath.ToSlash(filepath.Dir(rel)) == fn.PkgDir || (fn.PkgDir == "" && filepath.Dir(rel) == ".")
		if !sameDir && !fileImportsPath(path, wantPath) {
			return nil
		}
		if fileMentionsSymbol(path, fn.Name) {
			errFound = fmt.Sprintf("One catch: %s mentions %s, but that file isn't part of this platform's build (build tags or a nested module), so I can't check its call sites exactly — I won't change the signature blind. Update the use there by hand, or bring the file into the build, and retry.",
				p.LinkPath(rel), fn.Name)
		}
		return nil
	})
	return errFound
}

// --- use collection ---------------------------------------------------------

// sigCall is one located call site of the target: the call
// expression's argument span as byte offsets in its file, plus
// the typed context derivation needs while planning.
type sigCall struct {
	rel      string
	line     int
	lpOff    int // offset of the call's '('
	rpOff    int // offset of the call's ')'
	argSpan  [][2]int
	ellipsis bool
	encRel   string // enclosing function's file
	encLine  int    // enclosing FuncDecl's line; 0 at file scope
	encName  string // "shop.Checkout", for messages

	pkg     *packages.Package
	call    *ast.CallExpr
	encDecl *ast.FuncDecl // nil at file scope
}

// sigValueUse is one place the target is referenced as a value —
// its type consumed, not called. Any one of these refuses a
// reorder or an add.
type sigValueUse struct {
	rel  string
	line int
	what string
}

// collectFuncUses walks every loaded package's syntax and sorts
// each use of the target object into call sites and value uses,
// by object key (the identity that survives the several package
// instances a test-aware load produces).
func (p *Project) collectFuncUses(pkgs []*packages.Package, want objKey) (calls []*sigCall, values []sigValueUse, external []string) {
	callSeen := map[string]bool{}
	valSeen := map[string]bool{}
	extSeen := map[string]bool{}
	for _, pkg := range pkgs {
		if pkg.TypesInfo == nil || pkg.Fset == nil {
			continue
		}
		info := pkg.TypesInfo
		for _, file := range pkg.Syntax {
			var stack []ast.Node
			ast.Inspect(file, func(n ast.Node) bool {
				if n == nil {
					stack = stack[:len(stack)-1]
					return false
				}
				stack = append(stack, n)
				id, ok := n.(*ast.Ident)
				if !ok {
					return true
				}
				obj := info.Uses[id]
				if obj == nil {
					return true
				}
				key, ok := keyOfTypesObj(obj)
				if !ok || key != want {
					return true
				}
				tf := pkg.Fset.File(id.Pos())
				if tf == nil {
					return true
				}
				abs := tf.Name()
				if filepath.Base(abs) == "_testmain.go" {
					return true
				}
				rel, err := filepath.Rel(p.Root, abs)
				if err != nil || strings.HasPrefix(rel, "..") {
					if !extSeen[abs] {
						extSeen[abs] = true
						external = append(external, abs)
					}
					return true
				}
				rel = filepath.ToSlash(rel)
				// Classify by the use's parent: the Fun of a
				// call (directly, or as a selector's Sel) is a
				// call site; anything else consumes the value.
				var callExpr *ast.CallExpr
				if len(stack) >= 2 {
					switch par := stack[len(stack)-2].(type) {
					case *ast.CallExpr:
						if par.Fun == id {
							callExpr = par
						}
					case *ast.SelectorExpr:
						if par.Sel == id && len(stack) >= 3 {
							if ce, ok := stack[len(stack)-3].(*ast.CallExpr); ok && ce.Fun == par {
								callExpr = ce
							}
						}
					}
				}
				if callExpr == nil {
					vk := fmt.Sprintf("%s|%d", rel, tf.Offset(id.Pos()))
					if !valSeen[vk] {
						valSeen[vk] = true
						values = append(values, sigValueUse{rel: rel, line: tf.Line(id.Pos()), what: sigValueUseWhat(stack)})
					}
					return true
				}
				ck := fmt.Sprintf("%s|%d", rel, tf.Offset(callExpr.Lparen))
				if callSeen[ck] {
					return true
				}
				callSeen[ck] = true
				sc := &sigCall{
					rel:      rel,
					line:     tf.Line(callExpr.Pos()),
					lpOff:    tf.Offset(callExpr.Lparen),
					rpOff:    tf.Offset(callExpr.Rparen),
					ellipsis: callExpr.Ellipsis.IsValid(),
					pkg:      pkg,
					call:     callExpr,
				}
				for _, a := range callExpr.Args {
					sc.argSpan = append(sc.argSpan, [2]int{tf.Offset(a.Pos()), tf.Offset(a.End())})
				}
				sc.encRel = rel
				sc.encDecl, sc.encLine, sc.encName = enclosingFunc(stack, pkg)
				calls = append(calls, sc)
				return true
			})
		}
	}
	sort.Slice(calls, func(i, j int) bool {
		if calls[i].rel != calls[j].rel {
			return calls[i].rel < calls[j].rel
		}
		return calls[i].lpOff < calls[j].lpOff
	})
	sort.Slice(values, func(i, j int) bool {
		if values[i].rel != values[j].rel {
			return values[i].rel < values[j].rel
		}
		return values[i].line < values[j].line
	})
	return calls, values, external
}

// enclosingFunc names the FuncDecl a use sits inside (the
// nearest one on the parent stack) for manifests and messages,
// and gives its line in its own file (the use's file — a use's
// enclosing declaration is always in the same file).
func enclosingFunc(stack []ast.Node, pkg *packages.Package) (decl *ast.FuncDecl, line int, name string) {
	for i := len(stack) - 1; i >= 0; i-- {
		fd, ok := stack[i].(*ast.FuncDecl)
		if !ok {
			continue
		}
		name = pkg.Types.Name() + "." + fd.Name.Name
		if fd.Recv != nil && len(fd.Recv.List) > 0 {
			name = pkg.Types.Name() + "." + recvBaseName(exprText(pkg.Fset, fd.Recv.List[0].Type)) + "." + fd.Name.Name
		}
		for j := i + 1; j < len(stack); j++ {
			if _, ok := stack[j].(*ast.FuncLit); ok {
				name += " (in a closure)"
				break
			}
		}
		tf := pkg.Fset.File(fd.Pos())
		if tf != nil {
			line = tf.Line(fd.Pos())
		}
		return fd, line, name
	}
	return nil, 0, ""
}

// sigValueUseWhat phrases what a value use is doing, from the
// nearest meaningful ancestor.
func sigValueUseWhat(stack []ast.Node) string {
	for i := len(stack) - 2; i >= 0; i-- {
		switch stack[i].(type) {
		case *ast.AssignStmt, *ast.ValueSpec:
			return "assigned as a value"
		case *ast.ReturnStmt:
			return "returned as a value"
		case *ast.CallExpr:
			return "passed as a value"
		case *ast.CompositeLit:
			return "stored in a composite literal"
		case *ast.FuncDecl, *ast.File:
			return "used as a value"
		}
	}
	return "used as a value"
}

// checkCallers cross-checks the index's caller list against the
// located call sites: a caller the index believes in whose file
// the load covered, but in which no call site was found, means a
// site nobody can locate exactly — a refusal, never a miss.
func (p *Project) checkCallers(fn *Func, calls []*sigCall, covered map[string]bool) string {
	for _, id := range fn.Callers {
		cf := p.FuncByID(id)
		if cf == nil || !covered[cf.File] {
			continue
		}
		found := false
		for _, s := range calls {
			if s.encRel == cf.File && s.encLine == cf.Line {
				found = true
				break
			}
		}
		if !found {
			return fmt.Sprintf("My index says %s (%s:%d) calls %s, but I can't find that call exactly in the type-checked code — I won't change a signature when a call site can't be located. Re-analyze the project and retry.", cf.Display(), p.LinkPath(cf.File), cf.Line, fn.Display())
		}
	}
	return ""
}

// sigGuardUses runs the refusals every site-driven op shares:
// external uses, value uses, the caller cross-check, and the
// uncovered-file guard.
func (p *Project) sigGuardUses(fn *Func, calls []*sigCall, values []sigValueUse, external []string, covered map[string]bool) string {
	if len(external) > 0 {
		return fmt.Sprintf("Something outside this project (%s) also uses %s, and I can't rewrite code that isn't here — changing the signature would leave that behind.", external[0], fn.Name)
	}
	if len(values) > 0 {
		v := values[0]
		more := ""
		if len(values) > 1 {
			more = fmt.Sprintf(" (and %d more like it)", len(values)-1)
		}
		return fmt.Sprintf("%s is %s at %s:%d%s — its type is part of that code, so a signature change there isn't a call-site edit I can prove. Change that use by hand first, and I'll do the rest.", fn.Display(), v.what, p.LinkPath(v.rel), v.line, more)
	}
	if refusal := p.checkCallers(fn, calls, covered); refusal != "" {
		return refusal
	}
	return p.sigUncoveredGuard(fn, covered)
}

// collectParamOccurrences finds every identifier bound to the
// target's idx-th parameter — the declaration itself and every
// use inside the body — as (file, start, end) sites. Matching is
// by pointer identity within each home-package instance (the
// parameter's uses can only live in its own function's file,
// which only the home instances carry), deduped across them.
func (p *Project) collectParamOccurrences(pkgs []*packages.Package, tgt *sigTarget, idx int) []renameSite {
	wantPath := p.pkgPathOf(tgt.fn.PkgDir)
	var out []renameSite
	seen := map[string]bool{}
	for _, pkg := range pkgs {
		if pkg.PkgPath != wantPath || pkg.TypesInfo == nil || pkg.Fset == nil {
			continue
		}
		info := pkg.TypesInfo
		// This instance's own tuple var for the parameter.
		var tupleVar *types.Var
		for _, file := range pkg.Syntax {
			for _, d := range file.Decls {
				fd, ok := d.(*ast.FuncDecl)
				if !ok {
					continue
				}
				if fo, ok := info.Defs[fd.Name].(*types.Func); ok {
					if k, kok := keyOfTypesObj(fo); kok && k == tgt.key {
						if ps := fo.Signature().Params(); ps != nil && idx < ps.Len() {
							tupleVar = ps.At(idx)
						}
					}
				}
			}
		}
		if tupleVar == nil {
			continue
		}
		for _, file := range pkg.Syntax {
			ast.Inspect(file, func(n ast.Node) bool {
				id, ok := n.(*ast.Ident)
				if !ok {
					return true
				}
				obj := info.Defs[id]
				if obj == nil {
					obj = info.Uses[id]
				}
				if obj != tupleVar {
					return true
				}
				tf := pkg.Fset.File(id.Pos())
				if tf == nil {
					return true
				}
				rel, err := filepath.Rel(p.Root, tf.Name())
				if err != nil || strings.HasPrefix(rel, "..") {
					return true
				}
				rel = filepath.ToSlash(rel)
				s := renameSite{rel: rel, start: tf.Offset(id.Pos()), end: tf.Offset(id.End())}
				k := fmt.Sprintf("%s|%d|%d", rel, s.start, s.end)
				if !seen[k] {
					seen[k] = true
					out = append(out, s)
				}
				return true
			})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].rel != out[j].rel {
			return out[i].rel < out[j].rel
		}
		return out[i].start < out[j].start
	})
	return out
}

// --- the planner ------------------------------------------------------------

// planSignature is the PlanEdit path for the three signature
// kinds: resolve the target in the index, load the module with
// type info, pin the target's declaration in the typed project,
// then dispatch to the operation.
func (p *Project) planSignature(req EditRequest) (*EditPlan, string) {
	matches := p.findProjectFuncs(req.FuncName)
	if len(matches) == 0 {
		return nil, fmt.Sprintf("I couldn't find a function named %q in this project, so there's no signature to change.", req.FuncName)
	}
	if len(matches) > 1 {
		return nil, p.funcCandidates("name the receiver or package", req.FuncName, matches)
	}
	fn := matches[0]
	if fn.IsMain {
		return nil, "main's signature is fixed by the runtime — it can't take parameters, so there's nothing to rename, reorder, or add."
	}
	if fn.IsInit {
		return nil, "init's signature is fixed by the runtime — it can't take parameters, so there's nothing to rename, reorder, or add."
	}
	if why, blocked := p.fileBlocked(fn.File); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(fn.File), why)
	}
	src, _, _, refusal := p.readProjectFile(fn.File)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(fn.File))
	}
	pkgs, refusal := p.loadForSignature()
	if refusal != "" {
		return nil, refusal
	}
	tgt, refusal := p.findSigTarget(pkgs, fn, src)
	if refusal != "" {
		return nil, refusal
	}
	covered := p.loadedFileSet(pkgs)
	switch req.Kind {
	case EditRenameParam:
		return p.planRenameParam(req, tgt, pkgs, src)
	case EditReorderParams:
		return p.planReorderParams(req, tgt, pkgs, covered, src)
	case EditAddParam:
		return p.planAddParam(req, tgt, pkgs, covered, src)
	}
	return nil, "I don't know that kind of edit yet."
}

// sigHeaderText renders the target's declaration header (the
// func clause through the results) with a given parameter-list
// interior — the old → new signature strings for summaries.
func sigHeaderText(src []byte, tgt *sigTarget, interior string) string {
	fd := tgt.decl
	fset := tgt.declPkg.Fset
	start := offOf(fset, fd.Pos())
	end := offOf(fset, fd.End())
	if fd.Body != nil {
		end = offOf(fset, fd.Body.Pos())
	}
	head := string(src[start:end])
	lpRel := offOf(fset, fd.Type.Params.Opening) - start
	rpRel := offOf(fset, fd.Type.Params.Closing) - start
	return strings.Join(strings.Fields(head[:lpRel+1]+interior+head[rpRel:]), " ")
}

// declInterior is the declaration's current parameter-list text.
func (t *sigTarget) declInterior(src []byte) string {
	fset := t.declPkg.Fset
	lp := offOf(fset, t.decl.Type.Params.Opening)
	rp := offOf(fset, t.decl.Type.Params.Closing)
	return string(src[lp+1 : rp])
}

// gapHasComment reports whether any of the gaps around spans
// inside (lpOff, rpOff) holds a comment — permuting across a
// comment would silently drop it, so such a site refuses.
func gapHasComment(src []byte, lpOff, rpOff int, spans [][2]int) bool {
	gaps := [][2]int{}
	prev := lpOff + 1
	for _, sp := range spans {
		gaps = append(gaps, [2]int{prev, sp[0]})
		prev = sp[1]
	}
	gaps = append(gaps, [2]int{prev, rpOff})
	for _, g := range gaps {
		if g[0] < 0 || g[1] > len(src) || g[0] > g[1] {
			continue
		}
		text := string(src[g[0]:g[1]])
		if strings.Contains(text, "//") || strings.Contains(text, "/*") {
			return true
		}
	}
	return false
}

// --- rename a parameter -------------------------------------------------------

// planRenameParam renames one parameter and every use of it
// inside the function body. Parameter names are not part of a
// function's type, so interfaces and call sites are untouched —
// the occurrence set is binding-exact by object identity, which
// is what lets shadowed locals and closure captures sort
// themselves out: a use belongs to the parameter exactly when
// the type checker bound it to the parameter's *types.Var.
func (p *Project) planRenameParam(req EditRequest, tgt *sigTarget, pkgs []*packages.Package, src []byte) (*EditPlan, string) {
	fn := tgt.fn
	oldName, newName := req.SigOld, req.SigNew
	idx := tgt.paramIndex(oldName)
	if idx < 0 {
		return nil, fmt.Sprintf("%s has no parameter named %q — %s.", fn.Display(), oldName, tgt.paramNameList())
	}
	if oldName == "_" {
		return nil, fmt.Sprintf("The parameter of %s is the blank identifier — it has no uses to rename, and giving it a name is a different edit.", fn.Display())
	}
	if !token.IsIdentifier(newName) {
		return nil, fmt.Sprintf("%q isn't a name Go allows — a name is letters, digits and underscores and can't start with a digit or be a Go keyword.", newName)
	}
	if newName == oldName {
		return nil, fmt.Sprintf("It's already named %s — nothing to rename.", oldName)
	}
	if newName == "_" {
		return nil, fmt.Sprintf("Renaming %s to _ would leave the body's uses with no name to hold — I won't blank a parameter that code still uses.", oldName)
	}
	if tgt.recvName == newName {
		return nil, fmt.Sprintf("The receiver of %s is already named %s — a parameter can't take the receiver's name.", fn.Display(), newName)
	}
	for i, prm := range tgt.params {
		if i != idx && prm.name == newName {
			return nil, fmt.Sprintf("%s already has a parameter named %s.", fn.Display(), newName)
		}
	}
	for _, r := range tgt.results {
		if r.name == newName {
			return nil, fmt.Sprintf("%s already has a named result called %s — in Go a function's parameters and named results share one scope, so the two can't hold the same name.", fn.Display(), newName)
		}
	}
	if line, found := bodyTopDecl(tgt, newName); found {
		return nil, fmt.Sprintf("The body of %s declares %s itself at %s:%d — renaming the parameter to %s would collide with it.", fn.Display(), newName, p.LinkPath(tgt.declFile), line, newName)
	}
	occ := p.collectParamOccurrences(pkgs, tgt, idx)
	if len(occ) == 0 {
		return nil, sigDriftRefusal(fn)
	}
	var edits []textEdit
	for _, o := range occ {
		if o.rel != tgt.declFile {
			return nil, sigDriftRefusal(fn)
		}
		if o.end > len(src) || string(src[o.start:o.end]) != oldName {
			return nil, sigDriftRefusal(fn)
		}
		edits = append(edits, textEdit{start: o.start, end: o.end, text: newName})
	}
	prm := tgt.params[idx]
	interiorOld := tgt.declInterior(src)
	lpOff := offOf(tgt.declPkg.Fset, tgt.decl.Type.Params.Opening)
	relStart := prm.nameStart - (lpOff + 1)
	relEnd := prm.nameEnd - (lpOff + 1)
	interiorNew := interiorOld[:relStart] + newName + interiorOld[relEnd:]
	expect := tgt.expectParams()
	expect[idx].name = newName
	summary := fmt.Sprintf("Rename %s's parameter %s to %s — %s becomes %s. The name appears %d time(s), all inside the function itself; no call site changes.",
		fn.Display(), oldName, newName, sigHeaderText(src, tgt, interiorOld), sigHeaderText(src, tgt, interiorNew), len(occ))
	notes := []string{
		fmt.Sprintf("%s — the declaration and %d use(s), all inside %s.", p.LinkPath(tgt.declFile), len(occ)-1, fn.Name),
	}
	return p.finishSigPlan(fn, tgt,
		map[string][]textEdit{tgt.declFile: edits},
		map[string][]byte{tgt.declFile: src},
		summary, notes,
		fmt.Sprintf("%s's parameter is now named %s", fn.Display(), newName),
		expect)
}

// bodyTopDecl reports whether the function's own outermost block
// declares name (a := target, a var spec, a range variable) —
// such a declaration shares the parameters' scope, so a rename
// into it would collide. Only the body's direct statements are
// considered: declarations in nested blocks shadow legally, and
// the occurrence set already sorts those out.
func bodyTopDecl(tgt *sigTarget, name string) (int, bool) {
	fd := tgt.decl
	if fd.Body == nil {
		return 0, false
	}
	hit := func(id *ast.Ident) bool { return id != nil && id.Name == name }
	for _, stmt := range fd.Body.List {
		switch s := stmt.(type) {
		case *ast.DeclStmt:
			if gd, ok := s.Decl.(*ast.GenDecl); ok {
				for _, spec := range gd.Specs {
					if vs, ok := spec.(*ast.ValueSpec); ok {
						for _, nm := range vs.Names {
							if hit(nm) {
								return lineOfPos(tgt.declPkg.Fset, nm.Pos()), true
							}
						}
					}
				}
			}
		case *ast.AssignStmt:
			if s.Tok == token.DEFINE {
				for _, lhs := range s.Lhs {
					if id, ok := lhs.(*ast.Ident); ok && hit(id) {
						return lineOfPos(tgt.declPkg.Fset, id.Pos()), true
					}
				}
			}
		case *ast.RangeStmt:
			if s.Tok == token.DEFINE {
				if id, ok := s.Key.(*ast.Ident); ok && hit(id) {
					return lineOfPos(tgt.declPkg.Fset, id.Pos()), true
				}
				if id, ok := s.Value.(*ast.Ident); ok && hit(id) {
					return lineOfPos(tgt.declPkg.Fset, id.Pos()), true
				}
			}
		}
	}
	return 0, false
}

// --- reorder parameters -------------------------------------------------------

// reorderPlan computes one call site's new argument-list
// interior under a permutation, recursing into nested calls to
// the same function (a recursive call inside an argument of an
// outer call gets its interior from the same rule, so spans are
// never edited twice).
type reorderPlan struct {
	order  []int // new position -> old index
	byFile map[string][]*sigCall
	srcBy  map[string][]byte
	memo   map[*sigCall]string
}

func (r *reorderPlan) interior(sc *sigCall) string {
	if s, ok := r.memo[sc]; ok {
		return s
	}
	src := r.srcBy[sc.rel]
	texts := make([]string, len(sc.argSpan))
	for i, sp := range sc.argSpan {
		texts[i] = r.argText(sc, sp, src)
	}
	out := make([]string, len(texts))
	for newPos, oldIdx := range r.order {
		out[newPos] = texts[oldIdx]
	}
	s := strings.Join(out, ", ")
	r.memo[sc] = s
	return s
}

func (r *reorderPlan) argText(sc *sigCall, sp [2]int, src []byte) string {
	text := string(src[sp[0]:sp[1]])
	var inners []*sigCall
	for _, other := range r.byFile[sc.rel] {
		if other != sc && other.lpOff > sp[0] && other.rpOff < sp[1] {
			inners = append(inners, other)
		}
	}
	// Only the outermost inners are spliced here; each one's own
	// interior (computed by interior) already carries its
	// nested calls.
	var outer []*sigCall
	for _, in := range inners {
		contained := false
		for _, o := range inners {
			if o != in && o.lpOff < in.lpOff && in.rpOff < o.rpOff {
				contained = true
				break
			}
		}
		if !contained {
			outer = append(outer, in)
		}
	}
	sort.Slice(outer, func(i, j int) bool { return outer[i].lpOff > outer[j].lpOff })
	for _, in := range outer {
		text = text[:in.lpOff+1-sp[0]] + r.interior(in) + text[in.rpOff-sp[0]:]
	}
	return text
}

// outermostCalls returns the calls in a file not contained in
// any other call's span — the only ones that become file edits.
func outermostCalls(calls []*sigCall) []*sigCall {
	var out []*sigCall
	for _, sc := range calls {
		contained := false
		for _, o := range calls {
			if o != sc && o.lpOff < sc.lpOff && sc.rpOff < o.rpOff {
				contained = true
				break
			}
		}
		if !contained {
			out = append(out, sc)
		}
	}
	return out
}

func (p *Project) planReorderParams(req EditRequest, tgt *sigTarget, pkgs []*packages.Package, covered map[string]bool, src []byte) (*EditPlan, string) {
	fn := tgt.fn
	if refusal := p.sigInterfaceGuard(pkgs, tgt); refusal != "" {
		return nil, refusal
	}
	if tgt.isGeneric() {
		return nil, fmt.Sprintf("%s is generic — its call sites instantiate it with type arguments, and a parameter permutation can't be proven against every instantiation. Reordering generic functions isn't something I do yet.", fn.Display())
	}
	if tgt.sig.Variadic() {
		// A spread call is the sharpest form of the problem, and
		// the common one — name it when one exists.
		calls, _, _ := p.collectFuncUses(pkgs, tgt.key)
		for _, sc := range calls {
			if sc.ellipsis {
				return nil, fmt.Sprintf("The call at %s:%d spreads a slice into %s — a spread call has no fixed argument list to reorder, and I won't guess at it. Expand that call by hand and retry.", p.LinkPath(sc.rel), sc.line, fn.Name)
			}
		}
		return nil, fmt.Sprintf("%s is variadic — its call sites may pass any number of trailing arguments, so no fixed permutation can be proven at every site. Reordering a variadic function's parameters isn't something I do yet.", fn.Display())
	}
	a := tgt.paramIndex(req.SigOld)
	if a < 0 {
		return nil, fmt.Sprintf("%s has no parameter named %q — %s.", fn.Display(), req.SigOld, tgt.paramNameList())
	}
	n := len(tgt.params)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	switch req.SigMove {
	case "":
		b := tgt.paramIndex(req.SigNew)
		if b < 0 {
			return nil, fmt.Sprintf("%s has no parameter named %q — %s.", fn.Display(), req.SigNew, tgt.paramNameList())
		}
		if a == b {
			return nil, fmt.Sprintf("Both names are the same parameter (%s) — there's nothing to swap.", req.SigOld)
		}
		order[a], order[b] = b, a
	case "first", "last":
		rest := append([]int{}, order[:a]...)
		rest = append(rest, order[a+1:]...)
		if req.SigMove == "first" {
			order = append([]int{a}, rest...)
		} else {
			order = append(rest, a)
		}
	case "before", "after":
		b := tgt.paramIndex(req.SigNew)
		if b < 0 {
			return nil, fmt.Sprintf("%s has no parameter named %q — %s.", fn.Display(), req.SigNew, tgt.paramNameList())
		}
		if a == b {
			return nil, fmt.Sprintf("Moving %s %s itself leaves the order unchanged.", req.SigOld, req.SigMove)
		}
		rest := append([]int{}, order[:a]...)
		rest = append(rest, order[a+1:]...)
		bi := 0
		for i, v := range rest {
			if v == b {
				bi = i
			}
		}
		if req.SigMove == "after" {
			bi++
		}
		order = append(rest[:bi], append([]int{a}, rest[bi:]...)...)
	}
	identity := true
	for i, v := range order {
		if v != i {
			identity = false
		}
	}
	if identity {
		return nil, fmt.Sprintf("That leaves %s's parameter order exactly as it is — nothing to change.", fn.Display())
	}
	calls, values, external := p.collectFuncUses(pkgs, tgt.key)
	if refusal := p.sigGuardUses(fn, calls, values, external, covered); refusal != "" {
		return nil, refusal
	}
	srcBy := map[string][]byte{tgt.declFile: src}
	readSrc := func(rel string) ([]byte, string) {
		if b, ok := srcBy[rel]; ok {
			return b, ""
		}
		b, err := readFileBytes(filepath.Join(p.Root, rel))
		if err != nil {
			return nil, fmt.Sprintf("I couldn't read %s: %v", p.LinkPath(rel), err)
		}
		srcBy[rel] = b
		return b, ""
	}
	for _, sc := range calls {
		if sc.ellipsis {
			return nil, fmt.Sprintf("A call at %s:%d spreads a slice into %s (%s(...)) — a spread call has no fixed argument list to reorder, and I won't guess at it. Expand that call by hand and retry.", p.LinkPath(sc.rel), sc.line, fn.Name, fn.Name)
		}
		if len(sc.argSpan) != n {
			return nil, fmt.Sprintf("The call at %s:%d passes %d argument(s) but %s takes %d parameters — that doesn't line up with the type-checked signature, so I won't touch it.", p.LinkPath(sc.rel), sc.line, len(sc.argSpan), fn.Display(), n)
		}
		bsrc, r := readSrc(sc.rel)
		if r != "" {
			return nil, r
		}
		if gapHasComment(bsrc, sc.lpOff, sc.rpOff, sc.argSpan) {
			return nil, fmt.Sprintf("The call at %s:%d has a comment between its arguments — reordering would drop it, so I won't. Move the comment and retry.", p.LinkPath(sc.rel), sc.line)
		}
	}
	// A comment inside the declaration's own parameter list
	// would be dropped by the interior rewrite too.
	{
		fset := tgt.declPkg.Fset
		var spans [][2]int
		for _, f := range tgt.decl.Type.Params.List {
			spans = append(spans, [2]int{offOf(fset, f.Pos()), offOf(fset, f.End())})
		}
		lp := offOf(fset, tgt.decl.Type.Params.Opening)
		rp := offOf(fset, tgt.decl.Type.Params.Closing)
		if gapHasComment(src, lp, rp, spans) {
			return nil, fmt.Sprintf("The parameter list of %s has a comment inside it — reordering would drop it, so I won't. Move the comment and retry.", fn.Display())
		}
	}
	byFile := map[string][]*sigCall{}
	for _, sc := range calls {
		byFile[sc.rel] = append(byFile[sc.rel], sc)
	}
	rp := &reorderPlan{order: order, byFile: byFile, srcBy: srcBy, memo: map[*sigCall]string{}}
	editsByFile := map[string][]textEdit{}
	fset := tgt.declPkg.Fset
	newParams := make([]string, n)
	for newPos, oldIdx := range order {
		newParams[newPos] = renderSigParam(tgt.params[oldIdx])
	}
	editsByFile[tgt.declFile] = append(editsByFile[tgt.declFile], textEdit{
		start: offOf(fset, tgt.decl.Type.Params.Opening) + 1,
		end:   offOf(fset, tgt.decl.Type.Params.Closing),
		text:  strings.Join(newParams, ", "),
	})
	for rel, fileCalls := range byFile {
		for _, sc := range outermostCalls(fileCalls) {
			editsByFile[rel] = append(editsByFile[rel], textEdit{start: sc.lpOff + 1, end: sc.rpOff, text: rp.interior(sc)})
		}
	}
	expect := make([]sigNV, n)
	for newPos, oldIdx := range order {
		prm := tgt.params[oldIdx]
		expect[newPos] = sigNV{prm.name, prm.typ}
	}
	interiorOld := tgt.declInterior(src)
	summary := fmt.Sprintf("Change %s's signature — %s becomes %s. %d call site(s) across %d file(s) get their arguments reordered to match.",
		fn.Display(), sigHeaderText(src, tgt, interiorOld), sigHeaderText(src, tgt, strings.Join(newParams, ", ")),
		len(calls), countFiles(calls, tgt.declFile))
	notes := sigSiteNotes(p, tgt, calls, nil)
	return p.finishSigPlan(fn, tgt, editsByFile, srcBy, summary, notes,
		fmt.Sprintf("%s's parameters are reordered", fn.Display()), expect)
}

// renderSigParam renders one parameter the way it will be
// written: "name type", or just "type" when unnamed.
func renderSigParam(prm sigParam) string {
	if prm.name != "" {
		return prm.name + " " + prm.typ
	}
	return prm.typ
}

// countFiles counts the distinct files in a call set plus the
// declaration file when it isn't already among them.
func countFiles(calls []*sigCall, declFile string) int {
	set := map[string]bool{declFile: true}
	for _, sc := range calls {
		set[sc.rel] = true
	}
	return len(set)
}

// sigSiteNotes builds the impact manifest's per-file lines:
// the declaration file first, then every file with its site
// count, plus any per-site detail lines the caller computed
// (add-parameter derivations).
func sigSiteNotes(p *Project, tgt *sigTarget, calls []*sigCall, perSite map[*sigCall]string) []string {
	byFile := map[string][]*sigCall{}
	for _, sc := range calls {
		byFile[sc.rel] = append(byFile[sc.rel], sc)
	}
	// Declaration file first (listed even when it holds no call
	// site — it always holds the declaration), the rest sorted.
	ordered := []string{tgt.declFile}
	var rest []string
	for rel := range byFile {
		if rel != tgt.declFile {
			rest = append(rest, rel)
		}
	}
	sort.Strings(rest)
	ordered = append(ordered, rest...)
	var notes []string
	for _, rel := range ordered {
		sites := byFile[rel]
		var line string
		if rel == tgt.declFile {
			line = fmt.Sprintf("%s — the declaration", p.LinkPath(rel))
			if len(sites) > 0 {
				line += fmt.Sprintf(" and %d call site(s)", len(sites))
			}
		} else {
			line = fmt.Sprintf("%s — %d call site(s)", p.LinkPath(rel), len(sites))
		}
		notes = append(notes, line)
		if perSite != nil && len(calls) <= 12 {
			for _, sc := range sites {
				if s, ok := perSite[sc]; ok {
					notes = append(notes, "    "+s)
				}
			}
		}
	}
	if tgt.fn.Exported {
		notes = append(notes, fmt.Sprintf("%s is exported — code outside this project that calls it will need the same change on its side; this plan covers this project.", tgt.fn.Name))
	}
	return notes
}

// --- add a parameter -----------------------------------------------------------

// sigNV is one expected parameter (name, whitespace-stripped
// type text) for the structural verify.
type sigNV struct{ name, typ string }

func (t *sigTarget) expectParams() []sigNV {
	out := make([]sigNV, len(t.params))
	for i, prm := range t.params {
		out[i] = sigNV{prm.name, stripWS(prm.typ)}
	}
	return out
}

func stripWS(s string) string { return strings.Join(strings.Fields(s), "") }

// planAddParam adds a parameter and one argument at every call
// site. Two forms:
//
//	context form ("add a context parameter to F"): the new
//	parameter goes FIRST, and each site's argument is derived —
//	the caller's own in-scope context.Context value at that
//	point, by whatever name it goes under there.
//
//	using form ("add a T parameter named n to F using E"): the
//	new parameter goes LAST, and E is copied verbatim to every
//	site — after proving E's names resolve in every site's
//	file.
//
// Either way, a single site that can't supply the value refuses
// the WHOLE plan with every failing site listed: a signature
// change that lands in some files and not others is the one
// outcome this shape must never produce.
func (p *Project) planAddParam(req EditRequest, tgt *sigTarget, pkgs []*packages.Package, covered map[string]bool, src []byte) (*EditPlan, string) {
	fn := tgt.fn
	if refusal := p.sigInterfaceGuard(pkgs, tgt); refusal != "" {
		return nil, refusal
	}
	if req.SigNoUsing {
		return nil, fmt.Sprintf("Adding a %s parameter to %s needs a value at every call site — end the sentence with \"using <expression>\" to supply one (for example: add a %s parameter named %s to %s using someExpression), or say \"add a context parameter to %s\" and each caller passes its own context.",
			req.SigType, fn.Display(), req.SigType, req.SigName, req.FuncName, req.FuncName)
	}
	typeText := req.SigType
	if typeText == "context" {
		typeText = "context.Context"
	}
	isCtx := typeText == "context.Context" && req.SigUsing == ""
	name := req.SigName
	if isCtx && name == "" {
		name = "ctx"
	}
	if !token.IsIdentifier(name) {
		return nil, fmt.Sprintf("%q isn't a name Go allows for the new parameter.", name)
	}
	if tgt.isGeneric() {
		return nil, fmt.Sprintf("%s is generic — its call sites instantiate it with type arguments, and adding a parameter can't be proven against every instantiation. Adding parameters to generic functions isn't something I do yet.", fn.Display())
	}
	if !isCtx && tgt.sig.Variadic() {
		return nil, fmt.Sprintf("%s is variadic — a new trailing parameter would sit behind the variadic one, where call sites pass a variable number of values, so no single expression can be proven as the new argument. Adding a parameter to a variadic function this way isn't something I do yet.", fn.Display())
	}
	if tgt.recvName == name {
		return nil, fmt.Sprintf("The receiver of %s is already named %s — the new parameter can't take the receiver's name.", fn.Display(), name)
	}
	// For the context form, "already has one" is the precise
	// refusal whenever ANY parameter is a context — checked
	// before the name collisions so a ctx-named context parameter
	// gets this message rather than the name one.
	var ctxType types.Type
	if isCtx {
		ctxType = findContextType(pkgs)
		if ctxType != nil {
			for i, prm := range tgt.params {
				if prm.tuple != nil && types.Identical(prm.tuple.Type(), ctxType) {
					label := prm.name
					if label == "" {
						label = "(unnamed)"
					}
					return nil, fmt.Sprintf("%s already takes a context — %s context.Context is parameter %d. Adding a second one would leave callers guessing which is real.", fn.Display(), label, i+1)
				}
			}
		}
	}
	for _, prm := range tgt.params {
		if prm.name == name {
			if isCtx {
				return nil, fmt.Sprintf("%s already has a parameter named %s (%s) — it's the function's context already, or holds the name the new one would take. Rename that parameter first if you want a different context.", fn.Display(), name, prm.typ)
			}
			return nil, fmt.Sprintf("%s already has a parameter named %s.", fn.Display(), name)
		}
	}
	for _, r := range tgt.results {
		if r.name == name {
			return nil, fmt.Sprintf("%s already has a named result called %s — parameters and named results share one scope in Go.", fn.Display(), name)
		}
	}

	var declImportEdits []textEdit
	var derive func(sc *sigCall) (arg, how, fail string)

	if isCtx {
		tt, impEdits, refusal := p.sigContextTypeFor(tgt, src)
		if refusal != "" {
			return nil, refusal
		}
		typeText = tt
		declImportEdits = impEdits
		derive = func(sc *sigCall) (string, string, string) {
			if sc.encDecl != nil && sc.encDecl == tgt.decl {
				return name, "the new " + name + " parameter itself (recursive call)", ""
			}
			return deriveContextArg(ctxType, sc)
		}
	} else {
		typeExpr, err := parser.ParseExpr(typeText)
		if err != nil {
			return nil, fmt.Sprintf("I couldn't read %q as a Go type for the new parameter.", typeText)
		}
		expr, err := parser.ParseExpr(req.SigUsing)
		if err != nil {
			return nil, fmt.Sprintf("I couldn't read %q as a Go expression — the using value has to be one expression, copied verbatim to every call site.", req.SigUsing)
		}
		if exprHasFuncLit(expr) {
			return nil, "The using expression contains a function literal — it has a scope of its own, so I can't prove it names the same things at every call site."
		}
		impEdits, refusal := p.sigDeclTypeImports(tgt, typeExpr, src)
		if refusal != "" {
			return nil, refusal
		}
		declImportEdits = impEdits
		derive = func(sc *sigCall) (string, string, string) {
			if bad := p.exprUnsatisfiedIn(expr, sc); bad != "" {
				return "", "", bad
			}
			return req.SigUsing, "your expression, verbatim", ""
		}
	}

	calls, values, external := p.collectFuncUses(pkgs, tgt.key)
	if refusal := p.sigGuardUses(fn, calls, values, external, covered); refusal != "" {
		return nil, refusal
	}
	type sitePlan struct {
		sc  *sigCall
		arg string
		how string
	}
	var plans []sitePlan
	var fails []string
	perSite := map[*sigCall]string{}
	for _, sc := range calls {
		arg, how, fail := derive(sc)
		if fail != "" {
			fails = append(fails, fmt.Sprintf("%s:%d (in %s): %s", p.LinkPath(sc.rel), sc.line, sc.encName, fail))
			continue
		}
		plans = append(plans, sitePlan{sc, arg, how})
		perSite[sc] = fmt.Sprintf("line %d (in %s): passes %s — %s", sc.line, sc.encName, arg, how)
	}
	if len(fails) > 0 {
		return nil, fmt.Sprintf("I won't add the parameter — %d call site(s) can't supply the new argument, and a signature change has to land everywhere or nowhere:\n  - %s",
			len(fails), strings.Join(fails, "\n  - "))
	}

	srcBy := map[string][]byte{tgt.declFile: src}
	editsByFile := map[string][]textEdit{}
	fset := tgt.declPkg.Fset
	np := sigNV{name, stripWS(typeText)}
	expect := tgt.expectParams()
	interiorOld := tgt.declInterior(src)
	var interiorNew string
	if isCtx {
		insertAt := offOf(fset, tgt.decl.Type.Params.Opening) + 1
		text := name + " " + typeText
		if len(tgt.params) > 0 {
			text += ", "
			interiorNew = text + interiorOld
		} else {
			interiorNew = text
		}
		editsByFile[tgt.declFile] = append(editsByFile[tgt.declFile], textEdit{start: insertAt, end: insertAt, text: text})
		expect = append([]sigNV{np}, expect...)
	} else {
		insertAt := offOf(fset, tgt.decl.Type.Params.Closing)
		text := name + " " + typeText
		if len(tgt.params) > 0 {
			text = ", " + text
			interiorNew = interiorOld + text
		} else {
			interiorNew = text
		}
		editsByFile[tgt.declFile] = append(editsByFile[tgt.declFile], textEdit{start: insertAt, end: insertAt, text: text})
		expect = append(expect, np)
	}
	editsByFile[tgt.declFile] = append(editsByFile[tgt.declFile], declImportEdits...)
	for _, sp := range plans {
		sc := sp.sc
		if _, ok := srcBy[sc.rel]; !ok {
			bsrc, err := readFileBytes(filepath.Join(p.Root, sc.rel))
			if err != nil {
				return nil, fmt.Sprintf("I couldn't read %s: %v", p.LinkPath(sc.rel), err)
			}
			srcBy[sc.rel] = bsrc
		}
		if isCtx {
			text := sp.arg
			if len(sc.argSpan) > 0 {
				text += ", "
			}
			editsByFile[sc.rel] = append(editsByFile[sc.rel], textEdit{start: sc.lpOff + 1, end: sc.lpOff + 1, text: text})
		} else {
			text := sp.arg
			if len(sc.argSpan) > 0 {
				text = ", " + text
			}
			editsByFile[sc.rel] = append(editsByFile[sc.rel], textEdit{start: sc.rpOff, end: sc.rpOff, text: text})
		}
	}
	summary := fmt.Sprintf("Change %s's signature — %s becomes %s. %d call site(s) across %d file(s) each get the new argument.",
		fn.Display(), sigHeaderText(src, tgt, interiorOld), sigHeaderText(src, tgt, interiorNew),
		len(calls), countFiles(calls, tgt.declFile))
	notes := sigSiteNotes(p, tgt, calls, perSite)
	return p.finishSigPlan(fn, tgt, editsByFile, srcBy, summary, notes,
		fmt.Sprintf("%s takes the new %s parameter", fn.Display(), name), expect)
}

// findContextType returns the context.Context type from
// wherever the load resolved the standard package — nil when
// nothing in the module imports context at all.
func findContextType(pkgs []*packages.Package) types.Type {
	for _, pkg := range allLoadedPkgs(pkgs) {
		if pkg.PkgPath == "context" && pkg.Types != nil {
			if obj := pkg.Types.Scope().Lookup("Context"); obj != nil {
				return obj.Type()
			}
		}
	}
	return nil
}

// sigContextTypeFor works out how the declaration file can
// spell context.Context: already imported (possibly aliased),
// or a new import the plan adds. The other name-collision cases
// refuse — a file that imports something else AS "context", or
// is itself in package context, can't take the standard import.
func (p *Project) sigContextTypeFor(tgt *sigTarget, src []byte) (typeText string, importEdits []textEdit, refusal string) {
	if tgt.declAST == nil {
		return "", nil, sigDriftRefusal(tgt.fn)
	}
	for _, is := range tgt.declAST.Imports {
		path, _ := strconv.Unquote(is.Path.Value)
		if path != "context" {
			continue
		}
		if is.Name != nil {
			switch is.Name.Name {
			case "_", ".":
				return "", nil, fmt.Sprintf("context is imported into %s without a usable name (%s) — I can't spell context.Context there. Give the import a normal name and retry.", p.LinkPath(tgt.declFile), is.Name.Name)
			default:
				return is.Name.Name + ".Context", nil, ""
			}
		}
		return "context.Context", nil, ""
	}
	for _, is := range tgt.declAST.Imports {
		spec, eff := importSpecOf(is)
		if eff == "context" {
			return "", nil, fmt.Sprintf("%s already imports %s under the name context — adding the standard context import would collide with it.", p.LinkPath(tgt.declFile), spec)
		}
	}
	if tgt.declAST.Name.Name == "context" {
		return "", nil, fmt.Sprintf("This file's package is itself named context — the new parameter would need the standard library's context.Context, which can't be spelled from inside that package without an import cycle of names. I won't add it here.")
	}
	edits := insertImportSpecs(tgt.declAST, tgt.declPkg.Fset, src, []string{`"context"`})
	return "context.Context", edits, ""
}

// sigDeclTypeImports proves the using-form's TYPE can be
// spelled in the declaration file: every package it names is
// imported there already, or imported somewhere in the project
// (then the plan copies that import in). A bare type name must
// be a builtin or a top-level name of the declaration's own
// package.
func (p *Project) sigDeclTypeImports(tgt *sigTarget, typeExpr ast.Expr, src []byte) ([]textEdit, string) {
	if tgt.declAST == nil {
		return nil, sigDriftRefusal(tgt.fn)
	}
	imported := map[string]bool{}
	for _, is := range tgt.declAST.Imports {
		_, eff := importSpecOf(is)
		imported[eff] = true
	}
	roots := selectorRoots(typeExpr)
	var missing []string
	for _, root := range roots {
		if !imported[root] {
			missing = append(missing, root)
		}
	}
	var specs []string
	if len(missing) > 0 {
		specByRoot := map[string]string{}
		for _, pkg := range p.Packages {
			for _, rel := range pkg.Files {
				bsrc, _, _, refusal := p.readProjectFile(rel)
				if refusal != "" {
					continue
				}
				_, fileAST, err := parseGoSrc(rel, bsrc)
				if err != nil {
					continue
				}
				for _, is := range fileAST.Imports {
					spec, eff := importSpecOf(is)
					if _, ok := specByRoot[eff]; !ok {
						specByRoot[eff] = spec
					}
				}
			}
		}
		for _, root := range missing {
			spec, ok := specByRoot[root]
			if !ok {
				return nil, fmt.Sprintf("The type %s needs the package %q, and nothing in this project imports it — I only copy imports that already exist somewhere in the project, so import it somewhere first (even in another file), then retry.", exprText(tgt.declPkg.Fset, typeExpr), root)
			}
			specs = append(specs, spec)
		}
	}
	if id, ok := typeExpr.(*ast.Ident); ok {
		if !sigBuiltinTypes[id.Name] {
			found := false
			for _, pkg := range p.Packages {
				if pkg.Dir != tgt.fn.PkgDir {
					continue
				}
				for _, t := range pkg.Types {
					if t.Name == id.Name {
						found = true
					}
				}
				for _, f := range pkg.Funcs {
					if f.Name == id.Name {
						found = true
					}
				}
				for _, v := range pkg.Vars {
					if v.Name == id.Name {
						found = true
					}
				}
			}
			if !found {
				return nil, fmt.Sprintf("%q isn't a builtin type and isn't declared in package %s — the new parameter's type has to be spellable where %s is declared.", id.Name, tgt.fn.Pkg, tgt.fn.Name)
			}
		}
	}
	if len(specs) == 0 {
		return nil, ""
	}
	return insertImportSpecs(tgt.declAST, tgt.declPkg.Fset, src, specs), ""
}

// deriveContextArg finds the context a call site should pass:
// the in-scope context.Context variable (parameter or local,
// any name) declared before the call. Exactly one candidate
// wins outright; several are settled by the name "ctx" or by
// being the enclosing function's own parameter; none or an
// unsettled several fails the site.
func deriveContextArg(ctxType types.Type, sc *sigCall) (arg, how, fail string) {
	if ctxType == nil {
		return "", "", "nothing in this project imports context, so no caller can have one in scope"
	}
	info := sc.pkg.TypesInfo
	callPos := sc.call.Pos()
	best := map[string]*types.Var{}
	for _, obj := range info.Defs {
		v, ok := obj.(*types.Var)
		if !ok || !types.Identical(v.Type(), ctxType) {
			continue
		}
		vf := sc.pkg.Fset.File(v.Pos())
		cf := sc.pkg.Fset.File(callPos)
		if vf == nil || cf == nil || vf.Name() != cf.Name() {
			continue
		}
		if v.Parent() == nil || !v.Parent().Contains(callPos) {
			continue
		}
		if sc.pkg.Fset.Position(v.Pos()).Offset >= sc.lpOff {
			continue // declared at or after the call
		}
		if cur, ok := best[v.Name()]; !ok || v.Pos() > cur.Pos() {
			best[v.Name()] = v
		}
	}
	if len(best) == 0 {
		return "", "", "no context is in scope there — the caller has no context parameter or local to pass"
	}
	howFor := func(v *types.Var) string {
		if v.Parent() != nil && v.Parent() == v.Pkg().Scope() {
			return "the package-level " + v.Name()
		}
		if sc.encDecl != nil && info.Defs[sc.encDecl.Name] != nil {
			if fnObj, ok := info.Defs[sc.encDecl.Name].(*types.Func); ok {
				params := fnObj.Type().(*types.Signature).Params()
				for i := 0; i < params.Len(); i++ {
					if params.At(i) == v {
						return "its own context parameter"
					}
				}
			}
		}
		return "its local " + v.Name()
	}
	if len(best) == 1 {
		for name, v := range best {
			return name, howFor(v), ""
		}
	}
	if v, ok := best["ctx"]; ok {
		return "ctx", howFor(v) + " (the one named ctx)", ""
	}
	if sc.encDecl != nil {
		if fnObj, ok := info.Defs[sc.encDecl.Name].(*types.Func); ok {
			params := fnObj.Type().(*types.Signature).Params()
			inParams := map[string]bool{}
			for i := 0; i < params.Len(); i++ {
				if _, ok := best[params.At(i).Name()]; ok {
					inParams[params.At(i).Name()] = true
				}
			}
			if len(inParams) == 1 {
				for name := range inParams {
					return name, howFor(best[name]) + " (the enclosing function's parameter)", ""
				}
			}
		}
	}
	names := make([]string, 0, len(best))
	for name := range best {
		names = append(names, name)
	}
	sort.Strings(names)
	return "", "", fmt.Sprintf("%d contexts are in scope (%s) and none is named ctx or is the caller's own parameter — I can't tell which one this call should pass", len(best), strings.Join(names, ", "))
}

var sigBuiltinNames = map[string]bool{
	"nil": true, "true": true, "false": true, "iota": true,
	"append": true, "cap": true, "close": true, "complex": true,
	"copy": true, "delete": true, "imag": true, "len": true,
	"make": true, "new": true, "panic": true, "print": true,
	"println": true, "real": true, "recover": true,
}

var sigBuiltinTypes = map[string]bool{
	"any": true, "bool": true, "byte": true, "comparable": true,
	"complex64": true, "complex128": true, "error": true,
	"float32": true, "float64": true,
	"int": true, "int8": true, "int16": true, "int32": true, "int64": true,
	"rune": true, "string": true,
	"uint": true, "uint8": true, "uint16": true, "uint32": true, "uint64": true,
	"uintptr": true,
}

// exprUnsatisfiedIn proves the using-form's expression can be
// pasted into a call site's file: every name it uses freely
// (not a selector tail, not a composite-literal field) is an
// import of the site's file, a top-level name of the site's
// package, a name declared in the enclosing function, or a
// builtin. Returns the reason it can't, or "".
func (p *Project) exprUnsatisfiedIn(expr ast.Expr, sc *sigCall) string {
	allowed := map[string]bool{}
	for name := range sigBuiltinNames {
		allowed[name] = true
	}
	for name := range sigBuiltinTypes {
		allowed[name] = true
	}
	var siteAST *ast.File
	for _, file := range sc.pkg.Syntax {
		if file.Pos() <= sc.call.Pos() && sc.call.Pos() < file.End() {
			siteAST = file
			break
		}
	}
	if siteAST != nil {
		for _, is := range siteAST.Imports {
			_, eff := importSpecOf(is)
			allowed[eff] = true
		}
	}
	dir := filepath.ToSlash(filepath.Dir(sc.rel))
	if dir == "." {
		dir = ""
	}
	for _, pkg := range p.Packages {
		if pkg.Dir != dir {
			continue
		}
		for _, f := range pkg.Funcs {
			allowed[f.Name] = true
		}
		for _, t := range pkg.Types {
			allowed[t.Name] = true
		}
		for _, v := range pkg.Vars {
			allowed[v.Name] = true
		}
	}
	if sc.encDecl != nil {
		info := sc.pkg.TypesInfo
		for id, obj := range info.Defs {
			if obj != nil && sc.encDecl.Pos() <= id.Pos() && id.Pos() <= sc.encDecl.End() {
				allowed[obj.Name()] = true
			}
		}
	}
	skip := map[*ast.Ident]bool{}
	ast.Inspect(expr, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.SelectorExpr:
			skip[x.Sel] = true
		case *ast.KeyValueExpr:
			if id, ok := x.Key.(*ast.Ident); ok {
				skip[id] = true
			}
		}
		return true
	})
	bad := ""
	ast.Inspect(expr, func(n ast.Node) bool {
		if bad != "" {
			return false
		}
		id, ok := n.(*ast.Ident)
		if !ok || skip[id] {
			return true
		}
		if !allowed[id.Name] {
			bad = fmt.Sprintf("the expression names %s, which isn't in scope in %s — not imported there, not declared in that package or function, and not a builtin", id.Name, p.LinkPath(sc.rel))
			return false
		}
		return true
	})
	return bad
}

func exprHasFuncLit(expr ast.Expr) bool {
	found := false
	ast.Inspect(expr, func(n ast.Node) bool {
		if _, ok := n.(*ast.FuncLit); ok {
			found = true
			return false
		}
		return !found
	})
	return found
}

// --- finishing and verifying ---------------------------------------------------

// finishSigPlan assembles the plan's files: per file, the gate
// checks, the text edits applied to original bytes, and a
// parse check on the assembled text — a plan whose own assembly
// doesn't parse is refused here rather than shown broken at the
// gate. Files are ordered declaration-file first, then sorted.
func (p *Project) finishSigPlan(fn *Func, tgt *sigTarget, editsByFile map[string][]textEdit, srcByFile map[string][]byte, summary string, notes []string, verifyDesc string, expect []sigNV) (*EditPlan, string) {
	var rels []string
	for rel := range editsByFile {
		if rel != tgt.declFile {
			rels = append(rels, rel)
		}
	}
	sort.Strings(rels)
	rels = append([]string{tgt.declFile}, rels...)
	var files []EditFile
	for _, rel := range rels {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("A file this change has to touch is off-limits: %s — %s.", p.LinkPath(rel), why)
		}
		src, ok := srcByFile[rel]
		if !ok {
			return nil, fmt.Sprintf("I lost track of the current bytes of %s while planning — refusing rather than guessing.", p.LinkPath(rel))
		}
		if fileBytesBlocked(src) {
			return nil, fmt.Sprintf("A file this change has to touch is generated (marked DO NOT EDIT): %s.", p.LinkPath(rel))
		}
		after := applyTextEdits(src, editsByFile[rel])
		if _, _, err := parseGoSrc(rel, after); err != nil {
			return nil, fmt.Sprintf("I assembled an edit to %s that doesn't parse (%v) — refusing rather than showing you a broken plan.", p.LinkPath(rel), err)
		}
		files = append(files, finishFile(rel, src, after))
	}
	decl := files[0]
	siteLine := insertedLine(decl.After, fn.Name+"(")
	pkgDir, recvBase, name, file := fn.PkgDir, tgt.recvBase, fn.Name, tgt.declFile
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(file), siteLine),
		Files:      files,
		Notes:      notes,
		verify: func(after *Project) bool {
			return sigVerify(after, pkgDir, recvBase, name, file, expect)
		},
	}, ""
}

// sigVerify is the structural half of the verdict: on the
// post-write re-analysis, the target declaration (matched by
// package directory, receiver base, name, and file) must expand
// to exactly the expected parameter sequence — names and types,
// whitespace-insensitive. The spine's build over ./... is the
// other half: it covers every package the plan touched.
func sigVerify(after *Project, pkgDir, recvBase, name, file string, expect []sigNV) bool {
	for _, pkg := range after.Packages {
		if pkg.Dir != pkgDir {
			continue
		}
		fset := token.NewFileSet()
		var src []byte
		for _, rel := range pkg.Files {
			if rel != file {
				continue
			}
			bsrc, err := readFileBytes(filepath.Join(after.Root, rel))
			if err != nil {
				return false
			}
			src = bsrc
		}
		if src == nil {
			return false
		}
		fileAST, err := parser.ParseFile(fset, file, src, parser.SkipObjectResolution)
		if err != nil {
			return false
		}
		for _, d := range fileAST.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Name.Name != name {
				continue
			}
			if recvBaseOfDecl(fd) != recvBase {
				continue
			}
			got := expandDeclParams(fset, fd)
			if len(got) != len(expect) {
				return false
			}
			for i := range expect {
				if got[i].name != expect[i].name || got[i].typ != expect[i].typ {
					return false
				}
			}
			return true
		}
		return false
	}
	return false
}

// recvBaseOfDecl is the receiver's base type name for a
// declaration, "" for a plain function.
func recvBaseOfDecl(fd *ast.FuncDecl) string {
	if fd.Recv == nil || len(fd.Recv.List) == 0 {
		return ""
	}
	switch t := fd.Recv.List[0].Type.(type) {
	case *ast.StarExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name
		}
	case *ast.Ident:
		return t.Name
	case *ast.IndexExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name
		}
	case *ast.IndexListExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name
		}
	}
	return ""
}

// expandDeclParams expands a declaration's parameter list the
// way sigTarget does: grouped names share the type, every
// parameter gets its own entry, types whitespace-stripped.
func expandDeclParams(fset *token.FileSet, fd *ast.FuncDecl) []sigNV {
	if fd.Type.Params == nil {
		return nil
	}
	var out []sigNV
	for _, field := range fd.Type.Params.List {
		typ := stripWS(exprText(fset, field.Type))
		if len(field.Names) == 0 {
			out = append(out, sigNV{"", typ})
			continue
		}
		for _, nm := range field.Names {
			out = append(out, sigNV{nm.Name, typ})
		}
	}
	return out
}
