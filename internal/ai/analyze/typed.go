package analyze

// Typed refinement: best-effort compiler-grade type info via
// golang.org/x/tools/go/packages. The heuristic call graph guesses at
// method calls through interfaces, embedded structs, and constructor
// variables; when a repo loads with type info (its deps are available),
// selector calls are resolved through real method sets instead. When
// loading fails — missing deps, parse errors — the heuristic graph
// stands unchanged. This never fails the analysis.

import (
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/packages"
)

// refineTypes loads the project with type info and adds the call edges
// the heuristics missed or got wrong. It runs inside BuildGraph before
// effect propagation, so refined edges carry effects.
func (p *Project) refineTypes() {
	if p.Module == "" {
		return
	}
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedTypes | packages.NeedSyntax | packages.NeedTypesInfo,
		Dir:  p.Root,
	}
	pkgs, err := packages.Load(cfg, "./...")
	if err != nil {
		return
	}
	for _, pkg := range pkgs {
		if len(pkg.Errors) > 0 {
			return // partial type info is worse than the heuristics
		}
	}
	p.Typed = true
	for _, pkg := range pkgs {
		p.refinePackage(pkg)
	}
}

func (p *Project) refinePackage(pkg *packages.Package) {
	info := pkg.TypesInfo
	if info == nil {
		return
	}
	rel := strings.TrimPrefix(strings.TrimPrefix(pkg.PkgPath, p.Module), "/")
	for _, file := range pkg.Syntax {
		for _, decl := range file.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			fn := p.byID[funcID(rel, declRecv(fd), fd.Name.Name)]
			if fn == nil {
				continue
			}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				if target := p.typedTarget(info, call); target != nil && target != fn {
					p.addEdge(fn, target)
				}
				return true
			})
		}
	}
}

// typedTarget resolves a call through the type checker's method sets.
// Selections covers x.Method() for concrete types, interfaces, and
// embedded fields — the cases the name heuristics guess at.
func (p *Project) typedTarget(info *types.Info, call *ast.CallExpr) *Func {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return nil
	}
	s, ok := info.Selections[sel]
	if !ok {
		return nil
	}
	obj, ok := s.Obj().(*types.Func)
	if !ok || obj.Pkg() == nil {
		return nil
	}
	return p.funcForObject(obj)
}

// funcForObject maps a type-checker function object back to the model.
func (p *Project) funcForObject(obj *types.Func) *Func {
	rel := strings.TrimPrefix(strings.TrimPrefix(obj.Pkg().Path(), p.Module), "/")
	sig, ok := obj.Type().(*types.Signature)
	if !ok {
		return nil
	}
	recv := ""
	if r := sig.Recv(); r != nil {
		recv = namedTypeName(r.Type())
	}
	return p.byID[funcID(rel, recv, obj.Name())]
}

// namedTypeName extracts "Store" from *pkg.Store or pkg.Store.
func namedTypeName(t types.Type) string {
	if ptr, ok := t.(*types.Pointer); ok {
		t = ptr.Elem()
	}
	if named, ok := t.(*types.Named); ok {
		return named.Obj().Name()
	}
	return ""
}

// declRecv renders a FuncDecl receiver the way the model records it
// ("Store" for func (s *Store) Put).
func declRecv(fd *ast.FuncDecl) string {
	if fd.Recv == nil || len(fd.Recv.List) == 0 {
		return ""
	}
	return exprName(fd.Recv.List[0].Type)
}

// addEdge records a call edge the heuristics missed.
func (p *Project) addEdge(caller, callee *Func) {
	for _, id := range caller.Calls {
		if id == callee.ID {
			return
		}
	}
	caller.Calls = append(caller.Calls, callee.ID)
	callee.Callers = append(callee.Callers, caller.ID)
}

