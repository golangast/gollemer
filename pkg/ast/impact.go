// Impact analysis: which files reference a symbol?
//
// CalculateImpactRadius answers "if I change X, what else must I look
// at?" It reloads the module that owns the CodebaseContext with
// go/packages (pattern "./..." from the module root), resolves
// modifiedSymbol to its types.Object, and collects every referencing
// identifier — in the home package, its tests, and importing packages —
// from type-checked Uses/Defs, not text search. Shadowed locals and
// same-named symbols in other packages are therefore not false
// positives.
//
// Symbol forms: "User" (type, func, var, const) and "Store.Get" /
// "(*Store).Get" (methods).
//
// Honest limitations: the radius covers the module that owns the
// symbol; external modules importing it are out of scope. A
// function-local type deliberately shadowing the target name is a
// known (vanishingly rare) false-positive edge.
package ast

import (
	"fmt"
	"go/ast"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
)

// impactTarget identifies the symbol whose references we collect.
type impactTarget struct {
	obj      types.Object // canonical object from the home package
	pkgPath  string       // import path of the home package
	name     string       // symbol name ("User") or method name ("Get")
	kind     string       // e.g. "*types.TypeName", to tell a type from a func
	recv     string       // receiver type name for methods
	isMethod bool
}

// CalculateImpactRadius returns every reference to modifiedSymbol as
// "relative/path.go:line: identifier" entries, sorted and deduplicated,
// including the symbol's own declaration. It also logs the dependent
// file count for cascading refactors.
func CalculateImpactRadius(codeCtx *CodebaseContext, modifiedSymbol string) ([]string, error) {
	if codeCtx == nil {
		return nil, fmt.Errorf("impact: nil CodebaseContext")
	}
	if strings.TrimSpace(modifiedSymbol) == "" {
		return nil, fmt.Errorf("impact: empty modifiedSymbol")
	}
	if codeCtx.SourceDir == "" {
		return nil, fmt.Errorf("impact: CodebaseContext has no SourceDir (load it with LoadPackageContext)")
	}
	if codeCtx.ModulePath == "" {
		return nil, fmt.Errorf("impact: CodebaseContext has no ModulePath")
	}
	recv, name, isMethod, err := parseImpactSymbol(modifiedSymbol)
	if err != nil {
		return nil, err
	}
	moduleRoot, pkgs, err := loadModulePackages(codeCtx.SourceDir)
	if err != nil {
		return nil, fmt.Errorf("impact: %w", err)
	}
	targetPkgPath := codeCtx.ModulePath
	if rel, rerr := filepath.Rel(moduleRoot, codeCtx.SourceDir); rerr == nil && rel != "." {
		targetPkgPath += "/" + filepath.ToSlash(rel)
	}

	// Canonical target object from the home package, preferring the
	// non-test variant.
	var target *impactTarget
	for _, p := range pkgs {
		if p == nil || p.Types == nil || p.TypesInfo == nil {
			continue
		}
		if p.Types.Path() != targetPkgPath {
			continue
		}
		if t := findTargetObject(p, recv, name, isMethod); t != nil {
			t.pkgPath = targetPkgPath
			target = t
			if !strings.Contains(p.ID, "[") {
				break
			}
		}
	}
	if target == nil {
		what := fmt.Sprintf("%q", name)
		if isMethod {
			what = fmt.Sprintf("method %q on type %q", name, recv)
		}
		return nil, fmt.Errorf("impact: symbol %s not found in package %s", what, targetPkgPath)
	}

	// Load errors in the home package poison the type graph; errors
	// elsewhere are tolerated with a warning.
	for _, p := range pkgs {
		if p == nil || len(p.Errors) == 0 || p.Types == nil {
			continue
		}
		if p.Types.Path() != targetPkgPath {
			fmt.Printf("[impact] warning: ignoring load errors in %s\n", p.ID)
			continue
		}
		msgs := make([]string, 0, len(p.Errors))
		for _, e := range p.Errors {
			msgs = append(msgs, e.Msg)
		}
		return nil, fmt.Errorf("impact: home package %s has load errors: %s",
			targetPkgPath, strings.Join(msgs, "; "))
	}

	seen := make(map[string]bool)
	files := make(map[string]bool)
	var refs []string
	for _, p := range pkgs {
		if p == nil || p.TypesInfo == nil || p.Fset == nil || len(p.Syntax) == 0 {
			continue
		}
		info := p.TypesInfo
		fset := p.Fset
		for _, f := range p.Syntax {
			ast.Inspect(f, func(n ast.Node) bool {
				id, ok := n.(*ast.Ident)
				if !ok || id.Name != target.name {
					return true
				}
				var obj types.Object
				if obj = info.Uses[id]; obj == nil {
					obj = info.Defs[id]
				}
				if !referencesTarget(obj, target) {
					return true
				}
				pos := fset.Position(id.Pos())
				rel := pos.Filename
				if r, rerr := filepath.Rel(moduleRoot, pos.Filename); rerr == nil {
					rel = r
				}
				entry := fmt.Sprintf("%s:%d: %s", rel, pos.Line, id.Name)
				if !seen[entry] {
					seen[entry] = true
					refs = append(refs, entry)
					files[rel] = true
				}
				return true
			})
		}
	}
	sort.Strings(refs)

	fileList := make([]string, 0, len(files))
	for f := range files {
		fileList = append(fileList, f)
	}
	sort.Strings(fileList)
	fmt.Printf("[impact] Symbol %q modified -> %d dependent files identified for cascading updates\n",
		modifiedSymbol, len(fileList))
	for _, f := range fileList {
		fmt.Printf("[impact]   %s\n", f)
	}
	return refs, nil
}

// findTargetObject locates the symbol's object among a package's
// definitions. Bare symbols must be package-level; methods are matched
// by receiver type name because a method's *types.Func is not parented
// to the package scope.
func findTargetObject(p *packages.Package, recv, name string, isMethod bool) *impactTarget {
	scope := p.Types.Scope()
	for _, obj := range p.TypesInfo.Defs {
		if obj == nil || obj.Name() != name {
			continue
		}
		if isMethod {
			fn, ok := obj.(*types.Func)
			if !ok || methodReceiverName(fn) != recv {
				continue
			}
			return &impactTarget{obj: fn, name: name, kind: fmt.Sprintf("%T", fn), recv: recv, isMethod: true}
		}
		if obj.Parent() != scope {
			continue
		}
		return &impactTarget{obj: obj, name: name, kind: fmt.Sprintf("%T", obj)}
	}
	return nil
}

// referencesTarget reports whether obj is the target symbol. Object
// identity covers the common case; the fallback matches the same
// logical symbol across separate type-checkings (e.g. test variants),
// where identity does not hold. The kind check keeps a type distinct
// from a func/var of the same name.
func referencesTarget(obj types.Object, target *impactTarget) bool {
	if obj == nil {
		return false
	}
	if obj == target.obj {
		return true
	}
	if obj.Pkg() == nil || obj.Pkg().Path() != target.pkgPath {
		return false
	}
	if obj.Name() != target.name || fmt.Sprintf("%T", obj) != target.kind {
		return false
	}
	if target.isMethod {
		fn, ok := obj.(*types.Func)
		if !ok || methodReceiverName(fn) != target.recv {
			return false
		}
	}
	return true
}

// methodReceiverName returns the named receiver type of a method,
// dereferencing pointers: "(*Store).Get" -> "Store".
func methodReceiverName(fn *types.Func) string {
	sig, ok := fn.Type().(*types.Signature)
	if !ok || sig.Recv() == nil {
		return ""
	}
	t := sig.Recv().Type()
	if ptr, ok := t.(*types.Pointer); ok {
		t = ptr.Elem()
	}
	if named, ok := t.(*types.Named); ok && named.Obj() != nil {
		return named.Obj().Name()
	}
	return ""
}

// parseImpactSymbol splits "User" and "Store.Get" / "(*Store).Get"
// into receiver and name parts.
func parseImpactSymbol(s string) (recv, name string, isMethod bool, err error) {
	s = strings.TrimSpace(s)
	if i := strings.LastIndex(s, "."); i >= 0 {
		recv = strings.TrimSpace(s[:i])
		name = strings.TrimSpace(s[i+1:])
		recv = strings.Trim(recv, "()")
		recv = strings.TrimPrefix(recv, "*")
		isMethod = true
	} else {
		name = s
	}
	if !isGoIdent(name) || (isMethod && !isGoIdent(recv)) {
		return "", "", false, fmt.Errorf("impact: invalid symbol %q (want \"Name\" or \"Type.Method\")", s)
	}
	return recv, name, isMethod, nil
}

func isGoIdent(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		switch {
		case r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z'):
		case i > 0 && r >= '0' && r <= '9':
		default:
			return false
		}
	}
	return true
}

// findModuleRoot walks up from dir to the directory containing go.mod.
func findModuleRoot(dir string) (string, error) {
	d, err := filepath.Abs(dir)
	if err != nil {
		return "", fmt.Errorf("resolve dir %q: %w", dir, err)
	}
	for {
		if _, serr := os.Stat(filepath.Join(d, "go.mod")); serr == nil {
			return d, nil
		}
		parent := filepath.Dir(d)
		if parent == d {
			return "", fmt.Errorf("no go.mod found above %q", dir)
		}
		d = parent
	}
}
