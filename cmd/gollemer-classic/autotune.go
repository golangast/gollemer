// Genetic AST allocation auto-tuning.
//
// When the winning MCTS candidate still allocates, autoTuneAllocs
// applies a sequence of small, semantics-preserving AST mutations —
// slice pre-allocation, strings.Builder conversion, value-to-pointer
// receivers — re-validating (go test must still pass) and
// re-benchmarking after each one. A mutation is kept only if it
// strictly reduces AllocsPerOp without breaking tests; otherwise it is
// discarded. At most maxIter mutation attempts are made.
//
// Every mutation is a real AST rewrite (no text sed). Each runs on a
// freshly parsed copy of the files, so a rejected mutation leaves the
// input untouched. The unit-test gate is the soundness backstop: any
// transform that changes observable behavior fails validation and is
// thrown away.
package main

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"strconv"
	"strings"

	"golang.org/x/tools/go/ast/astutil"

	"github.com/golangast/gollemer/pkg/runner"
)

// mutation is one allocation-reducing AST transform. apply rewrites f
// in place and reports whether it changed anything; a false return
// means the pattern was absent and the file must be treated as
// unmodified.
type mutation struct {
	name  string
	apply func(f *ast.File) bool
}

// allocMutations is the ordered battery of micro-mutations tried by
// autoTuneAllocs.
var allocMutations = []mutation{
	{"prealloc-append", preallocAppend},
	{"strings-builder", stringBuilder},
	{"pointer-receiver", pointerReceiver},
}

// autoTuneAllocs runs the micro-mutation loop on files (module-relative
// path -> source). base carries the winner's benchmark metrics.
// It returns the tuned files, their metrics, and the names of the
// mutations that were kept.
func autoTuneAllocs(ctx context.Context, targetDir string, files map[string]string, base *runner.BenchmarkMetrics, maxIter int) (map[string]string, *runner.BenchmarkMetrics, []string) {
	best := cloneFiles(files)
	bestMetrics := base
	var kept []string

	attempts := maxIter
	if attempts > len(allocMutations) {
		attempts = len(allocMutations)
	}
	for i := 0; i < attempts; i++ {
		m := allocMutations[i]
		fmt.Printf("⚡ Auto-Tuning Memory [iter %d/%d]: trying %s...\n", i+1, attempts, m.name)
		mutated, changed := tryMutation(best, m)
		if !changed {
			fmt.Printf("   %s: pattern not present, skipping\n", m.name)
			continue
		}
		res, err := runner.ValidateGeneratedCode(targetDir, mutated)
		if err != nil {
			fmt.Printf("   %s: validation error (%v), discarding\n", m.name, err)
			continue
		}
		if !res.Passed {
			fmt.Printf("   %s: tests failed after mutation, discarding\n", m.name)
			continue
		}
		bm := benchmarkFiles(targetDir, mutated)
		if bm == nil {
			continue
		}
		fmt.Printf("   %s: allocs/op %d -> %d\n", m.name, bestMetrics.AllocsPerOp, bm.AllocsPerOp)
		if bm.Passed && bm.AllocsPerOp < bestMetrics.AllocsPerOp {
			best, bestMetrics = mutated, bm
			kept = append(kept, m.name)
			fmt.Printf("   %s: kept ✅\n", m.name)
		} else {
			fmt.Printf("   %s: no improvement, discarding\n", m.name)
		}
	}
	return best, bestMetrics, kept
}

// tryMutation parses every non-test Go file, applies m, and returns
// the rewritten file set. Files the mutation did not touch keep their
// original source. A false second return means nothing changed.
func tryMutation(files map[string]string, m mutation) (map[string]string, bool) {
	out := cloneFiles(files)
	changed := false
	for path, src := range out {
		if !isTunableFile(path) {
			continue
		}
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, path, src, parser.ParseComments)
		if err != nil {
			return nil, false
		}
		if !m.apply(f) {
			continue
		}
		var buf bytes.Buffer
		if err := format.Node(&buf, fset, f); err != nil {
			return nil, false
		}
		out[path] = buf.String()
		changed = true
	}
	return out, changed
}

// isTunableFile reports whether path is a candidate for mutation:
// Go source, not a test (tests are the harness, never the subject).
func isTunableFile(path string) bool {
	return strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go")
}

// benchmarkFiles profiles files in a throwaway sandbox and returns
// the first benchmark's metrics, or nil on any failure.
func benchmarkFiles(targetDir string, files map[string]string) *runner.BenchmarkMetrics {
	sandbox, cleanup, err := runner.CreateSandbox(targetDir, files)
	if err != nil {
		fmt.Printf("   benchmark sandbox: %v\n", err)
		return nil
	}
	defer cleanup()
	m, err := runner.RunBenchmarkValidation(sandbox, ".")
	if err != nil {
		fmt.Printf("   benchmark: %v\n", err)
		return nil
	}
	return m
}

func cloneFiles(files map[string]string) map[string]string {
	out := make(map[string]string, len(files))
	for k, v := range files {
		out[k] = v
	}
	return out
}

// ---------------------------------------------------------------------------
// Mutation 1: pre-allocate appended slices
// ---------------------------------------------------------------------------

// preallocAppend rewrites
//
//	var xs []T
//	for ... { xs = append(xs, ...) }
//
// into
//
//	xs := make([]T, 0, cap)
//	for ... { xs = append(xs, ...) }
//
// where cap is inferred from the loop: len(rng) for `for _, x := range
// rng`, or the bound N for `for i := 0; i < N; i++`. The declaration
// and the loop must be adjacent statements in the same block. The
// transform is rejected if the variable is redefined anywhere in the
// loop body (shadowing) or if nothing is appended inside the loop.
//
// Caveat: `var xs []T` is nil while `make` is not; code that tests
// `xs == nil` afterwards changes meaning. The validation gate
// (tests must still pass) catches that.
func preallocAppend(f *ast.File) bool {
	changed := false
	ast.Inspect(f, func(n ast.Node) bool {
		block, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i := 0; i+1 < len(block.List); i++ {
			name, elem := varSliceDecl(block.List[i])
			if name == "" {
				continue
			}
			capExpr := loopCap(block.List[i+1])
			if capExpr == nil {
				continue
			}
			if !loopAppendsTo(block.List[i+1], name) {
				continue
			}
			block.List[i] = &ast.AssignStmt{
				Lhs: []ast.Expr{ast.NewIdent(name)},
				Tok: token.DEFINE,
				Rhs: []ast.Expr{&ast.CallExpr{
					Fun: ast.NewIdent("make"),
					Args: []ast.Expr{
						&ast.ArrayType{Elt: elem},
						&ast.BasicLit{Kind: token.INT, Value: "0"},
						capExpr,
					},
				}},
			}
			changed = true
		}
		return true
	})
	return changed
}

// varSliceDecl matches `var <name> []T` (no initializer) and returns
// the name and element type.
func varSliceDecl(stmt ast.Stmt) (string, ast.Expr) {
	ds, ok := stmt.(*ast.DeclStmt)
	if !ok {
		return "", nil
	}
	gd, ok := ds.Decl.(*ast.GenDecl)
	if !ok || gd.Tok != token.VAR || len(gd.Specs) != 1 {
		return "", nil
	}
	vs, ok := gd.Specs[0].(*ast.ValueSpec)
	if !ok || len(vs.Names) != 1 || len(vs.Values) != 0 {
		return "", nil
	}
	at, ok := vs.Type.(*ast.ArrayType)
	if !ok || at.Len != nil {
		return "", nil
	}
	return vs.Names[0].Name, at.Elt
}

// loopCap infers an append capacity from a loop statement: len(rng)
// for range loops, the bound N for classic `for i := 0; i < N; i++`
// loops. It returns nil when the shape is not recognized.
func loopCap(stmt ast.Stmt) ast.Expr {
	switch l := stmt.(type) {
	case *ast.RangeStmt:
		return &ast.CallExpr{Fun: ast.NewIdent("len"), Args: []ast.Expr{l.X}}
	case *ast.ForStmt:
		init, ok := l.Init.(*ast.AssignStmt)
		if !ok || init.Tok != token.DEFINE || len(init.Lhs) != 1 || len(init.Rhs) != 1 {
			return nil
		}
		ctr, ok := init.Lhs[0].(*ast.Ident)
		if !ok {
			return nil
		}
		if lit, ok := init.Rhs[0].(*ast.BasicLit); !ok || lit.Kind != token.INT || lit.Value != "0" {
			return nil
		}
		cond, ok := l.Cond.(*ast.BinaryExpr)
		if !ok || cond.Op != token.LSS {
			return nil
		}
		cx, ok := cond.X.(*ast.Ident)
		if !ok || cx.Name != ctr.Name {
			return nil
		}
		inc, ok := l.Post.(*ast.IncDecStmt)
		if !ok || inc.Tok != token.INC {
			return nil
		}
		if ix, ok := inc.X.(*ast.Ident); !ok || ix.Name != ctr.Name {
			return nil
		}
		return cond.Y
	}
	return nil
}

// loopAppendsTo reports whether the loop body contains `name =
// append(name, ...)` (plain assignment, not :=) and no redefinition
// of name anywhere in the loop (which would shadow the outer var).
func loopAppendsTo(stmt ast.Stmt, name string) bool {
	var body *ast.BlockStmt
	switch l := stmt.(type) {
	case *ast.RangeStmt:
		body = l.Body
	case *ast.ForStmt:
		body = l.Body
	default:
		return false
	}
	appends := false
	ok := true
	ast.Inspect(body, func(n ast.Node) bool {
		if !ok {
			return false
		}
		as, isAssign := n.(*ast.AssignStmt)
		if !isAssign {
			return true
		}
		for _, lhs := range as.Lhs {
			id, isIdent := lhs.(*ast.Ident)
			if !isIdent || id.Name != name {
				continue
			}
			if as.Tok == token.DEFINE {
				ok = false // shadowed inside the loop; not our variable
				return false
			}
			if as.Tok == token.ASSIGN && len(as.Lhs) == 1 && len(as.Rhs) == 1 {
				if call, isCall := as.Rhs[0].(*ast.CallExpr); isCall {
					if fun, isIdent := call.Fun.(*ast.Ident); isIdent && fun.Name == "append" {
						if arg, isIdent := call.Args[0].(*ast.Ident); isIdent && arg.Name == name {
							appends = true
						}
					}
				}
			}
		}
		return true
	})
	return ok && appends
}

// ---------------------------------------------------------------------------
// Mutation 2: string concatenation loops -> strings.Builder
// ---------------------------------------------------------------------------

// stringBuilder rewrites
//
//	var s string
//	for ... { s += part }
//
// into
//
//	var s strings.Builder
//	for ... { s.WriteString(part) }
//
// and rewrites every remaining load of s into s.String(). It bails out
// (leaving the file untouched) if s is redefined anywhere in the
// function, assigned any other way, address-taken, or referenced
// inside an `s += ...` right-hand side. Because the variable keeps its
// name and every use is converted, closures over s stay correct.
func stringBuilder(f *ast.File) bool {
	changed := false
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		spec, nameIdent := findStringVar(fn.Body)
		if spec == nil {
			continue
		}
		if rewriteConcatLoop(f, fn, spec, nameIdent) {
			changed = true
		}
	}
	return changed
}

// findStringVar locates the first `var <name> string` (no
// initializer) in the function body.
func findStringVar(body *ast.BlockStmt) (*ast.ValueSpec, *ast.Ident) {
	var found *ast.ValueSpec
	var name *ast.Ident
	ast.Inspect(body, func(n ast.Node) bool {
		if found != nil {
			return false
		}
		vs, ok := n.(*ast.ValueSpec)
		if !ok || len(vs.Names) != 1 || len(vs.Values) != 0 {
			return true
		}
		if id, ok := vs.Type.(*ast.Ident); !ok || id.Name != "string" {
			return true
		}
		found, name = vs, vs.Names[0]
		return false
	})
	return found, name
}

// rewriteConcatLoop converts every `s += e` in fn into
// `s.WriteString(e)` and every other use of s into `s.String()`,
// then retypes the declaration. It returns false without effect when
// the pattern is unsafe or absent. The AST may be left partially
// rewritten on false; callers discard the file in that case.
func rewriteConcatLoop(f *ast.File, fn *ast.FuncDecl, spec *ast.ValueSpec, declName *ast.Ident) bool {
	name := declName.Name
	converted := 0
	safe := true
	astutil.Apply(fn, func(c *astutil.Cursor) bool {
		if !safe {
			return false
		}
		switch n := c.Node().(type) {
		case *ast.AssignStmt:
			for _, lhs := range n.Lhs {
				id, ok := lhs.(*ast.Ident)
				if !ok || id.Name != name {
					continue
				}
				if n.Tok == token.ADD_ASSIGN && len(n.Lhs) == 1 && len(n.Rhs) == 1 &&
					!referencesIdent(n.Rhs[0], name) {
					c.Replace(&ast.ExprStmt{X: &ast.CallExpr{
						Fun:  &ast.SelectorExpr{X: ast.NewIdent(name), Sel: ast.NewIdent("WriteString")},
						Args: []ast.Expr{n.Rhs[0]},
					}})
					converted++
					return false
				}
				safe = false // any other store to s: not convertible
				return false
			}
		case *ast.Ident:
			if n.Name != name || n == declName {
				return true
			}
			if isNameDefinition(n, c.Parent()) {
				safe = false // redefined: shadowing risk
				return false
			}
			if sel, ok := c.Parent().(*ast.SelectorExpr); ok && sel.Sel == n {
				return true // field/method name, not our variable
			}
			c.Replace(&ast.CallExpr{
				Fun: &ast.SelectorExpr{X: ast.NewIdent(name), Sel: ast.NewIdent("String")},
			})
			return false
		case *ast.UnaryExpr:
			if n.Op == token.AND {
				if id, ok := n.X.(*ast.Ident); ok && id.Name == name {
					safe = false // &s: address taken, type change unsafe
					return false
				}
			}
		}
		return true
	}, nil)
	if !safe || converted == 0 {
		return false
	}
	spec.Type = &ast.SelectorExpr{X: ast.NewIdent("strings"), Sel: ast.NewIdent("Builder")}
	ensureImport(f, "strings")
	return true
}

// referencesIdent reports whether e contains an identifier with the
// given name.
func referencesIdent(e ast.Expr, name string) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
			return false
		}
		return !found
	})
	return found
}

// isNameDefinition reports whether the identifier occurrence is a
// definition site (var spec, :=, range variable, parameter, func/type
// name, label) rather than a use.
func isNameDefinition(id *ast.Ident, parent ast.Node) bool {
	switch p := parent.(type) {
	case *ast.ValueSpec:
		for _, nm := range p.Names {
			if nm == id {
				return true
			}
		}
	case *ast.AssignStmt:
		if p.Tok == token.DEFINE {
			for _, lhs := range p.Lhs {
				if lhs == id {
					return true
				}
			}
		}
	case *ast.RangeStmt:
		if p.Key == id || p.Value == id {
			return true
		}
	case *ast.Field:
		for _, nm := range p.Names {
			if nm == id {
				return true
			}
		}
	case *ast.FuncDecl:
		return p.Name == id
	case *ast.TypeSpec:
		return p.Name == id
	case *ast.LabeledStmt:
		return p.Label == id
	case *ast.ImportSpec:
		return p.Name == id
	}
	return false
}

// ensureImport adds `import <path>` to f if not already imported.
// New nodes are anchored at real source positions (just after the
// package clause): go/printer interleaves comments by position, and
// unpositioned nodes make it emit the import split across the next
// declaration's doc comment.
func ensureImport(f *ast.File, path string) {
	if f == nil {
		return
	}
	for _, imp := range f.Imports {
		if unquote(imp.Path.Value) == path {
			return
		}
	}
	anchor := f.Name.End()
	mkSpec := func(pos token.Pos) *ast.ImportSpec {
		return &ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: strconv.Quote(path), ValuePos: pos}}
	}
	for _, decl := range f.Decls {
		gd, ok := decl.(*ast.GenDecl)
		if ok && gd.Tok == token.IMPORT {
			pos := anchor
			if n := len(gd.Specs); n > 0 {
				pos = gd.Specs[n-1].End()
			}
			gd.Specs = append(gd.Specs, mkSpec(pos))
			return
		}
	}
	f.Decls = append([]ast.Decl{&ast.GenDecl{
		Tok:    token.IMPORT,
		TokPos: anchor,
		Specs:  []ast.Spec{mkSpec(anchor)},
	}}, f.Decls...)
}

func unquote(s string) string {
	if u, err := strconv.Unquote(s); err == nil {
		return u
	}
	return strings.Trim(s, `"`)
}

// ---------------------------------------------------------------------------
// Mutation 3: value receivers -> pointer receivers
// ---------------------------------------------------------------------------

// pointerReceiver converts the first value-receiver method in the file
// to a pointer receiver: `func (s T) M()` becomes `func (s *T) M()`.
// Pointer receivers avoid copying the receiver on every call, which
// also avoids the heap allocation a copying call can cause. Methods
// already on pointer receivers are skipped. One method per call.
//
// Caveat: this changes T's method set (M moves from T to *T), which
// can break interface satisfaction elsewhere. The validation gate
// discards the mutation if anything stops compiling or tests fail.
func pointerReceiver(f *ast.File) bool {
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Recv == nil || len(fn.Recv.List) != 1 {
			continue
		}
		recv := fn.Recv.List[0]
		id, ok := recv.Type.(*ast.Ident)
		if !ok || id.Name == "" || id.Name == "_" {
			continue // already *T, or a complex receiver type
		}
		recv.Type = &ast.StarExpr{X: id}
		return true
	}
	return false
}
