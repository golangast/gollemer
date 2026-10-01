package synthesis

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"strings"
)

// OptimizationReport describes what OptimizeAllocations changed.
type OptimizationReport struct {
	// OptimizedCode is the rewritten source, gofmt-rendered.
	OptimizedCode string `json:"optimizedCode"`
	// AllocsBefore/AllocsAfter count unbounded allocation sites:
	// append calls on slices without a known capacity. A pre-sized
	// make counts as one bounded allocation and is not included.
	AllocsBefore int      `json:"allocsBefore"`
	AllocsAfter  int      `json:"allocsAfter"`
	Changes      []string `json:"changes"`

	coveredAppends int // unexported: appends neutralized by preallocation
}

// OptimizeAllocations rewrites file to reduce heap allocations and
// reports the before/after count of unbounded allocation sites.
// Two transforms, both conservative and documented:
//
//  1. Slice preallocation: `var s []T` followed in the same block by
//     a counted loop (`for i := 0; i < N; i++`) that appends to s
//     becomes `s := make([]T, 0, N)`. The loop bound may be an int
//     literal or a variable/selector expression.
//  2. Receiver narrowing: a value receiver on a struct declared in
//     the same file with more than four fields becomes a pointer
//     receiver, avoiding a copy of the whole struct on every call.
//
// The function mutates file in place and returns the rendered source.
// If no transform applies, OptimizedCode equals the input rendering
// and AllocsBefore == AllocsAfter.
func OptimizeAllocations(fset *token.FileSet, file *ast.File) *OptimizationReport {
	rep := &OptimizationReport{}
	rep.AllocsBefore = countUnboundedAllocs(file)
	rep.Changes = []string{}

	preallocateSlices(file, rep)
	narrowReceivers(file, rep)

	var buf bytes.Buffer
	if err := format.Node(&buf, fset, file); err != nil {
		rep.OptimizedCode = ""
	} else {
		rep.OptimizedCode = buf.String()
	}
	// Appends covered by a new preallocation no longer allocate.
	rep.AllocsAfter = rep.AllocsBefore - rep.coveredAppends
	if rep.AllocsAfter < 0 {
		rep.AllocsAfter = 0
	}
	if len(rep.Changes) == 0 {
		rep.Changes = []string{"no optimizable patterns found"}
	}
	return rep
}

// coveredAppends tracks appends neutralized by preallocation.
func (r *OptimizationReport) markCovered(n int) { r.coveredAppends += n }

// countUnboundedAllocs counts append calls whose target slice has no
// statically known capacity: each is a potential growth reallocation.
func countUnboundedAllocs(file *ast.File) int {
	n := 0
	ast.Inspect(file, func(x ast.Node) bool {
		if call, ok := x.(*ast.CallExpr); ok {
			if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "append" {
				n++
			}
		}
		return true
	})
	return n
}

// preallocateSlices finds `var s []T` + counted append loop patterns
// and rewrites them to `s := make([]T, 0, N)`.
func preallocateSlices(file *ast.File, rep *OptimizationReport) {
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		ast.Inspect(fn.Body, func(x ast.Node) bool {
			if blk, ok := x.(*ast.BlockStmt); ok {
				preallocateInBlock(blk, rep)
			}
			return true
		})
	}
}

func preallocateInBlock(body *ast.BlockStmt, rep *OptimizationReport) {
	for i, stmt := range body.List {
		ds, ok := stmt.(*ast.DeclStmt)
		if !ok {
			continue
		}
		gd, ok := ds.Decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.VAR || len(gd.Specs) != 1 {
			continue
		}
		vs, ok := gd.Specs[0].(*ast.ValueSpec)
		if !ok || len(vs.Names) != 1 || len(vs.Values) != 0 {
			continue
		}
		arr, ok := vs.Type.(*ast.ArrayType)
		if !ok || arr.Len != nil {
			continue // not a slice type
		}
		name := vs.Names[0].Name
		// Look for a counted append loop later in the same block.
		for j := i + 1; j < len(body.List); j++ {
			bound, covered := loopAppendsTo(body.List[j], name)
			if bound == nil {
				continue
			}
			mk := &ast.CallExpr{
				Fun: ast.NewIdent("make"),
				Args: []ast.Expr{
					&ast.ArrayType{Elt: arr.Elt},
					&ast.BasicLit{Kind: token.INT, Value: "0"},
					bound,
				},
			}
			body.List[i] = &ast.AssignStmt{
				Lhs:    []ast.Expr{ast.NewIdent(name)},
				TokPos: gd.Pos(), // keep position info so go/printer lays out correctly
				Tok:    token.DEFINE,
				Rhs:    []ast.Expr{mk},
			}
			rep.markCovered(covered)
			rep.Changes = append(rep.Changes,
				fmt.Sprintf("preallocated %s with make([]%s, 0, %s)", name, exprString(arr.Elt), exprString(bound)))
			break // one rewrite per declaration
		}
	}
}

// isPureBound reports whether e is safe to evaluate once up front:
// a literal, a variable, or a field selection. Anything else (a call,
// a function value, ...) might have side effects the loop relied on
// re-evaluating each iteration.
func isPureBound(e ast.Expr) bool {
	switch e.(type) {
	case *ast.BasicLit, *ast.Ident, *ast.SelectorExpr:
		return true
	}
	return false
}

// loopAppendsTo reports the bound expression of a `for i := 0; i < B;
// i++` loop whose body appends to name, plus how many distinct append
// call sites it covers. It returns (nil, 0) when stmt is not such a loop.
func loopAppendsTo(stmt ast.Stmt, name string) (ast.Expr, int) {
	fs, ok := stmt.(*ast.ForStmt)
	if !ok || fs.Init == nil || fs.Cond == nil || fs.Post == nil {
		return nil, 0
	}
	// Init: i := 0
	initAssign, ok := fs.Init.(*ast.AssignStmt)
	if !ok || initAssign.Tok != token.DEFINE || len(initAssign.Rhs) != 1 {
		return nil, 0
	}
	if lit, ok := initAssign.Rhs[0].(*ast.BasicLit); !ok || lit.Value != "0" {
		return nil, 0
	}
	// Cond: i < B
	cond, ok := fs.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.LSS {
		return nil, 0
	}
	if _, ok := cond.X.(*ast.Ident); !ok {
		return nil, 0
	}
	if !isPureBound(cond.Y) {
		return nil, 0
	}
	// Post: i++
	post, ok := fs.Post.(*ast.IncDecStmt)
	if !ok || post.Tok != token.INC {
		return nil, 0
	}
	covered := 0
	ast.Inspect(fs.Body, func(x ast.Node) bool {
		assign, ok := x.(*ast.AssignStmt)
		if !ok || len(assign.Rhs) != 1 {
			return true
		}
		call, ok := assign.Rhs[0].(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok || id.Name != "append" || len(call.Args) == 0 {
			return true
		}
		if target, ok := call.Args[0].(*ast.Ident); ok && target.Name == name {
			covered++
		}
		return true
	})
	if covered == 0 {
		return nil, 0
	}
	return cond.Y, covered
}

// narrowReceivers rewrites value receivers to pointer receivers for
// structs declared in this file with more than four fields.
func narrowReceivers(file *ast.File, rep *OptimizationReport) {
	fieldCounts := map[string]int{}
	for _, decl := range file.Decls {
		gd, ok := decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.TYPE {
			continue
		}
		for _, spec := range gd.Specs {
			ts, ok := spec.(*ast.TypeSpec)
			if !ok {
				continue
			}
			st, ok := ts.Type.(*ast.StructType)
			if !ok || st.Fields == nil {
				continue
			}
			n := 0
			for _, f := range st.Fields.List {
				n += len(f.Names)
			}
			fieldCounts[ts.Name.Name] = n
		}
	}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Recv == nil || len(fn.Recv.List) != 1 {
			continue
		}
		recv := fn.Recv.List[0]
		id, ok := recv.Type.(*ast.Ident)
		if !ok {
			continue // already a pointer or exotic receiver
		}
		if fieldCounts[id.Name] <= 4 {
			continue
		}
		recv.Type = &ast.StarExpr{Star: id.Pos(), X: &ast.Ident{NamePos: id.Pos(), Name: id.Name}}
		rep.Changes = append(rep.Changes,
			fmt.Sprintf("receiver of %s: %s → *%s (avoids copying %d fields per call)",
				fn.Name.Name, id.Name, id.Name, fieldCounts[id.Name]))
	}
}

// exprString renders a small expression for change descriptions.
func exprString(e ast.Expr) string {
	var buf bytes.Buffer
	if err := format.Node(&buf, token.NewFileSet(), e); err != nil {
		return "?"
	}
	return strings.TrimSpace(buf.String())
}
