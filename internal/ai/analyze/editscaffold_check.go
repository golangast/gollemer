package analyze

// The scaffold's self-check: the delegate's AST self-check,
// extended to the generated test file (the planDelegate bargain
// — generated code that fails our own checks is our bug, and no
// plan is shown carrying it). It holds the scaffold to what it
// claims to be: a test function with the scaffold's name that
// SKIPS before it runs anything, whose args struct's fields
// match the wrapper's signature field for field, and whose run
// loop calls the wrapper by its real name.

import (
	"go/ast"
	"go/token"
	"strings"
)

// fileDeclaresFunc reports whether a parsed file declares a
// top-level function with the given name.
func fileDeclaresFunc(f *ast.File, name string) bool {
	for _, d := range f.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == name {
			return true
		}
	}
	return false
}

// selfCheckScaffold checks the scaffold function on its own
// (before it is placed in a file) by parsing it under a package
// clause.
func selfCheckScaffold(body, stubName string, b *scaffoldBuilder) string {
	src := "package scaffoldcheck\n\n" + body
	fset, fAST, err := parseGoSrc("scaffold_test.go", []byte(src))
	if err != nil || fAST == nil {
		return "the assembled scaffold doesn't parse"
	}
	return checkScaffoldAST(fset, fAST, stubName, b)
}

// selfCheckScaffoldFile checks the scaffold as placed in the
// finished test file (created or appended).
func selfCheckScaffoldFile(after []byte, stubName string, b *scaffoldBuilder) string {
	fset, fAST, err := parseGoSrc("scaffold_test.go", after)
	if err != nil || fAST == nil {
		return "the assembled scaffold doesn't parse in its file"
	}
	return checkScaffoldAST(fset, fAST, stubName, b)
}

// checkScaffoldAST walks a parsed file holding the scaffold to
// its invariants. Returns "" when it is exactly what it claims.
func checkScaffoldAST(fset *token.FileSet, fileAST *ast.File, stubName string, b *scaffoldBuilder) string {
	var decl *ast.FuncDecl
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == stubName && fd.Recv == nil {
			decl = fd
			break
		}
	}
	if decl == nil || decl.Body == nil {
		return "the file has no test function named " + stubName
	}
	// It must skip before it runs anything: the first statement
	// is the t.Skip call. A scaffold that could run its
	// placeholder expectations is a liar in one direction or
	// the other (failing a suite, or passing as if it checked).
	skips := false
	if len(decl.Body.List) > 0 {
		if es, ok := decl.Body.List[0].(*ast.ExprStmt); ok {
			if call, ok := es.X.(*ast.CallExpr); ok {
				if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Skip" {
					if id, ok := sel.X.(*ast.Ident); ok && id.Name == "t" {
						skips = true
					}
				}
			}
		}
	}
	if !skips {
		return stubName + " doesn't skip itself before its cases — its placeholder expectations could run"
	}
	// The run loop must call the wrapper by its real name.
	calls := false
	runs := false
	ast.Inspect(decl, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if id, ok := call.Fun.(*ast.Ident); ok && id.Name == b.g.wrapper {
			calls = true
		}
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Run" {
			runs = true
		}
		return true
	})
	if !runs {
		return stubName + " has no t.Run loop"
	}
	if !calls {
		return stubName + " never calls " + b.g.wrapper
	}
	// The args struct must mirror the signature: same field
	// count, same names, same types (compared space-free, so
	// formatting never enters into it).
	var argsStruct *ast.StructType
	ast.Inspect(decl, func(n ast.Node) bool {
		if argsStruct != nil {
			return false
		}
		ts, ok := n.(*ast.TypeSpec)
		if ok && ts.Name.Name == "args" {
			if st, ok := ts.Type.(*ast.StructType); ok {
				argsStruct = st
			}
			return false
		}
		return true
	})
	if argsStruct == nil {
		return stubName + " has no args struct"
	}
	var fields []scaffoldField
	for _, f := range argsStruct.Fields.List {
		for _, name := range f.Names {
			fields = append(fields, scaffoldField{name: name.Name, expr: f.Type})
		}
	}
	if len(fields) != len(b.params) {
		return "the args struct's fields don't match the wrapper's parameters"
	}
	norm := func(s string) string { return strings.Join(strings.Fields(s), "") }
	for i, f := range fields {
		if f.name != b.params[i].name || norm(exprText(fset, f.expr)) != norm(b.params[i].typ) {
			return "the args struct's fields don't match the wrapper's parameters"
		}
	}
	return ""
}
