package ui

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
)

func (m *Mascot) AnalyzeComplexity(path string) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return
	}

	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok {
			complexity := 1
			ast.Inspect(fn.Body, func(bn ast.Node) bool {
				switch bn.(type) {
				case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.CaseClause, *ast.CommClause:
					complexity++
				}
				return true
			})

			if complexity > 10 {
				line := fset.Position(fn.Pos()).Line
				m.ConfirmRepair(fmt.Sprintf("Complexity alert: The '%s' function in %s:%d has a score of %d.", fn.Name.Name, path, line, complexity), func() error {
					m.Say(Happy, "Splitting this into multiple testable helpers will reduce your cognitive load. Working on a suggested refactor...")
					return nil
				})
			}
		}
		return true
	})
}
