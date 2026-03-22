package ui

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

func (m *Mascot) AuditMemoryLeaks(root string) {
	fset := token.NewFileSet()

	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") || strings.Contains(path, "vendor") || strings.Contains(path, "wasm") {
			return nil
		}

		node, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return nil
		}

		ast.Inspect(node, func(n ast.Node) bool {
			// Check for 'go func()' without context or cancellation
			if goStmt, ok := n.(*ast.GoStmt); ok {
				if _, ok := goStmt.Call.Fun.(*ast.FuncLit); ok {
					// Use heuristic: does it use a context?
					hasContext := false
					ast.Inspect(goStmt, func(cn ast.Node) bool {
						if id, ok := cn.(*ast.Ident); ok {
							if strings.Contains(strings.ToLower(id.Name), "ctx") || id.Name == "context" {
								hasContext = true
							}
						}
						return true
					})

					if !hasContext {
						m.Say(Alert, fmt.Sprintf("In %s, a goroutine is started without a clear 'context' or stop signal.", filepath.Base(path)))
						m.Say(Happy, "Adding a 'context.WithCancel' here will ensure the worker dies when you exit the app!")
					}
				}
			}
			return true
		})
		return nil
	})
}
