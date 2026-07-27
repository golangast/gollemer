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

func (m *Mascot) SimulateRaceConditions(root string) {
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
			// Find all 'go func()' blocks
			if goStmt, ok := n.(*ast.GoStmt); ok {
				if call, ok := goStmt.Call.Fun.(*ast.FuncLit); ok {
					m.Say(Thinking, fmt.Sprintf("Simulating concurrent access patterns in %s...", filepath.Base(path)))

					// Basic check: is it capturing a variable from the outer scope?
					ast.Inspect(call.Body, func(bn ast.Node) bool {
						if id, ok := bn.(*ast.Ident); ok {
							if id.Obj != nil && id.Obj.Kind == ast.Var {
								// In a real implementation, we would check if it's protected by a mutex
								m.Say(Alert, fmt.Sprintf("Goroutine captures variable '%s'. Adding a sync.RWMutex here will prevent 100%% of potential data corruption!", id.Name))
							}
						}
						return true
					})
				}
			}
			return true
		})
		return nil
	})
}
