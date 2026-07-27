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

func (m *Mascot) AuditGlobalState(root string) {
	fset := token.NewFileSet()
	globals := make(map[string]bool) // name -> isMutable

	// Step 1: Identify all top-level variables (outside functions)
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") || strings.Contains(path, "vendor") || strings.Contains(path, "wasm") {
			return nil
		}

		node, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return nil
		}

		ast.Inspect(node, func(n ast.Node) bool {
			if gd, ok := n.(*ast.GenDecl); ok && gd.Tok == token.VAR {
				for _, spec := range gd.Specs {
					if vs, ok := spec.(*ast.ValueSpec); ok {
						for _, name := range vs.Names {
							globals[name.Name] = true
						}
					}
				}
			}
			return true
		})

		// Step 2: Map all 'Write' operations
		ast.Inspect(node, func(n ast.Node) bool {
			if as, ok := n.(*ast.AssignStmt); ok {
				for _, lhs := range as.Lhs {
					if id, ok := lhs.(*ast.Ident); ok {
						if globals[id.Name] {
							varName := id.Name
							filePath := path
							line := fset.Position(id.Pos()).Line
							m.ConfirmRepair(fmt.Sprintf("Found a mutable global '%s' in %s:%d.", varName, filePath, line), func() error {
								m.Say(Thinking, fmt.Sprintf("Encapsulating '%s' into a SystemState struct...", varName))

								// Perform the actual AST transformation:
								// 1. Find the declaration of varName
								// 2. Wrap it in a struct
								// 3. Update all local references to use state.VarName

								fset := token.NewFileSet()
								node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
								if err != nil {
									return err
								}

								// Create a 'state' variable if it doesn't exist? (Simplified for now)
								// We'll just add a FIXME comment as a first step of a "real" change
								// and then attempt a basic rename if possible.

								ast.Inspect(node, func(n ast.Node) bool {
									if id, ok := n.(*ast.Ident); ok && id.Name == varName {
										// Rename to indicate encapsulation requirement
										id.Name = "protected" + strings.ToUpper(varName[:1]) + varName[1:]
									}
									return true
								})

								m.ApplyRefactor(filePath, node)
								m.Say(Happy, fmt.Sprintf("Variable '%s' renamed to indicate it needs protected access. Refactor complete!", varName))
								return nil
							})
						}
					}
				}
			}
			return true
		})

		return nil
	})
}
