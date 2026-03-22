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

// HuntDeadCode maps every exported and private identifier and cross-references them with their usage.
func (m *Mascot) HuntDeadCode(root string) {
	fset := token.NewFileSet()
	type declInfo struct {
		Path string
		Line int
	}
	declarations := make(map[string]declInfo)
	usageCounts := make(map[string]int)
	declPositions := make(map[token.Pos]bool)

	// Step 1: Scan all files to identify declarations and count all identifiers
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") || strings.Contains(path, "vendor") || strings.Contains(path, "wasm") {
			return nil
		}

		node, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			return nil
		}

		ast.Inspect(node, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.FuncDecl:
				if x.Name.Name != "main" && x.Name.Name != "init" {
					declarations[x.Name.Name] = declInfo{Path: path, Line: fset.Position(x.Name.Pos()).Line}
					declPositions[x.Name.Pos()] = true
				}
			case *ast.TypeSpec:
				declarations[x.Name.Name] = declInfo{Path: path, Line: fset.Position(x.Name.Pos()).Line}
				declPositions[x.Name.Pos()] = true
			case *ast.ValueSpec:
				for _, name := range x.Names {
					declarations[name.Name] = declInfo{Path: path, Line: fset.Position(name.Pos()).Line}
					declPositions[name.Pos()] = true
				}
			case *ast.AssignStmt:
				if x.Tok == token.DEFINE { // :=
					for _, lhs := range x.Lhs {
						if id, ok := lhs.(*ast.Ident); ok {
							declPositions[id.Pos()] = true
						}
					}
				}
			case *ast.Ident:
				usageCounts[x.Name]++
			}
			return true
		})
		return nil
	})

	// Step 2: Separate usage counts from declarations
	trueUsages := make(map[string]int)
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") || strings.Contains(path, "vendor") || strings.Contains(path, "wasm") {
			return nil
		}
		// Use local fset for secondary walk to avoid position overlap issues
		localFset := token.NewFileSet()
		node, _ := parser.ParseFile(localFset, path, nil, 0)
		ast.Inspect(node, func(n ast.Node) bool {
			if id, ok := n.(*ast.Ident); ok {
				// Search for the position in our declPositions map logic
				// In this refined pass, we use a simpler usage check after mapping all declarations
				trueUsages[id.Name]++
			}
			return true
		})
		return nil
	})

	// Step 3: Identify "Zero-Reference" items
	deadCount := 0
	for name, info := range declarations {
		// Heuristic: if trueUsages[name] == 1 (only the declaration itself), it's a ghost
		// Wait, trueUsages[name] counts ALL occurrences since Step 2 uses a fresh walk.
		// So we check if it's strictly > 1 (declaration + at least one usage).
		if trueUsages[name] <= 1 && !strings.HasPrefix(name, "Test") {
			deadCount++
			m.ConfirmRepair(fmt.Sprintf("Potential unused declaration '%s' in %s:%d.", name, info.Path, info.Line), func() error {
				m.Say(Happy, fmt.Sprintf("I've flagged '%s' for removal. This will prune your binary size! 👻", name))
				return nil
			})
		}
	}

	if deadCount > 0 {
		m.Say(Happy, fmt.Sprintf("Removing these items will reduce your binary size and cognitive load!"))
	}
}
