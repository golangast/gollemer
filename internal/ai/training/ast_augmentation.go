// Package training provides AST-based augmentation for Go code, including
// scope annotation, AST node masking for FIM training, and project context extraction.
package training

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
)

// ScopeAnnotation represents the scope context of a code location.
type ScopeAnnotation struct {
	Kind      string // "package", "import", "type", "struct", "interface", "func", "method", "if", "for", "switch", "select"
	Name      string // The name of the scope (e.g., function name, type name)
	Receiver  string // For methods: the receiver type (e.g., "*MoE")
	Signature string // Full signature for functions/methods
	Depth     int    // Nesting depth (0 = top-level)
	LineStart int    // Starting line number
	LineEnd   int    // Ending line number
}

// AnnotateScope analyzes Go source code and returns scope annotations for each
// significant code block. This is used to inject <SCOPE:...> tokens into FIM prompts.
func AnnotateScope(code string) ([]ScopeAnnotation, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	var annotations []ScopeAnnotation

	// Package scope
	pkgName := f.Name.Name
	annotations = append(annotations, ScopeAnnotation{
		Kind:      "package",
		Name:      pkgName,
		Depth:     0,
		LineStart: fset.Position(f.Package).Line,
	})

	ast.Inspect(f, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			start := fset.Position(node.Pos()).Line
			end := fset.Position(node.End()).Line
			receiver := ""
			if node.Recv != nil && len(node.Recv.List) > 0 {
				receiver = types.ExprString(node.Recv.List[0].Type)
			}

			kind := "func"
			if receiver != "" {
				kind = "method"
			}

			sig := buildFuncSignature(node, receiver)
			annotations = append(annotations, ScopeAnnotation{
				Kind:      kind,
				Name:      node.Name.Name,
				Receiver:  receiver,
				Signature: sig,
				Depth:     1,
				LineStart: start,
				LineEnd:   end,
			})

		case *ast.TypeSpec:
			start := fset.Position(node.Pos()).Line
			end := fset.Position(node.End()).Line
			kind := "type"
			if _, ok := node.Type.(*ast.StructType); ok {
				kind = "struct"
			} else if _, ok := node.Type.(*ast.InterfaceType); ok {
				kind = "interface"
			}
			annotations = append(annotations, ScopeAnnotation{
				Kind:      kind,
				Name:      node.Name.Name,
				Depth:     1,
				LineStart: start,
				LineEnd:   end,
			})
		}
		return true
	})

	return annotations, nil
}

// buildFuncSignature constructs a human-readable function/method signature.
func buildFuncSignature(fn *ast.FuncDecl, receiver string) string {
	var parts []string
	if receiver != "" {
		parts = append(parts, "func ("+receiver+")")
	} else {
		parts = append(parts, "func")
	}
	parts = append(parts, fn.Name.Name+"(")

	// Parameters
	var params []string
	if fn.Type.Params != nil {
		for _, p := range fn.Type.Params.List {
			paramNames := make([]string, len(p.Names))
			for i, n := range p.Names {
				paramNames[i] = n.Name
			}
			paramStr := strings.Join(paramNames, ", ")
			if paramStr != "" {
				paramStr += " "
			}
			paramStr += types.ExprString(p.Type)
			params = append(params, paramStr)
		}
	}
	parts = append(parts, strings.Join(params, ", ")+")")

	// Return types
	if fn.Type.Results != nil {
		var results []string
		for _, r := range fn.Type.Results.List {
			results = append(results, types.ExprString(r.Type))
		}
		if len(results) > 0 {
			if len(results) == 1 {
				parts = append(parts, " "+results[0])
			} else {
				parts = append(parts, " ("+strings.Join(results, ", ")+")")
			}
		}
	}

	return strings.Join(parts, "")
}

// FormatScopeAnnotation formats a scope annotation as a special token for FIM prompts.
func FormatScopeAnnotation(a ScopeAnnotation) string {
	switch a.Kind {
	case "method":
		return fmt.Sprintf("<SCOPE: method %s>", a.Signature)
	case "func":
		return fmt.Sprintf("<SCOPE: func %s>", a.Signature)
	case "struct":
		return fmt.Sprintf("<SCOPE: struct %s>", a.Name)
	case "interface":
		return fmt.Sprintf("<SCOPE: interface %s>", a.Name)
	case "package":
		return fmt.Sprintf("<SCOPE: package %s>", a.Name)
	default:
		return fmt.Sprintf("<SCOPE: %s %s>", a.Kind, a.Name)
	}
}

// InjectScopeAnnotations adds scope annotation tokens to a FIM prompt.
// The annotations are placed before the <FIM_PRE> section.
func InjectScopeAnnotations(code string, fimPrompt string) string {
	annotations, err := AnnotateScope(code)
	if err != nil {
		return fimPrompt // Return original if parsing fails
	}

	var sb strings.Builder
	for _, a := range annotations {
		sb.WriteString(FormatScopeAnnotation(a))
		sb.WriteString("\n")
	}
	sb.WriteString(fimPrompt)
	return sb.String()
}

// ExtractProjectContext extracts a compressed project outline from Go source code.
// Returns exported structs, function signatures, and type definitions.
func ExtractProjectContext(code string) string {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return ""
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("package %s\n", f.Name.Name))

	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			if d.Tok == token.TYPE {
				for _, spec := range d.Specs {
					if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name.IsExported() {
						sb.WriteString(fmt.Sprintf("type %s %s\n", ts.Name.Name, types.ExprString(ts.Type)))
					}
				}
			} else if d.Tok == token.VAR {
				for _, spec := range d.Specs {
					if vs, ok := spec.(*ast.ValueSpec); ok {
						for _, name := range vs.Names {
							if name.IsExported() {
								if vs.Type != nil {
									sb.WriteString(fmt.Sprintf("var %s %s\n", name.Name, types.ExprString(vs.Type)))
								}
							}
						}
					}
				}
			} else if d.Tok == token.CONST {
				for _, spec := range d.Specs {
					if vs, ok := spec.(*ast.ValueSpec); ok {
						for _, name := range vs.Names {
							if name.IsExported() {
								sb.WriteString(fmt.Sprintf("const %s\n", name.Name))
							}
						}
					}
				}
			}

		case *ast.FuncDecl:
			if d.Name.IsExported() {
				receiver := ""
				if d.Recv != nil && len(d.Recv.List) > 0 {
					receiver = types.ExprString(d.Recv.List[0].Type)
				}
				sig := buildFuncSignature(d, receiver)
				sb.WriteString(sig + "\n")
			}
		}
	}

	return sb.String()
}

// FormatProjectContext wraps the project context in the <PROJECT_CONTEXT> block.
func FormatProjectContext(context string) string {
	if context == "" {
		return ""
	}
	return fmt.Sprintf("<PROJECT_CONTEXT>\n%s</PROJECT_CONTEXT>\n", context)
}

// FormatContextTypes wraps type/function signatures in the <CONTEXT_TYPES> block.
func FormatContextTypes(signatures []string) string {
	if len(signatures) == 0 {
		return ""
	}
	return fmt.Sprintf("<CONTEXT_TYPES>\n%s\n</CONTEXT_TYPES>\n", strings.Join(signatures, "\n"))
}

// ExtractCallGraphSignatures extracts function signatures for all functions
// called within a given code block. This is used for call-graph ingestion.
func ExtractCallGraphSignatures(code string, targetFunc string) ([]string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	// Find the target function
	var targetBody *ast.BlockStmt
	ast.Inspect(f, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == targetFunc {
			targetBody = fn.Body
			return false
		}
		return true
	})

	if targetBody == nil {
		return nil, fmt.Errorf("function %q not found", targetFunc)
	}

	// Collect all function calls within the target function
	calledFuncs := make(map[string]bool)
	ast.Inspect(targetBody, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if ident, ok := call.Fun.(*ast.Ident); ok {
				calledFuncs[ident.Name] = true
			} else if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				calledFuncs[sel.Sel.Name] = true
			}
		}
		return true
	})

	// Find signatures for called functions
	var signatures []string
	ast.Inspect(f, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok {
			if calledFuncs[fn.Name.Name] {
				receiver := ""
				if fn.Recv != nil && len(fn.Recv.List) > 0 {
					receiver = types.ExprString(fn.Recv.List[0].Type)
				}
				sig := buildFuncSignature(fn, receiver)
				signatures = append(signatures, sig)
			}
		}
		return true
	})

	return signatures, nil
}
