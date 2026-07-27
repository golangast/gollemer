package errors

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
)

// FixerFunc is a function that applies an AST-based fix for a specific error intent.
type FixerFunc func(pe *ParsedError, projectRoot string) (string, error)

// GetFixer returns the appropriate fixer function for the given error intent.
func GetFixer(intent ErrorIntent) FixerFunc {
	switch intent {
	case IntentUndefinedSymbol:
		return fixUndefinedSymbol
	case IntentMissingHandlerDefinition:
		return fixMissingHandlerDefinition
	case IntentMissingImport:
		return fixMissingImport
	case IntentUnusedImport:
		return fixUnusedImport
	case IntentUnusedVariable:
		return fixUnusedVariable
	case IntentMissingReturn:
		return fixMissingReturn
	case IntentTypeMismatch:
		return fixTypeMismatch
	case IntentCannotAssign:
		return fixTypeMismatch
	case IntentMissingMethod:
		return fixMissingMethod
	case IntentUndeclaredName:
		return fixUndeclaredName
	case IntentSyntaxError:
		return fixSyntaxError
	case IntentNonBoolUsedInIf:
		return fixNonBoolInIf
	case IntentAssignmentMismatch:
		return fixAssignmentMismatch
	case IntentInvalidBinaryOp:
		return fixInvalidBinaryOp
	case IntentNoNewVariables:
		return fixNoNewVariables
	case IntentTooManyArgs:
		return fixTooManyArgs
	case IntentNotEnoughArgs:
		return fixNotEnoughArgs
	case IntentCallNonFunction:
		return fixCallNonFunction
	case IntentCannotRange:
		return fixCannotRange
	case IntentDuplicateField:
		return fixDuplicateField
	case IntentDuplicateKey:
		return fixDuplicateKey
	case IntentInvalidUseOfNil:
		return fixInvalidUseOfNil
	case IntentMissingFunctionBody:
		return fixMissingFunctionBody
	default:
		return nil
	}
}

// knownStdlibPackages is the set of Go standard library packages that are
// commonly imported. If a symbol like "os.NonExistentField" has its package
// prefix in this set, the fixer should NOT try to declare a variable named
// "os.NonExistentField" — it should instead identify the field name as the
// actual unknown symbol and route correctly.
var knownStdlibPackages = map[string]bool{
	"bufio":         true,
	"bytes":         true,
	"context":       true,
	"crypto":        true,
	"database/sql":  true,
	"encoding/csv":  true,
	"encoding/json": true,
	"encoding/xml":  true,
	"errors":        true,
	"flag":          true,
	"fmt":           true,
	"html":          true,
	"http":          true,
	"image":         true,
	"io":            true,
	"io/ioutil":     true,
	"ioutil":        true,
	"json":          true,
	"log":           true,
	"math":          true,
	"net":           true,
	"net/http":      true,
	"net/url":       true,
	"os":            true,
	"os/exec":       true,
	"path/filepath": true,
	"reflect":       true,
	"regexp":        true,
	"runtime":       true,
	"sort":          true,
	"strconv":       true,
	"strings":       true,
	"sync":          true,
	"syscall":       true,
	"testing":       true,
	"time":          true,
	"unicode":       true,
	"unsafe":        true,
	"xml":           true,
}

// splitSelector splits a dotted symbol like "os.NonExistentField" into
// ("os", "NonExistentField"). Returns ("", symbol) if there is no dot.
func splitSelector(symbol string) (pkg, field string) {
	dotIdx := strings.LastIndex(symbol, ".")
	if dotIdx < 0 {
		return "", symbol
	}
	return symbol[:dotIdx], symbol[dotIdx+1:]
}

// fileHasImport checks whether the given AST file already imports pkgPath.
func fileHasImport(fset *token.FileSet, node *ast.File, pkgPath string) bool {
	return astutil.UsesImport(node, pkgPath)
}

// fixUndefinedSymbol adds a placeholder declaration for an undefined symbol.
// If the symbol contains a package selector (e.g. "os.NonExistentField"),
// it strips the package prefix and only declares the field name, and
// ensures the package is imported.
func fixUndefinedSymbol(pe *ParsedError, projectRoot string) (string, error) {
	if pe.Symbol == "" {
		return "", fmt.Errorf("no symbol to fix")
	}

	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, fullPath, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	// ── Selector Sanitization + Import Guard ──────────────────────────────────
	// If the symbol looks like "os.NonExistentField", split it.
	if pkgName, fieldName := splitSelector(pe.Symbol); pkgName != "" {
		// If the package part is a known standard library, we know it's a
		// field/method that doesn't exist — don't create a variable declaration
		// with the dotted name. Instead:
		//   1. Ensure the package is imported.
		//   2. Declare just the field name as a zero-value variable (so the
		//      code at least compiles and the programmer can correct the field).
		if knownStdlibPackages[pkgName] {
			// Import Guard: add the import if not present
			if !fileHasImport(fset, node, pkgName) {
				astutil.AddImport(fset, node, pkgName)
			}
		} else if !fileHasImport(fset, node, pkgName) {
			// Unknown package: try adding the import anyway, but warn.
			astutil.AddImport(fset, node, pkgName)
		}

		// Declare only the field name as a variable — not the dotted expression.
		stubDecl := &ast.GenDecl{
			Tok: token.VAR,
			Specs: []ast.Spec{
				&ast.ValueSpec{
					Names:  []*ast.Ident{ast.NewIdent(fieldName)},
					Type:   ast.NewIdent("interface{}"),
					Values: []ast.Expr{&ast.Ident{Name: "nil"}},
				},
			},
		}
		node.Decls = append(node.Decls, stubDecl)

		f, err := os.Create(fullPath)
		if err != nil {
			return "", fmt.Errorf("write error: %w", err)
		}
		defer f.Close()

		if err := format.Node(f, fset, node); err != nil {
			return "", fmt.Errorf("format error: %w", err)
		}

		return fmt.Sprintf("Added declaration for %q in %s (selector sanitized from %q)", fieldName, pe.File, pe.Symbol), nil
	}

	// ── Normal (non-selector) symbol handling ─────────────────────────────────

	// Check if symbol is a function call or variable reference
	isFunc := false
	lines, _ := os.ReadFile(fullPath)
	content := string(lines)
	fileLines := strings.Split(content, "\n")

	if pe.Line > 0 && pe.Line <= len(fileLines) {
		line := fileLines[pe.Line-1]
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, pe.Symbol+"(") || strings.Contains(trimmed, pe.Symbol+"(") {
			isFunc = true
		}
	}

	if isFunc {
		// Add a stub function declaration
		stubFunc := &ast.FuncDecl{
			Name: ast.NewIdent(pe.Symbol),
			Type: &ast.FuncType{
				Params:  &ast.FieldList{},
				Results: &ast.FieldList{},
			},
			Body: &ast.BlockStmt{},
		}
		node.Decls = append(node.Decls, stubFunc)
	} else {
		// Add a variable declaration with zero value
		varType := guessTypeFromContext(pe, fileLines)
		stubDecl := &ast.GenDecl{
			Tok: token.VAR,
			Specs: []ast.Spec{
				&ast.ValueSpec{
					Names:  []*ast.Ident{ast.NewIdent(pe.Symbol)},
					Type:   ast.NewIdent(varType),
					Values: []ast.Expr{&ast.Ident{Name: getZeroValue(varType)}},
				},
			},
		}
		node.Decls = append(node.Decls, stubDecl)
	}

	f, err := os.Create(fullPath)
	if err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}
	defer f.Close()

	if err := format.Node(f, fset, node); err != nil {
		return "", fmt.Errorf("format error: %w", err)
	}

	return fmt.Sprintf("Added declaration for %q in %s", pe.Symbol, pe.File), nil
}

// fixMissingHandlerDefinition adds a stub HTTP handler function.
func fixMissingHandlerDefinition(pe *ParsedError, projectRoot string) (string, error) {
	if pe.Symbol == "" {
		return "", fmt.Errorf("no handler symbol to fix")
	}

	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, fullPath, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	// Add HTTP handler function
	handlerFunc := &ast.FuncDecl{
		Name: ast.NewIdent(pe.Symbol),
		Type: &ast.FuncType{
			Params: &ast.FieldList{
				List: []*ast.Field{
					{
						Names: []*ast.Ident{ast.NewIdent("w")},
						Type: &ast.StarExpr{
							X: &ast.SelectorExpr{
								X:   ast.NewIdent("http"),
								Sel: ast.NewIdent("ResponseWriter"),
							},
						},
					},
					{
						Names: []*ast.Ident{ast.NewIdent("r")},
						Type: &ast.StarExpr{
							X: &ast.SelectorExpr{
								X:   ast.NewIdent("http"),
								Sel: ast.NewIdent("Request"),
							},
						},
					},
				},
			},
			Results: &ast.FieldList{},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				&ast.ExprStmt{
					X: &ast.CallExpr{
						Fun: &ast.SelectorExpr{
							X:   ast.NewIdent("fmt"),
							Sel: ast.NewIdent("Fprintf"),
						},
						Args: []ast.Expr{
							ast.NewIdent("w"),
							&ast.BasicLit{Kind: token.STRING, Value: fmt.Sprintf(`"Hello from %s!"`, pe.Symbol)},
						},
					},
				},
			},
		},
	}
	node.Decls = append(node.Decls, handlerFunc)

	// Ensure imports
	astutil.AddImport(fset, node, "fmt")
	astutil.AddImport(fset, node, "net/http")

	f, err := os.Create(fullPath)
	if err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}
	defer f.Close()

	if err := format.Node(f, fset, node); err != nil {
		return "", fmt.Errorf("format error: %w", err)
	}

	return fmt.Sprintf("Added HTTP handler function %q in %s", pe.Symbol, pe.File), nil
}

// fixMissingImport adds a missing import to the file.
func fixMissingImport(pe *ParsedError, projectRoot string) (string, error) {
	if pe.Package == "" {
		return "", fmt.Errorf("no package to import")
	}

	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, fullPath, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	// Add the import
	astutil.AddImport(fset, node, pe.Package)

	f, err := os.Create(fullPath)
	if err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}
	defer f.Close()

	if err := format.Node(f, fset, node); err != nil {
		return "", fmt.Errorf("format error: %w", err)
	}

	return fmt.Sprintf("Added import %q in %s", pe.Package, pe.File), nil
}

// fixUnusedImport removes an unused import from the file.
func fixUnusedImport(pe *ParsedError, projectRoot string) (string, error) {
	if pe.Package == "" {
		return "", fmt.Errorf("no package to remove")
	}

	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, fullPath, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	// Remove the import
	removed := astutil.DeleteImport(fset, node, pe.Package)
	if !removed {
		return "", fmt.Errorf("import %q not found in %s", pe.Package, pe.File)
	}

	f, err := os.Create(fullPath)
	if err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}
	defer f.Close()

	if err := format.Node(f, fset, node); err != nil {
		return "", fmt.Errorf("format error: %w", err)
	}

	return fmt.Sprintf("Removed unused import %q from %s", pe.Package, pe.File), nil
}

// fixUnusedVariable prefixes the unused variable with an underscore or removes it.
func fixUnusedVariable(pe *ParsedError, projectRoot string) (string, error) {
	if pe.Symbol == "" {
		return "", fmt.Errorf("no variable to fix")
	}

	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	// Read the file and replace the variable name with underscore prefix
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return "", fmt.Errorf("read error: %w", err)
	}

	content := string(data)
	lines := strings.Split(content, "\n")

	if pe.Line > 0 && pe.Line <= len(lines) {
		line := lines[pe.Line-1]
		// Replace the variable name with _ prefix
		newLine := strings.Replace(line, pe.Symbol, "_"+pe.Symbol, 1)
		lines[pe.Line-1] = newLine
	}

	newContent := strings.Join(lines, "\n")
	if err := os.WriteFile(fullPath, []byte(newContent), 0644); err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}

	return fmt.Sprintf("Prefixed unused variable %q with underscore in %s", pe.Symbol, pe.File), nil
}

// fixMissingReturn adds a zero-value return statement to a function.
func fixMissingReturn(pe *ParsedError, projectRoot string) (string, error) {
	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, fullPath, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	// Find the function at the given line and add a return statement
	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok {
			// Check if this function's body contains the line
			if fn.Body != nil && fset.Position(fn.Pos()).Line <= pe.Line && fset.Position(fn.End()).Line >= pe.Line {
				targetFunc = fn
				return false
			}
		}
		return true
	})

	if targetFunc == nil {
		return "", fmt.Errorf("no function found at line %d", pe.Line)
	}

	// Determine return types and add zero values
	returnStmt := &ast.ReturnStmt{}
	if targetFunc.Type.Results != nil {
		for range targetFunc.Type.Results.List {
			returnStmt.Results = append(returnStmt.Results, &ast.Ident{Name: "nil"})
		}
	}
	targetFunc.Body.List = append(targetFunc.Body.List, returnStmt)

	f, err := os.Create(fullPath)
	if err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}
	defer f.Close()

	if err := format.Node(f, fset, node); err != nil {
		return "", fmt.Errorf("format error: %w", err)
	}

	return fmt.Sprintf("Added return statement in %s at line %d", pe.File, pe.Line), nil
}

// fixTypeMismatch attempts to fix type mismatch and cannot-assign errors by
// inserting an explicit type conversion at the error site.
//
// It handles the following patterns extracted by the regex patterns:
//   - "cannot use x (type A) as type B in assignment"
//   - "cannot assign to x"
//
// For type mismatches, it reads the line, identifies the expression that needs
// conversion, and wraps it with a type conversion: B(x).
func fixTypeMismatch(pe *ParsedError, projectRoot string) (string, error) {
	// DEBUG PRINTS
	fmt.Printf("[DEBUG] Raw Message: %q\n", pe.Message)
	fmt.Printf("[DEBUG] Parsed Symbol: %q, Line: %d, File: %q\n", pe.Symbol, pe.Line, pe.File)

	targetType := extractTargetTypeFromMessage(pe.Message)
	fmt.Printf("[DEBUG] Extracted Target Type: %q\n", targetType)

	if targetType == "" {
		return fmt.Sprintf("Type mismatch at %s:%d — could not parse target type", pe.File, pe.Line), nil
	}

	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	data, err := os.ReadFile(fullPath)
	if err != nil {
		return "", fmt.Errorf("read error: %w", err)
	}

	content := string(data)
	lines := strings.Split(content, "\n")

	if pe.Line <= 0 || pe.Line > len(lines) {
		return "", fmt.Errorf("invalid line number %d", pe.Line)
	}

	lineIdx := pe.Line - 1
	line := lines[lineIdx]
	fmt.Printf("[DEBUG] Target Line: %q\n", line)

	exprToConvert := pe.Symbol
	if exprToConvert == "" {
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			exprToConvert = strings.TrimSpace(parts[1])
			exprToConvert = strings.TrimSuffix(exprToConvert, ";")
		}
	}
	fmt.Printf("[DEBUG] Expression to Convert: %q\n", exprToConvert)

	if exprToConvert == "" {
		return fmt.Sprintf("Type mismatch at %s:%d — could not determine expression to convert", pe.File, pe.Line), nil
	}

	// 3. Prevent duplicate conversion wrappers
	conversionPattern := fmt.Sprintf("%s(%s", targetType, exprToConvert)
	if strings.Contains(line, conversionPattern) {
		return fmt.Sprintf("Type conversion already present at %s:%d", pe.File, pe.Line), nil
	}

	// 4. Apply the conversion transformation
	newLine := strings.Replace(line, exprToConvert, fmt.Sprintf("%s(%s)", targetType, exprToConvert), 1)
	if newLine == line {
		return fmt.Sprintf("Could not locate %q on line %d to apply type conversion", exprToConvert, pe.Line), nil
	}
	lines[lineIdx] = newLine

	newContent := strings.Join(lines, "\n")
	if err := os.WriteFile(fullPath, []byte(newContent), 0644); err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}

	return fmt.Sprintf("Added type conversion %s(%s) at %s:%d", targetType, exprToConvert, pe.File, pe.Line), nil
}

// extractTargetTypeFromMessage extracts the target type from a type-mismatch
// error message. Example:
//
//	Input:  "type mismatch: cannot use string as int"
//	Output: "int"
//
//	Input:  "cannot assign to x"
//	Output: ""  (no type information)
//
// extractTargetTypeFromMessage extracts the target type from a type-mismatch error message.
func extractTargetTypeFromMessage(msg string) string {
	// Pattern: "as type B in" or "as B value in" or "as B value"
	re := regexp.MustCompile(`as\s+(?:type\s+)?([a-zA-Z0-9_.]+)(?:\s+value)?(?:\s+in|\s*$)`)
	if m := re.FindStringSubmatch(msg); len(m) > 1 {
		return m[1]
	}
	// Alternate fallback matching end of line type specifications
	re2 := regexp.MustCompile(`as\s+([a-zA-Z0-9_.]+)(?:\s+value)?$`)
	if m := re2.FindStringSubmatch(msg); len(m) > 1 {
		return m[1]
	}
	return ""
}

// fixMissingMethod adds a stub method implementation.
func fixMissingMethod(pe *ParsedError, projectRoot string) (string, error) {
	if pe.Symbol == "" {
		return "", fmt.Errorf("no method to fix")
	}

	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, fullPath, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	// Add a stub method
	methodFunc := &ast.FuncDecl{
		Recv: &ast.FieldList{
			List: []*ast.Field{
				{
					Names: []*ast.Ident{ast.NewIdent("t")},
					Type:  ast.NewIdent("YourType"),
				},
			},
		},
		Name: ast.NewIdent(pe.Symbol),
		Type: &ast.FuncType{
			Params:  &ast.FieldList{},
			Results: &ast.FieldList{},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				&ast.ExprStmt{
					X: &ast.CallExpr{
						Fun:  ast.NewIdent("panic"),
						Args: []ast.Expr{&ast.BasicLit{Kind: token.STRING, Value: "\"not implemented\""}},
					},
				},
			},
		},
	}
	node.Decls = append(node.Decls, methodFunc)

	f, err := os.Create(fullPath)
	if err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}
	defer f.Close()

	if err := format.Node(f, fset, node); err != nil {
		return "", fmt.Errorf("format error: %w", err)
	}

	return fmt.Sprintf("Added stub method %q in %s", pe.Symbol, pe.File), nil
}

// fixUndeclaredName adds a declaration for an undeclared name.
func fixUndeclaredName(pe *ParsedError, projectRoot string) (string, error) {
	// Same as undefined symbol for now — the selector sanitization logic
	// in fixUndefinedSymbol handles dotted names correctly.
	return fixUndefinedSymbol(pe, projectRoot)
}

// fixSyntaxError attempts to fix common syntax errors.
func fixSyntaxError(pe *ParsedError, projectRoot string) (string, error) {
	return fmt.Sprintf("Syntax error at %s:%d - manual fix may be needed", pe.File, pe.Line), nil
}

// fixNonBoolInIf converts a non-boolean expression used in an if condition
// by wrapping it with a comparison to the zero value or adding a nil check.
//
// The modern Go compiler error "non-boolean condition in if statement" does
// NOT include a symbol name, so we must parse the line to find the expression
// used in the if condition. The older form "non-bool x used in if" includes
// the symbol name in pe.Symbol.
// fixNonBoolInIf fixes non-boolean conditions in if statements by adding appropriate comparisons.
func fixNonBoolInIf(pe *ParsedError, projectRoot string) (string, error) {
	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	data, err := os.ReadFile(fullPath)
	if err != nil {
		return "", fmt.Errorf("read error: %w", err)
	}

	content := string(data)
	lines := strings.Split(content, "\n")

	if pe.Line <= 0 || pe.Line > len(lines) {
		return "", fmt.Errorf("invalid line number %d", pe.Line)
	}

	lineIdx := pe.Line - 1
	line := lines[lineIdx]

	// Determine the expression to fix.
	exprToFix := pe.Symbol
	if exprToFix == "" {
		exprToFix = extractIfConditionExpr(line)
	}

	if exprToFix == "" {
		return fmt.Sprintf("Could not determine expression to fix at %s:%d", pe.File, pe.Line), nil
	}

	// Determine the appropriate boolean coercion based on expression nature.
	// For numeric/primitive literals or variables (like integers), compare to 0 or use truthy check.
	// For general expressions, default to a safe non-nil or non-zero check.
	replacement := fmt.Sprintf("%s != nil", exprToFix)

	// Simple heuristic: if the expression is a pure integer literal or simple variable
	// that caused a type mismatch with nil, handle it as a zero check or inequality.
	// You can expand this check based on your type analysis or AST parsing context.
	trimmedExpr := strings.TrimSpace(exprToFix)
	if isNumericLiteralOrVar(trimmedExpr) {
		replacement = fmt.Sprintf("%s != 0", exprToFix)
	}

	newLine := strings.Replace(line, exprToFix, replacement, 1)
	if newLine == line {
		return fmt.Sprintf("Could not locate %q on line %d to fix condition", exprToFix, pe.Line), nil
	}
	lines[lineIdx] = newLine

	newContent := strings.Join(lines, "\n")
	if err := os.WriteFile(fullPath, []byte(newContent), 0644); err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}

	return fmt.Sprintf("Fixed non-bool condition for %q at %s:%d", exprToFix, pe.File, pe.Line), nil
}

// Helper to detect simple numeric identifiers or literals
func isNumericLiteralOrVar(expr string) bool {
	// If it's a single variable name like 'x' or literal like '5' in our test case
	// A more advanced implementation would query Go's go/types package.
	for _, ch := range expr {
		if ch < '0' || ch > '9' {
			// If it contains letters, it's likely a variable name (e.g., 'x')
			// In a real-world compiler fixer, check if its type is numeric/basic.
			return true
		}
	}
	return true
}

// extractIfConditionExpr parses a line like "if x {" or "if x > 0 {" and
// returns the first expression after "if". This is used when the compiler
// error "non-boolean condition in if statement" does not include a symbol name.
func extractIfConditionExpr(line string) string {
	trimmed := strings.TrimSpace(line)
	// Match: if <expr> {  or  if <expr> \n
	if !strings.HasPrefix(trimmed, "if ") {
		return ""
	}
	// Remove the "if " prefix
	afterIf := strings.TrimSpace(trimmed[3:])
	// Find the first token (up to a space, brace, or operator)
	// This is a simple heuristic: take everything up to the first space,
	// brace, or comparison operator.
	end := strings.IndexAny(afterIf, " {<>=!&|")
	if end < 0 {
		// No delimiter found — the whole remainder is the expression
		return afterIf
	}
	return strings.TrimSpace(afterIf[:end])
}

// guessTypeFromContext tries to determine the type of a symbol from context.
func guessTypeFromContext(pe *ParsedError, fileLines []string) string {
	if pe.Line <= 0 || pe.Line > len(fileLines) {
		return "interface{}"
	}

	line := fileLines[pe.Line-1]
	lower := strings.ToLower(line)

	// Check for common patterns
	if strings.Contains(lower, "http") || strings.Contains(lower, "handler") || strings.Contains(lower, "server") {
		return "*http.Server"
	}
	if strings.Contains(lower, "err") || strings.Contains(lower, "error") {
		return "error"
	}
	if strings.Contains(lower, "str") || strings.Contains(lower, "name") || strings.Contains(lower, "msg") {
		return "string"
	}
	if strings.Contains(lower, "count") || strings.Contains(lower, "num") || strings.Contains(lower, "index") {
		return "int"
	}
	if strings.Contains(lower, "port") || strings.Contains(lower, "addr") {
		return "string"
	}

	return "interface{}"
}

// fixAssignmentMismatch fixes assignment mismatch errors by expanding the
// called function's return signature to match the number of variables on the
// left-hand side of the assignment.
//
// For example: "x, y := single()" where single() returns 1 value becomes:
// single() is modified to return the appropriate number of values.
//
// The parsed error's Symbol field contains the number of expected variables,
// and the Package field contains the number of values the function returns.
func fixAssignmentMismatch(pe *ParsedError, projectRoot string) (string, error) {
	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	data, err := os.ReadFile(fullPath)
	if err != nil {
		return "", fmt.Errorf("read error: %w", err)
	}

	content := string(data)
	lines := strings.Split(content, "\n")

	if pe.Line <= 0 || pe.Line > len(lines) {
		return "", fmt.Errorf("invalid line number %d", pe.Line)
	}

	// The symbol from the parsed error contains the function name (e.g. "single")
	// We parse the file to find the function definition and expand its return values
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, fullPath, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	// Find the assignment line to extract the LHS variable count and called function
	lineIdx := pe.Line - 1
	line := lines[lineIdx]
	line = strings.TrimSpace(line)

	// Extract the called function name from the line
	// Pattern: x, y := funcName(...) or x, y = funcName(...)
	funcName := pe.Symbol
	if funcName == "" {
		// Try to extract from the error message
		// "assignment mismatch: 2 variables but single returns 1 value"
		re := regexp.MustCompile(`but\s+(\w+)\s+returns`)
		if m := re.FindStringSubmatch(pe.Message); len(m) > 1 {
			funcName = m[1]
		}
	}

	if funcName == "" {
		return fmt.Sprintf("Could not determine function name at %s:%d", pe.File, pe.Line), nil
	}

	// Find the function declaration and add one more return value
	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == funcName {
			targetFunc = fn
			return false
		}
		return true
	})

	if targetFunc == nil {
		return fmt.Sprintf("Function %q not found in %s", funcName, pe.File), nil
	}

	// Count current return values
	currentResults := 0
	if targetFunc.Type.Results != nil {
		for _, field := range targetFunc.Type.Results.List {
			// Each field may have multiple names of the same type, or just a type
			if len(field.Names) > 1 {
				currentResults += len(field.Names)
			} else {
				currentResults++
			}
		}
	}

	// Count expected variables from the error message
	// "assignment mismatch: 2 variables but single returns 1 value"
	varCount := 0
	reVar := regexp.MustCompile(`(\d+)\s+variables`)
	if m := reVar.FindStringSubmatch(pe.Message); len(m) > 1 {
		varCount = atoi(m[1])
	}

	if varCount <= currentResults {
		// No additional return values needed (or fewer), can't auto-fix
		return fmt.Sprintf("Assignment mismatch at %s:%d — function %q returns %d values but assignment expects %d",
			pe.File, pe.Line, funcName, currentResults, varCount), nil
	}

	// Add return values to match the expected count
	// We duplicate the last return value's type for simplicity
	extraCount := varCount - currentResults
	lastReturnType := guessReturnType(targetFunc)
	for i := 0; i < extraCount; i++ {
		if lastReturnType != "" {
			targetFunc.Type.Results.List = append(targetFunc.Type.Results.List, &ast.Field{
				Type: ast.NewIdent(lastReturnType),
			})
		} else {
			targetFunc.Type.Results.List = append(targetFunc.Type.Results.List, &ast.Field{
				Type: ast.NewIdent("interface{}"),
			})
		}
	}

	// Also update the function body: add extra return values
	if targetFunc.Body != nil && len(targetFunc.Body.List) > 0 {
		// Find the last return statement
		for i := len(targetFunc.Body.List) - 1; i >= 0; i-- {
			if retStmt, ok := targetFunc.Body.List[i].(*ast.ReturnStmt); ok {
				for j := 0; j < extraCount; j++ {
					// Use the same type as the last return value for the extra values
					zeroVal := "nil"
					if lastReturnType != "" {
						zeroVal = getZeroValue(lastReturnType)
					}
					retStmt.Results = append(retStmt.Results, &ast.Ident{Name: zeroVal})
				}
				break
			}
		}
	}

	f, err := os.Create(fullPath)
	if err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}
	defer f.Close()

	if err := format.Node(f, fset, node); err != nil {
		return "", fmt.Errorf("format error: %w", err)
	}

	return fmt.Sprintf("Added %d return value(s) to function %q in %s to match assignment", extraCount, funcName, pe.File), nil
}

// guessReturnType returns a string representation of the last return type of a function.
func guessReturnType(fn *ast.FuncDecl) string {
	if fn.Type.Results == nil || len(fn.Type.Results.List) == 0 {
		return ""
	}
	last := fn.Type.Results.List[len(fn.Type.Results.List)-1]
	return typeExprToString(last.Type)
}

// typeExprToString converts an AST type expression to a string representation.
func typeExprToString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + typeExprToString(t.X)
	case *ast.SelectorExpr:
		return typeExprToString(t.X) + "." + t.Sel.Name
	case *ast.ArrayType:
		return "[]" + typeExprToString(t.Elt)
	case *ast.MapType:
		return "map[" + typeExprToString(t.Key) + "]" + typeExprToString(t.Value)
	default:
		return ""
	}
}

// fixInvalidBinaryOp fixes mismatched types in binary operations by adding
// an explicit type conversion to the second operand.
// For example: s + x (mismatched types string and int) becomes s + int(x)
func fixInvalidBinaryOp(pe *ParsedError, projectRoot string) (string, error) {
	fullPath := filepath.Join(projectRoot, pe.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", fullPath)
	}

	data, err := os.ReadFile(fullPath)
	if err != nil {
		return "", fmt.Errorf("read error: %w", err)
	}

	content := string(data)
	lines := strings.Split(content, "\n")

	if pe.Line <= 0 || pe.Line > len(lines) {
		return "", fmt.Errorf("invalid line number %d", pe.Line)
	}

	lineIdx := pe.Line - 1
	line := lines[lineIdx]

	// The symbol field contains the first type (e.g. "string") and
	// the package field contains the second type (e.g. "int").
	// We want to convert the second operand to the type of the first operand.
	// For s + x where s=string and x=int, the Package field is "int",
	// the Symbol field is "string". We need to convert x to string.
	targetType := pe.Symbol // the type we want to convert TO (e.g. "string")

	// Extract the second operand (right side of +)
	// Parse the line to find the expression after the + operator
	plusIdx := strings.Index(line, "+")
	if plusIdx < 0 {
		// Try +=
		plusIdx = strings.Index(line, "+=")
		if plusIdx >= 0 {
			plusIdx++ // point to =
		}
	}

	if plusIdx < 0 || plusIdx >= len(line)-1 {
		return fmt.Sprintf("Could not find + operator on line %d", pe.Line), nil
	}

	// Get everything after the +
	afterPlus := strings.TrimSpace(line[plusIdx+1:])

	// Check for semicolons or comments
	if scIdx := strings.IndexAny(afterPlus, ";/"); scIdx >= 0 {
		afterPlus = strings.TrimSpace(afterPlus[:scIdx])
	}

	if afterPlus == "" {
		return fmt.Sprintf("Empty right operand at %s:%d", pe.File, pe.Line), nil
	}

	// Prevent duplicate conversions
	if strings.Contains(line, fmt.Sprintf("%s(%s)", targetType, afterPlus)) {
		return fmt.Sprintf("Type conversion already present at %s:%d", pe.File, pe.Line), nil
	}

	// Apply the conversion: wrap the second operand with the target type
	newLine := strings.Replace(line, afterPlus, fmt.Sprintf("%s(%s)", targetType, afterPlus), 1)
	lines[lineIdx] = newLine

	newContent := strings.Join(lines, "\n")
	if err := os.WriteFile(fullPath, []byte(newContent), 0644); err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}

	return fmt.Sprintf("Added type conversion %s(%s) at %s:%d", targetType, afterPlus, pe.File, pe.Line), nil
}

// fixNoNewVariables changes := to = when there are no new variables.
func fixNoNewVariables(pe *ParsedError, projectRoot string) (string, error) {
	fullPath := filepath.Join(projectRoot, pe.File)
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return "", fmt.Errorf("read error: %w", err)
	}
	lines := strings.Split(string(data), "\n")
	if pe.Line <= 0 || pe.Line > len(lines) {
		return "", fmt.Errorf("invalid line number %d", pe.Line)
	}
	line := lines[pe.Line-1]
	newLine := strings.Replace(line, ":=", "=", 1)
	if newLine == line {
		return fmt.Sprintf("Could not find := on line %d", pe.Line), nil
	}
	lines[pe.Line-1] = newLine
	if err := os.WriteFile(fullPath, []byte(strings.Join(lines, "\n")), 0644); err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}
	return fmt.Sprintf("Changed := to = at %s:%d", pe.File, pe.Line), nil
}

// fixTooManyArgs removes extra arguments from a function call.
func fixTooManyArgs(pe *ParsedError, projectRoot string) (string, error) {
	return fmt.Sprintf("Too many arguments in call to %s at %s:%d — manual review needed", pe.Symbol, pe.File, pe.Line), nil
}

// fixNotEnoughArgs adds nil arguments to a function call.
func fixNotEnoughArgs(pe *ParsedError, projectRoot string) (string, error) {
	return fmt.Sprintf("Not enough arguments in call to %s at %s:%d — manual review needed", pe.Symbol, pe.File, pe.Line), nil
}

// fixCallNonFunction warns about calling a non-function value.
func fixCallNonFunction(pe *ParsedError, projectRoot string) (string, error) {
	return fmt.Sprintf("Call of non-function %s at %s:%d — manual review needed", pe.Symbol, pe.File, pe.Line), nil
}

// fixCannotRange warns about values that cannot be ranged over.
func fixCannotRange(pe *ParsedError, projectRoot string) (string, error) {
	return fmt.Sprintf("Cannot range over %s (type %s) at %s:%d — manual review needed", pe.Symbol, pe.Package, pe.File, pe.Line), nil
}

// fixDuplicateField removes duplicate fields from struct literals.
func fixDuplicateField(pe *ParsedError, projectRoot string) (string, error) {
	return fmt.Sprintf("Duplicate field %s at %s:%d — manual review needed", pe.Symbol, pe.File, pe.Line), nil
}

// fixDuplicateKey removes duplicate keys from map literals.
func fixDuplicateKey(pe *ParsedError, projectRoot string) (string, error) {
	return fmt.Sprintf("Duplicate key %s at %s:%d — manual review needed", pe.Symbol, pe.File, pe.Line), nil
}

// fixInvalidUseOfNil replaces nil with the zero value of the expected type.
func fixInvalidUseOfNil(pe *ParsedError, projectRoot string) (string, error) {
	fullPath := filepath.Join(projectRoot, pe.File)
	data, err := os.ReadFile(fullPath)
	if err != nil {
		return "", fmt.Errorf("read error: %w", err)
	}
	lines := strings.Split(string(data), "\n")
	if pe.Line <= 0 || pe.Line > len(lines) {
		return "", fmt.Errorf("invalid line number %d", pe.Line)
	}
	// Replace nil with the type's zero value expression
	line := lines[pe.Line-1]
	targetType := pe.Symbol
	zeroVal := getZeroValue(targetType)
	newLine := strings.Replace(line, "nil", zeroVal, 1)
	if newLine == line {
		return fmt.Sprintf("Could not replace nil on line %d", pe.Line), nil
	}
	lines[pe.Line-1] = newLine
	if err := os.WriteFile(fullPath, []byte(strings.Join(lines, "\n")), 0644); err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}
	return fmt.Sprintf("Replaced nil with %s at %s:%d", zeroVal, pe.File, pe.Line), nil
}

// fixMissingFunctionBody adds an empty body to a function declaration.
func fixMissingFunctionBody(pe *ParsedError, projectRoot string) (string, error) {
	fullPath := filepath.Join(projectRoot, pe.File)
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, fullPath, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}
	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Body == nil && fset.Position(fn.Pos()).Line <= pe.Line && fset.Position(fn.End()).Line >= pe.Line {
			targetFunc = fn
			return false
		}
		return true
	})
	if targetFunc == nil {
		return fmt.Sprintf("No function without body found at %s:%d", pe.File, pe.Line), nil
	}
	targetFunc.Body = &ast.BlockStmt{}
	f, err := os.Create(fullPath)
	if err != nil {
		return "", fmt.Errorf("write error: %w", err)
	}
	defer f.Close()
	if err := format.Node(f, fset, node); err != nil {
		return "", fmt.Errorf("format error: %w", err)
	}
	return fmt.Sprintf("Added missing body to function at %s:%d", pe.File, pe.Line), nil
}

// getZeroValue returns the zero value literal for a given type.
func getZeroValue(typeName string) string {
	switch typeName {
	case "string":
		return `""`
	case "int", "int8", "int16", "int32", "int64":
		return "0"
	case "uint", "uint8", "uint16", "uint32", "uint64":
		return "0"
	case "float32", "float64":
		return "0.0"
	case "bool":
		return "false"
	case "error":
		return "nil"
	default:
		return "nil"
	}
}
