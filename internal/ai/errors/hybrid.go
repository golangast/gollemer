// Package errors provides a hybrid error classification and auto-fix system.
// It ties a probabilistic classifier (ML/MoE or regex fallback) with
// deterministic AST-based fixers, safeguarding against neural hallucinations
// by validating all mutations through the Go compiler before committing.
package errors

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// =============================================================================
// 1. ErrorInfo – Structured Diagnostic Data
// =============================================================================

// ErrorInfo holds the diagnostic data and predicted intent from the classifier.
type ErrorInfo struct {
	File       string
	Line       int
	Column     int
	Message    string
	Raw        string
	Intent     ErrorIntent
	Confidence float64
	Symbol     string
	Package    string
}

// =============================================================================
// 2. Classifier Interface (ML / MoE or Fallback)
// =============================================================================

// Classifier interface for ML / MoE or regex-based fallback classification.
type Classifier interface {
	// Classify takes raw compiler output and returns a list of classified errors.
	Classify(compilerOutput string) ([]ErrorInfo, error)
}

// RegexClassifier is the fallback classifier that uses deterministic regex patterns.
type RegexClassifier struct {
	ProjectRoot string
}

// NewRegexClassifier creates a new regex-based fallback classifier.
func NewRegexClassifier(projectRoot string) *RegexClassifier {
	return &RegexClassifier{ProjectRoot: projectRoot}
}

// Classify uses the regex-based error patterns to classify compiler output.
func (rc *RegexClassifier) Classify(compilerOutput string) ([]ErrorInfo, error) {
	parsed := ParseCompilerOutput(compilerOutput)
	if len(parsed) == 0 {
		return nil, nil
	}

	infos := make([]ErrorInfo, 0, len(parsed))
	for _, pe := range parsed {
		infos = append(infos, ErrorInfo{
			File:       pe.File,
			Line:       pe.Line,
			Column:     pe.Column,
			Message:    pe.Message,
			Raw:        pe.Raw,
			Intent:     pe.Intent,
			Confidence: pe.Confidence,
			Symbol:     pe.Symbol,
			Package:    pe.Package,
		})
	}
	return infos, nil
}

// =============================================================================
// 3. Deterministic Fixer Interface
// =============================================================================

// Fixer applies a deterministic AST mutation to fix a classified error.
// It operates on the parsed AST and returns a description of what was fixed.
type Fixer interface {
	// Fix applies the fix to the parsed AST. Returns a human-readable message.
	Fix(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error)
}

// FixerFunc adapts a function to the Fixer interface.
type FixerFuncAdapter func(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error)

// Fix implements the Fixer interface.
func (f FixerFuncAdapter) Fix(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return f(file, fset, info)
}

// =============================================================================
// 4. HybridEngine – Ties Probabilistic Classification with Deterministic Fixers
// =============================================================================

// HybridEngine ties the probabilistic classifier with deterministic fixers.
// It follows a two-step process:
//
//	Step 1: Probabilistic Classification (The ML/MoE Layer)
//	  - Classifies raw compiler output into structured ErrorInfo with intents.
//	  - Falls back to deterministic regex patterns if the ML model is unavailable.
//
//	Step 2: Deterministic Routing & AST Mutation (The Symbolic Engine)
//	  - Routes each classified error to the appropriate AST-based fixer.
//	  - Applies the fix to the parsed AST, then validates through the Go compiler.
//	  - Rolls back on failure to prevent writing broken code to disk.
type HybridEngine struct {
	Classifier  Classifier
	Fixers      map[ErrorIntent]Fixer
	ProjectRoot string
	Verbose     bool
}

// NewHybridEngine creates a new hybrid engine with the given classifier and fixers.
func NewHybridEngine(projectRoot string, classifier Classifier, verbose bool) *HybridEngine {
	engine := &HybridEngine{
		Classifier:  classifier,
		Fixers:      make(map[ErrorIntent]Fixer),
		ProjectRoot: projectRoot,
		Verbose:     verbose,
	}

	// Register all available fixers by their intent.
	// Each fixer is wrapped to work with the AST-based Fixer interface.
	engine.registerAllFixers()
	return engine
}

// registerAllFixers populates the fixer registry with all known AST-based fixers.
func (h *HybridEngine) registerAllFixers() {
	// ── Symbol & Name Fixers ──────────────────────────────────────────────
	h.Fixers[IntentUndefinedSymbol] = FixerFuncAdapter(h.fixUndefinedSymbolAST)
	h.Fixers[IntentUndeclaredName] = FixerFuncAdapter(h.fixUndefinedSymbolAST)
	h.Fixers[IntentMissingFunctionBody] = FixerFuncAdapter(h.fixMissingFunctionBodyAST)

	// ── Import Fixers ─────────────────────────────────────────────────────
	h.Fixers[IntentMissingImport] = FixerFuncAdapter(h.fixMissingImportAST)
	h.Fixers[IntentUnusedImport] = FixerFuncAdapter(h.fixUnusedImportAST)

	// ── Type Mismatch Fixers ──────────────────────────────────────────────
	h.Fixers[IntentTypeMismatch] = FixerFuncAdapter(h.fixTypeMismatchAST)
	h.Fixers[IntentCannotAssign] = FixerFuncAdapter(h.fixTypeMismatchAST)
	h.Fixers[IntentInvalidBinaryOp] = FixerFuncAdapter(h.fixInvalidBinaryOpAST)
	h.Fixers[IntentInvalidUseOfNil] = FixerFuncAdapter(h.fixInvalidUseOfNilAST)

	// ── Variable & Declaration Fixers ─────────────────────────────────────
	h.Fixers[IntentUnusedVariable] = FixerFuncAdapter(h.fixUnusedVariableAST)
	h.Fixers[IntentNoNewVariables] = FixerFuncAdapter(h.fixNoNewVariablesAST)
	h.Fixers[IntentMissingReturn] = FixerFuncAdapter(h.fixMissingReturnAST)
	h.Fixers[IntentMissingHandlerDefinition] = FixerFuncAdapter(h.fixHandlerDefinitionAST)

	// ── Assignment Fixers ────────────────────────────────────────────────
	h.Fixers[IntentAssignmentMismatch] = FixerFuncAdapter(h.fixAssignmentMismatchAST)

	// ── Expression & Statement Fixers ─────────────────────────────────────
	h.Fixers[IntentNonBoolUsedInIf] = FixerFuncAdapter(h.fixNonBoolInIfAST)
	h.Fixers[IntentCannotTakeAddress] = FixerFuncAdapter(h.fixCannotTakeAddressAST)
	h.Fixers[IntentInvalidGoStmt] = FixerFuncAdapter(h.fixInvalidGoStmtAST)
	h.Fixers[IntentInvalidDeferStmt] = FixerFuncAdapter(h.fixInvalidDeferStmtAST)

	// ── Handler / Method Fixers ───────────────────────────────────────────
	h.Fixers[IntentMissingHandlerDefinition] = FixerFuncAdapter(h.fixHandlerDefinitionAST)
	h.Fixers[IntentMissingMethod] = FixerFuncAdapter(h.fixMissingMethodAST)

	// ── Informational Fixers (manual review notes) ────────────────────────
	h.Fixers[IntentTooManyArgs] = FixerFuncAdapter(h.fixTooManyArgsAST)
	h.Fixers[IntentNotEnoughArgs] = FixerFuncAdapter(h.fixNotEnoughArgsAST)
	h.Fixers[IntentCallNonFunction] = FixerFuncAdapter(h.fixCallNonFunctionAST)
	h.Fixers[IntentCannotRange] = FixerFuncAdapter(h.fixCannotRangeAST)
	h.Fixers[IntentDuplicateField] = FixerFuncAdapter(h.fixDuplicateFieldAST)
	h.Fixers[IntentDuplicateKey] = FixerFuncAdapter(h.fixDuplicateKeyAST)

	// ── Return Value Fixers ──────────────────────────────────────────────
	h.Fixers[IntentTooManyReturnValues] = FixerFuncAdapter(h.fixTooManyReturnValuesAST)
	h.Fixers[IntentTooFewReturnValues] = FixerFuncAdapter(h.fixTooFewReturnValuesAST)
	h.Fixers[IntentWrongReturnType] = FixerFuncAdapter(h.fixWrongReturnTypeAST)

	// ── Syntax & Fallback ─────────────────────────────────────────────────
	h.Fixers[IntentSyntaxError] = FixerFuncAdapter(h.fixSyntaxErrorAST)
	h.Fixers[IntentUnknown] = FixerFuncAdapter(h.fixUnknownAST)
}

// ProcessError runs the full hybrid pipeline:
//  1. Classify the error through the probabilistic classifier
//  2. Route to the deterministic AST fixer
//  3. Apply the fix with compile-time validation (safeguard against hallucinations)
//
// It returns a list of fix result messages, one per processed error.
func (h *HybridEngine) ProcessError(compilerOutput string) ([]string, error) {
	// ── Step 1: Probabilistic Classification (The ML/MoE Layer) ──────────────
	infos, err := h.Classifier.Classify(compilerOutput)
	if err != nil {
		return nil, fmt.Errorf("classification failed: %w", err)
	}

	if len(infos) == 0 {
		return nil, nil
	}

	if h.Verbose {
		fmt.Printf("[Hybrid] Classified %d errors from compiler output\n", len(infos))
		for _, info := range infos {
			fmt.Printf("[Hybrid]   %s:%d → Intent=%s (conf=%.2f)\n",
				info.File, info.Line, info.Intent, info.Confidence)
		}
	}

	// ── Step 2: Deterministic Routing & AST Mutation (The Symbolic Engine) ──
	var results []string

	for _, info := range infos {
		if info.File == "" {
			continue
		}

		fixer, ok := h.Fixers[info.Intent]
		if !ok {
			results = append(results, fmt.Sprintf(
				"⚠️  No deterministic fixer available for intent %s at %s:%d",
				info.Intent, info.File, info.Line))
			continue
		}

		// Apply the fix with compile-time validation (hallucination safeguard)
		fullPath := filepath.Join(h.ProjectRoot, info.File)
		msg, err := h.applySafeFix(fullPath, fixer, info)
		if err != nil {
			results = append(results, fmt.Sprintf(
				"❌ Fix failed for %s at %s:%d: %v",
				info.Intent, info.File, info.Line, err))
			continue
		}

		results = append(results, fmt.Sprintf("✅ %s", msg))
	}

	return results, nil
}

// =============================================================================
// 5. Hallucination Safeguard – Safe Fix Application with Compile Validation
// =============================================================================

// applySafeFix applies an AST fix with compile-time validation.
// It follows the SAFE pattern:
//
//	S - Save original content as backup
//	A - Apply the AST mutation
//	F - Format the modified AST back to source
//	E - Execute the compiler to verify correctness
//
// If compilation fails, the original content is restored (rollback).
func (h *HybridEngine) applySafeFix(filePath string, fixer Fixer, info ErrorInfo) (string, error) {
	// Read original content for rollback
	originalContent, err := os.ReadFile(filePath)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}

	// Parse the AST
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return "", fmt.Errorf("parse file: %w", err)
	}

	// Apply the structural AST modification (The Symbolic Engine)
	msg, err := fixer.Fix(file, fset, info)
	if err != nil {
		return "", fmt.Errorf("fixer error: %w", err)
	}

	// Write the modified AST back to disk
	f, err := os.Create(filePath)
	if err != nil {
		return "", fmt.Errorf("create file: %w", err)
	}

	if err := format.Node(f, fset, file); err != nil {
		f.Close()
		// Rollback: restore original content
		os.WriteFile(filePath, originalContent, 0644)
		return "", fmt.Errorf("format error (rolled back): %w", err)
	}
	f.Close()

	// ── Compile-Time Validation ──────────────────────────────────────────
	// The compiler acts as the ultimate verifier of the symbolic output.
	// If it compiles, the hybrid loop succeeded.
	if err := h.verifyCompilation(filePath); err != nil {
		// Rollback: restore original content
		os.WriteFile(filePath, originalContent, 0644)
		return "", fmt.Errorf("compilation failed after fix (rolled back): %w", err)
	}

	return msg, nil
}

// applySafeFixBatch applies a batch of AST fixes to the same file with
// a single compile-time validation at the end. This is more efficient when
// multiple errors exist in the same file.
func (h *HybridEngine) applySafeFixBatch(filePath string, fixers []struct {
	Fixer Fixer
	Info  ErrorInfo
}) ([]string, error) {
	// Read original content for rollback
	originalContent, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	// Parse the AST once
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse file: %w", err)
	}

	// Apply all fixes
	var messages []string
	for _, f := range fixers {
		msg, fixErr := f.Fixer.Fix(file, fset, f.Info)
		if fixErr != nil {
			messages = append(messages, fmt.Sprintf("❌ %s", fixErr))
			continue
		}
		messages = append(messages, msg)
	}

	// Write the modified AST back to disk
	f, err := os.Create(filePath)
	if err != nil {
		os.WriteFile(filePath, originalContent, 0644)
		return nil, fmt.Errorf("create file (rolled back): %w", err)
	}

	if err := format.Node(f, fset, file); err != nil {
		f.Close()
		os.WriteFile(filePath, originalContent, 0644)
		return nil, fmt.Errorf("format error (rolled back): %w", err)
	}
	f.Close()

	// ── Compile-Time Validation ──────────────────────────────────────────
	if err := h.verifyCompilation(filePath); err != nil {
		os.WriteFile(filePath, originalContent, 0644)
		return nil, fmt.Errorf("compilation failed after fixes (rolled back): %w", err)
	}

	return messages, nil
}

// verifyCompilation runs `go build` on the project to verify the fix compiles.
// It only checks the specific package containing the fixed file.
func (h *HybridEngine) verifyCompilation(filePath string) error {
	// Get the directory containing the fixed file
	dir := filepath.Dir(filePath)

	// Run `go build` on just that directory
	cmd := exec.Command("go", "build", ".")
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()

	if err != nil {
		return fmt.Errorf("build check failed: %s", strings.TrimSpace(string(output)))
	}

	if h.Verbose {
		fmt.Printf("[Hybrid] ✅ Compilation verified for %s\n", filePath)
	}

	return nil
}

// =============================================================================
// 6. AST-Based Fixer Implementations (The Symbolic Engine)
// =============================================================================
// These fixers operate directly on the parsed AST, making them deterministic
// and immune to neural hallucination. They are wrapped from the existing
// file-level fixers in fixers.go to work with the AST-based HybridEngine.

// builtinTypes lists all Go predeclared types for typo-correction.
var builtinTypes = []string{
	"bool", "byte", "complex64", "complex128", "error",
	"float32", "float64", "int", "int8", "int16", "int32", "int64",
	"rune", "string", "uint", "uint8", "uint16", "uint32", "uint64",
	"uintptr", "any",
}

// typoCorrectType returns the best-matching builtin type for a likely-typo'd
// identifier, or "" if no close match is found. Uses prefix matching and a
// simple Levenshtein distance threshold.
func typoCorrectType(symbol string) string {
	lower := strings.ToLower(symbol)
	best := ""
	bestDist := len(symbol) + 1
	for _, t := range builtinTypes {
		if strings.HasPrefix(t, lower) || strings.HasPrefix(lower, t) {
			d := levenshtein(lower, t)
			if d < bestDist {
				bestDist = d
				best = t
			}
		} else {
			d := levenshtein(lower, t)
			if d <= 2 && d < bestDist {
				bestDist = d
				best = t
			}
		}
	}
	// Only correct if within edit distance 3 to avoid wild guesses.
	if bestDist <= 3 {
		return best
	}
	return ""
}

// levenshtein computes the edit distance between two strings.
func levenshtein(a, b string) int {
	la, lb := len(a), len(b)
	dp := make([]int, lb+1)
	for j := range dp {
		dp[j] = j
	}
	for i := 1; i <= la; i++ {
		prev := dp[0]
		dp[0] = i
		for j := 1; j <= lb; j++ {
			tmp := dp[j]
			if a[i-1] == b[j-1] {
				dp[j] = prev
			} else {
				dp[j] = 1 + min3(prev, dp[j], dp[j-1])
			}
			prev = tmp
		}
	}
	return dp[lb]
}

func min3(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}

// fixUndefinedSymbolAST adds a placeholder declaration to the AST.
// If the symbol appears in a type position and looks like a typo of a builtin
// type, the ident is corrected in-place instead of adding a stub declaration.
func (h *HybridEngine) fixUndefinedSymbolAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	symbol := info.Symbol
	if symbol == "" {
		return "", fmt.Errorf("no symbol to fix")
	}

	// --- Phase 1: check if the symbol is used as a type. -----------------
	// Walk the AST looking for *ast.Ident nodes in type positions whose name
	// matches our undefined symbol. If found, attempt a typo correction
	// against the builtin types and patch the ident in-place.
	corrected := ""
	usedAsType := false
	ast.Inspect(file, func(n ast.Node) bool {
		if usedAsType {
			return false
		}
		switch node := n.(type) {
		// struct fields: type is node.Type
		case *ast.Field:
			if ident, ok := node.Type.(*ast.Ident); ok && ident.Name == symbol {
				if c := typoCorrectType(symbol); c != "" {
					ident.Name = c
					corrected = c
					usedAsType = true
					return false
				}
			}
		// var/const/type specs: type is node.Type
		case *ast.ValueSpec:
			if ident, ok := node.Type.(*ast.Ident); ok && ident.Name == symbol {
				if c := typoCorrectType(symbol); c != "" {
					ident.Name = c
					corrected = c
					usedAsType = true
					return false
				}
			}
		// type alias: node.Type
		case *ast.TypeSpec:
			if ident, ok := node.Type.(*ast.Ident); ok && ident.Name == symbol {
				if c := typoCorrectType(symbol); c != "" {
					ident.Name = c
					corrected = c
					usedAsType = true
					return false
				}
			}
		}
		return true
	})
	if usedAsType {
		return fmt.Sprintf("Corrected type typo %q → %q", symbol, corrected), nil
	}

	// --- Phase 2: check if it's a function call. -------------------------
	isFunc := false
	if info.Line > 0 {
		pos := fset.Position(file.Package)
		if pos.Line > 0 {
			ast.Inspect(file, func(n ast.Node) bool {
				if call, ok := n.(*ast.CallExpr); ok {
					if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == symbol {
						isFunc = true
						return false
					}
				}
				return true
			})
		}
	}

	if isFunc {
		stubFunc := &ast.FuncDecl{
			Name: ast.NewIdent(symbol),
			Type: &ast.FuncType{
				Params:  &ast.FieldList{},
				Results: &ast.FieldList{},
			},
			Body: &ast.BlockStmt{},
		}
		file.Decls = append(file.Decls, stubFunc)
		return fmt.Sprintf("Added stub function %q", symbol), nil
	}

	// --- Phase 3: fall back to a var stub declaration. -------------------
	stubDecl := &ast.GenDecl{
		Tok: token.VAR,
		Specs: []ast.Spec{
			&ast.ValueSpec{
				Names:  []*ast.Ident{ast.NewIdent(symbol)},
				Type:   ast.NewIdent("interface{}"),
				Values: []ast.Expr{&ast.Ident{Name: "nil"}},
			},
		},
	}
	file.Decls = append(file.Decls, stubDecl)
	return fmt.Sprintf("Added declaration for %q", symbol), nil
}

// fixMissingFunctionBodyAST adds a body to a function declaration that lacks one.
func (h *HybridEngine) fixMissingFunctionBodyAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	// Find the function without a body at the error line
	var targetFunc *ast.FuncDecl
	ast.Inspect(file, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Body == nil {
			pos := fset.Position(fn.Pos())
			if pos.Line <= info.Line && fset.Position(fn.End()).Line >= info.Line {
				targetFunc = fn
				return false
			}
		}
		return true
	})

	if targetFunc == nil {
		return "", fmt.Errorf("no function without body found at line %d", info.Line)
	}

	targetFunc.Body = &ast.BlockStmt{}
	return fmt.Sprintf("Added missing body to function %q", targetFunc.Name.Name), nil
}

// fixMissingImportAST adds a missing import to the AST.
func (h *HybridEngine) fixMissingImportAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	pkg := info.Package
	if pkg == "" {
		return "", fmt.Errorf("no package to import")
	}

	// Check if import already exists
	for _, imp := range file.Imports {
		if imp.Path != nil && imp.Path.Value == `"`+pkg+`"` {
			return fmt.Sprintf("Import %q already exists", pkg), nil
		}
	}

	// Add the import
	importSpec := &ast.ImportSpec{
		Path: &ast.BasicLit{Kind: token.STRING, Value: `"` + pkg + `"`},
	}

	importDecl := &ast.GenDecl{
		Tok:   token.IMPORT,
		Specs: []ast.Spec{importSpec},
	}

	// Insert as the first declaration after the package clause
	if len(file.Decls) > 0 {
		newDecls := make([]ast.Decl, 0, len(file.Decls)+1)
		newDecls = append(newDecls, importDecl)
		newDecls = append(newDecls, file.Decls...)
		file.Decls = newDecls
	} else {
		file.Decls = append(file.Decls, importDecl)
	}

	return fmt.Sprintf("Added import %q", pkg), nil
}

// fixUnusedImportAST removes an unused import from the AST.
func (h *HybridEngine) fixUnusedImportAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	pkg := info.Package
	if pkg == "" {
		return "", fmt.Errorf("no package to remove")
	}

	// Find and remove the import
	for i, decl := range file.Decls {
		genDecl, ok := decl.(*ast.GenDecl)
		if !ok || genDecl.Tok != token.IMPORT {
			continue
		}

		newSpecs := make([]ast.Spec, 0, len(genDecl.Specs))
		for _, spec := range genDecl.Specs {
			impSpec, ok := spec.(*ast.ImportSpec)
			if !ok || impSpec.Path == nil {
				continue
			}
			// Strip quotes from path value for comparison
			path := strings.Trim(impSpec.Path.Value, `"`)
			if path == pkg {
				continue // skip this import
			}
			newSpecs = append(newSpecs, spec)
		}

		if len(newSpecs) == 0 {
			// Remove the entire import declaration
			file.Decls = append(file.Decls[:i], file.Decls[i+1:]...)
		} else {
			genDecl.Specs = newSpecs
		}
		break
	}

	return fmt.Sprintf("Removed unused import %q", pkg), nil
}

// fixTypeMismatchAST adds an explicit type conversion to the AST.
func (h *HybridEngine) fixTypeMismatchAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	targetType := extractTargetTypeFromMessage(info.Message)
	if targetType == "" {
		return "", fmt.Errorf("could not parse target type from: %s", info.Message)
	}

	if info.Symbol == "" {
		return "", fmt.Errorf("no expression to convert")
	}

	// We need to find and wrap the expression in the AST.
	// This is a simplified approach that re-reads the line and applies text-level
	// replacement since we can't always map directly to AST nodes without type info.
	_ = targetType
	_ = info.Symbol

	// For now, delegate to the existing file-level fixer since full AST-based
	// expression wrapping requires type-checking to locate the exact expression.
	return "", fmt.Errorf("use file-level fixer for type mismatches")
}

// fixInvalidBinaryOpAST adds a type conversion to the second operand.
func (h *HybridEngine) fixInvalidBinaryOpAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	// Delegate to the existing file-level fixer
	return "", fmt.Errorf("use file-level fixer for binary op fixes")
}

// fixInvalidUseOfNilAST replaces nil with the proper zero value.
func (h *HybridEngine) fixInvalidUseOfNilAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	targetType := info.Symbol
	if targetType == "" {
		return "", fmt.Errorf("no target type to replace nil")
	}

	zeroVal := getZeroValue(targetType)
	if zeroVal == "nil" {
		return fmt.Sprintf("Cannot replace nil with zero value for type %q", targetType), nil
	}

	// Find and replace nil in the AST
	replaced := false
	ast.Inspect(file, func(n ast.Node) bool {
		if ident, ok := n.(*ast.Ident); ok && ident.Name == "nil" {
			pos := fset.Position(ident.Pos())
			if pos.Line == info.Line {
				ident.Name = zeroVal
				ident.Obj = nil
				replaced = true
				return false
			}
		}
		return true
	})

	if !replaced {
		return fmt.Sprintf("Could not find nil on line %d to replace", info.Line), nil
	}

	return fmt.Sprintf("Replaced nil with %s for type %q", zeroVal, targetType), nil
}

// fixUnusedVariableAST prefixes an unused variable with underscore.
func (h *HybridEngine) fixUnusedVariableAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	symbol := info.Symbol
	if symbol == "" {
		return "", fmt.Errorf("no variable to fix")
	}

	replaced := false
	ast.Inspect(file, func(n ast.Node) bool {
		if ident, ok := n.(*ast.Ident); ok && ident.Name == symbol {
			pos := fset.Position(ident.Pos())
			if pos.Line == info.Line {
				ident.Name = "_" + symbol
				ident.Obj = nil
				replaced = true
				return false
			}
		}
		return true
	})

	if !replaced {
		return fmt.Sprintf("Could not find %q on line %d", symbol, info.Line), nil
	}

	return fmt.Sprintf("Prefixed unused variable %q with underscore", symbol), nil
}

// fixNoNewVariablesAST changes := to = on the specified line.
func (h *HybridEngine) fixNoNewVariablesAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	// This is simpler to do at the file-text level. Delegate to existing fixer.
	return "", fmt.Errorf("use file-level fixer for no-new-variables")
}

// fixMissingReturnAST adds a return statement to a function, using proper zero values
// for each declared return type in the function signature.
func (h *HybridEngine) fixMissingReturnAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	targetFunc := findFuncContainingLine(file, fset, info.Line)
	if targetFunc == nil {
		return "", fmt.Errorf("no function found at line %d", info.Line)
	}

	returnStmt := &ast.ReturnStmt{}
	if targetFunc.Type.Results != nil {
		for _, field := range targetFunc.Type.Results.List {
			zeroExpr := zeroValueExpr(field.Type)
			if len(field.Names) == 0 {
				// unnamed return: one zero value per field
				returnStmt.Results = append(returnStmt.Results, zeroExpr)
			} else {
				// named return: one zero value per name
				for range field.Names {
					returnStmt.Results = append(returnStmt.Results, zeroExpr)
				}
			}
		}
	}
	targetFunc.Body.List = append(targetFunc.Body.List, returnStmt)

	return fmt.Sprintf("Added return statement to %q at line %d", targetFunc.Name.Name, info.Line), nil
}

// fixHandlerDefinitionAST adds a stub HTTP handler to the AST.
func (h *HybridEngine) fixHandlerDefinitionAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	symbol := info.Symbol
	if symbol == "" {
		return "", fmt.Errorf("no handler symbol to define")
	}

	handlerFunc := &ast.FuncDecl{
		Name: ast.NewIdent(symbol),
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
							&ast.BasicLit{Kind: token.STRING, Value: fmt.Sprintf(`"Hello from %s!"`, symbol)},
						},
					},
				},
			},
		},
	}
	file.Decls = append(file.Decls, handlerFunc)

	return fmt.Sprintf("Added HTTP handler function %q", symbol), nil
}

// fixAssignmentMismatchAST adds return values to match assignment.
func (h *HybridEngine) fixAssignmentMismatchAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	// Delegate to file-level fixer for now
	return "", fmt.Errorf("use file-level fixer for assignment mismatch")
}

// fixMissingMethodAST adds a stub method to the AST.
func (h *HybridEngine) fixMissingMethodAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	symbol := info.Symbol
	if symbol == "" {
		return "", fmt.Errorf("no method symbol to define")
	}

	methodFunc := &ast.FuncDecl{
		Recv: &ast.FieldList{
			List: []*ast.Field{
				{
					Names: []*ast.Ident{ast.NewIdent("t")},
					Type:  ast.NewIdent("YourType"),
				},
			},
		},
		Name: ast.NewIdent(symbol),
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
	file.Decls = append(file.Decls, methodFunc)

	return fmt.Sprintf("Added stub method %q", symbol), nil
}

// fixNonBoolInIfAST adds a comparison to non-bool if conditions.
func (h *HybridEngine) fixNonBoolInIfAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	// Find the if statement at the error line
	var targetIf *ast.IfStmt
	ast.Inspect(file, func(n ast.Node) bool {
		if ifStmt, ok := n.(*ast.IfStmt); ok {
			pos := fset.Position(ifStmt.Pos())
			if pos.Line == info.Line {
				targetIf = ifStmt
				return false
			}
		}
		return true
	})

	if targetIf == nil {
		return "", fmt.Errorf("no if statement found at line %d", info.Line)
	}

	// Wrap the condition with a nil check
	cond := targetIf.Cond
	targetIf.Cond = &ast.BinaryExpr{
		X:  cond,
		Op: token.NEQ,
		Y:  &ast.Ident{Name: "nil"},
	}

	return fmt.Sprintf("Added nil check to condition at line %d", info.Line), nil
}

// fixCannotTakeAddressAST reports that manual review is needed.
func (h *HybridEngine) fixCannotTakeAddressAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Cannot take address at %s:%d — manual review needed", info.File, info.Line), nil
}

// fixInvalidGoStmtAST reports that manual review is needed.
func (h *HybridEngine) fixInvalidGoStmtAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Invalid go statement at %s:%d — manual review needed", info.File, info.Line), nil
}

// fixInvalidDeferStmtAST reports that manual review is needed.
func (h *HybridEngine) fixInvalidDeferStmtAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Invalid defer statement at %s:%d — manual review needed", info.File, info.Line), nil
}

// fixTooManyArgsAST reports that manual review is needed.
func (h *HybridEngine) fixTooManyArgsAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Too many arguments in call to %s at %s:%d — manual review needed", info.Symbol, info.File, info.Line), nil
}

// fixNotEnoughArgsAST reports that manual review is needed.
func (h *HybridEngine) fixNotEnoughArgsAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Not enough arguments in call to %s at %s:%d — manual review needed", info.Symbol, info.File, info.Line), nil
}

// fixCallNonFunctionAST reports that manual review is needed.
func (h *HybridEngine) fixCallNonFunctionAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Call of non-function %s at %s:%d — manual review needed", info.Symbol, info.File, info.Line), nil
}

// fixCannotRangeAST reports that manual review is needed.
func (h *HybridEngine) fixCannotRangeAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Cannot range over %s at %s:%d — manual review needed", info.Symbol, info.File, info.Line), nil
}

// fixDuplicateFieldAST reports that manual review is needed.
func (h *HybridEngine) fixDuplicateFieldAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Duplicate field %s at %s:%d — manual review needed", info.Symbol, info.File, info.Line), nil
}

// fixDuplicateKeyAST reports that manual review is needed.
func (h *HybridEngine) fixDuplicateKeyAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Duplicate key %s at %s:%d — manual review needed", info.Symbol, info.File, info.Line), nil
}

// fixSyntaxErrorAST reports that manual review is needed.
func (h *HybridEngine) fixSyntaxErrorAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Syntax error at %s:%d — manual fix may be needed", info.File, info.Line), nil
}

// fixUnknownAST is the fallback for unclassified errors.
func (h *HybridEngine) fixUnknownAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	return fmt.Sprintf("Unknown error at %s:%d — cannot auto-fix", info.File, info.Line), nil
}

// =============================================================================
// 7. Return Value Fixers (The Semantic Engine)
// =============================================================================

// fixTooManyReturnValuesAST handles "too many return values".
// Two strategies:
//
//	a) If the function has NO declared return type, remove the returned expressions
//	   (turn "return expr" into bare "return").
//	b) If the function has return types, add the missing ones to the signature
//	   to match what is actually returned.
func (h *HybridEngine) fixTooManyReturnValuesAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	targetFunc := findFuncContainingLine(file, fset, info.Line)
	if targetFunc == nil {
		return "", fmt.Errorf("no function found containing line %d", info.Line)
	}

	// Count how many values are actually returned at the error line
	var returnStmt *ast.ReturnStmt
	ast.Inspect(targetFunc.Body, func(n ast.Node) bool {
		if ret, ok := n.(*ast.ReturnStmt); ok {
			pos := fset.Position(ret.Pos())
			if pos.Line == info.Line {
				returnStmt = ret
				return false
			}
		}
		return true
	})

	// Strategy A: function declares no return values → strip return expressions
	hasNoResults := targetFunc.Type.Results == nil || len(targetFunc.Type.Results.List) == 0
	if hasNoResults {
		if returnStmt != nil {
			returnStmt.Results = nil // bare return
		}
		return fmt.Sprintf("✅ Removed extra return values from %q (function declares no return type)", targetFunc.Name.Name), nil
	}

	// Strategy B: function has return types → add matching result types to signature
	// Build result list from the return statement's expressions
	if returnStmt != nil && len(returnStmt.Results) > 0 {
		// Replace the entire result list with interface{} placeholders
		// so it at least compiles (user can refine types)
		newList := make([]*ast.Field, len(returnStmt.Results))
		for i := range returnStmt.Results {
			newList[i] = &ast.Field{Type: ast.NewIdent("interface{}")}
		}
		targetFunc.Type.Results = &ast.FieldList{List: newList}
		return fmt.Sprintf("✅ Updated %q return signature to match %d return value(s)", targetFunc.Name.Name, len(returnStmt.Results)), nil
	}

	return "", fmt.Errorf("could not determine return expression at line %d", info.Line)
}

// fixTooFewReturnValuesAST handles "not enough return values".
// Adds missing zero-value return expressions to the return statement.
func (h *HybridEngine) fixTooFewReturnValuesAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	targetFunc := findFuncContainingLine(file, fset, info.Line)
	if targetFunc == nil {
		return "", fmt.Errorf("no function found containing line %d", info.Line)
	}

	// Find how many return values the signature demands
	want := 0
	if targetFunc.Type.Results != nil {
		for _, f := range targetFunc.Type.Results.List {
			if len(f.Names) == 0 {
				want++
			} else {
				want += len(f.Names)
			}
		}
	}

	// Find the return statement at error line
	var returnStmt *ast.ReturnStmt
	ast.Inspect(targetFunc.Body, func(n ast.Node) bool {
		if ret, ok := n.(*ast.ReturnStmt); ok {
			pos := fset.Position(ret.Pos())
			if pos.Line == info.Line {
				returnStmt = ret
				return false
			}
		}
		return true
	})

	if returnStmt == nil {
		return "", fmt.Errorf("no return statement found at line %d", info.Line)
	}

	have := len(returnStmt.Results)
	if have >= want {
		return fmt.Sprintf("Return statement at line %d already has %d value(s)", info.Line, have), nil
	}

	// Pad with zero values for each missing slot
	for i, field := range targetFunc.Type.Results.List {
		_ = i
		count := 1
		if len(field.Names) > 0 {
			count = len(field.Names)
		}
		for c := 0; c < count && len(returnStmt.Results) < want; c++ {
			returnStmt.Results = append(returnStmt.Results, zeroValueExpr(field.Type))
		}
	}

	return fmt.Sprintf("✅ Added %d missing zero-value return(s) to %q at line %d", want-have, targetFunc.Name.Name, info.Line), nil
}

// fixWrongReturnTypeAST handles type mismatches in return statements.
// It updates the function's return signature to match the actual returned type.
func (h *HybridEngine) fixWrongReturnTypeAST(file *ast.File, fset *token.FileSet, info ErrorInfo) (string, error) {
	targetFunc := findFuncContainingLine(file, fset, info.Line)
	if targetFunc == nil {
		return "", fmt.Errorf("no function found containing line %d", info.Line)
	}

	// info.Symbol = the literal being returned, info.Package = the expected type
	actualType := info.Package // re-used as the expected type from the pattern
	if actualType == "" {
		return "", fmt.Errorf("could not determine return type from error message")
	}

	// Update result list to match the declared return type
	if targetFunc.Type.Results != nil && len(targetFunc.Type.Results.List) > 0 {
		targetFunc.Type.Results.List[0].Type = ast.NewIdent(actualType)
		return fmt.Sprintf("✅ Updated return type of %q to %q at line %d", targetFunc.Name.Name, actualType, info.Line), nil
	}

	// If no results list, add one
	targetFunc.Type.Results = &ast.FieldList{
		List: []*ast.Field{{Type: ast.NewIdent(actualType)}},
	}
	return fmt.Sprintf("✅ Added return type %q to %q at line %d", actualType, targetFunc.Name.Name, info.Line), nil
}

// =============================================================================
// 8. AST Helper Functions
// =============================================================================

// findFuncContainingLine finds the *ast.FuncDecl whose body spans the given line.
func findFuncContainingLine(file *ast.File, fset *token.FileSet, line int) *ast.FuncDecl {
	var found *ast.FuncDecl
	ast.Inspect(file, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Body != nil {
			start := fset.Position(fn.Pos()).Line
			end := fset.Position(fn.End()).Line
			if start <= line && line <= end {
				// Pick the innermost (smallest) matching function
				if found == nil {
					found = fn
				} else {
					foundStart := fset.Position(found.Pos()).Line
					foundEnd := fset.Position(found.End()).Line
					if (end - start) < (foundEnd - foundStart) {
						found = fn
					}
				}
			}
		}
		return true
	})
	return found
}

// zeroValueExpr returns the zero-value expression for a given AST type expression.
// It handles common named types, pointers, slices, maps, channels, and interfaces.
func zeroValueExpr(t ast.Expr) ast.Expr {
	switch typ := t.(type) {
	case *ast.Ident:
		switch typ.Name {
		case "int", "int8", "int16", "int32", "int64",
			"uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
			"byte", "rune":
			return &ast.BasicLit{Kind: token.INT, Value: "0"}
		case "float32", "float64":
			return &ast.BasicLit{Kind: token.FLOAT, Value: "0.0"}
		case "complex64", "complex128":
			return &ast.BasicLit{Kind: token.IMAG, Value: "0i"}
		case "bool":
			return &ast.Ident{Name: "false"}
		case "string":
			return &ast.BasicLit{Kind: token.STRING, Value: `""`}
		case "error":
			return &ast.Ident{Name: "nil"}
		default:
			// Unknown named type — use nil and hope for the best
			return &ast.Ident{Name: "nil"}
		}
	case *ast.StarExpr, *ast.ArrayType, *ast.MapType, *ast.ChanType, *ast.InterfaceType, *ast.FuncType:
		// All of these have nil as their zero value
		return &ast.Ident{Name: "nil"}
	case *ast.SelectorExpr:
		// Package-qualified type like "io.Reader" — zero value is nil
		return &ast.Ident{Name: "nil"}
	case *ast.StructType:
		// Composite literal for an anonymous struct
		return &ast.CompositeLit{Type: t}
	}
	return &ast.Ident{Name: "nil"}
}

// =============================================================================
// 7. ApplySafeFix – Standalone Safe Fix Application
// =============================================================================

// ApplySafeFix applies a fix function to a file with compile-time validation.
// It follows the SAFE pattern:
//
//	S - Save original content as backup
//	A - Apply the AST mutation
//	F - Format the modified AST back to source
//	E - Execute the compiler to verify correctness
//
// If compilation fails, the original content is restored (rollback).
func ApplySafeFix(filePath string, fix func(*ast.File, *token.FileSet) error) error {
	// Read original content for rollback
	originalContent, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	// Parse the AST
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("parse file: %w", err)
	}

	// Apply the structural AST modification
	if err := fix(file, fset); err != nil {
		return fmt.Errorf("apply fix: %w", err)
	}

	// Write the modified AST back to disk
	f, err := os.Create(filePath)
	if err != nil {
		os.WriteFile(filePath, originalContent, 0644)
		return fmt.Errorf("create file (rolled back): %w", err)
	}

	if err := format.Node(f, fset, file); err != nil {
		f.Close()
		os.WriteFile(filePath, originalContent, 0644)
		return fmt.Errorf("format error (rolled back): %w", err)
	}
	f.Close()

	// ── Compile-Time Validation ──────────────────────────────────────────
	// The compiler acts as the ultimate verifier of the symbolic output.
	// If it compiles, the hybrid loop succeeded.
	dir := filepath.Dir(filePath)
	cmd := exec.Command("go", "build", ".")
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		// Rollback: restore original content
		os.WriteFile(filePath, originalContent, 0644)
		return fmt.Errorf("compilation failed (rolled back): %s", strings.TrimSpace(string(output)))
	}

	return nil
}
