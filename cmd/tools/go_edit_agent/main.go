package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/tools/go/ast/astutil"
)

// ─── Data Types ───────────────────────────────────────────────────────────────

// EditOperation describes a single AST-level edit to apply to a Go source file.
type EditOperation struct {
	Type       string `json:"type"`        // "insert_func", "modify_func", "add_field", "add_import", "replace_code", "delete_func"
	TargetFile string `json:"target_file"` // Path to the .go file
	FuncName   string `json:"func_name,omitempty"`
	StructName string `json:"struct_name,omitempty"`
	FieldName  string `json:"field_name,omitempty"`
	FieldType  string `json:"field_type,omitempty"`
	FieldTag   string `json:"field_tag,omitempty"`
	ImportPath string `json:"import_path,omitempty"`
	Code       string `json:"code,omitempty"`      // New function body or replacement code
	InsertAt   string `json:"insert_at,omitempty"` // "beginning", "end", or line number
	OldCode    string `json:"old_code,omitempty"`  // For replace_code
	NewCode    string `json:"new_code,omitempty"`  // For replace_code
}

// EditResult captures the outcome of applying an edit.
type EditResult struct {
	Success  bool   `json:"success"`
	File     string `json:"file"`
	Message  string `json:"message"`
	Error    string `json:"error,omitempty"`
	Duration string `json:"duration"`
}

// ValidationResult captures the outcome of the verification loop.
type ValidationResult struct {
	Success bool   `json:"success"`
	GoFmt   string `json:"gofmt,omitempty"`
	GoVet   string `json:"govet,omitempty"`
	GoBuild string `json:"gobuild,omitempty"`
	GoTest  string `json:"gotest,omitempty"`
}

// AgentRequest is the JSON input format for the editing agent.
type AgentRequest struct {
	File       string          `json:"file"`        // Target .go file
	Edits      []EditOperation `json:"edits"`       // List of edits to apply
	RunTest    bool            `json:"run_test"`    // Whether to run go test after edits
	MaxRetries int             `json:"max_retries"` // Self-correction retries (default 3)
}

// AgentResponse is the JSON output format.
type AgentResponse struct {
	Success      bool              `json:"success"`
	File         string            `json:"file"`
	EditsApplied int               `json:"edits_applied"`
	Results      []EditResult      `json:"results"`
	Validation   *ValidationResult `json:"validation,omitempty"`
	Error        string            `json:"error,omitempty"`
	Duration     string            `json:"duration"`
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	filePath := flag.String("file", "", "Target Go source file to edit")
	editsJSON := flag.String("edits", "", "JSON array of EditOperation objects")
	runTest := flag.Bool("test", false, "Run go test after edits")
	maxRetries := flag.Int("retries", 3, "Max self-correction retries")
	interactive := flag.Bool("interactive", false, "Read edits from stdin as JSON")
	flag.Parse()

	startTime := time.Now()

	var req AgentRequest

	if *interactive {
		// Read JSON request from stdin
		inputBytes, err := io.ReadAll(os.Stdin)
		if err != nil {
			log.Fatalf("Error reading stdin: %v", err)
		}
		if err := json.Unmarshal(inputBytes, &req); err != nil {
			log.Fatalf("Error parsing JSON from stdin: %v", err)
		}
	} else {
		req.File = *filePath
		if *editsJSON != "" {
			if err := json.Unmarshal([]byte(*editsJSON), &req.Edits); err != nil {
				log.Fatalf("Error parsing edits JSON: %v", err)
			}
		}
		req.RunTest = *runTest
		req.MaxRetries = *maxRetries
	}

	if req.MaxRetries <= 0 {
		req.MaxRetries = 3
	}

	if req.File == "" {
		log.Fatal("No target file specified. Use -file or provide in JSON.")
	}

	resp := executeAgent(req)
	resp.Duration = time.Since(startTime).Round(time.Millisecond).String()

	output, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(output))

	if !resp.Success {
		os.Exit(1)
	}
}

// ─── Agent Execution ──────────────────────────────────────────────────────────

func executeAgent(req AgentRequest) AgentResponse {
	resp := AgentResponse{
		File:    req.File,
		Success: true,
	}

	// Resolve absolute path
	absPath, err := filepath.Abs(req.File)
	if err != nil {
		resp.Success = false
		resp.Error = fmt.Sprintf("cannot resolve path: %v", err)
		return resp
	}
	resp.File = absPath

	// Verify file exists and is a .go file
	if !strings.HasSuffix(absPath, ".go") {
		resp.Success = false
		resp.Error = fmt.Sprintf("not a .go file: %s", absPath)
		return resp
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		resp.Success = false
		resp.Error = fmt.Sprintf("file not found: %s", absPath)
		return resp
	}

	// Apply edits with self-correction loop
	backupBytes, _ := os.ReadFile(absPath)

	for attempt := 0; attempt <= req.MaxRetries; attempt++ {
		if attempt > 0 {
			// Restore original before retry
			os.WriteFile(absPath, backupBytes, 0644)
			log.Printf("[go-edit-agent] Self-correction attempt %d/%d", attempt, req.MaxRetries)
		}

		results := applyEdits(absPath, req.Edits)
		resp.Results = results
		resp.EditsApplied = countSuccesses(results)

		// Run validation
		valResult := validateGoCode(absPath, req.RunTest)
		resp.Validation = &valResult

		if valResult.Success {
			resp.Success = true
			return resp
		}

		// Validation failed — attempt self-correction
		if attempt < req.MaxRetries {
			errMsg := buildErrorSummary(valResult)
			log.Printf("[go-edit-agent] Validation failed: %s", errMsg)

			// Generate corrective edits based on error messages
			correctiveEdits := generateCorrectiveEdits(absPath, errMsg)
			if len(correctiveEdits) > 0 {
				log.Printf("[go-edit-agent] Generated %d corrective edits", len(correctiveEdits))
				req.Edits = append(req.Edits, correctiveEdits...)
			}
		} else {
			resp.Success = false
			resp.Error = fmt.Sprintf("validation failed after %d retries", req.MaxRetries)
		}
	}

	return resp
}

// ─── AST Edit Application ─────────────────────────────────────────────────────

func applyEdits(filePath string, edits []EditOperation) []EditResult {
	var results []EditResult

	for _, edit := range edits {
		startTime := time.Now()
		result := EditResult{File: filePath}

		switch edit.Type {
		case "insert_func":
			result = applyInsertFunc(filePath, edit)
		case "modify_func":
			result = applyModifyFunc(filePath, edit)
		case "add_field":
			result = applyAddField(filePath, edit)
		case "add_import":
			result = applyAddImport(filePath, edit)
		case "replace_code":
			result = applyReplaceCode(filePath, edit)
		case "delete_func":
			result = applyDeleteFunc(filePath, edit)
		default:
			result = EditResult{
				Success: false,
				File:    filePath,
				Error:   fmt.Sprintf("unknown edit type: %s", edit.Type),
			}
		}

		result.Duration = time.Since(startTime).Round(time.Microsecond).String()
		results = append(results, result)
	}

	return results
}

// applyInsertFunc inserts a new function into the Go file using AST manipulation.
func applyInsertFunc(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("parse error: %v", err)}
	}

	// Check if function already exists
	exists := false
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == edit.FuncName {
			exists = true
			return false
		}
		return true
	})
	if exists {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("function %q already exists", edit.FuncName)}
	}

	// Parse the function code
	funcCode := edit.Code
	if funcCode == "" {
		funcCode = fmt.Sprintf("func %s() {\n\t// TODO: implement\n}\n", edit.FuncName)
	}

	// Wrap in a package to parse as a file
	src := fmt.Sprintf("package main\n\n%s", funcCode)
	funcFset := token.NewFileSet()
	funcNode, err := parser.ParseFile(funcFset, "", src, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("cannot parse function code: %v", err)}
	}

	// Extract the function declaration
	var newFunc *ast.FuncDecl
	for _, decl := range funcNode.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			newFunc = fn
			break
		}
	}
	if newFunc == nil {
		return EditResult{Success: false, File: filePath, Error: "no function declaration found in provided code"}
	}

	// Extract imports from the function code and add them to the target file
	for _, imp := range funcNode.Imports {
		if imp.Path != nil {
			path := strings.Trim(imp.Path.Value, "\"")
			astutil.AddImport(fset, node, path)
		}
	}

	// Add the function to the file
	node.Decls = append(node.Decls, newFunc)

	// Write back
	if err := writeFormattedFile(filePath, fset, node); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("inserted function %q", edit.FuncName)}
}

// applyModifyFunc modifies the body of an existing function.
func applyModifyFunc(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("parse error: %v", err)}
	}

	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == edit.FuncName {
			targetFunc = fn
			return false
		}
		return true
	})

	if targetFunc == nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("function %q not found", edit.FuncName)}
	}

	// If new code is provided, replace the function body
	if edit.Code != "" {
		// Parse the new body
		src := fmt.Sprintf("package main\nfunc _() {\n%s\n}", edit.Code)
		bodyFset := token.NewFileSet()
		bodyNode, err := parser.ParseFile(bodyFset, "", src, 0)
		if err != nil {
			return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("cannot parse new body: %v", err)}
		}

		for _, decl := range bodyNode.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "_" {
				targetFunc.Body = fn.Body
				break
			}
		}

		// Extract and add any new imports from the code
		for _, imp := range bodyNode.Imports {
			if imp.Path != nil {
				path := strings.Trim(imp.Path.Value, "\"")
				astutil.AddImport(fset, node, path)
			}
		}
	}

	if err := writeFormattedFile(filePath, fset, node); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("modified function %q", edit.FuncName)}
}

// applyAddField adds a field to a struct using AST manipulation.
func applyAddField(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("parse error: %v", err)}
	}

	var structType *ast.StructType
	ast.Inspect(node, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == edit.StructName {
			if st, ok := ts.Type.(*ast.StructType); ok {
				structType = st
				return false
			}
		}
		return true
	})

	if structType == nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("struct %q not found", edit.StructName)}
	}

	// Check if field already exists
	for _, f := range structType.Fields.List {
		for _, name := range f.Names {
			if name.Name == edit.FieldName {
				return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("field %q already exists in struct %q", edit.FieldName, edit.StructName)}
			}
		}
	}

	// Parse the field type expression
	typeExpr, err := parser.ParseExpr(edit.FieldType)
	if err != nil {
		typeExpr = ast.NewIdent(edit.FieldType)
	}

	newField := &ast.Field{
		Names: []*ast.Ident{ast.NewIdent(edit.FieldName)},
		Type:  typeExpr,
	}

	if edit.FieldTag != "" {
		tagVal := edit.FieldTag
		if !strings.HasPrefix(tagVal, "`") {
			tagVal = "`" + tagVal + "`"
		}
		newField.Tag = &ast.BasicLit{
			Kind:  token.STRING,
			Value: tagVal,
		}
	}

	structType.Fields.List = append(structType.Fields.List, newField)

	if err := writeFormattedFile(filePath, fset, node); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("added field %q %q to struct %q", edit.FieldName, edit.FieldType, edit.StructName)}
}

// applyAddImport adds an import to the Go file using astutil.
func applyAddImport(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("parse error: %v", err)}
	}

	if edit.ImportPath == "" {
		return EditResult{Success: false, File: filePath, Error: "import path is required"}
	}

	astutil.AddImport(fset, node, edit.ImportPath)

	if err := writeFormattedFile(filePath, fset, node); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("added import %q", edit.ImportPath)}
}

// applyReplaceCode replaces old code with new code using AST-level text replacement.
func applyReplaceCode(filePath string, edit EditOperation) EditResult {
	if edit.OldCode == "" || edit.NewCode == "" {
		return EditResult{Success: false, File: filePath, Error: "old_code and new_code are required for replace_code"}
	}

	content, err := os.ReadFile(filePath)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("read error: %v", err)}
	}

	newContent := strings.Replace(string(content), edit.OldCode, edit.NewCode, 1)
	if newContent == string(content) {
		return EditResult{Success: false, File: filePath, Error: "old_code not found in file"}
	}

	// Verify the result is still valid Go by parsing it
	fset := token.NewFileSet()
	_, err = parser.ParseFile(fset, filePath, newContent, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("replacement produces invalid Go: %v", err)}
	}

	if err := os.WriteFile(filePath, []byte(newContent), 0644); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	// Run gofmt on the result
	exec.Command("gofmt", "-w", filePath).Run()

	return EditResult{Success: true, File: filePath, Message: "code replacement applied"}
}

// applyDeleteFunc removes a function from the Go file.
func applyDeleteFunc(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("parse error: %v", err)}
	}

	found := false
	var newDecls []ast.Decl
	for _, decl := range node.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == edit.FuncName {
			found = true
			continue // Skip this declaration
		}
		newDecls = append(newDecls, decl)
	}

	if !found {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("function %q not found", edit.FuncName)}
	}

	node.Decls = newDecls

	if err := writeFormattedFile(filePath, fset, node); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("deleted function %q", edit.FuncName)}
}

// ─── File Writing ─────────────────────────────────────────────────────────────

func writeFormattedFile(filePath string, fset *token.FileSet, node *ast.File) error {
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	return format.Node(f, fset, node)
}

// ─── Validation Loop ──────────────────────────────────────────────────────────

func validateGoCode(filePath string, runTest bool) ValidationResult {
	result := ValidationResult{Success: true}
	dir := filepath.Dir(filePath)

	// 1. gofmt
	if out, err := exec.Command("gofmt", "-d", filePath).CombinedOutput(); err != nil {
		result.Success = false
		result.GoFmt = fmt.Sprintf("gofmt error: %v", err)
	} else if len(out) > 0 {
		// Apply formatting
		exec.Command("gofmt", "-w", filePath).Run()
	}

	// 2. go vet
	vetOut, err := exec.Command("go", "vet", filePath).CombinedOutput()
	if err != nil {
		result.Success = false
		result.GoVet = strings.TrimSpace(string(vetOut))
	}

	// 3. go build (compile check)
	buildOut, err := exec.Command("go", "build", "-o", os.DevNull, filePath).CombinedOutput()
	if err != nil {
		result.Success = false
		result.GoBuild = strings.TrimSpace(string(buildOut))
	}

	// 4. go test (optional)
	if runTest {
		testOut, err := exec.Command("go", "test", dir).CombinedOutput()
		if err != nil {
			result.Success = false
			result.GoTest = strings.TrimSpace(string(testOut))
		} else {
			result.GoTest = "PASS"
		}
	}

	return result
}

// ─── Self-Correction ──────────────────────────────────────────────────────────

func buildErrorSummary(val ValidationResult) string {
	var parts []string
	if val.GoVet != "" {
		parts = append(parts, "go vet: "+val.GoVet)
	}
	if val.GoBuild != "" {
		parts = append(parts, "go build: "+val.GoBuild)
	}
	if val.GoTest != "" {
		parts = append(parts, "go test: "+val.GoTest)
	}
	return strings.Join(parts, "; ")
}

// generateCorrectiveEdits attempts to fix common compilation errors automatically.
func generateCorrectiveEdits(filePath, errMsg string) []EditOperation {
	var edits []EditOperation

	errLower := strings.ToLower(errMsg)

	// Missing import
	if strings.Contains(errLower, "undefined:") || strings.Contains(errLower, "undeclared name:") {
		// Extract the undefined symbol
		parts := strings.Fields(errMsg)
		for i, p := range parts {
			if p == "undefined:" || p == "undeclared" {
				if i+1 < len(parts) {
					symbol := strings.TrimRight(parts[i+1], ".")
					// Try to guess the import based on common patterns
					if imp := guessImport(symbol); imp != "" {
						edits = append(edits, EditOperation{
							Type:       "add_import",
							TargetFile: filePath,
							ImportPath: imp,
						})
					}
				}
				break
			}
		}
	}

	// Unused import or variable
	if strings.Contains(errLower, "imported and not used") {
		// Extract the unused import
		parts := strings.Fields(errMsg)
		for i, p := range parts {
			if p == "imported" && i > 0 {
				unusedImport := strings.Trim(parts[i-1], "\"")
				edits = append(edits, EditOperation{
					Type:       "replace_code",
					TargetFile: filePath,
					OldCode:    fmt.Sprintf("\"%s\"", unusedImport),
					NewCode:    fmt.Sprintf("_ \"%s\"", unusedImport),
				})
			}
		}
	}

	// Unused variable — prefix with underscore
	if strings.Contains(errLower, "declared and not used") {
		parts := strings.Fields(errMsg)
		for i, p := range parts {
			if p == "declared" && i > 0 {
				varName := parts[i-1]
				edits = append(edits, EditOperation{
					Type:       "replace_code",
					TargetFile: filePath,
					OldCode:    varName + " ",
					NewCode:    "_ ",
				})
				break
			}
		}
	}

	return edits
}

// guessImport attempts to guess the import path for a symbol used but not imported.
func guessImport(symbol string) string {
	commonImports := map[string]string{
		"http":     "net/http",
		"fmt":      "fmt",
		"json":     "encoding/json",
		"os":       "os",
		"io":       "io",
		"strings":  "strings",
		"strconv":  "strconv",
		"time":     "time",
		"math":     "math",
		"sort":     "sort",
		"log":      "log",
		"filepath": "path/filepath",
		"ioutil":   "io/ioutil",
		"context":  "context",
		"sql":      "database/sql",
		"regexp":   "regexp",
		"sync":     "sync",
		"errors":   "errors",
		"flag":     "flag",
		"rand":     "math/rand",
		"atomic":   "sync/atomic",
		"template": "text/template",
		"html":     "html",
		"crypto":   "crypto/rand",
		"base64":   "encoding/base64",
		"csv":      "encoding/csv",
		"xml":      "encoding/xml",
		"gob":      "encoding/gob",
		"hex":      "encoding/hex",
		"gzip":     "compress/gzip",
		"tar":      "archive/tar",
		"zip":      "archive/zip",
		"bufio":    "bufio",
		"bytes":    "bytes",
		"exec":     "os/exec",
		"signal":   "os/signal",
		"user":     "os/user",
		"net":      "net",
		"url":      "net/url",
		"rpc":      "net/rpc",
		"smtp":     "net/smtp",
		"mail":     "net/mail",
		"tls":      "crypto/tls",
		"sha256":   "crypto/sha256",
		"md5":      "crypto/md5",
		"aes":      "crypto/aes",
		"rsa":      "crypto/rsa",
		"x509":     "crypto/x509",
		"tensor":   "github.com/golangast/gollemer/internal/ai/neural/tensor",
		"nn":       "github.com/golangast/gollemer/internal/ai/neural/nn",
		"moe":      "github.com/golangast/gollemer/internal/ai/moe",
		"semantic": "github.com/golangast/gollemer/internal/ai/neural/semantic",
		"ner":      "github.com/golangast/gollemer/internal/ai/neural/nn/ner",
		"astutil":  "golang.org/x/tools/go/ast/astutil",
	}
	if imp, ok := commonImports[symbol]; ok {
		return imp
	}
	return ""
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func countSuccesses(results []EditResult) int {
	count := 0
	for _, r := range results {
		if r.Success {
			count++
		}
	}
	return count
}

// ─── Tool Handler Interface ───────────────────────────────────────────────────

// ToolHandler provides a standardized interface for the Gollemer supervisor
// to call the Go edit agent as a tool.
type ToolHandler struct {
	Name        string
	Description string
}

// NewToolHandler creates a new ToolHandler for the Go edit agent.
func NewToolHandler() *ToolHandler {
	return &ToolHandler{
		Name:        "go_edit_agent",
		Description: "Edits Go source files using AST-level manipulation with validation and self-correction. Supports: insert_func, modify_func, add_field, add_import, replace_code, delete_func.",
	}
}

// Handle processes a tool call from the supervisor and returns the result.
// Input is a JSON-encoded AgentRequest, output is a JSON-encoded AgentResponse.
func (h *ToolHandler) Handle(inputJSON []byte) ([]byte, error) {
	var req AgentRequest
	if err := json.Unmarshal(inputJSON, &req); err != nil {
		return nil, fmt.Errorf("invalid request: %w", err)
	}

	if req.MaxRetries <= 0 {
		req.MaxRetries = 3
	}

	startTime := time.Now()
	resp := executeAgent(req)
	resp.Duration = time.Since(startTime).Round(time.Millisecond).String()

	return json.MarshalIndent(resp, "", "  ")
}
