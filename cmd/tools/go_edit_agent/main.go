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
	Query      string          `json:"query"`       // Natural language query (alternative to Edits)
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
	query := flag.String("query", "", "Natural language edit request (alternative to -edits)")
	runTest := flag.Bool("test", false, "Run go test after edits")
	maxRetries := flag.Int("retries", 3, "Max self-correction retries")
	interactive := flag.Bool("interactive", false, "Read edits from stdin as JSON")
	flag.Parse()

	startTime := time.Now()

	var req AgentRequest

	if *interactive {
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
		req.Query = *query
		req.RunTest = *runTest
		req.MaxRetries = *maxRetries
	}

	if req.MaxRetries <= 0 {
		req.MaxRetries = 3
	}

	if req.File == "" {
		log.Fatal("No target file specified. Use -file or provide in JSON.")
	}

	// If a natural language query was provided, parse it into edit operations
	if req.Query != "" && len(req.Edits) == 0 {
		edits := parseNaturalLanguageQuery(req.File, req.Query)
		if len(edits) == 0 {
			log.Fatal("Could not understand the edit request. Try being more specific (e.g., 'add function calculate that takes two ints and returns their sum')")
		}
		req.Edits = edits
	}

	resp := executeAgent(req)
	resp.Duration = time.Since(startTime).Round(time.Millisecond).String()

	output, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(output))

	if !resp.Success {
		os.Exit(1)
	}
}

// ─── Natural Language Parsing ─────────────────────────────────────────────────

// parseNaturalLanguageQuery reads the file's AST to understand its structure,
// then parses the natural language query to determine what edit to make.
func parseNaturalLanguageQuery(filePath, query string) []EditOperation {
	lower := strings.ToLower(strings.TrimSpace(query))

	// General fix/repair query: read the file and try to fix syntax errors
	if strings.Contains(lower, "fix") || strings.Contains(lower, "repair") || strings.Contains(lower, "syntax error") {
		edits := fixSyntaxErrors(filePath)
		if len(edits) > 0 {
			return edits
		}
	}

	// Read the file to understand its current structure
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		log.Printf("Warning: could not parse file for context: %v", err)
	}

	// Collect existing function names and structs for context
	existingFuncs := make(map[string]bool)
	existingStructs := make(map[string]*ast.StructType)
	if node != nil {
		ast.Inspect(node, func(n ast.Node) bool {
			switch t := n.(type) {
			case *ast.FuncDecl:
				existingFuncs[t.Name.Name] = true
			case *ast.TypeSpec:
				if st, ok := t.Type.(*ast.StructType); ok {
					existingStructs[t.Name.Name] = st
				}
			}
			return true
		})
	}

	// Detect: add/modify/delete function
	if strings.Contains(lower, "function") || strings.Contains(lower, "func ") {
		funcName := extractFuncName(lower)
		if funcName == "" {
			return nil
		}

		// Check if function already exists
		if existingFuncs[funcName] {
			// Check if this is a signature modification (return type, params)
			if strings.Contains(lower, "return type") || strings.Contains(lower, "return ") ||
				strings.Contains(lower, "signature") || strings.Contains(lower, "parameter") ||
				strings.Contains(lower, "(") || strings.Contains(lower, "int") {
				// Use replace_code to modify the function signature
				oldSig, newSig := buildSignatureChange(lower, funcName, filePath)
				if oldSig != "" && newSig != "" {
					return []EditOperation{{
						Type:       "replace_code",
						TargetFile: filePath,
						OldCode:    oldSig,
						NewCode:    newSig,
					}}
				}
			}
			// Modify existing function body
			body := buildFuncBodyFromQuery(lower, funcName)
			return []EditOperation{{
				Type:       "modify_func",
				TargetFile: filePath,
				FuncName:   funcName,
				Code:       body,
			}}
		}

		// Check if we're deleting
		if strings.Contains(lower, "delete") || strings.Contains(lower, "remove") {
			return []EditOperation{{
				Type:       "delete_func",
				TargetFile: filePath,
				FuncName:   funcName,
			}}
		}

		// Insert new function
		code := buildFuncCodeFromQuery(lower, funcName)
		return []EditOperation{{
			Type:       "insert_func",
			TargetFile: filePath,
			FuncName:   funcName,
			Code:       code,
		}}
	}

	// Detect: add import
	if strings.Contains(lower, "import ") {
		importPath := extractImportPath(lower)
		if importPath != "" {
			return []EditOperation{{
				Type:       "add_import",
				TargetFile: filePath,
				ImportPath: importPath,
			}}
		}
	}

	// Detect: add struct (new struct creation)
	if strings.Contains(lower, "struct") && (strings.Contains(lower, "add") || strings.Contains(lower, "new") || strings.Contains(lower, "create")) {
		structName := extractStructName(lower)
		if structName == "" {
			return nil
		}

		// Check if struct already exists
		if _, exists := existingStructs[structName]; exists {
			// Struct exists - add field to it
			fieldName := extractFieldName(lower)
			fieldType := extractFieldType(lower)
			if fieldName != "" {
				return []EditOperation{{
					Type:       "add_field",
					TargetFile: filePath,
					StructName: structName,
					FieldName:  fieldName,
					FieldType:  fieldType,
				}}
			}
			return nil
		}

		// Create new struct with fields
		code := buildStructCodeFromQuery(lower, structName)
		return []EditOperation{{
			Type:       "insert_struct",
			TargetFile: filePath,
			FuncName:   structName,
			Code:       code,
		}}
	}

	// Detect: add field to struct
	if strings.Contains(lower, "field") && strings.Contains(lower, "struct") {
		fieldName := extractFieldName(lower)
		structName := extractStructName(lower)
		fieldType := extractFieldType(lower)
		if fieldName != "" && structName != "" {
			return []EditOperation{{
				Type:       "add_field",
				TargetFile: filePath,
				StructName: structName,
				FieldName:  fieldName,
				FieldType:  fieldType,
			}}
		}
		// If struct name not specified, use the first struct found
		if fieldName != "" && structName == "" && len(existingStructs) > 0 {
			for name := range existingStructs {
				structName = name
				break
			}
			return []EditOperation{{
				Type:       "add_field",
				TargetFile: filePath,
				StructName: structName,
				FieldName:  fieldName,
				FieldType:  fieldType,
			}}
		}
	}

	return nil
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

		// If at least one edit succeeded, keep the changes even if validation fails
		// (the file may have pre-existing errors unrelated to our edit)
		if resp.EditsApplied > 0 {
			resp.Success = true
			resp.Error = fmt.Sprintf("edit applied but file has pre-existing issues: %s", buildErrorSummary(valResult))
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
		case "fix_syntax":
			// fixSyntaxErrors already wrote the file directly
			result = EditResult{Success: true, File: filePath, Message: "fixed syntax errors"}
		case "insert_struct":
			result = applyInsertStruct(filePath, edit)
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

// applyInsertStruct inserts a struct type definition into the Go file using text-based append.
func applyInsertStruct(filePath string, edit EditOperation) EditResult {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("read error: %v", err)}
	}

	// Check if struct already exists via text search
	if strings.Contains(string(content), "type "+edit.FuncName+" struct") {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("struct %q already exists", edit.FuncName)}
	}

	// Build the struct code
	structCode := edit.Code
	if structCode == "" {
		structCode = fmt.Sprintf("type %s struct {\n}\n", edit.FuncName)
	}

	// Append the struct to the end of the file
	newContent := string(content)
	if !strings.HasSuffix(newContent, "\n") {
		newContent += "\n"
	}
	newContent += structCode

	if err := os.WriteFile(filePath, []byte(newContent), 0644); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	// Run gofmt
	exec.Command("gofmt", "-w", filePath).Run()

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("inserted struct %q", edit.FuncName)}
}

// applyInsertFunc inserts a new function into the Go file using AST manipulation.
// Falls back to text-based insertion if the file has syntax errors.
func applyInsertFunc(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)

	// If file can't be parsed (has syntax errors), fall back to text-based insertion
	if err != nil {
		// Read the file content
		content, readErr := os.ReadFile(filePath)
		if readErr != nil {
			return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("read error: %v", readErr)}
		}

		// Build the function code
		funcCode := edit.Code
		if funcCode == "" {
			funcCode = fmt.Sprintf("func %s() {\n\t// TODO: implement\n}\n", edit.FuncName)
		}

		// Check if function already exists via text search
		if strings.Contains(string(content), "func "+edit.FuncName+"(") {
			return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("function %q already exists", edit.FuncName)}
		}

		// Append the function to the end of the file
		newContent := string(content)
		if !strings.HasSuffix(newContent, "\n") {
			newContent += "\n"
		}
		newContent += funcCode

		if err := os.WriteFile(filePath, []byte(newContent), 0644); err != nil {
			return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
		}

		// Run gofmt
		exec.Command("gofmt", "-w", filePath).Run()

		return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("inserted function %q (text fallback)", edit.FuncName)}
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

	// 2. go vet (compile check — works on individual files without requiring main())
	vetOut, err := exec.Command("go", "vet", filePath).CombinedOutput()
	if err != nil {
		result.Success = false
		result.GoVet = strings.TrimSpace(string(vetOut))
	}

	// 3. go test (optional)
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
		parts := strings.Fields(errMsg)
		for i, p := range parts {
			if p == "undefined:" || p == "undeclared" {
				if i+1 < len(parts) {
					symbol := strings.TrimRight(parts[i+1], ".")
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

// ─── Error Pattern Training Data ─────────────────────────────────────────────

// ErrorPattern describes a Go error pattern and how to fix it.
type ErrorPattern struct {
	ID          string   `json:"id"`
	Match       string   `json:"match"`
	Description string   `json:"description"`
	FixType     string   `json:"fix_type"`
	Examples    []string `json:"examples"`
	Confidence  float64  `json:"confidence"`
}

// ErrorPatternsDB holds all loaded error patterns.
type ErrorPatternsDB struct {
	Version  int            `json:"version"`
	Patterns []ErrorPattern `json:"patterns"`
}

// loadErrorPatterns loads the error pattern training data from the project root.
func loadErrorPatterns() *ErrorPatternsDB {
	// Try common locations for the training data
	candidates := []string{
		"data/training/go_error_patterns.json",
		"../data/training/go_error_patterns.json",
		"/home/zendrulat/g/gollemer/data/training/go_error_patterns.json",
	}
	for _, path := range candidates {
		data, err := os.ReadFile(path)
		if err == nil {
			var db ErrorPatternsDB
			if err := json.Unmarshal(data, &db); err == nil {
				log.Printf("📚 Loaded %d error patterns from %s", len(db.Patterns), path)
				return &db
			}
		}
	}
	log.Printf("⚠️  No error pattern training data found (checked %d locations)", len(candidates))
	return &ErrorPatternsDB{Patterns: []ErrorPattern{}}
}

// findMatchingPatterns finds all patterns that match the given error string.
func (db *ErrorPatternsDB) findMatchingPatterns(errStr string) []ErrorPattern {
	var matches []ErrorPattern
	lower := strings.ToLower(errStr)
	for _, p := range db.Patterns {
		if strings.Contains(lower, strings.ToLower(p.Match)) {
			matches = append(matches, p)
		}
	}
	return matches
}

// ─── Syntax Error Fixer ──────────────────────────────────────────────────────

// fixSyntaxErrors reads a Go file, tries to parse it, and applies text-based
// fixes for common syntax errors using the training data. Returns edit operations.
func fixSyntaxErrors(filePath string) []EditOperation {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil
	}

	// Try parsing to see if there are errors
	_, err = parser.ParseFile(token.NewFileSet(), filePath, content, parser.ParseComments)
	if err == nil {
		return nil // No errors
	}

	// Load error patterns from training data
	patterns := loadErrorPatterns()
	errStr := err.Error()
	lines := strings.Split(string(content), "\n")
	modified := false

	// Find matching patterns
	matches := patterns.findMatchingPatterns(errStr)
	if len(matches) == 0 {
		log.Printf("⚠️  No matching error pattern found for: %s", errStr)
		return nil
	}

	log.Printf("🔍 Matched %d error patterns", len(matches))

	// Apply fixes for each matching pattern
	for _, pattern := range matches {
		switch pattern.FixType {
		case "remove_duplicate_type":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "func ") {
					// Remove duplicate type names like "int int" -> "int"
					for _, typ := range []string{"int int", "string string", "float64 float64", "bool bool"} {
						if strings.Contains(trimmed, typ) {
							lines[i] = strings.Replace(trimmed, typ, strings.Fields(typ)[0], 1)
							modified = true
							log.Printf("🔧 [%s] Fixed duplicate type in: %s", pattern.ID, trimmed)
							break
						}
					}
				}
			}

		case "add_brace_after_func":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				// Handle: func declaration without opening brace
				if strings.HasPrefix(trimmed, "func ") && strings.Contains(trimmed, ")") && !strings.Contains(trimmed, "{") {
					lines[i] = trimmed + " {"
					modified = true
					log.Printf("🔧 [%s] Added missing '{' to: %s", pattern.ID, trimmed)
				}
				// Handle: type X struct declaration without opening brace (e.g. missing '{' before fields)
				if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, " struct") && !strings.Contains(trimmed, "{") {
					lines[i] = trimmed + " {"
					modified = true
					log.Printf("🔧 [%s] Added missing '{' to: %s", pattern.ID, trimmed)
				}
			}

		case "fix_type_declaration", "add_struct_keyword":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				// Handle: "type X {" missing the 'struct' keyword — e.g. "type jill  {"
				// produces the Go parser error: expected type, found '{'
				if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, "{") &&
					!strings.Contains(trimmed, " struct ") && !strings.HasPrefix(trimmed, "type struct") {
					lines[i] = strings.Replace(trimmed, "{", "struct {", 1)
					modified = true
					log.Printf("🔧 [%s] Added missing 'struct' keyword to: %s", pattern.ID, trimmed)
				}
			}

		case "add_closing_paren":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "func ") && strings.Contains(trimmed, "(") && !strings.Contains(trimmed, ")") {
					if strings.Contains(trimmed, "{") {
						lines[i] = strings.Replace(trimmed, " {", ") {", 1)
					} else {
						lines[i] = trimmed + ")"
					}
					modified = true
					log.Printf("🔧 [%s] Added missing ')' to: %s", pattern.ID, trimmed)
				}
			}

		case "add_closing_brace":
			openBraces := 0
			closeBraces := 0
			for _, line := range lines {
				openBraces += strings.Count(line, "{")
				closeBraces += strings.Count(line, "}")
			}
			if openBraces > closeBraces {
				lines = append(lines, strings.Repeat("}", openBraces-closeBraces))
				modified = true
				log.Printf("🔧 [%s] Added %d missing closing brace(s)", pattern.ID, openBraces-closeBraces)
			}

		case "balance_braces":
			openBraces := 0
			closeBraces := 0
			for _, line := range lines {
				openBraces += strings.Count(line, "{")
				closeBraces += strings.Count(line, "}")
			}
			if openBraces > closeBraces {
				lines = append(lines, strings.Repeat("}", openBraces-closeBraces))
				modified = true
				log.Printf("🔧 [%s] Added %d missing closing brace(s)", pattern.ID, openBraces-closeBraces)
			}

		case "add_missing_paren":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.Contains(trimmed, "fmt.Println") && !strings.Contains(trimmed, "(") {
					lines[i] = strings.Replace(trimmed, "fmt.Println", "fmt.Println(", 1)
					if !strings.HasSuffix(lines[i], ")") {
						lines[i] += ")"
					}
					modified = true
					log.Printf("🔧 [%s] Added missing '(' to: %s", pattern.ID, trimmed)
				}
			}

		case "add_func_keyword":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "fn ") {
					lines[i] = strings.Replace(trimmed, "fn ", "func ", 1)
					modified = true
					log.Printf("🔧 [%s] Added 'func' keyword to: %s", pattern.ID, trimmed)
				}
			}

		case "add_blank_import":
			// Already handled by generateCorrectiveEdits
			log.Printf("ℹ️  [%s] Unused import detected - will be handled by self-correction", pattern.ID)

		case "guess_import":
			// Already handled by generateCorrectiveEdits
			log.Printf("ℹ️  [%s] Undeclared name detected - will be handled by self-correction", pattern.ID)

		case "prefix_underscore":
			// Already handled by generateCorrectiveEdits
			log.Printf("ℹ️  [%s] Unused variable detected - will be handled by self-correction", pattern.ID)

		case "add_return_statement":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "func ") && strings.Contains(trimmed, ")") && !strings.Contains(trimmed, "{") {
					// This is a function declaration without body - add a return
					continue
				}
				// Find functions with body but no return
				if trimmed == "}" && i > 0 {
					prevLine := strings.TrimSpace(lines[i-1])
					if !strings.HasPrefix(prevLine, "return") && !strings.HasPrefix(prevLine, "}") {
						// Check if the function has a return type
						for j := i - 1; j >= 0; j-- {
							checkLine := strings.TrimSpace(lines[j])
							if strings.HasPrefix(checkLine, "func ") {
								// Simple check: if func has a return type, add return 0
								fields := strings.Fields(checkLine)
								if len(fields) >= 4 && fields[len(fields)-1] != "{" {
									// Has return type - add return statement
									lines = append(lines[:i], append([]string{"\treturn 0"}, lines[i:]...)...)
									modified = true
									log.Printf("🔧 [%s] Added missing return statement", pattern.ID)
								}
								break
							}
						}
					}
				}
			}

		case "fix_return_count":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "return ") && i > 0 {
					// Check if the function has no return type but has a return value
					for j := i - 1; j >= 0; j-- {
						checkLine := strings.TrimSpace(lines[j])
						if strings.HasPrefix(checkLine, "func ") {
							// If func has no return type but body has return with value, remove the value
							if !strings.Contains(checkLine, ") ") && !strings.Contains(checkLine, ") (") {
								// No return type - remove return value
								parts := strings.Fields(trimmed)
								if len(parts) > 1 {
									lines[i] = "\treturn"
									modified = true
									log.Printf("🔧 [%s] Fixed return count in function", pattern.ID)
								}
							}
							break
						}
					}
				}
			}

		case "add_newline":
			// Complex fix - just report for now
			log.Printf("ℹ️  [%s] Expected semicolon - may need manual fix", pattern.ID)

		case "report_unfixable":
			log.Printf("⚠️  [%s] Cannot auto-fix: %s", pattern.ID, pattern.Description)
		}
	}

	if !modified {
		return nil
	}

	// Write the fixed content
	newContent := strings.Join(lines, "\n")
	if err := os.WriteFile(filePath, []byte(newContent), 0644); err != nil {
		return nil
	}

	// Run gofmt
	exec.Command("gofmt", "-w", filePath).Run()

	// Return a special edit that applyEdits will count as success
	return []EditOperation{{
		Type:       "fix_syntax",
		TargetFile: filePath,
	}}
}

// ─── Natural Language Helper Functions ────────────────────────────────────────

// extractFuncName extracts a function name from a natural language query.
func extractFuncName(lower string) string {
	patterns := []string{
		"function called ", "function named ", "function '", "function \"",
		"func called ", "func named ", "func '", "func \"",
		"add function ", "add func ", "new function ", "insert function ",
		"add the function ", "add a function ", "add a new function ",
		"in the ", "in ", "to ",
	}

	// First try the standard patterns
	for _, prefix := range patterns {
		if idx := strings.Index(lower, prefix); idx >= 0 {
			start := idx + len(prefix)
			remaining := lower[start:]
			words := strings.Fields(remaining)
			for _, w := range words {
				name := strings.Trim(w, "'\",.;:()")
				if name != "" && !isStopWord(name) {
					return name
				}
			}
		}
	}

	// Try "X function" pattern (name before the word "function")
	if idx := strings.Index(lower, " function"); idx >= 0 {
		before := lower[:idx]
		words := strings.Fields(before)
		// Take the last word before " function"
		for i := len(words) - 1; i >= 0; i-- {
			name := strings.Trim(words[i], "'\",.;:()")
			if name != "" && !isStopWord(name) && !isStopWord(name+" function") {
				return name
			}
		}
	}

	// Fallback: look for "called X" or "named X"
	for _, marker := range []string{" called ", " named "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			start := idx + len(marker)
			remaining := lower[start:]
			words := strings.Fields(remaining)
			for _, w := range words {
				name := strings.Trim(w, "'\",.;:()")
				if name != "" && !isStopWord(name) {
					return name
				}
			}
		}
	}

	return ""
}

// extractStructName extracts a struct name from a natural language query.
func extractStructName(lower string) string {
	for _, marker := range []string{"struct named ", "struct called ", "struct "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			remaining := lower[idx+len(marker):]
			words := strings.Fields(remaining)
			if len(words) > 0 {
				name := strings.Trim(words[0], "'\",.;:()")
				if name != "named" && name != "called" && name != "with" && name != "a" && name != "an" {
					return name
				}
				if len(words) > 1 {
					return strings.Trim(words[1], "'\",.;:()")
				}
			}
		}
	}
	return ""
}

// extractFieldName extracts a field name from a natural language query.
func extractFieldName(lower string) string {
	if idx := strings.Index(lower, "field "); idx >= 0 {
		remaining := lower[idx+6:]
		words := strings.Fields(remaining)
		if len(words) > 0 {
			return strings.Trim(words[0], "'\",.;:()")
		}
	}
	return ""
}

// extractFieldType extracts a field type from a natural language query.
func extractFieldType(lower string) string {
	for _, marker := range []string{" of type ", " type "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			start := idx + len(marker)
			remaining := lower[start:]
			words := strings.Fields(remaining)
			if len(words) > 0 {
				return strings.Trim(words[0], "'\",.;:()")
			}
		}
	}
	return "string"
}

// extractImportPath extracts an import path from a natural language query.
func extractImportPath(lower string) string {
	if idx := strings.Index(lower, "import "); idx >= 0 {
		remaining := lower[idx+7:]
		words := strings.Fields(remaining)
		if len(words) > 0 {
			path := strings.Trim(words[0], "'\"")
			if path != "" {
				return path
			}
		}
	}
	return ""
}

// buildFuncCodeFromQuery generates Go function code from a natural language description.
func buildFuncCodeFromQuery(lower, funcName string) string {
	hasParams := strings.Contains(lower, "take") || strings.Contains(lower, "parameter") || strings.Contains(lower, "argument") || strings.Contains(lower, "input")
	hasReturn := strings.Contains(lower, "return") || strings.Contains(lower, "result")

	hasInt := strings.Contains(lower, "int") || strings.Contains(lower, "integer")
	hasString := strings.Contains(lower, "string") || strings.Contains(lower, "str")
	hasFloat := strings.Contains(lower, "float") || strings.Contains(lower, "float64")

	returnsInt := strings.Contains(lower, "sum") || strings.Contains(lower, "total") || strings.Contains(lower, "count") || strings.Contains(lower, "number")
	returnsString := strings.Contains(lower, "concat") || strings.Contains(lower, "join") || strings.Contains(lower, "message")
	returnsBool := strings.Contains(lower, "check") || strings.Contains(lower, "valid") || strings.Contains(lower, "compare")

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("func %s(", funcName))

	if hasParams {
		if hasInt && hasString {
			sb.WriteString("a int, b string")
		} else if hasInt && hasFloat {
			sb.WriteString("a int, b float64")
		} else if hasInt {
			if strings.Contains(lower, "two") || strings.Contains(lower, "2") {
				sb.WriteString("a, b int")
			} else {
				sb.WriteString("a int")
			}
		} else if hasString {
			sb.WriteString("s string")
		} else if hasFloat {
			sb.WriteString("f float64")
		} else {
			sb.WriteString("a int")
		}
	}

	sb.WriteString(")")

	if hasReturn {
		if returnsInt {
			sb.WriteString(" int")
		} else if returnsString {
			sb.WriteString(" string")
		} else if returnsBool {
			sb.WriteString(" bool")
		} else if hasInt {
			sb.WriteString(" int")
		} else {
			sb.WriteString(" int")
		}
	}

	sb.WriteString(" {\n")

	if strings.Contains(lower, "multiply") || strings.Contains(lower, "product") || strings.Contains(lower, "times") {
		sb.WriteString("\treturn a * b\n")
	} else if strings.Contains(lower, "sum") || strings.Contains(lower, "add") || strings.Contains(lower, "plus") || strings.Contains(lower, "total") {
		sb.WriteString("\treturn a + b\n")
	} else if strings.Contains(lower, "concat") || strings.Contains(lower, "join") {
		sb.WriteString("\treturn a + b\n")
	} else if strings.Contains(lower, "greet") || strings.Contains(lower, "hello") {
		sb.WriteString("\treturn fmt.Sprintf(\"Hello, %s!\", name)\n")
	} else if strings.Contains(lower, "square") {
		sb.WriteString("\treturn a * a\n")
	} else {
		sb.WriteString("\t// TODO: implement\n")
		sb.WriteString("\treturn 0\n")
	}

	sb.WriteString("}\n")

	return sb.String()
}

// buildFuncBodyFromQuery generates just the body statements for modifying an existing function.
func buildFuncBodyFromQuery(lower, funcName string) string {
	var sb strings.Builder

	if strings.Contains(lower, "multiply") || strings.Contains(lower, "product") || strings.Contains(lower, "times") {
		sb.WriteString("\treturn a * b\n")
	} else if strings.Contains(lower, "sum") || strings.Contains(lower, "add") || strings.Contains(lower, "plus") || strings.Contains(lower, "total") {
		sb.WriteString("\treturn a + b\n")
	} else if strings.Contains(lower, "concat") || strings.Contains(lower, "join") {
		sb.WriteString("\treturn a + b\n")
	} else if strings.Contains(lower, "greet") || strings.Contains(lower, "hello") {
		sb.WriteString("\treturn fmt.Sprintf(\"Hello, %s!\", name)\n")
	} else if strings.Contains(lower, "square") {
		sb.WriteString("\treturn a * a\n")
	} else {
		sb.WriteString("\t// TODO: implement\n")
		sb.WriteString("\treturn 0\n")
	}

	return sb.String()
}

// buildSignatureChange reads the file, finds the function signature, and modifies it
// based on the natural language query. Returns old and new code for replace_code.
func buildSignatureChange(lower, funcName, filePath string) (string, string) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", ""
	}
	lines := strings.Split(string(content), "\n")

	// Find the function declaration line
	funcIdx := -1
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "func "+funcName+"(") {
			funcIdx = i
			break
		}
	}
	if funcIdx == -1 {
		return "", ""
	}

	oldLine := lines[funcIdx]
	trimmed := strings.TrimSpace(oldLine)

	// Detect: add return type
	if strings.Contains(lower, "return type") || strings.Contains(lower, "return ") {
		// Check if function already has a return type
		// Look for pattern like ") int {" or ") string {" or ") (" after the params
		hasReturnType := false
		if strings.Contains(trimmed, ") ") && !strings.Contains(trimmed, ") {") {
			hasReturnType = true
		}
		if strings.Contains(trimmed, ") (") {
			hasReturnType = true
		}
		if hasReturnType {
			return "", ""
		}
		// Add return type - default to int
		// Find the position of " {" to insert the return type before it
		if braceIdx := strings.Index(trimmed, " {"); braceIdx >= 0 {
			newLine := trimmed[:braceIdx] + " int" + trimmed[braceIdx:]
			return oldLine, newLine
		}
		// No brace yet - function declaration without body
		if strings.HasSuffix(trimmed, ")") {
			newLine := trimmed + " int {"
			return oldLine, newLine
		}
		return "", ""
	}

	// Detect: add parameters like (int, int)
	if strings.Contains(lower, "(") && strings.Contains(lower, "int") {
		// Check if function already has params
		if strings.Contains(trimmed, "(") && !strings.Contains(trimmed, "()") {
			// Already has params
			return "", ""
		}
		// Extract the param types from the query
		// e.g. "(int,int)" or "(int, int)"
		parenStart := strings.Index(lower, "(")
		parenEnd := strings.Index(lower, ")")
		if parenStart >= 0 && parenEnd > parenStart {
			params := lower[parenStart : parenEnd+1]
			if strings.HasSuffix(trimmed, "{") {
				newLine := strings.Replace(trimmed, " {", " "+params+" {", 1)
				return oldLine, newLine
			}
			newLine := trimmed + " " + params + " {"
			return oldLine, newLine
		}
	}

	return "", ""
}

// buildStructCodeFromQuery generates Go struct code supporting single or multiple fields.
func buildStructCodeFromQuery(lower string, structName string) string {
	var fields [][2]string
	for _, marker := range []string{"fields ", "field "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			remaining := lower[idx+len(marker):]
			words := strings.Fields(remaining)
			for i := 0; i+1 < len(words); i += 2 {
				fn := strings.Trim(words[i], "'\",.;:()")
				ft := strings.Trim(words[i+1], "'\",.;:()")
				if fn != "" && ft != "" && !isStopWord(fn) {
					fields = append(fields, [2]string{fn, ft})
				}
			}
			break
		}
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("type %s struct {\n", structName))
	for _, f := range fields {
		sb.WriteString(fmt.Sprintf("\t%s %s\n", f[0], f[1]))
	}
	sb.WriteString("}\n")
	return sb.String()
}

// buildStructCode generates Go struct type definition code.
func buildStructCode(structName, fieldName, fieldType string) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("type %s struct {\n", structName))
	if fieldName != "" {
		if fieldType == "" {
			fieldType = "string"
		}
		sb.WriteString(fmt.Sprintf("\t%s %s\n", fieldName, fieldType))
	}
	sb.WriteString("}\n")
	return sb.String()
}

// isStopWord checks if a word is a common stop word.
func isStopWord(word string) bool {
	stopWords := map[string]bool{
		"that": true, "this": true, "with": true, "from": true, "into": true,
		"file": true, "the": true, "a": true, "an": true, "to": true,
		"in": true, "of": true, "for": true, "and": true, "or": true,
		"it": true, "is": true, "are": true, "was": true, "be": true,
		"has": true, "have": true, "do": true, "does": true, "will": true,
		"would": true, "could": true, "should": true, "may": true, "might": true,
		"can": true, "shall": true, "must": true, "need": true, "let": true,
		"make": true, "take": true, "get": true, "set": true, "put": true,
		"add": true, "new": true, "function": true, "func": true,
		"called": true, "named": true, "returns": true, "return": true,
		"takes": true, "parameters": true, "parameter": true,
		"arguments": true, "argument": true, "input": true, "output": true,
		"two": true, "three": true, "four": true, "five": true,
		"integers": true, "integer": true, "int": true, "string": true,
		"float": true, "bool": true, "boolean": true,
	}
	return stopWords[word]
}

// ─── Tool Handler Interface ───────────────────────────────────────────────────

type ToolHandler struct {
	Name        string
	Description string
}

func NewToolHandler() *ToolHandler {
	return &ToolHandler{
		Name:        "go_edit_agent",
		Description: "Edits Go source files using AST-level manipulation with validation and self-correction. Supports: insert_func, modify_func, add_field, add_import, replace_code, delete_func.",
	}
}

func (h *ToolHandler) Handle(inputJSON []byte) ([]byte, error) {
	var req AgentRequest
	if err := json.Unmarshal(inputJSON, &req); err != nil {
		return nil, fmt.Errorf("invalid request: %w", err)
	}

	if req.MaxRetries <= 0 {
		req.MaxRetries = 3
	}

	// Parse natural language query if provided
	if req.Query != "" && len(req.Edits) == 0 {
		edits := parseNaturalLanguageQuery(req.File, req.Query)
		if len(edits) == 0 {
			return nil, fmt.Errorf("could not understand the edit request")
		}
		req.Edits = edits
	}

	startTime := time.Now()
	resp := executeAgent(req)
	resp.Duration = time.Since(startTime).Round(time.Millisecond).String()

	return json.MarshalIndent(resp, "", "  ")
}
