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
	"regexp"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
)

type SemanticOutput struct {
	Operation      string          `json:"operation"`
	TargetResource *ResourceTarget `json:"target_resource"`
}

type ResourceTarget struct {
	Type       string                 `json:"type"`
	Name       string                 `json:"name"`
	Content    string                 `json:"content,omitempty"`
	Properties map[string]interface{} `json:"properties"`
}

type StructFieldDef struct {
	Name string `json:"name"`
	Type string `json:"type"`
	Tag  string `json:"tag,omitempty"`
}

func main() {
	targetFile := flag.String("target", "./main.go", "Target file path to apply code changes")
	validate := flag.Bool("validate", true, "Run validation steps (gofmt, go vet, go build)")
	rollback := flag.Bool("rollback", true, "Auto-rollback file changes on validation error")
	selfHeal := flag.Bool("self-heal", true, "Enable self-healing loop via moe_inference")
	maxRetries := flag.Int("max-retries", 3, "Maximum self-healing retries")
	moeBin := flag.String("moe-bin", "", "Path to moe_inference binary or script")
	flag.Parse()

	inputBytes, err := io.ReadAll(os.Stdin)
	if err != nil {
		log.Fatalf("Error reading stdin: %v", err)
	}

	inputText := string(inputBytes)
	jsonStart := strings.Index(inputText, "=== Generated Semantic Output ===")
	if jsonStart != -1 {
		inputText = inputText[jsonStart+len("=== Generated Semantic Output ==="):]
	}
	jsonEnd := strings.Index(inputText, "=================================")
	if jsonEnd != -1 {
		inputText = inputText[:jsonEnd]
	}

	inputText = strings.TrimSpace(inputText)
	if inputText == "" {
		log.Fatal("No valid JSON found in stdin stream")
	}

	var output SemanticOutput
	if err := json.Unmarshal([]byte(inputText), &output); err != nil {
		log.Fatalf("Failed to parse JSON input: %v", err)
	}

	if output.TargetResource == nil {
		log.Fatal("JSON output missing target_resource")
	}

	handlerName := getStringProp(output.TargetResource.Properties, "handler")
	if handlerName == "" {
		handlerName = output.TargetResource.Name
	}

	fmt.Printf("[apply_patch] Operation: %s | Target: %s | Component: %s\n", output.Operation, *targetFile, handlerName)

	if err := applyPatchWithSelfHealing(*targetFile, output, *maxRetries, *validate, *rollback, *selfHeal, *moeBin); err != nil {
		log.Fatalf("apply_patch failed for %s: %v", *targetFile, err)
	}

	fmt.Printf("Successfully applied patch and validated %s\n", *targetFile)
}

func applyPatchWithSelfHealing(targetFile string, output SemanticOutput, maxRetries int, enableValidation, enableRollback, enableSelfHeal bool, moeBin string) error {
	// Step 1: Rollback Buffer Setup
	var backupBytes []byte
	existed := false
	if _, err := os.Stat(targetFile); err == nil {
		existed = true
		b, err := os.ReadFile(targetFile)
		if err != nil {
			return fmt.Errorf("failed to read target file for backup: %w", err)
		}
		backupBytes = b
	}

	currentOutput := output

	for attempt := 0; attempt <= maxRetries; attempt++ {
		props := make(map[string]interface{})
		if currentOutput.TargetResource != nil && currentOutput.TargetResource.Properties != nil {
			props = currentOutput.TargetResource.Properties
		}

		handlerName := getStringProp(props, "handler")
		if handlerName == "" && currentOutput.TargetResource != nil {
			handlerName = currentOutput.TargetResource.Name
		}
		content := ""
		if currentOutput.TargetResource != nil {
			content = currentOutput.TargetResource.Content
		}
		if content == "" {
			content = getStringProp(props, "content")
		}
		urlPath := getStringProp(props, "url")

		if attempt > 0 {
			fmt.Printf("[apply_patch] (Self-Healing Attempt %d/%d) Re-applying patch to %s...\n", attempt, maxRetries, targetFile)
		}

		// Step 2: Apply Code Content & AST Injections
		if err := applyCodeChanges(targetFile, props, handlerName, content, urlPath); err != nil {
			if enableRollback {
				_ = rollbackFile(targetFile, backupBytes, existed)
			}
			return fmt.Errorf("failed to apply code changes: %w", err)
		}

		// Skip validation if not a .go file or validation disabled
		if !enableValidation || !strings.HasSuffix(targetFile, ".go") {
			return nil
		}

		// Step 3: Run Validation (gofmt, go vet, go build)
		valErr := validateGoCode(targetFile)
		if valErr == nil {
			fmt.Printf("[apply_patch] Validation Passed: gofmt, go vet, and go build succeeded for %s\n", targetFile)
			return nil
		}

		// Validation Failed
		errStr := valErr.Error()
		fmt.Printf("[apply_patch] Validation Error: %s\n", errStr)

		if !enableSelfHeal || attempt >= maxRetries {
			if enableRollback {
				fmt.Printf("[apply_patch] Auto-rolling back changes to %s...\n", targetFile)
				if err := rollbackFile(targetFile, backupBytes, existed); err != nil {
					fmt.Printf("[apply_patch] Rollback error: %v\n", err)
				} else {
					fmt.Printf("[apply_patch] Rollback complete.\n")
				}
			}
			return fmt.Errorf("validation failed: %s", errStr)
		}

		// Self-Healing Loop
		fmt.Printf("[apply_patch] Piping failure log to moe_inference for self-healing...\n")
		_ = rollbackFile(targetFile, backupBytes, existed)

		fixPrompt := fmt.Sprintf("fix compiler error: %s", errStr)
		newOutput, err := runMoeInference(fixPrompt, moeBin)
		if err != nil {
			fmt.Printf("[apply_patch] moe_inference self-healing query failed: %v\n", err)
			return fmt.Errorf("self-healing loop error: %w (original build error: %s)", err, errStr)
		}

		currentOutput = *newOutput
	}

	return nil
}

func applyCodeChanges(targetFile string, props map[string]interface{}, handlerName, content, urlPath string) error {
	// Ensure file exists
	if _, err := os.Stat(targetFile); os.IsNotExist(err) {
		if content != "" {
			if err := os.WriteFile(targetFile, []byte(content), 0644); err != nil {
				return err
			}
		} else {
			pkgName := filepath.Base(filepath.Dir(targetFile))
			if pkgName == "." || pkgName == "/" {
				pkgName = "main"
			}
			initialCode := fmt.Sprintf("package %s\n\n", pkgName)
			if err := os.WriteFile(targetFile, []byte(initialCode), 0644); err != nil {
				return err
			}
		}
	}

	existingBytes, err := os.ReadFile(targetFile)
	if err != nil {
		return err
	}
	existingContent := string(existingBytes)

	// Top-level function appending
	if content != "" {
		if handlerName != "" && !strings.Contains(existingContent, "func "+handlerName) {
			lines := strings.Split(content, "\n")
			var filteredLines []string
			var importsToAdd []string
			inImportBlock := false
			for _, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "package ") {
					continue
				}
				if strings.HasPrefix(trimmed, "import (") {
					inImportBlock = true
					continue
				}
				if inImportBlock {
					if trimmed == ")" {
						inImportBlock = false
					} else if trimmed != "" {
						impPath := strings.Trim(trimmed, "\"\t `")
						if impPath != "" {
							importsToAdd = append(importsToAdd, impPath)
						}
					}
					continue
				}
				if strings.HasPrefix(trimmed, "import ") {
					impPath := strings.Trim(strings.TrimPrefix(trimmed, "import "), "\"\t `")
					if impPath != "" {
						importsToAdd = append(importsToAdd, impPath)
					}
					continue
				}
				filteredLines = append(filteredLines, line)
			}
			cleanContent := strings.Join(filteredLines, "\n")
			existingContent = strings.TrimRight(existingContent, "\n\t ") + "\n\n" + strings.TrimSpace(cleanContent) + "\n"
			if err := os.WriteFile(targetFile, []byte(existingContent), 0644); err != nil {
				return err
			}

			if strings.HasSuffix(targetFile, ".go") && len(importsToAdd) > 0 {
				fset := token.NewFileSet()
				node, err := parser.ParseFile(fset, targetFile, nil, parser.ParseComments)
				if err == nil {
					for _, imp := range importsToAdd {
						astutil.AddImport(fset, node, imp)
					}
					if f, err := os.Create(targetFile); err == nil {
						_ = format.Node(f, fset, node)
						f.Close()
					}
				}
			}
		}
	}

	if !strings.HasSuffix(targetFile, ".go") {
		return nil
	}

	// Targeted Method / Function Injection
	targetFunc := getStringProp(props, "target_function", "target_func", "func")
	if targetFunc == "" {
		targetFunc = "main"
	}

	var stmtsToInject []ast.Stmt

	// Route registration
	if handlerName != "" && urlPath != "" {
		routeCallSnippet := fmt.Sprintf(`http.HandleFunc("%s", %s)`, urlPath, handlerName)
		parsedStmts, err := parseStatements(routeCallSnippet)
		if err == nil {
			stmtsToInject = append(stmtsToInject, parsedStmts...)
		}
		autoRegisterRoutes(targetFile, handlerName, urlPath)
	}

	// Middleware injection
	middleware := getStringProp(props, "middleware", "middleware_name")
	if middleware != "" {
		routerName := getStringProp(props, "router", "router_name")
		if routerName == "" {
			routerName = "r"
		}
		mwSnippet := fmt.Sprintf(`%s.Use(%s)`, routerName, middleware)
		parsedStmts, err := parseStatements(mwSnippet)
		if err == nil {
			stmtsToInject = append(stmtsToInject, parsedStmts...)
		}
	}

	// Custom code injection
	injectCode := getStringProp(props, "inject_code", "code_snippet", "injection")
	if injectCode != "" {
		parsedStmts, err := parseStatements(injectCode)
		if err == nil {
			stmtsToInject = append(stmtsToInject, parsedStmts...)
		}
	}

	if len(stmtsToInject) > 0 {
		if err := injectIntoFunction(targetFile, targetFunc, stmtsToInject); err != nil {
			log.Printf("[apply_patch] Notice (Function Injection): %v", err)
		}
	}

	// Type-Safe Struct Modification
	structName := getStringProp(props, "struct_name", "struct")
	var fieldsToAdd []StructFieldDef

	fieldName := getStringProp(props, "field_name", "field")
	fieldType := getStringProp(props, "field_type", "type")
	fieldTag := getStringProp(props, "field_tag", "tag")

	if fieldName != "" && fieldType != "" {
		fieldsToAdd = append(fieldsToAdd, StructFieldDef{Name: fieldName, Type: fieldType, Tag: fieldTag})
	}

	if rawFields, ok := props["struct_fields"].([]interface{}); ok {
		for _, rf := range rawFields {
			if fMap, ok := rf.(map[string]interface{}); ok {
				fName, _ := fMap["name"].(string)
				fType, _ := fMap["type"].(string)
				fTag, _ := fMap["tag"].(string)
				if fName != "" && fType != "" {
					fieldsToAdd = append(fieldsToAdd, StructFieldDef{Name: fName, Type: fType, Tag: fTag})
				}
			}
		}
	}

	if structName == "" || len(fieldsToAdd) == 0 {
		if promptStr := getStringProp(props, "prompt", "query", "description"); promptStr != "" {
			sName, fDef := parseStructModificationPrompt(promptStr)
			if sName != "" && fDef.Name != "" {
				if structName == "" {
					structName = sName
				}
				fieldsToAdd = append(fieldsToAdd, fDef)
			}
		}
	}

	if structName != "" && len(fieldsToAdd) > 0 {
		if err := modifyStructFields(targetFile, structName, fieldsToAdd); err != nil {
			log.Printf("[apply_patch] Notice (Struct Modification): %v", err)
		}
	}

	return nil
}

func parseStructModificationPrompt(promptStr string) (string, StructFieldDef) {
	re := regexp.MustCompile(`(?i)add\s+(\w+)\s+([\w\*\[\]\.]+)\s+field\s+to\s+(?:struct\s+)?(\w+)`)
	matches := re.FindStringSubmatch(promptStr)
	if len(matches) == 4 {
		fieldName := matches[1]
		fieldType := matches[2]
		structName := matches[3]
		return structName, StructFieldDef{Name: fieldName, Type: fieldType}
	}
	return "", StructFieldDef{}
}

func injectIntoFunction(targetFile, targetFunc string, stmtsToInject []ast.Stmt) error {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, targetFile, nil, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("failed to parse %s: %w", targetFile, err)
	}

	var targetDecl *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == targetFunc {
			targetDecl = fn
			return false
		}
		return true
	})

	if targetDecl == nil {
		if targetFunc == "main" {
			targetDecl = &ast.FuncDecl{
				Name: ast.NewIdent("main"),
				Type: &ast.FuncType{Params: &ast.FieldList{}},
				Body: &ast.BlockStmt{},
			}
			node.Decls = append(node.Decls, targetDecl)
		} else {
			return fmt.Errorf("target function %q not found in %s", targetFunc, targetFile)
		}
	}

	for _, stmt := range stmtsToInject {
		var buf strings.Builder
		if err := format.Node(&buf, token.NewFileSet(), stmt); err == nil {
			stmtStr := buf.String()
			if strings.Contains(stmtStr, "http.") {
				ensureImport(fset, node, "net/http")
			}
			if strings.Contains(stmtStr, "sql.") {
				ensureImport(fset, node, "database/sql")
			}
		}
	}

	for _, newStmt := range stmtsToInject {
		var newBuf strings.Builder
		_ = format.Node(&newBuf, token.NewFileSet(), newStmt)
		newStmtStr := strings.TrimSpace(newBuf.String())

		alreadyPresent := false
		for _, existingStmt := range targetDecl.Body.List {
			var existingBuf strings.Builder
			_ = format.Node(&existingBuf, fset, existingStmt)
			if strings.TrimSpace(existingBuf.String()) == newStmtStr {
				alreadyPresent = true
				break
			}
		}
		if !alreadyPresent {
			targetDecl.Body.List = append(targetDecl.Body.List, newStmt)
		}
	}

	f, err := os.Create(targetFile)
	if err != nil {
		return fmt.Errorf("failed to open %s for writing: %w", targetFile, err)
	}
	defer f.Close()

	return format.Node(f, fset, node)
}

func modifyStructFields(targetFile, structName string, fieldsToAdd []StructFieldDef) error {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, targetFile, nil, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("failed to parse %s: %w", targetFile, err)
	}

	var structType *ast.StructType
	ast.Inspect(node, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == structName {
			if st, ok := ts.Type.(*ast.StructType); ok {
				structType = st
				return false
			}
		}
		return true
	})

	if structType == nil {
		return fmt.Errorf("struct %q not found in %s", structName, targetFile)
	}

	for _, fieldDef := range fieldsToAdd {
		if fieldDef.Name == "" || fieldDef.Type == "" {
			continue
		}
		exists := false
		for _, f := range structType.Fields.List {
			for _, name := range f.Names {
				if name.Name == fieldDef.Name {
					exists = true
					break
				}
			}
			if exists {
				break
			}
		}
		if exists {
			continue
		}

		typeExpr, err := parser.ParseExpr(fieldDef.Type)
		if err != nil {
			typeExpr = ast.NewIdent(fieldDef.Type)
		}

		newField := &ast.Field{
			Names: []*ast.Ident{ast.NewIdent(fieldDef.Name)},
			Type:  typeExpr,
		}

		if fieldDef.Tag != "" {
			tagVal := fieldDef.Tag
			if !strings.HasPrefix(tagVal, "`") {
				tagVal = "`" + tagVal + "`"
			}
			newField.Tag = &ast.BasicLit{
				Kind:  token.STRING,
				Value: tagVal,
			}
		}

		structType.Fields.List = append(structType.Fields.List, newField)
	}

	f, err := os.Create(targetFile)
	if err != nil {
		return fmt.Errorf("failed to open %s for writing: %w", targetFile, err)
	}
	defer f.Close()

	return format.Node(f, fset, node)
}

func ensureImport(fset *token.FileSet, node *ast.File, importPath string) {
	astutil.AddImport(fset, node, importPath)
}

func parseStatements(snippet string) ([]ast.Stmt, error) {
	snippet = strings.TrimSpace(snippet)
	if snippet == "" {
		return nil, nil
	}
	src := fmt.Sprintf("package dummy\nfunc _() {\n%s\n}", snippet)
	fset := token.NewFileSet()
	fileNode, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to parse snippet %q: %w", snippet, err)
	}
	for _, decl := range fileNode.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "_" {
			return fn.Body.List, nil
		}
	}
	return nil, fmt.Errorf("failed to extract statements")
}

func validateGoCode(targetFile string) error {
	dir := filepath.Dir(targetFile)
	base := filepath.Base(targetFile)

	// 1. gofmt
	gofmtCmd := exec.Command("gofmt", "-w", targetFile)
	if out, err := gofmtCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("gofmt failed on %s: %s (%v)", base, string(out), err)
	}

	// 2. go vet
	vetCmd := exec.Command("go", "vet", ".")
	vetCmd.Dir = dir
	if out, err := vetCmd.CombinedOutput(); err != nil {
		cleanOut := strings.TrimSpace(string(out))
		return fmt.Errorf("go vet failed in %s: %s", dir, cleanOut)
	}

	// 3. go build
	tmpOutput := filepath.Join(os.TempDir(), fmt.Sprintf("apply_patch_build_check_%d", os.Getpid()))
	defer os.Remove(tmpOutput)

	buildCmd := exec.Command("go", "build", "-o", tmpOutput, ".")
	buildCmd.Dir = dir
	if out, err := buildCmd.CombinedOutput(); err != nil {
		cleanOut := strings.TrimSpace(string(out))
		if cleanOut == "" {
			cleanOut = err.Error()
		}
		return fmt.Errorf("go build failed: %s", cleanOut)
	}

	return nil
}

func rollbackFile(targetFile string, backupBytes []byte, existed bool) error {
	if existed {
		if err := os.WriteFile(targetFile, backupBytes, 0644); err != nil {
			return fmt.Errorf("failed to restore backup: %w", err)
		}
	} else {
		if err := os.Remove(targetFile); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("failed to remove created file: %w", err)
		}
	}

	if _, err := os.Stat(".git"); err == nil {
		_ = exec.Command("git", "checkout", "--", targetFile).Run()
	}

	return nil
}

func runMoeInference(prompt string, moeBin string) (*SemanticOutput, error) {
	var cmd *exec.Cmd
	if moeBin != "" {
		cmd = exec.Command(moeBin, "-prompt", prompt)
	} else if fi, err := os.Stat("./cmd/tools/moe_inference/moe_inference"); err == nil && !fi.IsDir() && (fi.Mode()&0111 != 0) {
		cmd = exec.Command("./cmd/tools/moe_inference/moe_inference", "-prompt", prompt)
	} else if fi, err := os.Stat("/tmp/moe_inference"); err == nil && !fi.IsDir() && (fi.Mode()&0111 != 0) {
		cmd = exec.Command("/tmp/moe_inference", "-prompt", prompt)
	} else {
		cmd = exec.Command("go", "run", "./cmd/tools/moe_inference", "-prompt", prompt)
	}

	outBytes, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("moe_inference failed: %v (output: %s)", err, string(outBytes))
	}

	outStr := string(outBytes)
	jsonStart := strings.Index(outStr, "=== Generated Semantic Output ===")
	if jsonStart != -1 {
		outStr = outStr[jsonStart+len("=== Generated Semantic Output ==="):]
	}
	jsonEnd := strings.Index(outStr, "=================================")
	if jsonEnd != -1 {
		outStr = outStr[:jsonEnd]
	}

	outStr = strings.TrimSpace(outStr)
	if outStr == "" {
		return nil, fmt.Errorf("no valid JSON output found from moe_inference")
	}

	var output SemanticOutput
	if err := json.Unmarshal([]byte(outStr), &output); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}

	return &output, nil
}

func getStringProp(props map[string]interface{}, keys ...string) string {
	if props == nil {
		return ""
	}
	for _, key := range keys {
		if val, ok := props[key].(string); ok && val != "" {
			return val
		}
	}
	return ""
}

func autoRegisterRoutes(targetFile string, handlerName, urlPath string) {
	if handlerName == "" || urlPath == "" {
		return
	}
	dir := filepath.Dir(targetFile)
	candidates := []string{
		filepath.Join(dir, "main.go"),
		filepath.Join(dir, "routes.go"),
		"./main.go",
	}

	routeSnippet := fmt.Sprintf(`http.HandleFunc("%s", %s)`, urlPath, handlerName)
	stmts, err := parseStatements(routeSnippet)
	if err != nil || len(stmts) == 0 {
		return
	}

	for _, cand := range candidates {
		if absCand, err := filepath.Abs(cand); err == nil {
			if absTarget, err := filepath.Abs(targetFile); err == nil && absCand == absTarget {
				continue
			}
		}
		if _, err := os.Stat(cand); err == nil {
			if err := injectIntoFunction(cand, "main", stmts); err == nil {
				fmt.Printf("[apply_patch] Multi-File AST: Automatically registered route http.HandleFunc(%q, %s) in %s\n", urlPath, handlerName, cand)
				_ = validateGoCode(cand)
				break
			}
		}
	}
}
