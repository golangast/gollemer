package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/scanner"
	"go/token"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// 1. Parse a target source file, print package name, list top-level declarations,
// and catch parse errors explicitly with line and column numbers.
func parseAndListDecls(filePath string) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)

	// Explicitly handle parse errors (which are usually scanner.ErrorList)
	if err != nil {
		if errList, ok := err.(scanner.ErrorList); ok {
			for _, e := range errList {
				fmt.Printf("Parse error at %s line %d, column %d: %s\n",
					e.Pos.Filename, e.Pos.Line, e.Pos.Column, e.Msg)
			}
		} else {
			fmt.Printf("Parse error: %v\n", err)
		}
		return
	}

	fmt.Printf("Package: %s\n", f.Name.Name)
	fmt.Println("Top-level declarations:")
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			fmt.Printf("- FuncDecl: %s\n", d.Name.Name)
		case *ast.GenDecl:
			fmt.Printf("- GenDecl: token=%v, specs=%d\n", d.Tok, len(d.Specs))
		default:
			fmt.Printf("- %T\n", d)
		}
	}
}

// 2. Struct and function to programmatically execute `go vet`,
// capture stderr, and parse diagnostics.
type VetDiagnostic struct {
	FilePath     string
	Line         int
	Column       int
	ErrorMessage string
}

func runGoVet(filePath string) ([]VetDiagnostic, error) {
	cmd := exec.Command("go", "vet", filePath)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	// go vet exits with code 1 if it finds issues, which returns an error here.
	// We capture stderr regardless to parse the output.
	err := cmd.Run()
	output := stderr.String()

	var diagnostics []VetDiagnostic

	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Expected format: file:line:col: message or file:line: message
		line = strings.TrimPrefix(line, "vet: ")
		parts := strings.SplitN(line, ":", 4)
		if len(parts) >= 4 {
			file := parts[0]
			lineNum, _ := strconv.Atoi(parts[1])
			colNum, _ := strconv.Atoi(parts[2])
			msg := strings.TrimSpace(parts[3])

			diagnostics = append(diagnostics, VetDiagnostic{
				FilePath:     file,
				Line:         lineNum,
				Column:       colNum,
				ErrorMessage: msg,
			})
		} else if len(parts) == 3 {
			file := parts[0]
			lineNum, _ := strconv.Atoi(parts[1])
			msg := strings.TrimSpace(parts[2])

			diagnostics = append(diagnostics, VetDiagnostic{
				FilePath:     file,
				Line:         lineNum,
				ErrorMessage: msg,
			})
		}
	}

	// If it fails for reasons other than finding issues (where it output things to stderr), report it.
	if err != nil && len(diagnostics) == 0 {
		return nil, fmt.Errorf("go vet command failed: %w (stderr: %s)", err, output)
	}

	return diagnostics, nil
}

// 3. Check for specific import, inject if not present, and append a struct declaration.
func modifyASTAndPrint(fset *token.FileSet, f *ast.File, importPath string) error {
	// Check if the specific package is already imported
	hasImport := false
	for _, imp := range f.Imports {
		if imp.Path.Value == `"`+importPath+`"` {
			hasImport = true
			break
		}
	}

	// Inject new import if missing
	if !hasImport {
		newImport := &ast.GenDecl{
			Tok: token.IMPORT,
			Specs: []ast.Spec{
				&ast.ImportSpec{
					Path: &ast.BasicLit{
						Kind:  token.STRING,
						Value: `"` + importPath + `"`,
					},
				},
			},
		}

		// Safely prepend the new import to the file's declarations
		f.Decls = append([]ast.Decl{newImport}, f.Decls...)
	}

	// Append a new struct declaration node: `type NewStruct struct {}`
	newStructDecl := &ast.GenDecl{
		Tok: token.TYPE,
		Specs: []ast.Spec{
			&ast.TypeSpec{
				Name: ast.NewIdent("NewStruct"),
				Type: &ast.StructType{
					Fields: &ast.FieldList{},
				},
			},
		},
	}
	f.Decls = append(f.Decls, newStructDecl)

	// Write the modified AST back out using go/printer
	// Using os.Stdout here for demonstration, but this can write to a file or buffer
	err := printer.Fprint(os.Stdout, fset, f)
	if err != nil {
		return fmt.Errorf("printer.Fprint failed: %w", err)
	}

	return nil
}

// 4. AutoFixPipeline integrates our tools into a closed-loop system:
// It runs go vet, parses the file, matches error strings to structural fixes,
// and automatically mutates and writes the AST back to disk.
func AutoFixPipeline(filePath string) error {
	for i := 0; i < 3; i++ {
		diagnostics, err := runGoVet(filePath)
		if err != nil && len(diagnostics) == 0 {
			return fmt.Errorf("runGoVet failed: %w", err)
		}

		if len(diagnostics) == 0 {
			return nil // No diagnostics to fix
		}

		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
		if err != nil {
			return fmt.Errorf("failed to parse file for fixing: %w", err)
		}

		fixed := false
		for _, diag := range diagnostics {
			// Example structural fix: If "math" package is missing based on vet/compiler output.
			if strings.Contains(diag.ErrorMessage, "undeclared name: math") || strings.Contains(diag.ErrorMessage, "undefined: math") {
				hasImport := false
				for _, imp := range f.Imports {
					if imp.Path.Value == `"math"` {
						hasImport = true
						break
					}
				}

				if !hasImport {
					newImport := &ast.GenDecl{
						Tok: token.IMPORT,
						Specs: []ast.Spec{
							&ast.ImportSpec{
								Path: &ast.BasicLit{
									Kind:  token.STRING,
									Value: `"math"`,
								},
							},
						},
					}
					// Safely prepend the missing import
					f.Decls = append([]ast.Decl{newImport}, f.Decls...)
					fixed = true
				}
			}

			// Also check for "http" missing import
			if strings.Contains(diag.ErrorMessage, "undeclared name: http") || strings.Contains(diag.ErrorMessage, "undefined: http") {
				hasImport := false
				for _, imp := range f.Imports {
					if imp.Path.Value == `"net/http"` {
						hasImport = true
						break
					}
				}

				if !hasImport {
					newImport := &ast.GenDecl{
						Tok: token.IMPORT,
						Specs: []ast.Spec{
							&ast.ImportSpec{
								Path: &ast.BasicLit{
									Kind:  token.STRING,
									Value: `"net/http"`,
								},
							},
						},
					}
					f.Decls = append([]ast.Decl{newImport}, f.Decls...)
					fixed = true
				}
			}

			// Also check for "fmt" missing import
			if strings.Contains(diag.ErrorMessage, "undeclared name: fmt") || strings.Contains(diag.ErrorMessage, "undefined: fmt") {
				hasImport := false
				for _, imp := range f.Imports {
					if imp.Path.Value == `"fmt"` {
						hasImport = true
						break
					}
				}

				if !hasImport {
					newImport := &ast.GenDecl{
						Tok: token.IMPORT,
						Specs: []ast.Spec{
							&ast.ImportSpec{
								Path: &ast.BasicLit{
									Kind:  token.STRING,
									Value: `"fmt"`,
								},
							},
						},
					}
					f.Decls = append([]ast.Decl{newImport}, f.Decls...)
					fixed = true
				}
			}

			// Also check for "sql" missing import
			if strings.Contains(diag.ErrorMessage, "undeclared name: sql") || strings.Contains(diag.ErrorMessage, "undefined: sql") {
				hasImport := false
				for _, imp := range f.Imports {
					if imp.Path.Value == `"database/sql"` {
						hasImport = true
						break
					}
				}

				if !hasImport {
					newImport := &ast.GenDecl{
						Tok: token.IMPORT,
						Specs: []ast.Spec{
							&ast.ImportSpec{
								Path: &ast.BasicLit{
									Kind:  token.STRING,
									Value: `"database/sql"`,
								},
							},
						},
					}
					f.Decls = append([]ast.Decl{newImport}, f.Decls...)
					fixed = true
				}
			}

			// Additional diagnostics string matching and AST node injection could be added here
		}

		if fixed {
			var buf bytes.Buffer
			if err := printer.Fprint(&buf, fset, f); err != nil {
				return fmt.Errorf("printer.Fprint failed during formatting: %w", err)
			}
			if err := os.WriteFile(filePath, buf.Bytes(), 0644); err != nil {
				return fmt.Errorf("failed to write fixed file: %w", err)
			}
		} else {
			break
		}
	}
	return nil
}

// --- Autonomous Engine Expansions ---

// IndexEntry represents a parsed semantic chunk
type IndexEntry struct {
	Type     string `json:"type"` // "func" or "struct"
	Name     string `json:"name"`
	Doc      string `json:"doc"`
	FilePath string `json:"file_path"`
}

// BuildSemanticIndex walks a directory, chunks Go files into function and struct blocks,
// and builds a simple local JSON index for intent querying.
func BuildSemanticIndex(dirPath string, indexPath string) error {
	var index []IndexEntry
	fset := token.NewFileSet()

	err := filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".go") {
			f, parseErr := parser.ParseFile(fset, path, nil, parser.ParseComments)
			if parseErr != nil {
				return nil // Skip files with fatal parse errors
			}

			for _, decl := range f.Decls {
				switch d := decl.(type) {
				case *ast.FuncDecl:
					doc := ""
					if d.Doc != nil {
						doc = d.Doc.Text()
					}
					index = append(index, IndexEntry{
						Type:     "func",
						Name:     d.Name.Name,
						Doc:      strings.TrimSpace(doc),
						FilePath: path,
					})
				case *ast.GenDecl:
					if d.Tok == token.TYPE {
						for _, spec := range d.Specs {
							if ts, ok := spec.(*ast.TypeSpec); ok {
								if _, isStruct := ts.Type.(*ast.StructType); isStruct {
									doc := ""
									if d.Doc != nil {
										doc = d.Doc.Text()
									} else if ts.Doc != nil {
										doc = ts.Doc.Text()
									}
									index = append(index, IndexEntry{
										Type:     "struct",
										Name:     ts.Name.Name,
										Doc:      strings.TrimSpace(doc),
										FilePath: path,
									})
								}
							}
						}
					}
				}
			}
		}
		return nil
	})

	if err != nil {
		return fmt.Errorf("walk directory failed: %w", err)
	}

	data, err := json.MarshalIndent(index, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal index: %w", err)
	}

	return os.WriteFile(indexPath, data, 0644)
}

// MutateByIntent dynamically appends AST nodes based on a natural language instruction.
func MutateByIntent(f *ast.File, instruction string) (bool, error) {
	mutated := false

	// Regex for "Add a <Name> struct"
	structRegex := regexp.MustCompile(`(?i)add a (\w+) struct`)
	if match := structRegex.FindStringSubmatch(instruction); match != nil {
		structName := match[1]
		newStruct := &ast.GenDecl{
			Tok: token.TYPE,
			Specs: []ast.Spec{
				&ast.TypeSpec{
					Name: ast.NewIdent(structName),
					Type: &ast.StructType{
						Fields: &ast.FieldList{},
					},
				},
			},
		}
		f.Decls = append(f.Decls, newStruct)
		mutated = true
	}

	// Regex for "Add a <Name> function"
	funcRegex := regexp.MustCompile(`(?i)add a (\w+) function`)
	if match := funcRegex.FindStringSubmatch(instruction); match != nil {
		funcName := match[1]
		newFunc := &ast.FuncDecl{
			Name: ast.NewIdent(funcName),
			Type: &ast.FuncType{
				Params: &ast.FieldList{}, // No args
			},
			Body: &ast.BlockStmt{
				List: []ast.Stmt{},
			},
		}
		f.Decls = append(f.Decls, newFunc)
		mutated = true
	}

	return mutated, nil
}

// ApplyAndVerify performs a mutation, writes to disk, runs a compilation check,
// and reverts if compilation fails, attempting self-correction where possible.
func ApplyAndVerify(filePath string, instruction string) error {
	// 1. Read original content as backup
	originalContent, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// 2. Parse AST
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("parse error: %w", err)
	}

	// 3. Apply intent-driven mutation
	mutated, err := MutateByIntent(f, instruction)
	if err != nil {
		return fmt.Errorf("mutation error: %w", err)
	}
	if !mutated {
		return fmt.Errorf("instruction '%s' did not trigger any mutations", instruction)
	}

	// 4. Write mutated AST back to file
	var buf bytes.Buffer
	if err := printer.Fprint(&buf, fset, f); err != nil {
		return fmt.Errorf("failed to print ast: %w", err)
	}
	if err := os.WriteFile(filePath, buf.Bytes(), 0644); err != nil {
		return fmt.Errorf("failed to write mutated file: %w", err)
	}

	// 5. Verification step via compilation
	// We run `go build` to verify syntax of the target file.
	cmd := exec.Command("go", "build", filePath)
	if output, err := cmd.CombinedOutput(); err != nil {
		// Compilation failed. Attempt self-correction using our existing AutoFixPipeline
		if fixErr := AutoFixPipeline(filePath); fixErr == nil {
			// Check again
			cmd2 := exec.Command("go", "build", filePath)
			if _, err2 := cmd2.CombinedOutput(); err2 == nil {
				return nil // Fixed and compiled successfully
			}
		}

		// If self-correction fails, revert permanently
		os.WriteFile(filePath, originalContent, 0644)
		return fmt.Errorf("verification failed and self-correction could not fix it. Error: %s", string(output))
	}

	return nil
}

// ApplyIntentMutation reads a natural language instruction and mutates the AST file accordingly.
func ApplyIntentMutation(filePath string, instruction string) error {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("failed to parse file for mutation: %w", err)
	}

	// Simple heuristic/intent matcher based on instructions
	if strings.Contains(instruction, "MetricsLogger struct") {
		// Check if MetricsLogger already exists to prevent duplicates
		exists := false
		for _, decl := range f.Decls {
			if genDecl, ok := decl.(*ast.GenDecl); ok && genDecl.Tok == token.TYPE {
				for _, spec := range genDecl.Specs {
					if typeSpec, ok := spec.(*ast.TypeSpec); ok && typeSpec.Name.Name == "MetricsLogger" {
						exists = true
					}
				}
			}
		}

		if !exists {
			// Construct: type MetricsLogger struct {}
			newStructDecl := &ast.GenDecl{
				Tok: token.TYPE,
				Specs: []ast.Spec{
					&ast.TypeSpec{
						Name: ast.NewIdent("MetricsLogger"),
						Type: &ast.StructType{
							Fields: &ast.FieldList{},
						},
					},
				},
			}
			// Append the new struct declaration to the file's top-level declarations
			f.Decls = append(f.Decls, newStructDecl)
		}
	}

	// Write the modified AST back to the file safely
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file for writing: %w", err)
	}
	defer file.Close()

	err = printer.Fprint(file, fset, f)
	if err != nil {
		return fmt.Errorf("failed to write modified AST to file: %w", err)
	}

	return nil
}

// ParseHandlerIntent extracts a handler name and target file path from a natural language string.
func ParseHandlerIntent(instruction string) (handlerName, filePath string, err error) {
	re := regexp.MustCompile(`(?i)add handler named\s+(\w+)\s+to\s+([a-zA-Z0-9_./-]+)`)
	matches := re.FindStringSubmatch(instruction)
	if len(matches) < 3 {
		return "", "", fmt.Errorf("could not parse intent from instruction: %s", instruction)
	}
	handlerName = matches[1]
	// Capitalize handler name to make it exported if desired, or just append "Handler"
	handlerName = strings.Title(handlerName) + "Handler"
	filePath = matches[2]
	return handlerName, filePath, nil
}

// SpliceHTTPHandler constructs and injects an HTTP handler into the specified AST file.
func SpliceHTTPHandler(f *ast.File, handlerName string) error {
	for _, decl := range f.Decls {
		if funcDecl, ok := decl.(*ast.FuncDecl); ok {
			if funcDecl.Name.Name == handlerName {
				return fmt.Errorf("handler %s already exists", handlerName)
			}
		}
	}

	handlerFunc := &ast.FuncDecl{
		Name: ast.NewIdent(handlerName),
		Type: &ast.FuncType{
			Params: &ast.FieldList{
				List: []*ast.Field{
					{
						Names: []*ast.Ident{ast.NewIdent("w")},
						Type: &ast.SelectorExpr{
							X:   ast.NewIdent("http"),
							Sel: ast.NewIdent("ResponseWriter"),
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
							X:   ast.NewIdent("w"),
							Sel: ast.NewIdent("Write"),
						},
						Args: []ast.Expr{
							&ast.CallExpr{
								Fun: &ast.ArrayType{
									Elt: ast.NewIdent("byte"),
								},
								Args: []ast.Expr{
									&ast.BasicLit{
										Kind:  token.STRING,
										Value: `"OK"`,
									},
								},
							},
						},
					},
				},
			},
		},
	}

	f.Decls = append(f.Decls, handlerFunc)
	return nil
}

// InjectHandlerAndVerify orchestrates parsing the intent, injecting the HTTP handler,
// auto-fixing imports, and verifying compilation.
func InjectHandlerAndVerify(instruction string) error {
	handlerName, filePath, err := ParseHandlerIntent(instruction)
	if err != nil {
		return err
	}

	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	var f *ast.File
	fset := token.NewFileSet()
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		f = &ast.File{
			Name: ast.NewIdent("main"),
		}
	} else {
		f, err = parser.ParseFile(fset, filePath, nil, parser.ParseComments)
		if err != nil {
			return err
		}
	}

	originalContent, _ := os.ReadFile(filePath)

	if err := SpliceHTTPHandler(f, handlerName); err != nil {
		return err
	}

	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	if err := printer.Fprint(file, fset, f); err != nil {
		file.Close()
		return err
	}
	file.Close()

	AutoFixPipeline(filePath)

	cmd := exec.Command("go", "build", filePath)
	if output, err := cmd.CombinedOutput(); err != nil {
		if len(originalContent) > 0 {
			os.WriteFile(filePath, originalContent, 0644)
		} else {
			os.Remove(filePath)
		}
		return fmt.Errorf("build failed: %s", string(output))
	}

	return nil
}

// --- Phase 2: Repository Learning & Safe-Injection Engine ---

type FuncMeta struct {
	Name     string
	Receiver string
	File     string
}

type StructMeta struct {
	Name   string
	Fields []string
	File   string
}

type PackageMeta struct {
	Name      string
	Files     []string
	Structs   map[string]StructMeta
	Functions map[string]FuncMeta
}

// LearnRepository parses the entire directory and returns an in-memory metadata mapping.
func LearnRepository(rootPath string) (map[string]*PackageMeta, error) {
	repo := make(map[string]*PackageMeta)
	fset := token.NewFileSet()

	err := filepath.WalkDir(rootPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".go") && !strings.HasSuffix(d.Name(), "_test.go") {
			f, parseErr := parser.ParseFile(fset, path, nil, 0)
			if parseErr != nil {
				return nil // Skip parse errors for resilient learning
			}

			pkgName := f.Name.Name
			if _, exists := repo[pkgName]; !exists {
				repo[pkgName] = &PackageMeta{
					Name:      pkgName,
					Structs:   make(map[string]StructMeta),
					Functions: make(map[string]FuncMeta),
				}
			}

			repo[pkgName].Files = append(repo[pkgName].Files, path)

			for _, decl := range f.Decls {
				switch d := decl.(type) {
				case *ast.FuncDecl:
					receiver := ""
					if d.Recv != nil && len(d.Recv.List) > 0 {
						// Extract receiver type name
						if starExp, ok := d.Recv.List[0].Type.(*ast.StarExpr); ok {
							if ident, ok := starExp.X.(*ast.Ident); ok {
								receiver = ident.Name
							}
						} else if ident, ok := d.Recv.List[0].Type.(*ast.Ident); ok {
							receiver = ident.Name
						}
					}
					repo[pkgName].Functions[d.Name.Name] = FuncMeta{
						Name:     d.Name.Name,
						Receiver: receiver,
						File:     path,
					}

				case *ast.GenDecl:
					if d.Tok == token.TYPE {
						for _, spec := range d.Specs {
							if ts, ok := spec.(*ast.TypeSpec); ok {
								if st, ok := ts.Type.(*ast.StructType); ok {
									var fields []string
									if st.Fields != nil {
										for _, field := range st.Fields.List {
											if len(field.Names) > 0 {
												fields = append(fields, field.Names[0].Name)
											}
										}
									}
									repo[pkgName].Structs[ts.Name.Name] = StructMeta{
										Name:   ts.Name.Name,
										Fields: fields,
										File:   path,
									}
								}
							}
						}
					}
				}
			}
		}
		return nil
	})

	return repo, err
}

// InjectSafeNode dynamically splices an ast.Decl or *ast.Field directly into the AST
// without corrupting existing code.
func InjectSafeNode(f *ast.File, targetStruct string, node interface{}) error {
	switch n := node.(type) {
	case ast.Decl:
		// Splicing a new top-level declaration (like a func or new struct)
		if targetStruct == "" {
			f.Decls = append(f.Decls, n)
			return nil
		}

		insertIdx := -1
		for i, decl := range f.Decls {
			if genDecl, ok := decl.(*ast.GenDecl); ok && genDecl.Tok == token.TYPE {
				for _, spec := range genDecl.Specs {
					if typeSpec, ok := spec.(*ast.TypeSpec); ok && typeSpec.Name.Name == targetStruct {
						insertIdx = i
						break
					}
				}
			}
		}

		if insertIdx == -1 {
			return fmt.Errorf("target struct %s not found in file", targetStruct)
		}

		// Splice into Decls right after the struct
		f.Decls = append(f.Decls[:insertIdx+1], append([]ast.Decl{n}, f.Decls[insertIdx+1:]...)...)
		return nil

	case *ast.Field:
		// Injecting a field into an existing struct
		if targetStruct == "" {
			return fmt.Errorf("targetStruct must be specified to inject a field")
		}

		for _, decl := range f.Decls {
			if genDecl, ok := decl.(*ast.GenDecl); ok && genDecl.Tok == token.TYPE {
				for _, spec := range genDecl.Specs {
					if typeSpec, ok := spec.(*ast.TypeSpec); ok && typeSpec.Name.Name == targetStruct {
						if st, ok := typeSpec.Type.(*ast.StructType); ok {
							if st.Fields == nil {
								st.Fields = &ast.FieldList{}
							}
							st.Fields.List = append(st.Fields.List, n)
							return nil
						}
					}
				}
			}
		}
		return fmt.Errorf("target struct %s not found in file", targetStruct)

	default:
		return fmt.Errorf("unsupported node type for injection: %T", node)
	}
}

// InjectAndValidate orchestrates AST injection, triggers AutoFixPipeline for missing imports,
// runs compilation validation, and safely rolls back on failure.
func InjectAndValidate(filePath string, targetStruct string, node interface{}) error {
	originalContent, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("parse error: %w", err)
	}

	err = InjectSafeNode(f, targetStruct, node)
	if err != nil {
		return fmt.Errorf("failed to inject node: %w", err)
	}

	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file for writing: %w", err)
	}

	if err := printer.Fprint(file, fset, f); err != nil {
		file.Close()
		return fmt.Errorf("failed to format and write AST: %w", err)
	}
	file.Close()

	// 1. Run AutoFixPipeline to resolve potential missing imports
	// (Silently swallows errors if no autofixes apply)
	AutoFixPipeline(filePath)

	// 2. Validate compilation
	cmd := exec.Command("go", "build", filePath)
	if output, err := cmd.CombinedOutput(); err != nil {
		// Compilation failed. Revert safely.
		os.WriteFile(filePath, originalContent, 0644)
		return fmt.Errorf("validation failed, reverted safely. Build error: %s", string(output))
	}

	return nil
}

// --- Phase 3: Corpus Learning & Context-Aware Synthesizer ---

type CorpusPattern struct {
	ID      string   `json:"id"`
	Type    string   `json:"type"` // "func" or "struct"
	Name    string   `json:"name"`
	RawCode string   `json:"raw_code"`
	Tags    []string `json:"tags"`
}

// BuildCodeCorpus extracts AST patterns from a directory and serializes them to JSON.
func BuildCodeCorpus(corpusDir string, outputJSON string) error {
	var patterns []CorpusPattern
	fset := token.NewFileSet()

	err := filepath.WalkDir(corpusDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".go") && !strings.HasSuffix(d.Name(), "_test.go") {
			f, parseErr := parser.ParseFile(fset, path, nil, parser.ParseComments)
			if parseErr != nil {
				return nil
			}

			for i, decl := range f.Decls {
				var buf bytes.Buffer
				if err := printer.Fprint(&buf, fset, decl); err != nil {
					continue
				}
				rawCode := buf.String()

				switch d := decl.(type) {
				case *ast.FuncDecl:
					doc := ""
					if d.Doc != nil {
						doc = d.Doc.Text()
					}
					tags := strings.Fields(strings.ToLower(d.Name.Name + " " + doc))
					patterns = append(patterns, CorpusPattern{
						ID:      fmt.Sprintf("%s_func_%d", filepath.Base(path), i),
						Type:    "func",
						Name:    d.Name.Name,
						RawCode: rawCode,
						Tags:    tags,
					})
				case *ast.GenDecl:
					if d.Tok == token.TYPE {
						for j, spec := range d.Specs {
							if ts, ok := spec.(*ast.TypeSpec); ok {
								if _, isStruct := ts.Type.(*ast.StructType); isStruct {
									doc := ""
									if d.Doc != nil {
										doc = d.Doc.Text()
									} else if ts.Doc != nil {
										doc = ts.Doc.Text()
									}
									tags := strings.Fields(strings.ToLower(ts.Name.Name + " " + doc))

									// Recapture just the struct spec if there are multiple specs in the block
									var sbuf bytes.Buffer
									// Re-wrap the spec in a pseudo-GenDecl to print it fully standalone
									standalone := &ast.GenDecl{
										Tok:   token.TYPE,
										Specs: []ast.Spec{ts},
									}
									if err := printer.Fprint(&sbuf, fset, standalone); err == nil {
										rawCode = sbuf.String()
									}

									patterns = append(patterns, CorpusPattern{
										ID:      fmt.Sprintf("%s_struct_%d_%d", filepath.Base(path), i, j),
										Type:    "struct",
										Name:    ts.Name.Name,
										RawCode: rawCode,
										Tags:    tags,
									})
								}
							}
						}
					}
				}
			}
		}
		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to walk corpus dir: %w", err)
	}

	data, err := json.MarshalIndent(patterns, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal corpus: %w", err)
	}

	return os.WriteFile(outputJSON, data, 0644)
}

// MatchAndLoadPattern heuristically matches instruction to a template and returns the ast.Decl
func MatchAndLoadPattern(instruction string, corpusJSON string) (ast.Decl, error) {
	data, err := os.ReadFile(corpusJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to read corpus JSON: %w", err)
	}

	var patterns []CorpusPattern
	if err := json.Unmarshal(data, &patterns); err != nil {
		return nil, fmt.Errorf("failed to parse corpus JSON: %w", err)
	}

	instruction = strings.ToLower(instruction)
	instructionWords := strings.Fields(instruction)

	var bestMatch *CorpusPattern
	bestScore := 0

	for i := range patterns {
		score := 0
		for _, word := range instructionWords {
			for _, tag := range patterns[i].Tags {
				if strings.Contains(tag, word) {
					score++
				}
			}
			if strings.Contains(strings.ToLower(patterns[i].Name), word) {
				score += 2
			}
		}
		if score > bestScore {
			bestScore = score
			bestMatch = &patterns[i]
		}
	}

	if bestMatch == nil || bestScore == 0 {
		return nil, fmt.Errorf("no matching pattern found in corpus")
	}

	// Now re-parse the raw code back into an AST Decl
	src := fmt.Sprintf("package main\n\n%s", bestMatch.RawCode)
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to re-parse corpus code: %w", err)
	}

	if len(f.Decls) == 0 {
		return nil, fmt.Errorf("parsed corpus code yielded no declarations")
	}

	return f.Decls[0], nil
}

// SynthesizeAndInject searches the corpus, loads the template, and integrates it securely
func SynthesizeAndInject(filePath string, instruction string, corpusJSON string) error {
	decl, err := MatchAndLoadPattern(instruction, corpusJSON)
	if err != nil {
		return fmt.Errorf("pattern match failed: %w", err)
	}

	return InjectAndValidate(filePath, "", decl)
}

// --- Phase 4: Autonomous Memory & Feedback Loop ---

type Experience struct {
	Timestamp   string `json:"timestamp"`
	Instruction string `json:"instruction"`
	TargetFile  string `json:"target_file"`
	PatternID   string `json:"pattern_id"`
}

// RecordSuccess saves a verified injection into the memory log.
func RecordSuccess(memoryPath string, instruction string, targetFile string, patternID string) error {
	var memory []Experience

	if data, err := os.ReadFile(memoryPath); err == nil {
		json.Unmarshal(data, &memory)
	}

	memory = append(memory, Experience{
		Timestamp:   time.Now().Format(time.RFC3339),
		Instruction: instruction,
		TargetFile:  targetFile,
		PatternID:   patternID,
	})

	data, err := json.MarshalIndent(memory, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal memory: %w", err)
	}

	return os.WriteFile(memoryPath, data, 0644)
}

// RecallExperience scans historical successes for an exact match of the instruction.
func RecallExperience(memoryPath string, instruction string) (string, error) {
	data, err := os.ReadFile(memoryPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("no memory exists")
		}
		return "", fmt.Errorf("failed to read memory: %w", err)
	}

	var memory []Experience
	if err := json.Unmarshal(data, &memory); err != nil {
		return "", fmt.Errorf("failed to parse memory: %w", err)
	}

	instruction = strings.ToLower(strings.TrimSpace(instruction))
	for i := len(memory) - 1; i >= 0; i-- { // Search from newest to oldest
		if strings.ToLower(strings.TrimSpace(memory[i].Instruction)) == instruction {
			return memory[i].PatternID, nil
		}
	}

	return "", fmt.Errorf("no matching experience found")
}

// OrchestrateAndLearn manages the end-to-end flow: Memory Recall -> Corpus Matching -> Synthesis -> Learning
func OrchestrateAndLearn(filePath string, instruction string, corpusJSON string, memoryJSON string) error {
	var decl ast.Decl
	var matchedPatternID string

	// 1. Semantic Memory Recall
	if patternID, err := RecallSimilarExperience(memoryJSON, instruction, 0.6); err == nil {
		// Attempt to load the exact pattern from the corpus using the patternID
		data, readErr := os.ReadFile(corpusJSON)
		if readErr == nil {
			var patterns []CorpusPattern
			if json.Unmarshal(data, &patterns) == nil {
				for _, p := range patterns {
					if p.ID == patternID {
						src := fmt.Sprintf("package main\n\n%s", p.RawCode)
						fset := token.NewFileSet()
						f, parseErr := parser.ParseFile(fset, "", src, 0)
						if parseErr == nil && len(f.Decls) > 0 {
							decl = f.Decls[0]
							matchedPatternID = patternID
							break
						}
					}
				}
			}
		}
	}

	// 2. Fallback Synthesis (Heuristic Search)
	if decl == nil {
		data, err := os.ReadFile(corpusJSON)
		if err != nil {
			return fmt.Errorf("failed to read corpus JSON: %w", err)
		}

		var patterns []CorpusPattern
		if err := json.Unmarshal(data, &patterns); err != nil {
			return fmt.Errorf("failed to parse corpus JSON: %w", err)
		}

		instructionLower := strings.ToLower(instruction)
		instructionWords := strings.Fields(instructionLower)

		var bestMatch *CorpusPattern
		bestScore := 0

		for i := range patterns {
			score := 0
			for _, word := range instructionWords {
				for _, tag := range patterns[i].Tags {
					if strings.Contains(tag, word) {
						score++
					}
				}
				if strings.Contains(strings.ToLower(patterns[i].Name), word) {
					score += 2
				}
			}
			if score > bestScore {
				bestScore = score
				bestMatch = &patterns[i]
			}
		}

		if bestMatch == nil || bestScore == 0 {
			return fmt.Errorf("no matching pattern found in corpus")
		}

		src := fmt.Sprintf("package main\n\n%s", bestMatch.RawCode)
		fset := token.NewFileSet()
		f, parseErr := parser.ParseFile(fset, "", src, 0)
		if parseErr != nil {
			return fmt.Errorf("failed to re-parse corpus code: %w", parseErr)
		}

		if len(f.Decls) == 0 {
			return fmt.Errorf("parsed corpus code yielded no declarations")
		}

		decl = f.Decls[0]
		matchedPatternID = bestMatch.ID
	}

	// 3. Injection and Verification
	if err := InjectAndValidate(filePath, "", decl); err != nil {
		return fmt.Errorf("injection/validation failed: %w", err)
	}

	// 4. Learning (Record Success)
	if err := RecordSuccess(memoryJSON, instruction, filePath, matchedPatternID); err != nil {
		return fmt.Errorf("succeeded but failed to record memory: %w", err)
	}

	return nil
}

// stopWords is a set of common English words to strip before similarity scoring.
var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "i": true, "to": true, "in": true,
	"my": true, "for": true, "and": true, "of": true, "with": true,
	"into": true, "from": true, "that": true, "this": true, "it": true,
	"is": true, "are": true, "want": true, "need": true, "add": true,
	"file": true, "on": true, "at": true, "by": true, "new": true,
}

// tokenize lowercases, strips stop words, and returns a de-duplicated token set.
func tokenize(s string) map[string]struct{} {
	tokens := make(map[string]struct{})
	for _, word := range strings.Fields(strings.ToLower(s)) {
		// Strip leading/trailing punctuation
		word = strings.Trim(word, ".,!?;:'\"()")
		if word != "" && !stopWords[word] {
			tokens[word] = struct{}{}
		}
	}
	return tokens
}

// CalculateSimilarity returns the Jaccard similarity (0.0–1.0) between two instructions.
func CalculateSimilarity(newInst, pastInst string) float64 {
	setA := tokenize(newInst)
	setB := tokenize(pastInst)

	if len(setA) == 0 && len(setB) == 0 {
		return 1.0
	}

	// Intersection
	intersection := 0
	for token := range setA {
		if _, ok := setB[token]; ok {
			intersection++
		}
	}

	// Union = |A| + |B| - |intersection|
	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

// RecallSimilarExperience finds the best-scoring past experience above the given threshold.
func RecallSimilarExperience(memoryPath string, instruction string, threshold float64) (string, error) {
	data, err := os.ReadFile(memoryPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("no memory exists")
		}
		return "", fmt.Errorf("failed to read memory: %w", err)
	}

	var memory []Experience
	if err := json.Unmarshal(data, &memory); err != nil {
		return "", fmt.Errorf("failed to parse memory: %w", err)
	}

	var bestPatternID string
	bestScore := 0.0

	for _, exp := range memory {
		score := CalculateSimilarity(instruction, exp.Instruction)
		if score > bestScore {
			bestScore = score
			bestPatternID = exp.PatternID
		}
	}

	if bestScore < threshold {
		return "", fmt.Errorf("best similarity %.2f below threshold %.2f", bestScore, threshold)
	}

	return bestPatternID, nil
}

// --- Phase 5: Struct Inspector and Generalized Database Code Generator ---

type StructField struct {
	Name string
	Type string
}

type StructModel struct {
	Name   string
	Fields []StructField
}

// InspectStructAST parses a Go file, locates a struct type by name, and extracts its fields and types.
func InspectStructAST(filePath, structName string) (*StructModel, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("failed to parse file: %w", err)
	}

	for _, decl := range f.Decls {
		if genDecl, ok := decl.(*ast.GenDecl); ok && genDecl.Tok == token.TYPE {
			for _, spec := range genDecl.Specs {
				if typeSpec, ok := spec.(*ast.TypeSpec); ok && typeSpec.Name.Name == structName {
					structType, ok := typeSpec.Type.(*ast.StructType)
					if !ok {
						return nil, fmt.Errorf("%s is not a struct", structName)
					}

					model := &StructModel{
						Name: structName,
					}

					for _, field := range structType.Fields.List {
						fieldTypeStr := ""
						if ident, ok := field.Type.(*ast.Ident); ok {
							fieldTypeStr = ident.Name
						} else if sel, ok := field.Type.(*ast.SelectorExpr); ok {
							if xIdent, ok := sel.X.(*ast.Ident); ok {
								fieldTypeStr = xIdent.Name + "." + sel.Sel.Name
							}
						}

						if len(field.Names) == 0 {
							model.Fields = append(model.Fields, StructField{Name: fieldTypeStr, Type: fieldTypeStr})
						} else {
							for _, name := range field.Names {
								model.Fields = append(model.Fields, StructField{Name: name.Name, Type: fieldTypeStr})
							}
						}
					}
					return model, nil
				}
			}
		}
	}
	return nil, fmt.Errorf("struct %s not found in file %s", structName, filePath)
}

// GenerateDatabaseCode generates a database/sql insert function for the given StructModel.
func GenerateDatabaseCode(model *StructModel) string {
	tableName := strings.ToLower(model.Name) + "s"
	var fieldNames []string
	var placeholders []string
	var argNames []string

	for _, field := range model.Fields {
		fieldNames = append(fieldNames, strings.ToLower(field.Name))
		placeholders = append(placeholders, "?")
		argNames = append(argNames, "obj."+field.Name)
	}

	query := fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)", tableName, strings.Join(fieldNames, ", "), strings.Join(placeholders, ", "))

	code := fmt.Sprintf(`
// Insert%s inserts a new %s record into the database.
func Insert%s(db *sql.DB, obj *%s) error {
	query := %c%s%c
	_, err := db.Exec(query, %s)
	return err
}
`, model.Name, model.Name, model.Name, model.Name, '`', query, '`', strings.Join(argNames, ", "))

	return code
}

// ParseDatabaseIntent extracts the struct name and file path for database code generation.
func ParseDatabaseIntent(instruction string) (structName, filePath string, err error) {
	re := regexp.MustCompile(`(?i)generate database code for\s+(\w+)\s+in\s+([a-zA-Z0-9_./-]+)`)
	matches := re.FindStringSubmatch(instruction)
	if len(matches) < 3 {
		return "", "", fmt.Errorf("could not parse database intent from instruction: %s", instruction)
	}
	return matches[1], matches[2], nil
}

// --- Phase 6: Remote Repository Cloner, Recursive Indexer & NLP Concept Mapper ---

// CloneOrUpdateRepo clones a public git repository to destDir, or pulls the latest
// changes if the repository already exists locally.
func CloneOrUpdateRepo(repoURL string, destDir string) error {
	gitDir := filepath.Join(destDir, ".git")
	if info, err := os.Stat(gitDir); err == nil && info.IsDir() {
		// Repo already exists — pull latest changes.
		cmd := exec.Command("git", "-C", destDir, "pull")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("git pull failed for %s: %w", destDir, err)
		}
		return nil
	}

	// Repo does not exist — clone it.
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory %s: %w", destDir, err)
	}

	cmd := exec.Command("git", "clone", repoURL, destDir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("git clone failed from %s: %w", repoURL, err)
	}
	return nil
}

// MapNLPToCodebase searches a corpus JSON file for patterns whose tags best match
// the keywords in the natural language instruction, then records the best match in
// the memory engine for future synthesis.
func MapNLPToCodebase(instruction string, corpusPath string, memoryPath string) (string, error) {
	data, err := os.ReadFile(corpusPath)
	if err != nil {
		return "", fmt.Errorf("failed to read corpus at %s: %w", corpusPath, err)
	}

	var patterns []CorpusPattern
	if err := json.Unmarshal(data, &patterns); err != nil {
		return "", fmt.Errorf("failed to parse corpus JSON: %w", err)
	}

	if len(patterns) == 0 {
		return "", fmt.Errorf("corpus is empty — index a repository first")
	}

	// Tokenize the instruction into a keyword set (reuses existing stop-word logic).
	instrTokens := tokenize(instruction)

	bestScore := -1.0
	var bestPattern CorpusPattern

	for _, p := range patterns {
		// Build the tag token set for this pattern.
		tagSet := make(map[string]struct{})
		for _, tag := range p.Tags {
			tagSet[strings.ToLower(tag)] = struct{}{}
		}
		// Also fold the pattern name in as an additional signal.
		for _, w := range strings.Fields(strings.ToLower(p.Name)) {
			tagSet[w] = struct{}{}
		}

		// Jaccard similarity between instruction tokens and pattern tags.
		intersection := 0
		for tok := range instrTokens {
			if _, ok := tagSet[tok]; ok {
				intersection++
			}
		}
		union := len(instrTokens) + len(tagSet) - intersection
		score := 0.0
		if union > 0 {
			score = float64(intersection) / float64(union)
		}

		if score > bestScore {
			bestScore = score
			bestPattern = p
		}
	}

	if bestScore <= 0.0 {
		return "", fmt.Errorf("no relevant pattern found for instruction: %q", instruction)
	}

	// Persist the successful NLP→code mapping to memory for future synthesis.
	if err := RecordSuccess(memoryPath, instruction, "", bestPattern.ID); err != nil {
		return "", fmt.Errorf("failed to record memory: %w", err)
	}

	return bestPattern.ID, nil
}

// ParseRemoteRepoIntent extracts a GitHub repository URL and optional target corpus
// path from a natural language instruction such as:
//
//	"clone github.com/user/repo into ./repos/myrepo"
func ParseRemoteRepoIntent(instruction string) (repoURL, destDir string, err error) {
	re := regexp.MustCompile(`(?i)(?:clone|index|fetch)\s+(https?://[^\s]+|github\.com/[^\s]+)\s+(?:into|to)?\s*([a-zA-Z0-9_./-]+)`)
	matches := re.FindStringSubmatch(instruction)
	if len(matches) < 3 {
		return "", "", fmt.Errorf("could not parse remote repo intent from: %q", instruction)
	}

	url := matches[1]
	// Normalise bare "github.com/..." to a full https URL.
	if !strings.HasPrefix(url, "http") {
		url = "https://" + url
	}
	return url, matches[2], nil
}

// --- Phase 7: Multi-Step Execution Planner ---

// StepKind describes what a PlanStep does.
type StepKind string

const (
	StepKindInspectStruct  StepKind = "inspect_struct"
	StepKindGenerateDB     StepKind = "generate_db"
	StepKindSynthesizeHTTP StepKind = "synthesize_http"
	StepKindOrchestrate    StepKind = "orchestrate"
)

// PlanStep represents one atomic unit of work inside an ExecutionPlan.
type PlanStep struct {
	Index       int      `json:"index"`
	Kind        StepKind `json:"kind"`
	Description string   `json:"description"`
	Instruction string   `json:"instruction"`
	TargetFile  string   `json:"target_file"`
	StructName  string   `json:"struct_name,omitempty"` // used by inspect_struct / generate_db
}

// StepResult captures the outcome of a single PlanStep execution.
type StepResult struct {
	Step    PlanStep `json:"step"`
	Success bool     `json:"success"`
	Output  string   `json:"output,omitempty"`
	Err     string   `json:"err,omitempty"`
}

// ExecutionPlan holds an ordered slice of steps derived from a high-level intent.
type ExecutionPlan struct {
	Goal    string     `json:"goal"`
	Steps   []PlanStep `json:"steps"`
	Results []StepResult
}

// PlanHighLevelIntent analyses a high-level instruction string and returns an
// ExecutionPlan with ordered PlanSteps. Rules are heuristic/keyword-driven.
//
// Recognised high-level templates (case-insensitive):
//
//   - "user auth api" / "auth api" → 3-step: struct → DB → HTTP
//   - "crud api for <Struct>" → 3-step: struct → DB → HTTP
//   - "database for <Struct>" → 2-step: struct → DB
//   - "http api for <Struct>" → 2-step: struct → HTTP
//
// For anything else, a single orchestration step is emitted.
func PlanHighLevelIntent(goal string, baseTargetDir string, corpusJSON string, memoryJSON string) (*ExecutionPlan, error) {
	lower := strings.ToLower(goal)

	// Helper to derive a sensible output file path.
	outFile := func(name string) string {
		if baseTargetDir == "" {
			baseTargetDir = "./ft"
		}
		return filepath.Join(baseTargetDir, name)
	}

	plan := &ExecutionPlan{Goal: goal}

	// Extract an optional struct name from the goal (e.g. "crud api for User").
	structName := ""
	reStruct := regexp.MustCompile(`(?i)(?:for|of)\s+([A-Z][a-zA-Z0-9]*)`)
	if m := reStruct.FindStringSubmatch(goal); len(m) >= 2 {
		structName = m[1]
	}

	switch {
	case strings.Contains(lower, "auth api") || strings.Contains(lower, "user auth"):
		structName = "User"
		plan.Steps = []PlanStep{
			{Index: 0, Kind: StepKindInspectStruct, Description: "Inspect or scaffold User struct", Instruction: "add User struct with ID, Username, PasswordHash, Email fields", TargetFile: outFile("models.go"), StructName: structName},
			{Index: 1, Kind: StepKindGenerateDB, Description: "Generate DB insert function for User", Instruction: fmt.Sprintf("generate database code for %s in %s", structName, outFile("models.go")), TargetFile: outFile("models.go"), StructName: structName},
			{Index: 2, Kind: StepKindSynthesizeHTTP, Description: "Synthesize HTTP auth handler", Instruction: "add http handler for auth login to " + outFile("auth_handler.go"), TargetFile: outFile("auth_handler.go")},
		}

	case strings.Contains(lower, "crud") && structName != "":
		plan.Steps = []PlanStep{
			{Index: 0, Kind: StepKindInspectStruct, Description: fmt.Sprintf("Scaffold %s struct", structName), Instruction: fmt.Sprintf("add %s struct", structName), TargetFile: outFile("models.go"), StructName: structName},
			{Index: 1, Kind: StepKindGenerateDB, Description: fmt.Sprintf("Generate DB operations for %s", structName), Instruction: fmt.Sprintf("generate database code for %s in %s", structName, outFile("models.go")), TargetFile: outFile("models.go"), StructName: structName},
			{Index: 2, Kind: StepKindSynthesizeHTTP, Description: fmt.Sprintf("Synthesize CRUD HTTP routes for %s", structName), Instruction: fmt.Sprintf("add http handler for %s to %s", strings.ToLower(structName), outFile(strings.ToLower(structName)+"_handler.go")), TargetFile: outFile(strings.ToLower(structName) + "_handler.go")},
		}

	case strings.Contains(lower, "database") && structName != "":
		plan.Steps = []PlanStep{
			{Index: 0, Kind: StepKindInspectStruct, Description: fmt.Sprintf("Scaffold %s struct", structName), Instruction: fmt.Sprintf("add %s struct", structName), TargetFile: outFile("models.go"), StructName: structName},
			{Index: 1, Kind: StepKindGenerateDB, Description: fmt.Sprintf("Generate DB operations for %s", structName), Instruction: fmt.Sprintf("generate database code for %s in %s", structName, outFile("models.go")), TargetFile: outFile("models.go"), StructName: structName},
		}

	case (strings.Contains(lower, "http") || strings.Contains(lower, "api")) && structName != "":
		plan.Steps = []PlanStep{
			{Index: 0, Kind: StepKindInspectStruct, Description: fmt.Sprintf("Scaffold %s struct", structName), Instruction: fmt.Sprintf("add %s struct", structName), TargetFile: outFile("models.go"), StructName: structName},
			{Index: 1, Kind: StepKindSynthesizeHTTP, Description: fmt.Sprintf("Synthesize HTTP routes for %s", structName), Instruction: fmt.Sprintf("add http handler for %s to %s", strings.ToLower(structName), outFile(strings.ToLower(structName)+"_handler.go")), TargetFile: outFile(strings.ToLower(structName) + "_handler.go")},
		}

	default:
		// Fallback: single generic orchestration step.
		plan.Steps = []PlanStep{
			{Index: 0, Kind: StepKindOrchestrate, Description: "Run semantic orchestrator for goal", Instruction: goal, TargetFile: outFile("generated.go")},
		}
	}

	return plan, nil
}

// RunExecutionPlan iterates through each PlanStep sequentially, executes the
// appropriate action, and runs AutoFixPipeline on failure before marking the
// step as failed. State (e.g. produced struct models) is threaded between steps.
func RunExecutionPlan(plan *ExecutionPlan, corpusJSON string, memoryJSON string) []StepResult {
	results := make([]StepResult, 0, len(plan.Steps))

	// Shared state carried forward between steps.
	var lastStructModel *StructModel

	for _, step := range plan.Steps {
		fmt.Printf("  [%d/%d] %s\n", step.Index+1, len(plan.Steps), step.Description)

		result := StepResult{Step: step}

		// Ensure the target file exists with a valid package header.
		if _, err := os.Stat(step.TargetFile); os.IsNotExist(err) {
			dir := filepath.Dir(step.TargetFile)
			os.MkdirAll(dir, 0755)
			pkg := filepath.Base(dir)

			// Clean up package name to be a valid Go identifier.
			re := regexp.MustCompile(`[^a-zA-Z0-9_]`)
			pkg = re.ReplaceAllString(pkg, "")

			// Cannot start with a digit.
			if len(pkg) > 0 && pkg[0] >= '0' && pkg[0] <= '9' {
				pkg = "p" + pkg
			}

			if pkg == "." || pkg == "" {
				pkg = "main"
			}
			os.WriteFile(step.TargetFile, []byte("package "+pkg+"\n"), 0644)
		}

		var stepErr error

		switch step.Kind {

		// ── Inspect / scaffold struct ──────────────────────────────────────────
		case StepKindInspectStruct:
			model, err := InspectStructAST(step.TargetFile, step.StructName)
			if err != nil {
				// Struct not found in file — synthesize it from corpus.
				orchErr := OrchestrateAndLearn(step.TargetFile, step.Instruction, corpusJSON, memoryJSON)
				if orchErr != nil {
					// Last-resort: inject a minimal bare struct.
					bareStruct := &ast.GenDecl{
						Tok: token.TYPE,
						Specs: []ast.Spec{
							&ast.TypeSpec{
								Name: ast.NewIdent(step.StructName),
								Type: &ast.StructType{Fields: &ast.FieldList{}},
							},
						},
					}
					stepErr = InjectAndValidate(step.TargetFile, "", bareStruct)
				}
				// Re-try inspection after synthesis.
				if stepErr == nil {
					model, stepErr = InspectStructAST(step.TargetFile, step.StructName)
				}
			}
			if stepErr == nil && model != nil {
				lastStructModel = model
				result.Output = fmt.Sprintf("struct %s with %d fields", model.Name, len(model.Fields))
			}

		// ── Generate DB operations ─────────────────────────────────────────────
		case StepKindGenerateDB:
			if lastStructModel == nil {
				// Try to inspect again from the target file.
				model, err := InspectStructAST(step.TargetFile, step.StructName)
				if err != nil {
					stepErr = fmt.Errorf("no struct model available for DB generation: %w", err)
					break
				}
				lastStructModel = model
			}

			dbCode := GenerateDatabaseCode(lastStructModel)

			// Parse the generated code into an AST FuncDecl.
			fset := token.NewFileSet()
			f, parseErr := parser.ParseFile(fset, "", "package dummy\n"+dbCode, 0)
			if parseErr != nil || len(f.Decls) == 0 {
				stepErr = fmt.Errorf("failed to parse generated DB code: %w", parseErr)
				break
			}

			funcDecl, ok := f.Decls[0].(*ast.FuncDecl)
			if !ok {
				stepErr = fmt.Errorf("expected *ast.FuncDecl in generated DB code")
				break
			}

			stepErr = InjectAndValidate(step.TargetFile, lastStructModel.Name, funcDecl)
			if stepErr == nil {
				result.Output = fmt.Sprintf("Insert%s injected", lastStructModel.Name)
			}

		// ── Synthesize HTTP handler ─────────────────────────────────────────────
		case StepKindSynthesizeHTTP:
			// Delegate to the corpus-based orchestrator which knows how to match
			// "http handler" patterns from the indexed corpus.
			stepErr = OrchestrateAndLearn(step.TargetFile, step.Instruction, corpusJSON, memoryJSON)
			if stepErr == nil {
				result.Output = step.TargetFile + " updated"
			}

		// ── Generic orchestration fallback ─────────────────────────────────────
		case StepKindOrchestrate:
			stepErr = OrchestrateAndLearn(step.TargetFile, step.Instruction, corpusJSON, memoryJSON)
			if stepErr == nil {
				result.Output = step.TargetFile + " updated"
			}
		}

		// ── Per-step error recovery via AutoFixPipeline ────────────────────────
		if stepErr != nil {
			fmt.Printf("    ⚠️  Step %d failed: %v — attempting AutoFix...\n", step.Index+1, stepErr)
			if fixErr := AutoFixPipeline(step.TargetFile); fixErr == nil {
				// AutoFix succeeded — re-run step validation.
				validateCmd := exec.Command("go", "build", step.TargetFile)
				if out, err := validateCmd.CombinedOutput(); err == nil {
					fmt.Printf("    ✅  AutoFix recovered step %d\n", step.Index+1)
					result.Success = true
					result.Output = "recovered via AutoFixPipeline"
					result.Err = stepErr.Error()
				} else {
					result.Success = false
					result.Err = fmt.Sprintf("%v | post-fix build: %s", stepErr, string(out))
				}
			} else {
				result.Success = false
				result.Err = fmt.Sprintf("%v | autofix: %v", stepErr, fixErr)
			}
		} else {
			result.Success = true
		}

		results = append(results, result)
		fmt.Printf("    %s\n", map[bool]string{true: "✅", false: "❌"}[result.Success])
	}

	return results
}
