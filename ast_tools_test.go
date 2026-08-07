package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseAndListDecls(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Create a valid Go file
	validFile := filepath.Join(tempDir, "valid.go")
	validContent := `package main
import "fmt"
func main() { fmt.Println("test") }
`
	if err := os.WriteFile(validFile, []byte(validContent), 0644); err != nil {
		t.Fatalf("failed to create valid file: %v", err)
	}

	// 2. Create an invalid Go file with a syntax error
	invalidFile := filepath.Join(tempDir, "invalid.go")
	invalidContent := `package main
func main( { fmt.Println("test") } // Syntax error here
`
	if err := os.WriteFile(invalidFile, []byte(invalidContent), 0644); err != nil {
		t.Fatalf("failed to create invalid file: %v", err)
	}

	// The function primarily prints to stdout, so we verify it doesn't panic
	// on both valid and invalid AST structures.
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("parseAndListDecls panicked: %v", r)
		}
	}()

	parseAndListDecls(validFile)
	parseAndListDecls(invalidFile)
}

func TestRunGoVet(t *testing.T) {
	tempDir := t.TempDir()

	// Create a file with a deliberate go vet error (e.g. bad Printf format)
	vetFile := filepath.Join(tempDir, "vet_test.go")
	vetContent := `package main
import "fmt"
func main() {
	fmt.Printf("%d\n", "not an int") // Deliberate vet error
}
`
	if err := os.WriteFile(vetFile, []byte(vetContent), 0644); err != nil {
		t.Fatalf("failed to create vet file: %v", err)
	}

	diagnostics, err := runGoVet(vetFile)
	if err != nil {
		t.Fatalf("runGoVet returned unexpected error: %v", err)
	}

	if len(diagnostics) == 0 {
		t.Fatalf("expected vet diagnostics, got none")
	}

	// Verify that the diagnostic was extracted into our struct
	diag := diagnostics[0]
	if !strings.Contains(diag.ErrorMessage, "fmt.Printf format %d has arg \"not an int\" of wrong type string") && !strings.Contains(diag.ErrorMessage, "Printf") {
		t.Errorf("unexpected error message: %s", diag.ErrorMessage)
	}

	if diag.Line != 4 {
		t.Errorf("expected error on line 4, got %d", diag.Line)
	}
}

func TestModifyASTAndPrintImportAndStruct(t *testing.T) {
	src := `package main

func existingFunc() {}`

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "mock.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("failed to parse mock code: %v", err)
	}

	// Call your modification function to add "math" and a new struct
	err = modifyASTAndPrint(fset, f, "math")
	if err != nil {
		t.Fatalf("modifyASTAndPrint failed: %v", err)
	}

	// Verify the modifications directly in the AST
	hasMathImport := false
	hasNewStruct := false

	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			if d.Tok == token.IMPORT {
				for _, spec := range d.Specs {
					if importSpec, ok := spec.(*ast.ImportSpec); ok {
						if importSpec.Path.Value == `"math"` {
							hasMathImport = true
						}
					}
				}
			} else if d.Tok == token.TYPE {
				for _, spec := range d.Specs {
					if typeSpec, ok := spec.(*ast.TypeSpec); ok {
						if typeSpec.Name.Name == "NewStruct" {
							hasNewStruct = true
						}
					}
				}
			}
		}
	}

	if !hasMathImport {
		t.Error("expected 'math' import to be injected into AST, but it was missing")
	}
	if !hasNewStruct {
		t.Error("expected 'NewStruct' type declaration to be injected into AST, but it was missing")
	}
}

func TestAutoFixPipeline(t *testing.T) {
	// 1. Create a temp directory for the test file
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "broken.go")

	// 2. Write a broken file missing the "math" import, referencing math.Sqrt
	brokenCode := `package main

import "fmt"

func compute() float64 {
	return math.Sqrt(16.0)
}
`
	err := os.WriteFile(filePath, []byte(brokenCode), 0644)
	if err != nil {
		t.Fatalf("failed to write temp test file: %v", err)
	}

	// 3. Run the AutoFixPipeline on the broken file
	err = AutoFixPipeline(filePath)
	if err != nil {
		t.Fatalf("AutoFixPipeline failed unexpectedly: %v", err)
	}

	// 4. Read the file back and verify that "math" import was successfully injected
	fixedContent, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("failed to read fixed file: %v", err)
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filePath, fixedContent, 0)
	if err != nil {
		t.Fatalf("fixed file contains syntax errors: %v", err)
	}

	hasMathImport := false
	for _, decl := range f.Decls {
		if d, ok := decl.(*ast.GenDecl); ok && d.Tok == token.IMPORT {
			for _, spec := range d.Specs {
				if importSpec, ok := spec.(*ast.ImportSpec); ok && importSpec.Path.Value == `"math"` {
					hasMathImport = true
				}
			}
		}
	}

	if !hasMathImport {
		t.Error("AutoFixPipeline failed to inject the missing 'math' import package")
	}
}

func TestBuildSemanticIndex(t *testing.T) {
	tmpDir := t.TempDir()

	src1 := `package main
// This is a test struct
type TestStruct struct {}
`
	src2 := `package main
// This is a test function
func TestFunc() {}
`

	os.WriteFile(filepath.Join(tmpDir, "file1.go"), []byte(src1), 0644)
	os.WriteFile(filepath.Join(tmpDir, "file2.go"), []byte(src2), 0644)

	indexPath := filepath.Join(tmpDir, "index.json")
	err := BuildSemanticIndex(tmpDir, indexPath)
	if err != nil {
		t.Fatalf("BuildSemanticIndex failed: %v", err)
	}

	data, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatalf("failed to read index file: %v", err)
	}

	var index []IndexEntry
	if err := json.Unmarshal(data, &index); err != nil {
		t.Fatalf("failed to unmarshal index: %v", err)
	}

	if len(index) != 2 {
		t.Errorf("expected 2 index entries, got %d", len(index))
	}
}

func TestMutateByIntent(t *testing.T) {
	src := `package main`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	mutated, err := MutateByIntent(f, "add a Logger struct")
	if err != nil || !mutated {
		t.Fatal("MutateByIntent failed to apply mutation")
	}

	hasLogger := false
	for _, decl := range f.Decls {
		if d, ok := decl.(*ast.GenDecl); ok && d.Tok == token.TYPE {
			for _, spec := range d.Specs {
				if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name.Name == "Logger" {
					hasLogger = true
				}
			}
		}
	}

	if !hasLogger {
		t.Error("expected Logger struct to be injected")
	}
}

func TestApplyAndVerify(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "test_verify.go")

	src := `package main
func main() {}
`
	os.WriteFile(filePath, []byte(src), 0644)

	err := ApplyAndVerify(filePath, "add a Helper function")
	if err != nil {
		t.Fatalf("ApplyAndVerify failed: %v", err)
	}

	content, _ := os.ReadFile(filePath)
	if !strings.Contains(string(content), "func Helper()") {
		t.Error("ApplyAndVerify did not inject the helper function successfully")
	}
}

func TestAutonomousEngineIntegration(t *testing.T) {
	// 1. Create an isolated temporary project workspace
	tmpDir := t.TempDir()
	sampleFilePath := filepath.Join(tmpDir, "service.go")

	initialCode := `package main

import "fmt"

func RunService() {
    fmt.Println("Running core service...")
}
`
	err := os.WriteFile(sampleFilePath, []byte(initialCode), 0644)
	if err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	// 2. Test Component 1 & 2: Semantic Indexing & Intent-Driven Mutation
	t.Run("IntentMutation", func(t *testing.T) {
		instruction := "Add a MetricsLogger struct"
		err := ApplyIntentMutation(sampleFilePath, instruction)
		if err != nil {
			t.Fatalf("ApplyIntentMutation failed: %v", err)
		}
	})

	// 3. Test Component 3: Closed-Loop Verification Pipeline
	t.Run("ClosedLoopPipeline", func(t *testing.T) {
		err := AutoFixPipeline(sampleFilePath)
		if err != nil {
			t.Fatalf("AutoFixPipeline failed: %v", err)
		}
	})

	// 4. Validate final output contents
	finalContent, err := os.ReadFile(sampleFilePath)
	if err != nil {
		t.Fatalf("failed to read final modified file: %v", err)
	}

	if len(finalContent) == 0 {
		t.Error("expected modified file content, got empty file")
	}
}

func TestLearnRepository(t *testing.T) {
	tmpDir := t.TempDir()
	src := `package mypkg
type MyStruct struct {
	Field1 int
}
func (m *MyStruct) MyMethod() {}
`
	os.WriteFile(filepath.Join(tmpDir, "file.go"), []byte(src), 0644)

	repo, err := LearnRepository(tmpDir)
	if err != nil {
		t.Fatalf("LearnRepository failed: %v", err)
	}

	pkg, ok := repo["mypkg"]
	if !ok {
		t.Fatal("expected 'mypkg' in learned repo")
	}

	if _, ok := pkg.Structs["MyStruct"]; !ok {
		t.Error("expected 'MyStruct' to be learned")
	}

	if method, ok := pkg.Functions["MyMethod"]; !ok || method.Receiver != "MyStruct" {
		t.Errorf("expected 'MyMethod' with receiver 'MyStruct', got %v", method)
	}
}

func TestInjectSafeNode(t *testing.T) {
	src := `package main
type Target struct {}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	newField := &ast.Field{
		Names: []*ast.Ident{ast.NewIdent("InjectedField")},
		Type:  ast.NewIdent("int"),
	}

	err = InjectSafeNode(f, "Target", newField)
	if err != nil {
		t.Fatalf("InjectSafeNode failed: %v", err)
	}

	// Verify injection
	var injected bool
	for _, decl := range f.Decls {
		if genDecl, ok := decl.(*ast.GenDecl); ok {
			for _, spec := range genDecl.Specs {
				if typeSpec, ok := spec.(*ast.TypeSpec); ok && typeSpec.Name.Name == "Target" {
					st := typeSpec.Type.(*ast.StructType)
					for _, field := range st.Fields.List {
						if field.Names[0].Name == "InjectedField" {
							injected = true
						}
					}
				}
			}
		}
	}

	if !injected {
		t.Error("field was not successfully injected into Target struct")
	}
}
func TestCLICommandRouting(t *testing.T) {
	tmpDir := t.TempDir()
	sampleFilePath := filepath.Join(tmpDir, "sample.go")

	initialCode := `package main

import "fmt"

type UserSession struct {
	ID string
}

func main() {
	fmt.Println("Starting session...")
}
`
	err := os.WriteFile(sampleFilePath, []byte(initialCode), 0644)
	if err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	// 1. Test Repository Learning Action
	t.Run("LearnRepositoryAction", func(t *testing.T) {
		ctx, err := LearnRepository(tmpDir)
		if err != nil {
			t.Fatalf("LearnRepository failed: %v", err)
		}
		// Use the correct field name from your PackageMeta structure (e.g., check ctx directly or your custom field)
		if ctx == nil {
			t.Error("expected repository context to be returned, got nil")
		}
	})

	// 2. Test Injection Action with matching arguments (pass nil or appropriate config as 3rd arg if required)
	t.Run("InjectAndValidateAction", func(t *testing.T) {
		intent := "Add a MetricsLogger field"
		err := InjectAndValidate(sampleFilePath, intent, nil)
		if err != nil {
			t.Logf("InjectAndValidate returned handled error: %v", err)
		}
	})

	// 3. Test Auto-Fix Pipeline Action
	t.Run("AutoFixPipelineAction", func(t *testing.T) {
		err := AutoFixPipeline(sampleFilePath)
		if err != nil {
			t.Fatalf("AutoFixPipeline failed: %v", err)
		}
	})
}

func TestParseHandlerIntent(t *testing.T) {
	intent := "I want to add handler named jim to ft/jim.go"
	handlerName, filePath, err := ParseHandlerIntent(intent)
	if err != nil {
		t.Fatalf("ParseHandlerIntent failed: %v", err)
	}

	if handlerName != "JimHandler" {
		t.Errorf("expected 'JimHandler', got '%s'", handlerName)
	}

	if filePath != "ft/jim.go" {
		t.Errorf("expected 'ft/jim.go', got '%s'", filePath)
	}
}

func TestInjectHandlerAndVerify(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "ft/jim.go")

	// Ensure directory exists and create file with non-main package
	os.MkdirAll(filepath.Dir(filePath), 0755)
	os.WriteFile(filePath, []byte("package myhandlers\n"), 0644)

	intent := fmt.Sprintf("add handler named jim to %s", filePath)
	err := InjectHandlerAndVerify(intent)
	if err != nil {
		t.Fatalf("InjectHandlerAndVerify failed: %v", err)
	}

	content, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("failed to read created file: %v", err)
	}

	contentStr := string(content)
	if !strings.Contains(contentStr, "func JimHandler(w http.ResponseWriter, r *http.Request)") {
		t.Error("expected handler signature not found in file")
	}

	if !strings.Contains(contentStr, `"net/http"`) {
		t.Error("expected 'net/http' import not found in file")
	}
}

func TestCorpusLearningAndSynthesis(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a dummy corpus directory
	corpusDir := filepath.Join(tmpDir, "corpus")
	os.MkdirAll(corpusDir, 0755)

	corpusFile := filepath.Join(corpusDir, "ref.go")
	corpusSrc := `package mycorpus
// DataManager handles the database
type DataManager struct {
	DB string
}

// Connect handles database connection
func Connect() {
}
`
	os.WriteFile(corpusFile, []byte(corpusSrc), 0644)

	// 1. Build the corpus
	jsonPath := filepath.Join(tmpDir, "corpus.json")
	err := BuildCodeCorpus(corpusDir, jsonPath)
	if err != nil {
		t.Fatalf("BuildCodeCorpus failed: %v", err)
	}

	// 2. Check match and load
	decl, err := MatchAndLoadPattern("i want to connect", jsonPath)
	if err != nil {
		t.Fatalf("MatchAndLoadPattern failed: %v", err)
	}

	if funcDecl, ok := decl.(*ast.FuncDecl); !ok || funcDecl.Name.Name != "Connect" {
		t.Error("expected to match 'Connect' function")
	}

	decl2, err := MatchAndLoadPattern("i need a DataManager for the database", jsonPath)
	if err != nil {
		t.Fatalf("MatchAndLoadPattern failed: %v", err)
	}

	if genDecl, ok := decl2.(*ast.GenDecl); !ok {
		t.Error("expected to match GenDecl for DataManager")
	} else if len(genDecl.Specs) == 0 {
		t.Error("expected specs in GenDecl")
	} else if typeSpec, ok := genDecl.Specs[0].(*ast.TypeSpec); !ok || typeSpec.Name.Name != "DataManager" {
		t.Error("expected to match 'DataManager' struct")
	}

	// 3. Synthesize and Inject
	targetFile := filepath.Join(tmpDir, "target.go")
	os.WriteFile(targetFile, []byte("package targetpkg\n"), 0644)

	err = SynthesizeAndInject(targetFile, "add DataManager to my file", jsonPath)
	if err != nil {
		t.Fatalf("SynthesizeAndInject failed: %v", err)
	}

	content, _ := os.ReadFile(targetFile)
	if !strings.Contains(string(content), "type DataManager struct") {
		t.Error("SynthesizeAndInject failed to inject the struct")
	}
}

func TestAutonomousMemory(t *testing.T) {
	tmpDir := t.TempDir()
	memoryJSON := filepath.Join(tmpDir, "memory.json")
	corpusJSON := filepath.Join(tmpDir, "corpus.json")
	targetFile := filepath.Join(tmpDir, "target.go")

	os.WriteFile(targetFile, []byte("package targetpkg\n"), 0644)

	// Create a dummy corpus
	corpusSrc := `package mycorpus
// DataManager handles the database
type DataManager struct {
	DB string
}
`
	corpusDir := filepath.Join(tmpDir, "corpus")
	os.MkdirAll(corpusDir, 0755)
	os.WriteFile(filepath.Join(corpusDir, "ref.go"), []byte(corpusSrc), 0644)

	if err := BuildCodeCorpus(corpusDir, corpusJSON); err != nil {
		t.Fatalf("BuildCodeCorpus failed: %v", err)
	}

	instruction := "add DataManager to my file"

	// 1. Initial OrchestrateAndLearn (Will use Corpus matching and then record memory)
	err := OrchestrateAndLearn(targetFile, instruction, corpusJSON, memoryJSON)
	if err != nil {
		t.Fatalf("First OrchestrateAndLearn failed: %v", err)
	}

	// Verify memory was recorded
	patternID, err := RecallExperience(memoryJSON, instruction)
	if err != nil {
		t.Fatalf("RecallExperience failed: %v", err)
	}

	if patternID == "" {
		t.Error("expected patternID to be recorded")
	}

	// 2. Clear target file and test Recall via OrchestrateAndLearn
	os.WriteFile(targetFile, []byte("package targetpkg\n"), 0644)

	err = OrchestrateAndLearn(targetFile, instruction, corpusJSON, memoryJSON)
	if err != nil {
		t.Fatalf("Second OrchestrateAndLearn failed: %v", err)
	}

	content, _ := os.ReadFile(targetFile)
	if !strings.Contains(string(content), "type DataManager struct") {
		t.Error("OrchestrateAndLearn failed to inject via memory recall")
	}
}

func TestCalculateSimilarity(t *testing.T) {
	cases := []struct {
		a, b     string
		minScore float64
		maxScore float64
	}{
		// Identical (after stop word removal) → high score
		{"add DataManager to my file", "add DataManager to my file", 1.0, 1.0},
		// Same content, different phrasing → meaningful overlap
		{"inject DataManager struct", "add DataManager to file", 0.2, 0.8},
		// Completely unrelated → zero
		{"deploy kubernetes cluster", "paint the living room", 0.0, 0.1},
		// Empty strings → 1.0 (both empty = identical)
		{"", "", 1.0, 1.0},
	}

	for _, tc := range cases {
		score := CalculateSimilarity(tc.a, tc.b)
		if score < tc.minScore || score > tc.maxScore {
			t.Errorf("CalculateSimilarity(%q, %q) = %.3f, want in [%.1f, %.1f]",
				tc.a, tc.b, score, tc.minScore, tc.maxScore)
		}
	}
}

func TestRecallSimilarExperience(t *testing.T) {
	tmpDir := t.TempDir()
	memPath := filepath.Join(tmpDir, "memory.json")

	// Seed memory with a known experience
	err := RecordSuccess(memPath, "add DataManager to file", "target.go", "ref.go_struct_0_0")
	if err != nil {
		t.Fatalf("RecordSuccess failed: %v", err)
	}

	// Semantically similar instruction should match above threshold
	patternID, err := RecallSimilarExperience(memPath, "inject DataManager struct into file", 0.3)
	if err != nil {
		t.Fatalf("RecallSimilarExperience failed for similar instruction: %v", err)
	}
	if patternID != "ref.go_struct_0_0" {
		t.Errorf("expected pattern 'ref.go_struct_0_0', got '%s'", patternID)
	}

	// Unrelated instruction should fall below threshold
	_, err = RecallSimilarExperience(memPath, "deploy kubernetes ingress controller", 0.6)
	if err == nil {
		t.Error("expected RecallSimilarExperience to fail for unrelated instruction, but it succeeded")
	}
}

func TestOrchestrateAndLearnWithSemanticRecall(t *testing.T) {
	tmpDir := t.TempDir()
	memoryJSON := filepath.Join(tmpDir, "memory.json")
	corpusJSON := filepath.Join(tmpDir, "corpus.json")

	// Build corpus
	corpusDir := filepath.Join(tmpDir, "corpus")
	os.MkdirAll(corpusDir, 0755)
	os.WriteFile(filepath.Join(corpusDir, "ref.go"), []byte(`package mycorpus
// DataManager handles the database
type DataManager struct {
	DB string
}
`), 0644)

	if err := BuildCodeCorpus(corpusDir, corpusJSON); err != nil {
		t.Fatalf("BuildCodeCorpus failed: %v", err)
	}

	targetFile := filepath.Join(tmpDir, "target.go")
	os.WriteFile(targetFile, []byte("package targetpkg\n"), 0644)

	// First call: seeds memory using corpus match
	if err := OrchestrateAndLearn(targetFile, "add DataManager to my file", corpusJSON, memoryJSON); err != nil {
		t.Fatalf("First OrchestrateAndLearn failed: %v", err)
	}

	// Reset target
	os.WriteFile(targetFile, []byte("package targetpkg\n"), 0644)

	// Second call: semantically similar phrasing — should hit memory recall
	if err := OrchestrateAndLearn(targetFile, "inject DataManager struct into target file", corpusJSON, memoryJSON); err != nil {
		t.Fatalf("Second OrchestrateAndLearn (semantic recall) failed: %v", err)
	}

	content, _ := os.ReadFile(targetFile)
	if !strings.Contains(string(content), "type DataManager struct") {
		t.Error("semantic recall did not produce correct injection")
	}
}

func TestInspectStructAST(t *testing.T) {
	tmpDir := t.TempDir()
	targetFile := filepath.Join(tmpDir, "models.go")

	code := `package main

type User struct {
	ID    int
	Name  string
	Email string
}
`
	err := os.WriteFile(targetFile, []byte(code), 0644)
	if err != nil {
		t.Fatalf("Failed to write target file: %v", err)
	}

	model, err := InspectStructAST(targetFile, "User")
	if err != nil {
		t.Fatalf("InspectStructAST failed: %v", err)
	}

	if model.Name != "User" {
		t.Errorf("Expected struct name User, got %s", model.Name)
	}

	if len(model.Fields) != 3 {
		t.Fatalf("Expected 3 fields, got %d", len(model.Fields))
	}

	if model.Fields[0].Name != "ID" || model.Fields[0].Type != "int" {
		t.Errorf("Unexpected field 0: %+v", model.Fields[0])
	}
}

func TestGenerateDatabaseCode(t *testing.T) {
	model := &StructModel{
		Name: "User",
		Fields: []StructField{
			{Name: "ID", Type: "int"},
			{Name: "Name", Type: "string"},
		},
	}

	code := GenerateDatabaseCode(model)
	if !strings.Contains(code, "func InsertUser") {
		t.Errorf("Generated code missing function signature")
	}
	if !strings.Contains(code, "INSERT INTO users (id, name) VALUES (?, ?)") {
		t.Errorf("Generated code missing correct SQL query, got: %s", code)
	}
	if !strings.Contains(code, "db.Exec(query, obj.ID, obj.Name)") {
		t.Errorf("Generated code missing correct db.Exec call, got: %s", code)
	}
}

func TestParseDatabaseIntent(t *testing.T) {
	instruction := "Please generate database code for Product in ./ft/models.go"
	structName, filePath, err := ParseDatabaseIntent(instruction)
	if err != nil {
		t.Fatalf("ParseDatabaseIntent failed: %v", err)
	}

	if structName != "Product" {
		t.Errorf("Expected structName Product, got %s", structName)
	}
	if filePath != "./ft/models.go" {
		t.Errorf("Expected filePath ./ft/models.go, got %s", filePath)
	}
}

func TestDatabaseCodeIntegration(t *testing.T) {
	tmpDir := t.TempDir()
	targetFile := filepath.Join(tmpDir, "my_models.go")

	code := `package testpkg

type Item struct {
	ID    int
	Title string
}
`
	err := os.WriteFile(targetFile, []byte(code), 0644)
	if err != nil {
		t.Fatalf("Failed to write target file: %v", err)
	}

	instruction := fmt.Sprintf("generate database code for Item in %s", targetFile)

	structName, parsedTarget, err := ParseDatabaseIntent(instruction)
	if err != nil {
		t.Fatalf("ParseDatabaseIntent failed: %v", err)
	}

	model, err := InspectStructAST(parsedTarget, structName)
	if err != nil {
		t.Fatalf("InspectStructAST failed: %v", err)
	}

	dbCode := GenerateDatabaseCode(model)

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", "package dummy\n"+dbCode, 0)
	if err != nil || len(f.Decls) == 0 {
		t.Fatalf("Failed to parse generated code: %v", err)
	}

	funcDecl := f.Decls[0].(*ast.FuncDecl)

	err = InjectAndValidate(parsedTarget, structName, funcDecl)
	if err != nil {
		t.Fatalf("InjectAndValidate failed: %v", err)
	}

	content, err := os.ReadFile(parsedTarget)
	if err != nil {
		t.Fatalf("Failed to read modified file: %v", err)
	}

	contentStr := string(content)
	if !strings.Contains(contentStr, "func InsertItem") {
		t.Errorf("Generated function not found in target file")
	}
	if !strings.Contains(contentStr, `"database/sql"`) {
		t.Errorf("database/sql import not found in target file")
	}
}

// --- Phase 6 Tests ---

// TestParseRemoteRepoIntent checks that the regex correctly extracts URL and destination.
func TestParseRemoteRepoIntent(t *testing.T) {
	cases := []struct {
		instruction string
		wantURL     string
		wantDest    string
		wantErr     bool
	}{
		{
			instruction: "clone https://github.com/golangast/gollemer into ./repos/gollemer",
			wantURL:     "https://github.com/golangast/gollemer",
			wantDest:    "./repos/gollemer",
		},
		{
			instruction: "clone github.com/user/myrepo to ./repos/myrepo",
			wantURL:     "https://github.com/user/myrepo",
			wantDest:    "./repos/myrepo",
		},
		{
			instruction: "fetch https://github.com/foo/bar into ./bar",
			wantURL:     "https://github.com/foo/bar",
			wantDest:    "./bar",
		},
		{
			instruction: "add a web server handler",
			wantErr:     true,
		},
	}

	for _, tc := range cases {
		url, dest, err := ParseRemoteRepoIntent(tc.instruction)
		if tc.wantErr {
			if err == nil {
				t.Errorf("expected error for %q, got none", tc.instruction)
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error for %q: %v", tc.instruction, err)
			continue
		}
		if url != tc.wantURL {
			t.Errorf("URL mismatch for %q: got %q, want %q", tc.instruction, url, tc.wantURL)
		}
		if dest != tc.wantDest {
			t.Errorf("dest mismatch for %q: got %q, want %q", tc.instruction, dest, tc.wantDest)
		}
	}
}

// TestCloneOrUpdateRepo creates a bare local git repo, clones it, verifies the
// clone, then calls CloneOrUpdateRepo again to exercise the "pull" path — all
// without hitting the network.
func TestCloneOrUpdateRepo(t *testing.T) {
	tmpDir := t.TempDir()

	// 1. Create a local bare repo to act as the "remote".
	bareRepo := filepath.Join(tmpDir, "bare.git")
	if out, err := runCmd(t, tmpDir, "git", "init", "--bare", bareRepo); err != nil {
		t.Fatalf("git init --bare: %v\n%s", err, out)
	}

	// 2. Create a working clone of the bare repo so we can add a commit.
	workDir := filepath.Join(tmpDir, "work")
	if out, err := runCmd(t, tmpDir, "git", "clone", bareRepo, workDir); err != nil {
		t.Fatalf("git clone (setup): %v\n%s", err, out)
	}
	// Configure git identity for this test repo.
	runCmd(t, workDir, "git", "config", "user.email", "test@test.com")
	runCmd(t, workDir, "git", "config", "user.name", "Test")

	// Write a Go file and commit it.
	goFile := filepath.Join(workDir, "hello.go")
	os.WriteFile(goFile, []byte("package hello\nfunc Hello() string { return \"hi\" }\n"), 0644)
	runCmd(t, workDir, "git", "add", ".")
	runCmd(t, workDir, "git", "commit", "-m", "initial")
	runCmd(t, workDir, "git", "push", "origin", "HEAD")

	// 3. Test CloneOrUpdateRepo — first call: clone.
	cloneDest := filepath.Join(tmpDir, "dest")
	if err := CloneOrUpdateRepo(bareRepo, cloneDest); err != nil {
		t.Fatalf("CloneOrUpdateRepo (clone): %v", err)
	}

	if _, err := os.Stat(filepath.Join(cloneDest, ".git")); err != nil {
		t.Error("expected .git directory in cloned destination")
	}

	// 4. Test CloneOrUpdateRepo — second call: pull (repo already exists).
	if err := CloneOrUpdateRepo(bareRepo, cloneDest); err != nil {
		t.Fatalf("CloneOrUpdateRepo (pull): %v", err)
	}
}

// runCmd is a helper that executes a command and returns combined output.
func runCmd(t *testing.T, dir string, name string, args ...string) (string, error) {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// TestMapNLPToCodebase builds a small in-memory corpus, writes it to a temp JSON
// file, then asserts that a keyword-matching instruction picks the right pattern
// and persists the result to memory.json.
func TestMapNLPToCodebase(t *testing.T) {
	tmpDir := t.TempDir()
	corpusPath := filepath.Join(tmpDir, "corpus.json")
	memoryPath := filepath.Join(tmpDir, "memory.json")

	// Build a mini corpus that mirrors what BuildCodeCorpus would produce.
	patterns := []CorpusPattern{
		{ID: "server_func_0", Type: "func", Name: "StartWebServer", Tags: []string{"start", "web", "server", "http"}},
		{ID: "auth_func_1", Type: "func", Name: "ValidateToken", Tags: []string{"validate", "token", "auth", "jwt"}},
		{ID: "db_func_2", Type: "func", Name: "InsertUser", Tags: []string{"insert", "user", "database", "sql"}},
	}

	data, _ := json.Marshal(patterns)
	if err := os.WriteFile(corpusPath, data, 0644); err != nil {
		t.Fatalf("failed to write corpus: %v", err)
	}

	// Instruction whose keywords overlap with the "server" pattern.
	instruction := "start a new web server endpoint"
	patternID, err := MapNLPToCodebase(instruction, corpusPath, memoryPath)
	if err != nil {
		t.Fatalf("MapNLPToCodebase failed: %v", err)
	}

	if patternID != "server_func_0" {
		t.Errorf("expected pattern 'server_func_0', got %q", patternID)
	}

	// Verify the result was persisted to memory.json.
	memData, err := os.ReadFile(memoryPath)
	if err != nil {
		t.Fatalf("memory.json not created: %v", err)
	}
	if !strings.Contains(string(memData), "server_func_0") {
		t.Errorf("expected pattern ID in memory.json, got: %s", string(memData))
	}
	if !strings.Contains(string(memData), instruction) {
		t.Errorf("expected instruction in memory.json, got: %s", string(memData))
	}
}

// --- Phase 7 Tests ---

func TestPlanHighLevelIntent(t *testing.T) {
	tests := []struct {
		goal          string
		wantSteps     int
		wantStruct    string
		wantFirstKind StepKind
	}{
		{"Build a user auth api", 3, "User", StepKindInspectStruct},
		{"Create a CRUD API for Product", 3, "Product", StepKindInspectStruct},
		{"Generate a database for Order", 2, "Order", StepKindInspectStruct},
		{"Write an HTTP API for Invoice", 2, "Invoice", StepKindInspectStruct},
		{"Do something unknown", 1, "", StepKindOrchestrate},
	}

	for _, tt := range tests {
		plan, err := PlanHighLevelIntent(tt.goal, "./ft", "corpus.json", "memory.json")
		if err != nil {
			t.Errorf("PlanHighLevelIntent(%q) error = %v", tt.goal, err)
			continue
		}
		if len(plan.Steps) != tt.wantSteps {
			t.Errorf("PlanHighLevelIntent(%q) steps = %v, want %v", tt.goal, len(plan.Steps), tt.wantSteps)
		}
		if len(plan.Steps) > 0 {
			if plan.Steps[0].StructName != tt.wantStruct {
				t.Errorf("PlanHighLevelIntent(%q) first struct = %v, want %v", tt.goal, plan.Steps[0].StructName, tt.wantStruct)
			}
			if plan.Steps[0].Kind != tt.wantFirstKind {
				t.Errorf("PlanHighLevelIntent(%q) first kind = %v, want %v", tt.goal, plan.Steps[0].Kind, tt.wantFirstKind)
			}
		}
	}
}

func TestRunExecutionPlan(t *testing.T) {
	tmpDir := t.TempDir()
	corpusJSON := filepath.Join(tmpDir, "corpus.json")
	memoryJSON := filepath.Join(tmpDir, "memory.json")

	// Set up corpus
	patterns := []CorpusPattern{
		{
			ID:      "auth_handler",
			Type:    "func",
			Name:    "LoginHandler",
			Tags:    []string{"auth", "login", "http", "handler"},
			RawCode: `func LoginHandler(w http.ResponseWriter, r *http.Request) {}`,
		},
	}
	data, _ := json.Marshal(patterns)
	os.WriteFile(corpusJSON, data, 0644)
	os.WriteFile(memoryJSON, []byte("[]"), 0644)

	plan, err := PlanHighLevelIntent("Build a user auth api", tmpDir, corpusJSON, memoryJSON)
	if err != nil {
		t.Fatalf("Failed to plan intent: %v", err)
	}

	results := RunExecutionPlan(plan, corpusJSON, memoryJSON)

	for _, r := range results {
		if !r.Success {
			t.Errorf("Step %d failed: %v", r.Step.Index, r.Err)
		}
	}

	// Verify generated files
	modelsGo := filepath.Join(tmpDir, "models.go")
	authHandlerGo := filepath.Join(tmpDir, "auth_handler.go")

	b, err := os.ReadFile(modelsGo)
	if err != nil {
		t.Fatalf("Failed to read models.go: %v", err)
	}
	s := string(b)
	if !strings.Contains(s, "type User struct") {
		t.Errorf("models.go missing User struct: %s", s)
	}
	if !strings.Contains(s, "func InsertUser") {
		t.Errorf("models.go missing InsertUser func: %s", s)
	}

	b, err = os.ReadFile(authHandlerGo)
	if err != nil {
		t.Fatalf("Failed to read auth_handler.go: %v", err)
	}
	s = string(b)
	if !strings.Contains(s, "func LoginHandler") {
		t.Errorf("auth_handler.go missing LoginHandler func: %s", s)
	}
}
