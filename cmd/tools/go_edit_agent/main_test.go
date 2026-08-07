package main

import (
	"encoding/json"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestApplyInsertFunc(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "main.go")

	initialCode := `package main

import "fmt"

func main() {
	fmt.Println("Hello")
}
`
	if err := os.WriteFile(targetFile, []byte(initialCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	edit := EditOperation{
		Type:     "insert_func",
		FuncName: "greet",
		Code:     "func greet(name string) string {\n\treturn \"Hello, \" + name\n}\n",
	}

	result := applyInsertFunc(targetFile, edit)
	if !result.Success {
		t.Fatalf("applyInsertFunc failed: %s", result.Error)
	}

	content, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read modified file: %v", err)
	}

	if !strings.Contains(string(content), "func greet") {
		t.Errorf("Expected greet function in file. Content:\n%s", string(content))
	}
}

func TestApplyModifyFunc(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "main.go")

	initialCode := `package main

import "fmt"

func main() {
	fmt.Println("Hello")
}
`
	if err := os.WriteFile(targetFile, []byte(initialCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	edit := EditOperation{
		Type:     "modify_func",
		FuncName: "main",
		Code:     `fmt.Println("Modified")`,
	}

	result := applyModifyFunc(targetFile, edit)
	if !result.Success {
		t.Fatalf("applyModifyFunc failed: %s", result.Error)
	}

	content, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read modified file: %v", err)
	}

	if !strings.Contains(string(content), "Modified") {
		t.Errorf("Expected modified content in file. Content:\n%s", string(content))
	}
}

func TestApplyAddField(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "models.go")

	initialCode := `package main

type User struct {
	Name string
}
`
	if err := os.WriteFile(targetFile, []byte(initialCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	edit := EditOperation{
		Type:       "add_field",
		StructName: "User",
		FieldName:  "Email",
		FieldType:  "string",
		FieldTag:   `json:"email"`,
	}

	result := applyAddField(targetFile, edit)
	if !result.Success {
		t.Fatalf("applyAddField failed: %s", result.Error)
	}

	content, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read modified file: %v", err)
	}

	if !strings.Contains(string(content), "Email") || !strings.Contains(string(content), "email") {
		t.Errorf("Expected Email field with json tag. Content:\n%s", string(content))
	}
}

func TestApplyAddImport(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "main.go")

	initialCode := `package main

func main() {
}
`
	if err := os.WriteFile(targetFile, []byte(initialCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	edit := EditOperation{
		Type:       "add_import",
		ImportPath: "net/http",
	}

	result := applyAddImport(targetFile, edit)
	if !result.Success {
		t.Fatalf("applyAddImport failed: %s", result.Error)
	}

	content, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read modified file: %v", err)
	}

	if !strings.Contains(string(content), `"net/http"`) {
		t.Errorf("Expected net/http import. Content:\n%s", string(content))
	}
}

func TestApplyReplaceCode(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "main.go")

	initialCode := `package main

import "fmt"

func main() {
	fmt.Println("Hello")
}
`
	if err := os.WriteFile(targetFile, []byte(initialCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	edit := EditOperation{
		Type:    "replace_code",
		OldCode: `fmt.Println("Hello")`,
		NewCode: `fmt.Println("World")`,
	}

	result := applyReplaceCode(targetFile, edit)
	if !result.Success {
		t.Fatalf("applyReplaceCode failed: %s", result.Error)
	}

	content, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read modified file: %v", err)
	}

	if !strings.Contains(string(content), `fmt.Println("World")`) {
		t.Errorf("Expected replaced content. Content:\n%s", string(content))
	}
}

func TestApplyDeleteFunc(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "main.go")

	initialCode := `package main

import "fmt"

func greet() string {
	return "hi"
}

func main() {
	fmt.Println(greet())
}
`
	if err := os.WriteFile(targetFile, []byte(initialCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	edit := EditOperation{
		Type:     "delete_func",
		FuncName: "greet",
	}

	result := applyDeleteFunc(targetFile, edit)
	if !result.Success {
		t.Fatalf("applyDeleteFunc failed: %s", result.Error)
	}

	content, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read modified file: %v", err)
	}

	if strings.Contains(string(content), "func greet") {
		t.Errorf("Expected greet function to be deleted. Content:\n%s", string(content))
	}
}

func TestExecuteAgent(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "main.go")

	initialCode := `package main

import "fmt"

func main() {
	fmt.Println("Hello")
}
`
	if err := os.WriteFile(targetFile, []byte(initialCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	req := AgentRequest{
		File: targetFile,
		Edits: []EditOperation{
			{
				Type:       "add_import",
				ImportPath: "net/http",
			},
			{
				Type:     "insert_func",
				FuncName: "handler",
				Code:     "func handler(w http.ResponseWriter, r *http.Request) {\n\tfmt.Fprintf(w, \"OK\")\n}\n",
			},
		},
		MaxRetries: 1,
	}

	resp := executeAgent(req)
	if !resp.Success {
		t.Fatalf("executeAgent failed: %s", resp.Error)
	}

	if resp.EditsApplied != 2 {
		t.Errorf("Expected 2 edits applied, got %d", resp.EditsApplied)
	}

	content, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read modified file: %v", err)
	}

	if !strings.Contains(string(content), `"net/http"`) {
		t.Errorf("Expected net/http import. Content:\n%s", string(content))
	}
	if !strings.Contains(string(content), "func handler") {
		t.Errorf("Expected handler function. Content:\n%s", string(content))
	}
}

func TestFixSyntaxErrors_MissingStructKeyword(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "jim.go")

	// Mirrors ft/jim.go: "type jill  {" is missing the 'struct' keyword
	brokenContent := `package main

type named struct {
	name string
}

type jill  {
	cat string
	age int
}
`
	if err := os.WriteFile(targetFile, []byte(brokenContent), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	edits := fixSyntaxErrors(targetFile)
	if len(edits) == 0 {
		t.Fatalf("Expected fixSyntaxErrors to produce edits for missing 'struct' keyword")
	}

	content, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read fixed file: %v", err)
	}

	if !strings.Contains(string(content), "type jill struct {") {
		t.Errorf("Expected file to contain 'type jill struct {', got:\n%s", string(content))
	}

	// The fixed file must parse cleanly
	fset := token.NewFileSet()
	if _, err := parser.ParseFile(fset, targetFile, content, parser.ParseComments); err != nil {
		t.Errorf("Fixed file should parse cleanly, got: %v", err)
	}
}

func TestGuessImport(t *testing.T) {
	tests := []struct {
		symbol   string
		expected string
	}{
		{"http", "net/http"},
		{"fmt", "fmt"},
		{"json", "encoding/json"},
		{"tensor", "github.com/golangast/gollemer/internal/ai/neural/tensor"},
		{"moe", "github.com/golangast/gollemer/internal/ai/moe"},
		{"unknown", ""},
	}

	for _, tt := range tests {
		result := guessImport(tt.symbol)
		if result != tt.expected {
			t.Errorf("guessImport(%q) = %q, want %q", tt.symbol, result, tt.expected)
		}
	}
}

func TestToolHandler(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "main.go")

	initialCode := `package main

import "fmt"

func main() {
	fmt.Println("Hello")
}
`
	if err := os.WriteFile(targetFile, []byte(initialCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	handler := NewToolHandler()
	if handler.Name != "go_edit_agent" {
		t.Errorf("Expected name 'go_edit_agent', got %q", handler.Name)
	}

	req := AgentRequest{
		File: targetFile,
		Edits: []EditOperation{
			{
				Type:       "add_import",
				ImportPath: "os",
			},
		},
		MaxRetries: 1,
	}

	reqJSON, _ := json.Marshal(req)
	respJSON, err := handler.Handle(reqJSON)
	if err != nil {
		t.Fatalf("Handler.Handle failed: %v", err)
	}

	var resp AgentResponse
	if err := json.Unmarshal(respJSON, &resp); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	if !resp.Success {
		t.Errorf("Expected success, got error: %s", resp.Error)
	}
}

func TestValidateGoCode(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "main.go")

	validCode := `package main

import "fmt"

func main() {
	fmt.Println("OK")
}
`
	if err := os.WriteFile(targetFile, []byte(validCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	result := validateGoCode(targetFile, false)
	if !result.Success {
		t.Errorf("validateGoCode failed for valid code: %s", result.GoBuild)
	}
}
