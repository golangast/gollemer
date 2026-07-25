package main

import (
	"go/ast"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseStructModificationPrompt(t *testing.T) {
	tests := []struct {
		prompt     string
		wantStruct string
		wantField  string
		wantType   string
	}{
		{
			prompt:     "add UserID string field to Struct User",
			wantStruct: "User",
			wantField:  "UserID",
			wantType:   "string",
		},
		{
			prompt:     "add Email string field to User",
			wantStruct: "User",
			wantField:  "Email",
			wantType:   "string",
		},
		{
			prompt:     "ADD Age int FIELD TO STRUCT Profile",
			wantStruct: "Profile",
			wantField:  "Age",
			wantType:   "int",
		},
	}

	for _, tt := range tests {
		structName, fieldDef := parseStructModificationPrompt(tt.prompt)
		if structName != tt.wantStruct {
			t.Errorf("prompt %q: got structName=%q, want %q", tt.prompt, structName, tt.wantStruct)
		}
		if fieldDef.Name != tt.wantField {
			t.Errorf("prompt %q: got fieldName=%q, want %q", tt.prompt, fieldDef.Name, tt.wantField)
		}
		if fieldDef.Type != tt.wantType {
			t.Errorf("prompt %q: got fieldType=%q, want %q", tt.prompt, fieldDef.Type, tt.wantType)
		}
	}
}

func TestInjectIntoFunction(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "main.go")

	initialCode := `package main

import "fmt"

func main() {
	fmt.Println("Hello World")
}
`
	if err := os.WriteFile(targetFile, []byte(initialCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	// Prepare statements to inject: http.HandleFunc("/api", handleApi)
	snippet := `http.HandleFunc("/api", handleApi)`
	stmts, err := parseStatements(snippet)
	if err != nil {
		t.Fatalf("Failed to parse statements: %v", err)
	}

	if err := injectIntoFunction(targetFile, "main", stmts); err != nil {
		t.Fatalf("injectIntoFunction failed: %v", err)
	}

	modifiedBytes, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read modified file: %v", err)
	}
	content := string(modifiedBytes)

	if !strings.Contains(content, "HandleFunc") || !strings.Contains(content, "/api") {
		t.Errorf("Expected injected statement in modified file. Content:\n%s", content)
	}
	if !strings.Contains(content, `"net/http"`) {
		t.Errorf("Expected net/http import to be added automatically. Content:\n%s", content)
	}
}

func TestModifyStructFields(t *testing.T) {
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

	fields := []StructFieldDef{
		{Name: "UserID", Type: "string", Tag: `json:"user_id"`},
		{Name: "Age", Type: "int"},
	}

	if err := modifyStructFields(targetFile, "User", fields); err != nil {
		t.Fatalf("modifyStructFields failed: %v", err)
	}

	modifiedBytes, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read modified file: %v", err)
	}
	content := string(modifiedBytes)

	if !strings.Contains(content, "UserID") || !strings.Contains(content, "user_id") {
		t.Errorf("Expected UserID field in struct User. Content:\n%s", content)
	}
	if !strings.Contains(content, "Age") || !strings.Contains(content, "int") {
		t.Errorf("Expected Age field in struct User. Content:\n%s", content)
	}
}

func TestValidateGoCode(t *testing.T) {
	tempDir := t.TempDir()
	// Write dummy go.mod in tempDir so go vet and go build work
	goModContent := "module testmod\n\ngo 1.20\n"
	_ = os.WriteFile(filepath.Join(tempDir, "go.mod"), []byte(goModContent), 0644)

	targetFile := filepath.Join(tempDir, "main.go")

	// Valid Go file
	validCode := `package main

import "fmt"

func main() {
	fmt.Println("OK")
}
`
	if err := os.WriteFile(targetFile, []byte(validCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	if err := validateGoCode(targetFile); err != nil {
		t.Errorf("validateGoCode failed for valid code: %v", err)
	}

	// Invalid Go file (compilation error)
	invalidCode := `package main

func main() {
	undefinedSymbol()
}
`
	if err := os.WriteFile(targetFile, []byte(invalidCode), 0644); err != nil {
		t.Fatalf("Failed to write temp file: %v", err)
	}

	err := validateGoCode(targetFile)
	if err == nil {
		t.Errorf("validateGoCode expected error for invalid code, got nil")
	} else if !strings.Contains(err.Error(), "undefinedSymbol") && !strings.Contains(err.Error(), "build failed") {
		t.Errorf("Expected build failure message containing undefinedSymbol or build failed, got: %v", err)
	}
}

func TestRollbackFile(t *testing.T) {
	tempDir := t.TempDir()
	targetFile := filepath.Join(tempDir, "test.txt")

	originalContent := "original content"
	if err := os.WriteFile(targetFile, []byte(originalContent), 0644); err != nil {
		t.Fatalf("Failed to write original file: %v", err)
	}

	// Modify file
	if err := os.WriteFile(targetFile, []byte("broken content"), 0644); err != nil {
		t.Fatalf("Failed to modify file: %v", err)
	}

	// Rollback
	if err := rollbackFile(targetFile, []byte(originalContent), true); err != nil {
		t.Fatalf("rollbackFile failed: %v", err)
	}

	readContent, err := os.ReadFile(targetFile)
	if err != nil {
		t.Fatalf("Failed to read after rollback: %v", err)
	}

	if string(readContent) != originalContent {
		t.Errorf("Got %q after rollback, want %q", string(readContent), originalContent)
	}
}

func TestParseStatements(t *testing.T) {
	snippet := `http.HandleFunc("/test", testHandler)`
	stmts, err := parseStatements(snippet)
	if err != nil {
		t.Fatalf("parseStatements failed: %v", err)
	}
	if len(stmts) != 1 {
		t.Fatalf("Expected 1 statement, got %d", len(stmts))
	}

	exprStmt, ok := stmts[0].(*ast.ExprStmt)
	if !ok {
		t.Fatalf("Expected *ast.ExprStmt, got %T", stmts[0])
	}
	callExpr, ok := exprStmt.X.(*ast.CallExpr)
	if !ok {
		t.Fatalf("Expected *ast.CallExpr, got %T", exprStmt.X)
	}

	fset := token.NewFileSet()
	_ = fset
	_ = callExpr
}
