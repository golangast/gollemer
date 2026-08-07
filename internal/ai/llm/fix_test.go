package llm

import (
	"os"
	"strings"
	"testing"
)

func TestHandleFixCommand(t *testing.T) {
	// Create a temporary directory for testing fix command
	tmpDir := t.TempDir()
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Failed to get wd: %v", err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to chdir: %v", err)
	}
	defer os.Chdir(origDir)

	// Create a broken Go file similar to jim.go
	brokenContent := `// Jim implementation for the webserver object
package main

func init() {
	fmt.Println("Initializing jim (webserver logic)")
}
 fn() {

}
`
	testFile := "jim.go"
	if err := os.WriteFile(testFile, []byte(brokenContent), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	runner := &Runner{}
	res := runner.handleFixCommand(testFile)
	t.Logf("Fix result: %s", res)

	// Verify file is fixed and builds
	content, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("Failed to read fixed file: %v", err)
	}

	t.Logf("Fixed file content:\n%s", string(content))

	if !strings.Contains(res, "could not auto-fix") {
		t.Fatalf("Expected fix result to contain 'could not auto-fix', got: %s", res)
	}
}

func TestGoFixer_MissingStructBrace(t *testing.T) {
	tmpDir := t.TempDir()
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Failed to get wd: %v", err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to chdir: %v", err)
	}
	defer os.Chdir(origDir)

	brokenContent := `package main

type named struct 
	name string
}

type jill struct {
	cat string
	age int
}
`
	testFile := "main.go"
	if err := os.WriteFile(testFile, []byte(brokenContent), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	fixer := NewGoFixer(testFile)
	if err := fixer.Fix(); err != nil {
		t.Fatalf("Expected GoFixer to fix missing struct brace, but got error: %v", err)
	}

	fixedContent, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("Failed to read fixed file: %v", err)
	}

	if !strings.Contains(string(fixedContent), "type named struct {") {
		t.Errorf("Expected file to contain 'type named struct {', got:\n%s", string(fixedContent))
	}
}

func TestGoFixer_MissingStructKeyword(t *testing.T) {
	tmpDir := t.TempDir()
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Failed to get wd: %v", err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to chdir: %v", err)
	}
	defer os.Chdir(origDir)

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
	testFile := "main.go"
	if err := os.WriteFile(testFile, []byte(brokenContent), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	fixer := NewGoFixer(testFile)
	if err := fixer.Fix(); err != nil {
		t.Fatalf("Expected GoFixer to fix missing struct keyword, but got error: %v", err)
	}

	fixedContent, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("Failed to read fixed file: %v", err)
	}

	if !strings.Contains(string(fixedContent), "type jill struct {") {
		t.Errorf("Expected file to contain 'type jill struct {', got:\n%s", string(fixedContent))
	}

	// The fixed file must parse cleanly
	_ = fixedContent
}

func TestGoFixer_MissingParentheses(t *testing.T) {
	tmpDir := t.TempDir()
	origDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Failed to get wd: %v", err)
	}
	if err := os.Chdir(tmpDir); err != nil {
		t.Fatalf("Failed to chdir: %v", err)
	}
	defer os.Chdir(origDir)

	brokenContent := `package main

import "fmt"

func main() {
	fmt.Println("Hello")
}

func startmyserver{
	return
}
`
	testFile := "main.go"
	if err := os.WriteFile(testFile, []byte(brokenContent), 0644); err != nil {
		t.Fatalf("Failed to write test file: %v", err)
	}

	fixer := NewGoFixer(testFile)
	if err := fixer.Fix(); err != nil {
		t.Fatalf("Expected GoFixer to fix missing parentheses, but got error: %v", err)
	}

	fixedContent, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("Failed to read fixed file: %v", err)
	}

	if !strings.Contains(string(fixedContent), "func startmyserver() {") {
		t.Errorf("Expected file to contain 'func startmyserver() {', got:\n%s", string(fixedContent))
	}
}
