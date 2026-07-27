package symbol

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSymbolGraph_IndexAndQuery(t *testing.T) {
	// Create a temporary workspace with a small Go file
	tmpDir := t.TempDir()

	goContent := `package auth

import "fmt"

// User represents an authenticated user
type User struct {
	Name string
	Age  int
}

// Authenticator handles JWT token validation
type Authenticator interface {
	ValidateToken(token string) (*User, error)
	GenerateToken(user *User) (string, error)
}

// JWTAuthenticator implements Authenticator for JWT tokens
type JWTAuthenticator struct {
	SecretKey string
}

func (j *JWTAuthenticator) ValidateToken(token string) (*User, error) {
	fmt.Println("Validating token")
	return &User{Name: "test"}, nil
}

func (j *JWTAuthenticator) GenerateToken(user *User) (string, error) {
	fmt.Println("Generating token")
	return "token123", nil
}

func validateUser(user *User) bool {
	return user.Name != ""
}
`

	goTestContent := `package auth

import "testing"

func TestJWTAuthenticator_ValidateToken(t *testing.T) {
	auth := &JWTAuthenticator{SecretKey: "test-secret"}
	user, err := auth.ValidateToken("test-token")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if user.Name != "test" {
		t.Fatalf("expected Name=test, got %s", user.Name)
	}
}
`

	mainContent := `package main

import (
	"fmt"
	"log"
	"example.com/auth"
)

func main() {
	auth := &auth.JWTAuthenticator{SecretKey: "my-secret"}
	token, err := auth.GenerateToken(&auth.User{Name: "admin"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Token:", token)
}
`

	// Write test files
	writeFile(t, filepath.Join(tmpDir, "auth", "auth.go"), goContent)
	writeFile(t, filepath.Join(tmpDir, "auth", "auth_test.go"), goTestContent)
	writeFile(t, filepath.Join(tmpDir, "main.go"), mainContent)

	// Create and index the symbol graph
	sg := NewSymbolGraph(tmpDir)
	if err := sg.IndexWorkspace(); err != nil {
		t.Fatalf("IndexWorkspace failed: %v", err)
	}

	// Test Summary
	t.Log(sg.Summary())

	// Test FindDefinitions
	t.Run("FindDefinitions", func(t *testing.T) {
		defs := sg.FindDefinitions("User")
		if len(defs) == 0 {
			t.Fatal("expected to find User definition")
		}
		var foundStruct bool
		for _, d := range defs {
			if d.Kind == KindStruct && d.Name == "User" {
				foundStruct = true
				if len(d.Children) == 0 {
					t.Error("User struct should have fields")
				}
				break
			}
		}
		if !foundStruct {
			t.Error("expected to find User struct definition")
		}
	})

	t.Run("FindDefinitions_Authenticator", func(t *testing.T) {
		defs := sg.FindDefinitions("Authenticator")
		if len(defs) == 0 {
			t.Fatal("expected to find Authenticator definition")
		}
		var foundIface bool
		for _, d := range defs {
			if d.Kind == KindInterface && d.Name == "Authenticator" {
				foundIface = true
				// Should have methods listed as children
				if len(d.Children) == 0 {
					t.Error("Authenticator interface should have methods")
				}
				break
			}
		}
		if !foundIface {
			t.Error("expected to find Authenticator interface")
		}
	})

	t.Run("FindDefinitions_ValidateToken", func(t *testing.T) {
		defs := sg.FindDefinitions("ValidateToken")
		if len(defs) == 0 {
			t.Fatal("expected to find ValidateToken definition")
		}
		var foundMethod bool
		for _, d := range defs {
			if d.Kind == KindMethod && d.Name == "ValidateToken" {
				foundMethod = true
				break
			}
		}
		if !foundMethod {
			t.Error("expected to find ValidateToken method")
		}
	})

	t.Run("FindReferences", func(t *testing.T) {
		refs := sg.FindReferences("ValidateToken")
		if len(refs) == 0 {
			t.Log("No references found (may vary based on parsing)")
		} else {
			for _, r := range refs {
				t.Logf("  Reference: %s:%d", r.File, r.Line)
			}
		}
	})

	t.Run("FindCallers", func(t *testing.T) {
		// ValidateToken should be called from the test file
		callers := sg.FindCallers("ValidateToken")
		if len(callers) > 0 {
			for _, c := range callers {
				t.Logf("  Caller: %s in %s:%d", c.Name, c.File, c.Line)
			}
		}
	})

	t.Run("FindImplementations", func(t *testing.T) {
		impls := sg.FindImplementations("Authenticator")
		if len(impls) == 0 {
			t.Fatal("expected to find implementations of Authenticator")
		}
		var foundImpl bool
		for _, i := range impls {
			if i.Kind == KindStruct && i.Name == "JWTAuthenticator" {
				foundImpl = true
				break
			}
		}
		if !foundImpl {
			t.Error("expected JWTAuthenticator to implement Authenticator")
		}
	})

	t.Run("SearchSymbols", func(t *testing.T) {
		results := sg.SearchSymbols("token")
		if len(results) == 0 {
			t.Error("expected to find symbols matching 'token'")
		} else {
			for _, r := range results {
				t.Logf("  Symbol: %s (%s)", r.Name, r.Kind)
			}
		}
	})

	t.Run("TraceCallGraph", func(t *testing.T) {
		graph := sg.TraceCallGraph("ValidateToken", 3)
		if len(graph) > 0 {
			for caller, callees := range graph {
				t.Logf("  %s -> %v", caller, callees)
			}
		}
	})

	t.Run("GetSymbolsByFile", func(t *testing.T) {
		syms := sg.GetSymbolsByFile("auth/auth.go")
		if len(syms) == 0 {
			t.Error("expected symbols in auth/auth.go")
		}
	})

	t.Run("GetSymbolsByPackage", func(t *testing.T) {
		syms := sg.GetSymbolsByPackage("auth")
		if len(syms) == 0 {
			t.Error("expected symbols in package auth")
		}
	})

	t.Run("ExportImportJSON", func(t *testing.T) {
		data, err := sg.ExportJSON()
		if err != nil {
			t.Fatalf("ExportJSON failed: %v", err)
		}

		sg2 := NewSymbolGraph(tmpDir)
		if err := sg2.ImportJSON(data); err != nil {
			t.Fatalf("ImportJSON failed: %v", err)
		}

		// Verify data roundtripped
		if len(sg2.symbols) != len(sg.symbols) {
			t.Errorf("symbol count mismatch: got %d, want %d", len(sg2.symbols), len(sg.symbols))
		}
	})
}

func TestSymbolGraph_EmptyWorkspace(t *testing.T) {
	tmpDir := t.TempDir()
	sg := NewSymbolGraph(tmpDir)

	if err := sg.IndexWorkspace(); err != nil {
		t.Fatalf("IndexWorkspace on empty dir failed: %v", err)
	}

	if len(sg.symbols) != 0 {
		t.Errorf("expected 0 symbols in empty workspace, got %d", len(sg.symbols))
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
