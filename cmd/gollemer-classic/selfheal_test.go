package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseFilesFromResponse(t *testing.T) {
	resp := "Here is the code:\n```go\n=== FILE: a.go ===\npackage demo\n\n=== FILE: sub/b.go ===\npackage sub\n```\n"
	files, err := parseFilesFromResponse(resp)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("got %d files, want 2", len(files))
	}
	if !strings.HasPrefix(files["a.go"], "package demo") {
		t.Fatalf("a.go content wrong: %q", files["a.go"])
	}
	if !strings.Contains(files["sub/b.go"], "package sub") {
		t.Fatalf("sub/b.go content wrong: %q", files["sub/b.go"])
	}
	for name := range files {
		if strings.Contains(name, "```") {
			t.Fatalf("fence leaked into path %q", name)
		}
	}
}

func TestParseFilesFromResponseErrors(t *testing.T) {
	for _, tc := range []struct {
		name string
		resp string
	}{
		{"no markers", "package demo\n"},
		{"absolute path", "=== FILE: /etc/evil.go ===\npackage evil\n"},
		{"escape", "=== FILE: ../evil.go ===\npackage evil\n"},
		{"non-go", "=== FILE: notes.txt ===\nhello\n"},
	} {
		if _, err := parseFilesFromResponse(tc.resp); err == nil {
			t.Errorf("%s: expected error, got nil", tc.name)
		}
	}
}

// fixture builds a tiny module for sandbox tests.
func fixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	write := func(name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/demo\n\ngo 1.26.0\n")
	write("add.go", "package demo\n\nfunc Add(a, b int) int { return a + b }\n")
	return dir
}

func TestGenerateAndSelfHealExhaustsAttempts(t *testing.T) {
	dir := fixture(t)
	// Always returns code with a type error; never heals.
	broken := "=== FILE: broken.go ===\npackage demo\n\nfunc Broken() int { return undefinedVar }\n"
	SetLLMClient(NewMockClient(broken))
	defer SetLLMClient(NewLLMClientFromEnv())

	_, err := GenerateAndSelfHeal(dir, "break things", nil, nil, 2)
	if err == nil {
		t.Fatal("expected exhaustion error, got nil")
	}
	if !strings.Contains(err.Error(), "all 2 attempts failed") {
		t.Fatalf("error should mention exhaustion, got: %v", err)
	}
}

func TestGenerateAndSelfHealHealsFormatViolation(t *testing.T) {
	dir := fixture(t)
	garbage := "Here is some prose with no file blocks at all."
	fixed := "=== FILE: add_test.go ===\npackage demo\n\nimport \"testing\"\n\nfunc TestAdd(t *testing.T) {\n\tif got := Add(2, 2); got != 4 {\n\t\tt.Errorf(\"got %d, want 4\", got)\n\t}\n}\n"
	SetLLMClient(NewMockClient(garbage, fixed))
	defer SetLLMClient(NewLLMClientFromEnv())

	files, err := GenerateAndSelfHeal(dir, "write a test for Add", nil, nil, 3)
	if err != nil {
		t.Fatalf("expected heal after format violation, got: %v", err)
	}
	if _, ok := files["add_test.go"]; !ok {
		t.Fatalf("expected add_test.go in %v", files)
	}
}
