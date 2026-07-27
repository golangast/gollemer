// Package agent tests the LLM tool execution and auto-fix loop.
// Uses a mock LLM callback to verify the 5-iteration fallback logic
// and tool execution paths without needing a real LLM API.
package agent

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunAutoFix_PassesImmediately verifies the loop exits immediately when tests pass.
func TestRunAutoFix_PassesImmediately(t *testing.T) {
	tmpDir := t.TempDir()
	err := os.WriteFile(filepath.Join(tmpDir, "pass_test.go"), []byte(`package pass

import "testing"

func TestPass(t *testing.T) {
}
`), 0644)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	modInit := runCmd(t, tmpDir, "go", "mod", "init", "pass")
	if !modInit {
		t.Fatal("Failed to init go module")
	}

	llmCalled := false
	callLLM := func(prompt string, tools []LLMTool) ([]ToolCall, string, error) {
		llmCalled = true
		return nil, "unexpected call", nil
	}

	err = RunAutoFix(".", tmpDir, callLLM)
	if err != nil {
		t.Fatalf("RunAutoFix should have passed immediately: %v", err)
	}
	if llmCalled {
		t.Error("LLM should not have been called when tests pass")
	}
}

// TestRunAutoFix_SuccessOnFirstFix verifies the loop succeeds when a fix is applied.
func TestRunAutoFix_SuccessOnFirstFix(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a simple test that always fails due to a hardcoded error
	err := os.WriteFile(filepath.Join(tmpDir, "bug_test.go"), []byte(`package bug

import "testing"

func TestBug(t *testing.T) {
	t.Error("this test is broken")
}
`), 0644)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	modInit := runCmd(t, tmpDir, "go", "mod", "init", "bug")
	if !modInit {
		t.Fatal("Failed to init go module")
	}

	// Mock LLM: each call returns a fix that removes the t.Error line.
	// The loop keeps retrying until the patch actually makes the test pass.
	// Use a simple fix: completely replace the file
	callLLM := func(prompt string, tools []LLMTool) ([]ToolCall, string, error) {
		return []ToolCall{
			{
				Name:      "apply_patch",
				Arguments: json.RawMessage(`{"file":"bug_test.go","patch":"<<<<<<< SEARCH\nimport \"testing\"\n\nfunc TestBug(t *testing.T) {\n\tt.Error(\"this test is broken\")\n}\n=======\nimport \"testing\"\n\nfunc TestBug(t *testing.T) {\n}\n>>>>>>> REPLACE"}`),
			},
		}, "removing the error", nil
	}

	err = RunAutoFix(".", tmpDir, callLLM)
	if err != nil {
		t.Fatalf("RunAutoFix should have succeeded: %v", err)
	}
}

// TestRunAutoFix_MaxRetries verifies the loop stops after 5 iterations.
func TestRunAutoFix_MaxRetries(t *testing.T) {
	tmpDir := t.TempDir()

	err := os.WriteFile(filepath.Join(tmpDir, "fail_test.go"), []byte(`package fail

import "testing"

func TestAlwaysFail(t *testing.T) {
	t.Error("This test always fails")
}
`), 0644)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	modInit := runCmd(t, tmpDir, "go", "mod", "init", "fail")
	if !modInit {
		t.Fatal("Failed to init go module")
	}

	attempts := 0
	callLLM := func(prompt string, tools []LLMTool) ([]ToolCall, string, error) {
		attempts++
		return nil, "I cannot fix this test", nil
	}

	err = RunAutoFix(".", tmpDir, callLLM)
	if err == nil {
		t.Fatal("RunAutoFix should have failed after max retries")
	}
	if !strings.Contains(err.Error(), "5 iterations") {
		t.Errorf("Expected error about 5 iterations, got: %v", err)
	}
	if attempts != 5 {
		t.Errorf("Expected 5 LLM calls, got %d", attempts)
	}
}

// TestRunAutoFix_LLMFailure verifies the loop handles LLM errors gracefully.
func TestRunAutoFix_LLMFailure(t *testing.T) {
	tmpDir := t.TempDir()

	err := os.WriteFile(filepath.Join(tmpDir, "err_test.go"), []byte(`package err

import "testing"

func TestError(t *testing.T) {
	t.Error("failing")
}
`), 0644)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	modInit := runCmd(t, tmpDir, "go", "mod", "init", "err")
	if !modInit {
		t.Fatal("Failed to init go module")
	}

	attempts := 0
	callLLM := func(prompt string, tools []LLMTool) ([]ToolCall, string, error) {
		attempts++
		if attempts == 1 {
			return nil, "", nil
		}
		return nil, "still can't fix", nil
	}

	err = RunAutoFix(".", tmpDir, callLLM)
	if err == nil {
		t.Fatal("RunAutoFix should have failed")
	}
}

// TestToolExecutor_RunTests verifies the run_tests tool works correctly.
func TestToolExecutor_RunTests(t *testing.T) {
	tmpDir := t.TempDir()

	err := os.WriteFile(filepath.Join(tmpDir, "sum_test.go"), []byte(`package sum

import "testing"

func TestSum(t *testing.T) {
	if 1+1 != 2 {
		t.Error("basic math broken")
	}
}
`), 0644)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	modInit := runCmd(t, tmpDir, "go", "mod", "init", "sum")
	if !modInit {
		t.Fatal("Failed to init go module")
	}

	executor := NewLLMToolExecutor(tmpDir)
	result := executor.executeRunTests(json.RawMessage(`{"package": "."}`))
	if !result.Success {
		t.Fatalf("Tests should pass: %s", result.Output)
	}
}

// TestToolExecutor_RunTestsWithFilter verifies the -run flag works.
func TestToolExecutor_RunTestsWithFilter(t *testing.T) {
	tmpDir := t.TempDir()

	err := os.WriteFile(filepath.Join(tmpDir, "multi_test.go"), []byte(`package multi

import "testing"

func TestFoo(t *testing.T) {}
func TestBar(t *testing.T) {}
func TestBaz(t *testing.T) {}
`), 0644)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	modInit := runCmd(t, tmpDir, "go", "mod", "init", "multi")
	if !modInit {
		t.Fatal("Failed to init go module")
	}

	executor := NewLLMToolExecutor(tmpDir)

	result := executor.executeRunTests(json.RawMessage(`{"package": ".", "run": "TestFoo"}`))
	if !result.Success {
		t.Fatalf("TestFoo should pass: %s", result.Output)
	}
	if strings.Contains(result.Output, "TestBar") {
		t.Error("TestBar should not have run")
	}
}

// TestToolExecutor_ReadFile verifies the read_file tool.
func TestToolExecutor_ReadFile(t *testing.T) {
	tmpDir := t.TempDir()

	content := "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n"
	err := os.WriteFile(filepath.Join(tmpDir, "main.go"), []byte(content), 0644)
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}

	executor := NewLLMToolExecutor(tmpDir)

	// Read entire file
	result := executor.executeReadFile(json.RawMessage(`{"path": "main.go"}`))
	if !result.Success {
		t.Fatalf("ReadFile failed: %v", result.Error)
	}
	if result.Output != content {
		t.Errorf("ReadFile content mismatch.\nExpected:\n%s\nGot:\n%s", content, result.Output)
	}

	// Read with line range (lines 1-2: package + blank)
	result = executor.executeReadFile(json.RawMessage(`{"path": "main.go", "start_line": 1, "end_line": 2}`))
	if !result.Success {
		t.Fatalf("ReadFile with range failed: %v", result.Error)
	}
	// strings.Join doesn't add trailing newline, so we get "package main\n"
	expected := "package main\n"
	if result.Output != expected {
		t.Errorf("Expected lines 1-2:\n%q\nGot:\n%q", expected, result.Output)
	}

	// Read non-existent file
	result = executor.executeReadFile(json.RawMessage(`{"path": "nonexistent.go"}`))
	if result.Success {
		t.Error("ReadFile should fail for non-existent file")
	}
}

// TestToolExecutor_ApplyPatch verifies the apply_patch tool.
func TestToolExecutor_ApplyPatch(t *testing.T) {
	tmpDir := t.TempDir()

	original := "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n"
	err := os.WriteFile(filepath.Join(tmpDir, "main.go"), []byte(original), 0644)
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}

	executor := NewLLMToolExecutor(tmpDir)

	// Apply a patch to change "hello" to "world"
	// Use \n escape sequences in the JSON string
	result := executor.executeApplyPatch(json.RawMessage(`{"file":"main.go","patch":"<<<<<<< SEARCH\n\tfmt.Println(\"hello\")\n=======\n\tfmt.Println(\"world\")\n>>>>>>> REPLACE"}`))
	if !result.Success {
		t.Fatalf("ApplyPatch failed: %v", result.Error)
	}

	// Verify the file was changed
	data, _ := os.ReadFile(filepath.Join(tmpDir, "main.go"))
	if !strings.Contains(string(data), `fmt.Println("world")`) {
		t.Errorf("File should contain 'world'. Content:\n%s", string(data))
	}
	if strings.Contains(string(data), `fmt.Println("hello")`) {
		t.Errorf("File should not contain 'hello'. Content:\n%s", string(data))
	}
}

// TestToolExecutor_ApplyPatchNonexistent verifies error handling for missing files.
func TestToolExecutor_ApplyPatchNonexistent(t *testing.T) {
	executor := NewLLMToolExecutor(t.TempDir())
	result := executor.executeApplyPatch(json.RawMessage(`{"file":"no_such_file.go","patch":"x"}`))
	if result.Success {
		t.Error("ApplyPatch should fail for non-existent file")
	}
}

// TestToolExecutor_ApplyPatchInvalid verifies error handling for invalid patches.
func TestToolExecutor_ApplyPatchInvalid(t *testing.T) {
	tmpDir := t.TempDir()
	os.WriteFile(filepath.Join(tmpDir, "main.go"), []byte("package main"), 0644)

	executor := NewLLMToolExecutor(tmpDir)

	result := executor.executeApplyPatch(json.RawMessage(`{"file":"main.go","patch":"not a valid patch"}`))
	if result.Success {
		t.Error("ApplyPatch should fail for invalid patch format")
	}
}

// TestToolDefinitions_AllPresent verifies all three tool definitions exist.
func TestToolDefinitions_AllPresent(t *testing.T) {
	executor := NewLLMToolExecutor(".")
	tools := executor.GetToolDefinitions()

	expected := map[string]bool{
		"run_tests":   false,
		"read_file":   false,
		"apply_patch": false,
	}

	for _, tool := range tools {
		if _, ok := expected[tool.Name]; ok {
			expected[tool.Name] = true
		}
	}

	for name, found := range expected {
		if !found {
			t.Errorf("Missing tool definition: %s", name)
		}
	}
}

// TestToolDefinitions_HaveParameters verifies all tool definitions have JSON schemas.
func TestToolDefinitions_HaveParameters(t *testing.T) {
	executor := NewLLMToolExecutor(".")
	tools := executor.GetToolDefinitions()

	for _, tool := range tools {
		if len(tool.Parameters) == 0 {
			t.Errorf("Tool %s has no parameters", tool.Name)
		}
		var params map[string]interface{}
		if err := json.Unmarshal(tool.Parameters, &params); err != nil {
			t.Errorf("Tool %s has invalid parameter JSON: %v", tool.Name, err)
		}
		if params["type"] != "object" {
			t.Errorf("Tool %s parameters should be type 'object'", tool.Name)
		}
	}
}

// runCmd is a helper to run a command in a directory.
func runCmd(t *testing.T, dir, name string, args ...string) bool {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("Command %s %v failed: %s\nOutput: %s", name, args, err, out)
		return false
	}
	return true
}
