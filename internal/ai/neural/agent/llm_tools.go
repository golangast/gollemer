// Package agent provides LLM function calling tools for the auto-fix loop.
// These tools give the LLM access to read files, run tests, and apply patches
// to Go source code. The agent loop calls the LLM with tool definitions,
// executes the tool choices, and iterates until tests pass.
package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// LLMTool defines a function that the LLM can call.
type LLMTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ToolCall represents a specific invocation of a tool by the LLM.
type ToolCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// LLMToolExecutor executes LLM tool calls against the local environment.
type LLMToolExecutor struct {
	WorkingDir string
}

// NewLLMToolExecutor creates a new tool executor.
func NewLLMToolExecutor(workingDir string) *LLMToolExecutor {
	return &LLMToolExecutor{
		WorkingDir: workingDir,
	}
}

// GetToolDefinitions returns the LLM function calling tool definitions.
func (e *LLMToolExecutor) GetToolDefinitions() []LLMTool {
	return []LLMTool{
		e.defineRunTests(),
		e.defineReadFile(),
		e.defineApplyPatch(),
	}
}

// defineRunTests returns the run_tests tool definition.
func (e *LLMToolExecutor) defineRunTests() LLMTool {
	return LLMTool{
		Name:        "run_tests",
		Description: "Runs 'go test -v <package>' on the specified Go package and returns the stdout/stderr output. Use this to verify code changes or check test failures.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"package": {
					"type": "string",
					"description": "The Go package path to test (e.g., './cmd/tools/multi_orchestrator' or just '.')"
				},
				"run": {
					"type": "string",
					"description": "Optional: Specific test function to run (e.g., 'TestApplyPatch'). Uses -run flag."
				},
				"timeout": {
					"type": "string",
					"description": "Optional: Test timeout (e.g., '30s', '2m'). Defaults to '30s'.",
					"default": "30s"
				}
			},
			"required": ["package"]
		}`),
	}
}

// defineReadFile returns the read_file tool definition.
func (e *LLMToolExecutor) defineReadFile() LLMTool {
	return LLMTool{
		Name:        "read_file",
		Description: "Reads the content of a specific file in the project. Use this to inspect code before making changes.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {
					"type": "string",
					"description": "Relative path to the file from the project root (e.g., 'cmd/tools/multi_orchestrator/main.go')"
				},
				"start_line": {
					"type": "integer",
					"description": "Optional: Starting line number (1-based) to read from",
					"default": 1
				},
				"end_line": {
					"type": "integer",
					"description": "Optional: Ending line number (1-based) to read up to",
					"default": 0
				}
			},
			"required": ["path"]
		}`),
	}
}

// defineApplyPatch returns the apply_patch tool definition.
func (e *LLMToolExecutor) defineApplyPatch() LLMTool {
	return LLMTool{
		Name:        "apply_patch",
		Description: "Applies code changes to a Go source file. Uses SEARCH/REPLACE blocks to make targeted edits. The SEARCH block must match the existing code exactly. After applying, runs gofmt to ensure formatting is correct.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"file": {
					"type": "string",
					"description": "Path to the file to modify (e.g., 'cmd/tools/multi_orchestrator/main.go')"
				},
				"patch": {
					"type": "string",
					"description": "The SEARCH/REPLACE patch in the format:\n<<<<<<< SEARCH\nexact code to find\n=======\nreplacement code\n>>>>>>> REPLACE"
				}
			},
			"required": ["file", "patch"]
		}`),
	}
}

// ExecuteToolCall executes a single LLM tool call and returns the result.
// Uses the existing ToolResult type from tool_registry.go (Error is error type).
func (e *LLMToolExecutor) ExecuteToolCall(call ToolCall) ToolResult {
	switch call.Name {
	case "run_tests":
		return e.executeRunTests(call.Arguments)
	case "read_file":
		return e.executeReadFile(call.Arguments)
	case "apply_patch":
		return e.executeApplyPatch(call.Arguments)
	default:
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("unknown tool: %s", call.Name),
		}
	}
}

// executeRunTests runs go test on the specified package.
func (e *LLMToolExecutor) executeRunTests(args json.RawMessage) ToolResult {
	var params struct {
		Package string `json:"package"`
		Run     string `json:"run,omitempty"`
		Timeout string `json:"timeout,omitempty"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return ToolResult{Success: false, Error: fmt.Errorf("invalid arguments: %v", err)}
	}

	if params.Timeout == "" {
		params.Timeout = "30s"
	}

	cmdArgs := []string{"test", "-v"}
	if params.Run != "" {
		cmdArgs = append(cmdArgs, "-run", params.Run)
	}
	cmdArgs = append(cmdArgs, "-timeout", params.Timeout, params.Package)

	cmd := exec.Command("go", cmdArgs...)
	cmd.Dir = e.WorkingDir

	output, err := cmd.CombinedOutput()
	outputStr := string(output)

	if err != nil {
		return ToolResult{
			Success: false,
			Output:  outputStr,
			Error:   fmt.Errorf("tests failed (exit: %v)", err),
		}
	}

	return ToolResult{
		Success: true,
		Output:  outputStr,
	}
}

// executeReadFile reads a file's contents.
func (e *LLMToolExecutor) executeReadFile(args json.RawMessage) ToolResult {
	var params struct {
		Path      string `json:"path"`
		StartLine int    `json:"start_line,omitempty"`
		EndLine   int    `json:"end_line,omitempty"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return ToolResult{Success: false, Error: fmt.Errorf("invalid arguments: %v", err)}
	}

	fullPath := filepath.Join(e.WorkingDir, params.Path)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return ToolResult{Success: false, Error: fmt.Errorf("file not found: %s", params.Path)}
	}

	data, err := os.ReadFile(fullPath)
	if err != nil {
		return ToolResult{Success: false, Error: fmt.Errorf("read error: %v", err)}
	}

	content := string(data)
	lines := strings.Split(content, "\n")

	if params.StartLine > 0 {
		start := params.StartLine - 1
		if start >= len(lines) {
			return ToolResult{Success: false, Error: fmt.Errorf("start line %d exceeds file length %d", params.StartLine, len(lines))}
		}
		if params.EndLine > 0 && params.EndLine <= len(lines) {
			content = strings.Join(lines[start:params.EndLine], "\n")
		} else {
			content = strings.Join(lines[start:], "\n")
		}
	}

	return ToolResult{
		Success: true,
		Output:  content,
	}
}

// executeApplyPatch applies a SEARCH/REPLACE patch to a file.
// Delegates to the search_replace binary if available, otherwise does manual patching.
func (e *LLMToolExecutor) executeApplyPatch(args json.RawMessage) ToolResult {
	var params struct {
		File  string `json:"file"`
		Patch string `json:"patch"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return ToolResult{Success: false, Error: fmt.Errorf("invalid arguments: %v", err)}
	}

	fullPath := filepath.Join(e.WorkingDir, params.File)
	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		return ToolResult{Success: false, Error: fmt.Errorf("file not found: %s", params.File)}
	}

	// Read the original file for backup
	original, err := os.ReadFile(fullPath)
	if err != nil {
		return ToolResult{Success: false, Error: fmt.Errorf("read error: %v", err)}
	}

	// Try to use the search_replace binary if available
	searchReplaceBin, err := exec.LookPath("search_replace")
	if err == nil {
		cmd := exec.Command(searchReplaceBin,
			"-file", fullPath,
			"-patch", params.Patch,
			"-apply",
			"-gofmt=true",
			"-verify=false",
		)
		output, execErr := cmd.CombinedOutput()
		if execErr != nil {
			os.WriteFile(fullPath, original, 0644)
			return ToolResult{
				Success: false,
				Output:  string(output),
				Error:   fmt.Errorf("apply patch failed: %v", execErr),
			}
		}
		return ToolResult{
			Success: true,
			Output:  fmt.Sprintf("Patch applied successfully to %s\n%s", params.File, string(output)),
		}
	}

	// Fallback: manual patch application
	newContent := string(original)
	patch := params.Patch

	// Normalize patch: replace literal \n (backslash-n) with actual newlines
	// This handles JSON-encoded patches where \n was not decoded
	patch = strings.ReplaceAll(patch, "\\n", "\n")

	// Find the SEARCH/REPLACE markers
	searchMarker := "<<<<<<< SEARCH\n"
	sepMarker := "=======\n"
	replaceMarker := "\n>>>>>>> REPLACE"

	searchIdx := strings.Index(patch, searchMarker)
	if searchIdx == -1 {
		return ToolResult{Success: false, Error: fmt.Errorf("invalid patch format: missing SEARCH marker")}
	}
	searchStart := searchIdx + len(searchMarker)

	sepIdx := strings.Index(patch[searchStart:], sepMarker)
	if sepIdx == -1 {
		return ToolResult{Success: false, Error: fmt.Errorf("invalid patch format: missing separator")}
	}

	search := patch[searchStart : searchStart+sepIdx]
	replaceStart := searchStart + sepIdx + len(sepMarker)

	replaceEnd := strings.Index(patch[replaceStart:], replaceMarker)
	if replaceEnd == -1 {
		replaceEnd = strings.Index(patch[replaceStart:], ">>>>>>> REPLACE")
		if replaceEnd == -1 {
			return ToolResult{Success: false, Error: fmt.Errorf("invalid patch format: missing REPLACE marker")}
		}
	}

	replace := patch[replaceStart : replaceStart+replaceEnd]

	// Also normalize tab escapes
	search = strings.ReplaceAll(search, "\\t", "\t")
	replace = strings.ReplaceAll(replace, "\\t", "\t")

	if !strings.Contains(newContent, search) {
		return ToolResult{Success: false, Error: fmt.Errorf("search text not found in file:\n%s", truncateString(search, 200))}
	}

	newContent = strings.Replace(newContent, search, replace, 1)

	if err := os.WriteFile(fullPath, []byte(newContent), 0644); err != nil {
		os.WriteFile(fullPath, original, 0644)
		return ToolResult{Success: false, Error: fmt.Errorf("write error: %v", err)}
	}

	return ToolResult{
		Success: true,
		Output:  fmt.Sprintf("Patch applied successfully to %s", params.File),
	}
}

// RunAutoFix runs the LLM-driven auto-fix loop for a Go package.
// It runs tests, sends failures to an LLM with tool access, executes tool
// choices, and repeats until tests pass or max iterations are reached.
func RunAutoFix(pkgTarget string, workingDir string, callLLM func(prompt string, tools []LLMTool) ([]ToolCall, string, error)) error {
	maxRetries := 5
	executor := NewLLMToolExecutor(workingDir)

	fmt.Printf("🔧 Auto-fix loop for %s (max %d iterations)\n", pkgTarget, maxRetries)

	for i := 0; i < maxRetries; i++ {
		fmt.Printf("\n📋 Iteration %d/%d: Running go test %s...\n", i+1, maxRetries, pkgTarget)

		// Step 1: Run tests
		result := executor.executeRunTests(json.RawMessage(
			fmt.Sprintf(`{"package": "%s", "timeout": "60s"}`, pkgTarget),
		))

		if result.Success {
			fmt.Printf("✅ All tests passed!\n%s\n", truncateString(result.Output, 500))
			return nil
		}

		fmt.Printf("❌ Tests failed (iteration %d/%d)\n", i+1, maxRetries)

		// Step 2: Build the prompt with test failure context
		prompt := fmt.Sprintf(
			"The command 'go test -v %s' failed with the following output:\n\n%s\n\n"+
				"Inspect the relevant files using 'read_file' and apply code fixes using 'apply_patch' to fix the failing tests.\n"+
				"The patch must use the SEARCH/REPLACE format:\n"+
				"<<<<<<< SEARCH\ncode that currently exists\n=======\nreplacement code\n>>>>>>> REPLACE\n\n"+
				"Make sure the SEARCH block matches the existing code exactly.",
			pkgTarget,
			truncateString(result.Output, 4000),
		)

		// Step 3: Call LLM with tool definitions
		tools := executor.GetToolDefinitions()
		toolCalls, responseText, err := callLLM(prompt, tools)
		if err != nil {
			fmt.Printf("⚠️  LLM call failed: %v\n", err)
			continue
		}

		if len(toolCalls) == 0 {
			fmt.Printf("⚠️  LLM returned no tool calls. Response: %s\n", truncateString(responseText, 300))
			continue
		}

		// Step 4: Execute all tool calls from the LLM
		for _, call := range toolCalls {
			fmt.Printf("  🛠️  Executing: %s(%s)\n", call.Name, truncateString(string(call.Arguments), 100))
			toolResult := executor.ExecuteToolCall(call)
			if toolResult.Success {
				fmt.Printf("  ✅ %s succeeded\n", call.Name)
				if len(toolResult.Output) > 0 {
					fmt.Printf("     Output: %s\n", truncateString(toolResult.Output, 200))
				}
			} else {
				errMsg := ""
				if toolResult.Error != nil {
					errMsg = toolResult.Error.Error()
				}
				fmt.Printf("  ❌ %s failed: %s\n", call.Name, errMsg)
				if len(toolResult.Output) > 0 {
					fmt.Printf("     Output: %s\n", truncateString(toolResult.Output, 200))
				}
			}
		}
	}

	return fmt.Errorf("failed to fix tests within %d iterations", maxRetries)
}

// truncateString truncates a string to the specified max length, adding "..." if truncated.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
