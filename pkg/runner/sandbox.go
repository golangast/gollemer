// Package runner executes generated Go code in an isolated sandbox and
// turns `go test` output into structured feedback for code generation.
//
// ValidateGeneratedCode copies a project's module context (go.mod,
// sources, testdata, vendor tree) into a fresh temporary directory,
// overlays candidate files (candidates win on path conflicts), runs
// `go test -json ./...` with a 30-second timeout, and parses failures
// into TestErrors classified as syntax, compiler, or assertion
// problems so an LLM can tell a typo from a type error from a broken
// expectation.
//
// The sandbox is removed on return. The `go` toolchain must be on
// PATH. Timeouts kill the whole test process group (unix) so a
// runaway generated binary cannot linger.
package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Error categories for TestError.Category.
const (
	// CategorySyntax marks files that do not parse.
	CategorySyntax = "syntax"
	// CategoryCompiler marks files that parse but do not type-check or
	// build, including import errors and infrastructure failures (such
	// as a validation timeout) that prevent tests from running.
	CategoryCompiler = "compiler"
	// CategoryAssertion marks code that built and ran but failed a test
	// expectation (including panics).
	CategoryAssertion = "assertion"
)

// validationTimeout bounds `go test` so generated code cannot hang the
// feedback loop.
const validationTimeout = 30 * time.Second

// maxOutputBytes caps ExecutionResult.Output; the parsed Errors slice
// always carries the actionable details.
const maxOutputBytes = 32 * 1024

// TestError is one parsed failure: where it happened, what the tool
// said, and which bucket it falls in for the generation feedback loop.
type TestError struct {
	FilePath   string `json:"filePath"`
	LineNumber int    `json:"lineNumber"`
	Message    string `json:"message"`
	Category   string `json:"category"` // "syntax" | "compiler" | "assertion"
}

// ExecutionResult is the outcome of validating candidate files in the
// sandbox. Passed is true only when `go test` exited zero before the
// timeout.
type ExecutionResult struct {
	Passed bool        `json:"passed"`
	Output string      `json:"output"`
	Errors []TestError `json:"errors"`
}

// ValidateGeneratedCode validates candidate Go files against the module
// rooted at targetDir in an isolated temporary sandbox.
//
// targetDir must be a Go module root (contain go.mod); its full context
// is copied into the sandbox. candidateFiles maps module-relative paths
// ("pkg/foo/gen.go") to file contents and is written over the copied
// context. Relative paths that escape the sandbox ("../x") or absolute
// paths are rejected.
//
// The error return is reserved for validation that could not run at all
// (bad targetDir, copy failure, missing go toolchain, path escape). A
// `go test` failure is a successful validation with Passed=false.
func ValidateGeneratedCode(targetDir string, candidateFiles map[string]string) (*ExecutionResult, error) {
	if len(candidateFiles) == 0 {
		return nil, fmt.Errorf("runner: no candidate files provided")
	}
	sandbox, cleanup, err := CreateSandbox(targetDir, candidateFiles)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), validationTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "go", "test", "-json", "./...")
	cmd.Dir = sandbox
	// New process group so a timeout reaps the whole test tree (the
	// test binary is a child of `go`), not just the `go` process.
	// Unix-specific; this package targets unix environments.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	var execErr *exec.Error
	if errors.As(runErr, &execErr) {
		return nil, fmt.Errorf("runner: go toolchain not available: %w", execErr)
	}
	timedOut := ctx.Err() == context.DeadlineExceeded
	if timedOut && cmd.Process != nil {
		// The leader is already dead via CommandContext; kill the
		// rest of the group (runaway test binary).
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}

	text, failedTests := splitTestOutput(stdout.String(), stderr.String())
	result := &ExecutionResult{Output: truncate(text), Errors: []TestError{}}
	result.Errors = parseErrors(text, failedTests)
	result.Passed = runErr == nil && !timedOut
	if timedOut {
		result.Errors = append(result.Errors, TestError{
			Message:  fmt.Sprintf("go test timed out after %s (possible infinite loop or deadlock)", validationTimeout),
			Category: CategoryCompiler,
		})
	}
	if !result.Passed && len(result.Errors) == 0 {
		msg := lastMeaningfulLine(text)
		if msg == "" {
			msg = "go test failed with no parseable output"
		}
		result.Errors = append(result.Errors, TestError{Message: msg, Category: CategoryCompiler})
	}
	return result, nil
}

// CreateSandbox builds an isolated copy of the targetDir module with
// the given files applied on top (candidates win on conflicts), and
// returns the sandbox directory plus a cleanup function that removes
// it. Unlike ValidateGeneratedCode it does not run anything; the
// caller owns execution (e.g. benchmark validation).
func CreateSandbox(targetDir string, candidateFiles map[string]string) (sandboxDir string, cleanup func(), err error) {
	if targetDir == "" {
		return "", nil, fmt.Errorf("runner: empty targetDir")
	}
	info, err := os.Stat(targetDir)
	if err != nil {
		return "", nil, fmt.Errorf("runner: targetDir: %w", err)
	}
	if !info.IsDir() {
		return "", nil, fmt.Errorf("runner: targetDir %q is not a directory", targetDir)
	}
	if _, err := os.Stat(filepath.Join(targetDir, "go.mod")); err != nil {
		return "", nil, fmt.Errorf("runner: no go.mod in %q: targetDir must be a Go module root", targetDir)
	}

	sandbox, err := os.MkdirTemp("", "gollemer-sandbox-*")
	if err != nil {
		return "", nil, fmt.Errorf("runner: create sandbox: %w", err)
	}
	cleanup = func() { os.RemoveAll(sandbox) }

	if err := copyProject(targetDir, sandbox); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("runner: copy project context: %w", err)
	}
	for rel, content := range candidateFiles {
		path, err := safeJoin(sandbox, rel)
		if err != nil {
			cleanup()
			return "", nil, fmt.Errorf("runner: candidate %q: %w", rel, err)
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			cleanup()
			return "", nil, fmt.Errorf("runner: create dir for %q: %w", rel, err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			cleanup()
			return "", nil, fmt.Errorf("runner: write candidate %q: %w", rel, err)
		}
	}
	return sandbox, cleanup, nil
}

// copyProject replicates the module context into the sandbox: every
// regular file (go.mod, sources, testdata, vendor tree) with its
// permissions. .git is skipped; non-regular files (symlinks, sockets)
// are skipped so nothing outside targetDir leaks in.
func copyProject(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		if rel == ".git" || strings.HasPrefix(rel, ".git"+string(filepath.Separator)) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		if !d.Type().IsRegular() {
			return nil
		}
		return copyFile(path, target)
	})
}

// copyFile copies one regular file preserving its permission bits.
func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	info, err := os.Stat(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, info.Mode().Perm())
}

// safeJoin joins a candidate-relative path onto the sandbox, rejecting
// absolute paths and ".." escapes so generated content can never be
// written outside the sandbox.
func safeJoin(sandbox, rel string) (string, error) {
	if rel == "" {
		return "", fmt.Errorf("empty path")
	}
	if filepath.IsAbs(rel) {
		return "", fmt.Errorf("absolute paths not allowed")
	}
	clean := filepath.Clean(rel)
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes sandbox")
	}
	return filepath.Join(sandbox, clean), nil
}

// testEvent is the subset of `go test -json` output we care about.
type testEvent struct {
	Action  string `json:"Action"`
	Package string `json:"Package"`
	Test    string `json:"Test"`
	Output  string `json:"Output"`
}

// splitTestOutput unwraps `go test -json` events into human-readable
// text and harvests the names of failed tests (including subtests).
// Non-JSON lines and stderr are passed through verbatim.
func splitTestOutput(stdout, stderr string) (string, map[string]bool) {
	var sb strings.Builder
	failed := make(map[string]bool)
	for _, line := range strings.Split(stdout, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var ev testEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			sb.WriteString(line)
			sb.WriteByte('\n')
			continue
		}
		if ev.Action == "fail" && ev.Test != "" {
			failed[ev.Test] = true
		}
		if ev.Output != "" {
			sb.WriteString(ev.Output)
		}
	}
	if s := strings.TrimSpace(stderr); s != "" {
		sb.WriteString(s)
		sb.WriteByte('\n')
	}
	return sb.String(), failed
}

var (
	// Matches `path/to/file.go:12:4: message` and `file.go:12: message`.
	errLineRe = regexp.MustCompile(`^(.+?\.go):(\d+)(?::(\d+))?:\s*(.+)$`)
	// Matches `--- FAIL: TestName` (and indented subtests).
	failLineRe = regexp.MustCompile(`^--- FAIL:\s*(\S+)`)
	// Matches goroutine stack frames (`/path/f.go:9 +0x1a`), which are
	// context for a panic, not actionable errors.
	frameRe = regexp.MustCompile(`\.go:\d+\s+\+?0x[0-9a-fA-F]`)
)

// compilerMarkers are message fragments that identify type-check /
// build failures as opposed to failed test expectations.
var compilerMarkers = []string{
	"undefined", "cannot use", "cannot convert", "mismatched types",
	"not enough arguments", "too many arguments", "declared and not used",
	"imported and not used", "missing return", "is not a type",
	"invalid operation", "cannot assign", "multiple-value",
	"no new variables", "redeclared", "not defined",
}

// parseErrors converts unwrapped `go test` text into structured errors.
// Classification: syntax errors by message; `file_test.go:line` lines
// are assertions when tests actually ran (a package that does not
// compile produces no test events, so its errors stay compiler);
// everything else with a file:line is a compiler error. Each failed
// test also yields one summary error carrying its name.
func parseErrors(text string, failedTests map[string]bool) []TestError {
	errs := []TestError{}
	seen := make(map[string]bool)
	add := func(e TestError) {
		key := e.Category + "|" + e.FilePath + "|" + strconv.Itoa(e.LineNumber) + "|" + e.Message
		if seen[key] {
			return
		}
		seen[key] = true
		errs = append(errs, e)
	}

	sawTestFail := len(failedTests) > 0
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if m := failLineRe.FindStringSubmatch(line); m != nil {
			failedTests[m[1]] = true
			sawTestFail = true
			continue
		}
		if strings.HasPrefix(line, "panic:") {
			sawTestFail = true
			add(TestError{Message: line, Category: CategoryAssertion})
			continue
		}
		if frameRe.MatchString(line) {
			continue
		}
		l := strings.TrimPrefix(line, "vet: ")
		m := errLineRe.FindStringSubmatch(l)
		if m == nil {
			continue
		}
		file := strings.TrimPrefix(m[1], "./")
		var lineNo int
		fmt.Sscanf(m[2], "%d", &lineNo)
		msg := m[4]
		category := CategoryCompiler
		switch {
		case isSyntaxError(msg):
			category = CategorySyntax
		case sawTestFail && strings.HasSuffix(file, "_test.go") && !looksLikeCompilerError(msg):
			category = CategoryAssertion
		}
		add(TestError{FilePath: file, LineNumber: lineNo, Message: msg, Category: category})
	}

	names := make([]string, 0, len(failedTests))
	for name := range failedTests {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		add(TestError{Message: "FAIL: " + name, Category: CategoryAssertion})
	}
	return errs
}

// isSyntaxError reports whether a compiler message is a parse failure.
func isSyntaxError(msg string) bool {
	lower := strings.ToLower(msg)
	return strings.Contains(lower, "syntax error") || strings.HasPrefix(lower, "expected")
}

// looksLikeCompilerError reports whether a message reads as a
// type-check/build failure rather than a failed test expectation.
func looksLikeCompilerError(msg string) bool {
	lower := strings.ToLower(msg)
	for _, marker := range compilerMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

// lastMeaningfulLine returns the last non-empty output line that is not
// a bare go test summary marker, for failures with no parseable errors.
func lastMeaningfulLine(text string) string {
	lines := strings.Split(text, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		l := strings.TrimSpace(lines[i])
		if l == "" || l == "FAIL" || strings.HasPrefix(l, "ok ") || strings.HasPrefix(l, "? ") {
			continue
		}
		return l
	}
	return ""
}

// truncate caps output length; parsed Errors carry the details.
func truncate(s string) string {
	if len(s) <= maxOutputBytes {
		return s
	}
	return s[:maxOutputBytes] + "\n[output truncated]"
}

/*
Example: validate a generated test file against a small project.

	package main

	import (
		"fmt"
		"log"
		"os"
		"path/filepath"

		"github.com/golangast/gollemer/pkg/runner"
	)

	func main() {
		// 1. Project context: a module with one source file.
		proj, err := os.MkdirTemp("", "runner-proj-*")
		if err != nil {
			log.Fatal(err)
		}
		defer os.RemoveAll(proj)
		write := func(name, content string) {
			if err := os.WriteFile(filepath.Join(proj, name), []byte(content), 0o644); err != nil {
				log.Fatal(err)
			}
		}
		write("go.mod", "module example.com/demo\n\ngo 1.26.0\n")
		write("add.go", "package demo\n\nfunc Add(a, b int) int { return a + b }\n")

		// 2. Candidate: a generated test with a deliberate assertion failure.
		candidates := map[string]string{
			"add_test.go": `package demo

	import "testing"

	func TestAdd(t *testing.T) {
		if got := Add(2, 2); got != 5 {
			t.Errorf("got %d, want 5", got)
		}
	}
	`,
		}

		// 3. Validate in an isolated sandbox.
		res, err := runner.ValidateGeneratedCode(proj, candidates)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("passed:", res.Passed)
		for _, e := range res.Errors {
			fmt.Printf("[%s] %s:%d: %s\n", e.Category, e.FilePath, e.LineNumber, e.Message)
		}
	}
*/
