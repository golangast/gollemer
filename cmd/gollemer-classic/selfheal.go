// Self-healing Go code generation pipeline.
//
// GenerateAndSelfHeal asks an LLM for candidate Go files, validates
// them in the runner sandbox with `go test`, and feeds compiler/test
// failures back to the LLM for repair until the code passes or
// attempts run out.
//
// The LLM is reached through the LLMClient interface. By default
// callLLM uses an OpenAI-compatible HTTP client configured from the
// environment (LLM_API_URL, LLM_API_KEY, LLM_MODEL; defaults to a local
// Ollama at http://localhost:11434/v1). Tests and demos swap in
// MockClient via SetLLMClient. No API keys are hardcoded or logged.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/format"
	"go/parser"
	"go/token"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/golangast/gollemer/pkg/ast"
	"github.com/golangast/gollemer/pkg/runner"
)

// ---------------------------------------------------------------------------
// LLM client
// ---------------------------------------------------------------------------

// LLMClient generates text from a prompt. Implementations must be safe
// for sequential use; GenerateAndSelfHeal calls Complete serially.
type LLMClient interface {
	Complete(prompt string) (string, error)
	Describe() string
}

// openAICompatClient talks to any OpenAI-compatible chat-completions
// endpoint (OpenAI, Ollama's /v1, OpenRouter, ...). Pure stdlib HTTP.
type openAICompatClient struct {
	baseURL string
	apiKey  string
	model   string
	http    *http.Client
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// NewLLMClientFromEnv builds the default client from LLM_API_URL,
// LLM_API_KEY and LLM_MODEL. Defaults target a local Ollama.
func NewLLMClientFromEnv() LLMClient {
	return &openAICompatClient{
		baseURL: strings.TrimRight(envOr("LLM_API_URL", "http://localhost:11434/v1"), "/"),
		apiKey:  os.Getenv("LLM_API_KEY"),
		model:   envOr("LLM_MODEL", "qwen2.5-coder:7b"),
		http:    &http.Client{Timeout: 120 * time.Second},
	}
}

// Complete sends one chat request and returns the assistant's text.
func (c *openAICompatClient) Complete(prompt string) (string, error) {
	body, err := json.Marshal(chatRequest{
		Model: c.model,
		Messages: []chatMessage{
			{Role: "system", Content: "You are a senior Go engineer. Follow the requested output format exactly."},
			{Role: "user", Content: prompt},
		},
		Temperature: 0.2,
		Stream:      false,
	})
	if err != nil {
		return "", fmt.Errorf("llm: encode request: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("llm: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("llm: POST %s: %w (is an LLM serving at LLM_API_URL? see -mock for a demo)", c.baseURL, err)
	}
	defer resp.Body.Close()
	var cr chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return "", fmt.Errorf("llm: decode response (status %s): %w", resp.Status, err)
	}
	if resp.StatusCode != http.StatusOK {
		msg := ""
		if cr.Error != nil {
			msg = ": " + cr.Error.Message
		}
		return "", fmt.Errorf("llm: status %s%s", resp.Status, msg)
	}
	if cr.Error != nil {
		return "", fmt.Errorf("llm: API error: %s", cr.Error.Message)
	}
	if len(cr.Choices) == 0 {
		return "", fmt.Errorf("llm: empty response (no choices)")
	}
	return strings.TrimSpace(cr.Choices[0].Message.Content), nil
}

// Describe names the endpoint without exposing the API key.
func (c *openAICompatClient) Describe() string {
	return fmt.Sprintf("openai-compatible %s model=%s", c.baseURL, c.model)
}

// MockClient replays scripted responses, one per call; the last
// response repeats once the script is exhausted. For tests and demos.
type MockClient struct {
	mu        sync.Mutex
	calls     int
	responses []string
}

// NewMockClient builds a mock that returns responses in order.
func NewMockClient(responses ...string) *MockClient {
	return &MockClient{responses: responses}
}

// Complete returns the next scripted response.
func (m *MockClient) Complete(prompt string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.responses) == 0 {
		return "", fmt.Errorf("mock: no responses scripted")
	}
	i := m.calls
	if i >= len(m.responses) {
		i = len(m.responses) - 1
	}
	m.calls++
	return m.responses[i], nil
}

// Describe identifies the mock.
func (m *MockClient) Describe() string {
	return "mock (scripted responses)"
}

// llmClient is the active client. Initialized from the environment;
// replace via SetLLMClient (tests, demos, alternate providers).
var llmClient LLMClient = NewLLMClientFromEnv()

// SetLLMClient swaps the client used by callLLM.
func SetLLMClient(c LLMClient) {
	llmClient = c
}

// callLLM sends prompt to the configured LLM and returns its raw text.
func callLLM(prompt string) (string, error) {
	if llmClient == nil {
		return "", fmt.Errorf("llm: no client configured")
	}
	return llmClient.Complete(prompt)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

// maxPromptChunks bounds how many code chunks enter the generation
// prompt; callers select relevance, this caps cost.
const maxPromptChunks = 80

// buildSystemPrompt renders the codebase context and relevant chunks
// into the system prompt for the initial generation attempt.
func buildSystemPrompt(codeCtx *ast.CodebaseContext, chunks []ast.CodeChunk) string {
	var sb strings.Builder
	sb.WriteString("You are a senior Go software engineer. Generate complete, working Go source files for the task below.\n\n")
	if codeCtx != nil {
		fmt.Fprintf(&sb, "Module: %s\nPackage: %s\n\n", codeCtx.ModulePath, codeCtx.PackageName)
		if len(codeCtx.Imports) > 0 {
			sb.WriteString("Imports already used in this package (path -> local name):\n")
			paths := make([]string, 0, len(codeCtx.Imports))
			for p := range codeCtx.Imports {
				paths = append(paths, p)
			}
			sort.Strings(paths)
			for _, p := range paths {
				fmt.Fprintf(&sb, "  %q -> %s\n", p, codeCtx.Imports[p])
			}
			sb.WriteString("\n")
		}
		if len(codeCtx.Structs) > 0 {
			sb.WriteString("Structs:\n")
			for _, st := range codeCtx.Structs {
				if st.DocComment != "" {
					fmt.Fprintf(&sb, "  // %s\n", firstLine(st.DocComment))
				}
				fmt.Fprintf(&sb, "  type %s struct {\n", st.Name)
				for _, f := range st.Fields {
					tag := ""
					if f.JSONTag != "" {
						tag = fmt.Sprintf(" `json:\"%s\"`", f.JSONTag)
					}
					fmt.Fprintf(&sb, "    %s %s%s\n", f.Name, f.TypeString, tag)
				}
				sb.WriteString("  }\n")
			}
			sb.WriteString("\n")
		}
		if len(codeCtx.Interfaces) > 0 {
			sb.WriteString("Interfaces:\n")
			for _, iface := range codeCtx.Interfaces {
				fmt.Fprintf(&sb, "  type %s interface {\n", iface.Name)
				for _, m := range iface.Methods {
					fmt.Fprintf(&sb, "    %s(%s)", m.Name, strings.Join(m.Parameters, ", "))
					switch len(m.Results) {
					case 0:
					case 1:
						fmt.Fprintf(&sb, " %s", m.Results[0])
					default:
						fmt.Fprintf(&sb, " (%s)", strings.Join(m.Results, ", "))
					}
					sb.WriteString("\n")
				}
				sb.WriteString("  }\n")
			}
			sb.WriteString("\n")
		}
	}
	// Repository style mining is best-effort: mined conventions make
	// generated code match the repo's habits, but mining must never
	// break generation.
	if codeCtx != nil {
		if style, serr := ast.InferRepositoryStyle(codeCtx); serr == nil && style != nil {
			if g := style.SystemPromptGuidelines(); g != "" {
				sb.WriteString("Repository conventions (follow these):\n")
				sb.WriteString(g)
				sb.WriteString("\n")
			}
		} else if serr != nil {
			fmt.Printf("[selfheal] style mining skipped: %v\n", serr)
		}
	}
	if len(chunks) > 0 {
		sb.WriteString("Relevant code:\n")
		n := len(chunks)
		if n > maxPromptChunks {
			n = maxPromptChunks
		}
		for _, c := range chunks[:n] {
			fmt.Fprintf(&sb, "--- %s: %s (%s, lines %d-%d) ---\n%s\n",
				c.FilePath, c.SymbolName, c.Kind, c.StartLine, c.EndLine, c.CodeContent)
		}
		if len(chunks) > maxPromptChunks {
			fmt.Fprintf(&sb, "... (%d more chunks omitted)\n", len(chunks)-maxPromptChunks)
		}
		sb.WriteString("\n")
	}
	sb.WriteString(`Output format (strict):
Return one or more file blocks, each EXACTLY like this, with a module-relative path:

=== FILE: pkg/store/memory.go ===
package store
...

Rules:
- Return ONLY the file blocks. No explanations, no markdown fences.
- Paths are relative to the module root, e.g. "store.go" or "pkg/store/memory.go".
- The code must compile with go build ./... and pass go test ./....
- Do not redeclare symbols that already exist in the package.
`)
	return sb.String()
}

// buildRepairPrompt renders sandbox failures and the previous code
// into a follow-up repair prompt.
func buildRepairPrompt(task string, candidates map[string]string, errs []runner.TestError) string {
	var sb strings.Builder
	sb.WriteString("The generated Go code failed validation with the following compiler/test errors:\n")
	for _, e := range errs {
		fmt.Fprintf(&sb, "[File: %s, Line: %d, Error: %s]\n", e.FilePath, e.LineNumber, e.Message)
	}
	sb.WriteString("\nPrevious Code:\n")
	names := sortedKeys(candidates)
	if len(names) == 0 {
		sb.WriteString("(none)\n")
	}
	for _, name := range names {
		fmt.Fprintf(&sb, "=== FILE: %s ===\n%s\n", name, candidates[name])
	}
	fmt.Fprintf(&sb, "\nOriginal task: %s\n\n", task)
	sb.WriteString("Please fix the Go syntax/type errors and return only the corrected Go code, ")
	sb.WriteString("using the same === FILE: <module-relative path> === format (one block per file, no explanations, no markdown fences).")
	return sb.String()
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSpace(strings.TrimPrefix(s, "//"))
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

// fileMarkerRe matches the per-file delimiter the LLM is instructed to
// emit: === FILE: <module-relative path> ===
var fileMarkerRe = regexp.MustCompile(`^===\s*FILE:\s*(.+?)\s*===\s*$`)

// parseFilesFromResponse splits a raw LLM response into module-relative
// path -> Go source. Markdown fences are stripped; chatter before the
// first marker is ignored. Paths are validated with the same jail
// rules the sandbox enforces.
func parseFilesFromResponse(resp string) (map[string]string, error) {
	resp = stripCodeFences(resp)
	files := make(map[string]string)
	var cur string
	var sb strings.Builder
	flush := func() {
		if cur != "" {
			files[cur] = strings.Trim(sb.String(), "\n") + "\n"
			sb.Reset()
		}
	}
	for _, line := range strings.Split(resp, "\n") {
		if m := fileMarkerRe.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
			flush()
			path := filepath.Clean(m[1])
			if err := validCandidatePath(path); err != nil {
				return nil, fmt.Errorf("invalid file path %q: %w", m[1], err)
			}
			cur = path
			continue
		}
		if cur == "" {
			continue
		}
		sb.WriteString(line)
		sb.WriteByte('\n')
	}
	flush()
	if len(files) == 0 {
		return nil, fmt.Errorf("no === FILE: <path> === blocks found")
	}
	return files, nil
}

// stripCodeFences removes markdown fence lines LLMs like to wrap code
// in; the file markers carry the structure.
func stripCodeFences(s string) string {
	lines := strings.Split(s, "\n")
	out := lines[:0]
	for _, line := range lines {
		t := strings.TrimSpace(line)
		if t == "```" || t == "```go" {
			continue
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

// validCandidatePath mirrors the sandbox jail: module-relative .go
// paths only, no absolute paths, no ".." escapes.
func validCandidatePath(p string) error {
	if p == "" {
		return fmt.Errorf("empty path")
	}
	if filepath.IsAbs(p) {
		return fmt.Errorf("absolute paths not allowed")
	}
	if p == ".." || strings.HasPrefix(p, ".."+string(filepath.Separator)) {
		return fmt.Errorf("path escapes module")
	}
	if filepath.Ext(p) != ".go" {
		return fmt.Errorf("not a Go file")
	}
	return nil
}

// formatCandidates runs go/format over each candidate. Files that do
// not parse are passed through raw so the sandbox reports the exact
// syntax error for the repair loop.
func formatCandidates(files map[string]string) map[string]string {
	out := make(map[string]string, len(files))
	for _, name := range sortedKeys(files) {
		code := files[name]
		if formatted, err := format.Source([]byte(code)); err == nil {
			out[name] = string(formatted)
		} else {
			fmt.Printf("[selfheal] warning: gofmt failed for %s (%v); validating raw source\n", name, err)
			out[name] = code
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Self-healing loop
// ---------------------------------------------------------------------------

// GenerateAndSelfHeal generates Go files for prompt with the LLM and
// repairs them against the runner sandbox until `go test` passes or
// maxAttempts is exhausted. codeCtx and chunks seed the generation
// prompt (both may be nil/empty). Returns the validated file contents
// keyed by module-relative path.
func GenerateAndSelfHeal(targetDir string, prompt string, codeCtx *ast.CodebaseContext, chunks []ast.CodeChunk, maxAttempts int) (map[string]string, error) {
	if targetDir == "" {
		return nil, fmt.Errorf("selfheal: empty targetDir")
	}
	if maxAttempts < 1 {
		return nil, fmt.Errorf("selfheal: maxAttempts must be >= 1, got %d", maxAttempts)
	}
	system := buildSystemPrompt(codeCtx, chunks)

	var candidates map[string]string
	var lastErrs []runner.TestError
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		var raw string
		var err error
		if attempt == 1 {
			fmt.Printf("[selfheal] attempt %d/%d: generating code...\n", attempt, maxAttempts)
			raw, err = callLLM(system + "\nTask:\n" + prompt + "\n")
		} else {
			fmt.Printf("[selfheal] attempt %d/%d: requesting repair...\n", attempt, maxAttempts)
			raw, err = callLLM(buildRepairPrompt(prompt, candidates, lastErrs))
		}
		if err != nil {
			return nil, fmt.Errorf("selfheal: attempt %d: LLM call failed: %w", attempt, err)
		}
		files, perr := parseFilesFromResponse(raw)
		if perr != nil {
			// A format violation is repairable: feed it back like a
			// validation error instead of dying.
			lastErrs = []runner.TestError{{
				Message:  fmt.Sprintf("LLM response format error: %v. Reply with one === FILE: <module-relative path> === block per file and no other text.", perr),
				Category: runner.CategoryCompiler,
			}}
			fmt.Printf("[selfheal] attempt %d/%d: response had no file blocks; asking for repair...\n", attempt, maxAttempts)
			continue
		}
		candidates = formatCandidates(files)
		fmt.Printf("[selfheal] attempt %d/%d: validating %d file(s) in sandbox...\n", attempt, maxAttempts, len(candidates))
		res, verr := runner.ValidateGeneratedCode(targetDir, candidates)
		if verr != nil {
			return nil, fmt.Errorf("selfheal: attempt %d: sandbox validation failed: %w", attempt, verr)
		}
		if res.Passed {
			fmt.Printf("[selfheal] validation PASSED on attempt %d\n", attempt)
			return candidates, nil
		}
		lastErrs = res.Errors
		fmt.Printf("[selfheal] attempt %d/%d: FAILED with %d error(s):\n", attempt, maxAttempts, len(lastErrs))
		for _, e := range lastErrs {
			fmt.Printf("[selfheal]   [%s] %s:%d: %s\n", e.Category, e.FilePath, e.LineNumber, e.Message)
		}
	}
	return nil, fmt.Errorf("selfheal: all %d attempts failed; last errors:\n%s", maxAttempts, formatErrors(lastErrs))
}

// formatErrors renders errors for terminal output and final errors.
func formatErrors(errs []runner.TestError) string {
	var sb strings.Builder
	for _, e := range errs {
		fmt.Fprintf(&sb, "  [%s] %s:%d: %s\n", e.Category, e.FilePath, e.LineNumber, e.Message)
	}
	return sb.String()
}

// ---------------------------------------------------------------------------
// Context loading helpers
// ---------------------------------------------------------------------------

// loadChunks parses every .go file under dir into semantic chunks for
// prompt context. Unparseable files are skipped. At most max chunks are
// returned.
func loadChunks(dir string, max int) []ast.CodeChunk {
	var chunks []ast.CodeChunk
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if d.Name() == ".git" || d.Name() == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(d.Name(), ".go") {
			return nil
		}
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(dir, path)
		cs, err := ast.ChunkFile(fset, f, rel)
		if err != nil {
			return nil
		}
		chunks = append(chunks, cs...)
		return nil
	})
	if len(chunks) > max {
		chunks = chunks[:max]
	}
	return chunks
}

func sortedKeys(m map[string]string) []string {
	names := make([]string, 0, len(m))
	for name := range m {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
