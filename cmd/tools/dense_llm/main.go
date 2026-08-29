// Command dense_llm runs interactive inference on the simplified dense MLP
// trained on the basic go_edit_agent update-command corpus.
//
// For social prompts it prints the corpus response; for code_update prompts it
// prints the transformed code snippet produced by the matched command and, when
// a target Go file is available, applies the change directly to disk.
//
// The interactive shell maintains multiple named conversations so responses can
// be context-aware across multiple turns and across multiple independent
// dialogue threads (e.g. follow-up questions, references to previous code edits,
// and social continuity). Each conversation can target a different Go file,
// so multiple conversations can independently update different files.
//
// Conversation management commands (interactive mode):
//
//	/new [name]        start a new conversation (auto-generates a name if omitted)
//	/list              list all conversations
//	/switch <name>     switch to an existing conversation
//	/delete <name>     delete a conversation
//	/current           show the active conversation name
//	/file <path>       set the target Go file to update for the active conversation
//	/help              show this help
//
// Usage:
//
//	go run ./cmd/tools/dense_llm -model=data/models/dense/model.gob \
//	    [-prompt "..."] [-file=path/to/target.go]
package main

import (
	"bufio"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/dense"
)

// ChatTurn is a single user/assistant exchange in the conversation history.
type ChatTurn struct {
	User      string
	Assistant string
	Type      string // "social" or "code_update"
}

// Conversation tracks the multi-turn dialogue state for a single thread.
type Conversation struct {
	ID         string
	Turns      []ChatTurn
	TargetFile string // Go file this conversation applies code updates to
}

// AddTurn appends a new exchange to the conversation history.
func (c *Conversation) AddTurn(user, assistant, cmdType string) {
	c.Turns = append(c.Turns, ChatTurn{User: user, Assistant: assistant, Type: cmdType})
	// Keep only the last 10 turns to bound memory.
	if len(c.Turns) > 10 {
		c.Turns = c.Turns[len(c.Turns)-10:]
	}
}

// LastUser returns the most recent user message, or "" if none.
func (c *Conversation) LastUser() string {
	if len(c.Turns) == 0 {
		return ""
	}
	return c.Turns[len(c.Turns)-1].User
}

// LastAssistant returns the most recent assistant response, or "" if none.
func (c *Conversation) LastAssistant() string {
	if len(c.Turns) == 0 {
		return ""
	}
	return c.Turns[len(c.Turns)-1].Assistant
}

// LastType returns the type of the most recent exchange.
func (c *Conversation) LastType() string {
	if len(c.Turns) == 0 {
		return ""
	}
	return c.Turns[len(c.Turns)-1].Type
}

// HasContext returns true when there is at least one prior exchange.
func (c *Conversation) HasContext() bool {
	return len(c.Turns) > 0
}

// ContextString builds a compact summary of recent history for context-aware
// response generation.
func (c *Conversation) ContextString() string {
	if len(c.Turns) == 0 {
		return ""
	}
	var sb strings.Builder
	for i, t := range c.Turns {
		if i > 0 {
			sb.WriteString(" | ")
		}
		sb.WriteString(fmt.Sprintf("U:%s A:%s", t.User, t.Assistant))
	}
	return sb.String()
}

// SetTargetFile validates and sets this conversation's target file for create,
// edit, or delete operations. Go-specific edits still use .go target files.
func (c *Conversation) SetTargetFile(path string) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("target file path cannot be empty")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("resolve target file path: %w", err)
	}
	c.TargetFile = abs
	return nil
}

// TargetGoFile returns the conversation's target file, or "" if none set.
func (c *Conversation) TargetGoFile() string {
	return c.TargetFile
}

// ConversationManager holds multiple named conversations.
type ConversationManager struct {
	conversations map[string]*Conversation
	active        string
	nextID        int
}

// NewConversationManager creates a manager with a default conversation.
func NewConversationManager() *ConversationManager {
	m := &ConversationManager{
		conversations: make(map[string]*Conversation),
		nextID:        1,
	}
	m.New("default")
	return m
}

// New creates a new conversation with the given name (or auto-generates one).
// Returns the conversation and its name.
func (m *ConversationManager) New(name string) (*Conversation, string) {
	if name == "" {
		name = fmt.Sprintf("conv-%d", m.nextID)
		m.nextID++
	}
	// Ensure uniqueness.
	base := name
	for i := 2; ; i++ {
		if _, exists := m.conversations[name]; !exists {
			break
		}
		name = fmt.Sprintf("%s-%d", base, i)
	}
	conv := &Conversation{ID: name}
	m.conversations[name] = conv
	m.active = name
	return conv, name
}

// Get returns the active conversation.
func (m *ConversationManager) Get() *Conversation {
	if conv, ok := m.conversations[m.active]; ok {
		return conv
	}
	// Fallback: create a new one.
	conv, _ := m.New("")
	return conv
}

// GetByName returns a conversation by name, or nil.
func (m *ConversationManager) GetByName(name string) *Conversation {
	if conv, ok := m.conversations[name]; ok {
		return conv
	}
	return nil
}

// Switch sets the active conversation by name. Returns false if not found.
func (m *ConversationManager) Switch(name string) bool {
	if _, ok := m.conversations[name]; ok {
		m.active = name
		return true
	}
	return false
}

// Delete removes a conversation by name. Returns false if not found or if it's
// the last remaining conversation.
func (m *ConversationManager) Delete(name string) bool {
	if len(m.conversations) <= 1 {
		return false
	}
	if _, ok := m.conversations[name]; !ok {
		return false
	}
	delete(m.conversations, name)
	if m.active == name {
		// Switch to any remaining conversation.
		for k := range m.conversations {
			m.active = k
			break
		}
	}
	return true
}

// Active returns the active conversation name.
func (m *ConversationManager) Active() string {
	return m.active
}

// List returns all conversation names in insertion order.
func (m *ConversationManager) List() []string {
	names := make([]string, 0, len(m.conversations))
	for k := range m.conversations {
		names = append(names, k)
	}
	return names
}

// isFollowUp detects if the current prompt is a follow-up to the previous turn
// (e.g. "what did you say", "can you repeat", "show me again", "more").
func isFollowUp(prompt string, conv *Conversation) bool {
	lower := strings.ToLower(strings.TrimSpace(prompt))
	followUpPhrases := []string{
		"what did you say", "what did you mean", "can you repeat",
		"show me again", "say that again", "more", "again",
		"what was that", "explain that", "what does that mean",
		"how does that work", "can you elaborate", "tell me more",
		"what about", "and then", "what next", "what now",
		"what did i say", "what was my last message",
	}
	for _, phrase := range followUpPhrases {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

// isGreeting detects common greeting/social openers.
func isGreeting(prompt string) bool {
	lower := strings.ToLower(strings.TrimSpace(prompt))
	greetings := []string{
		"hello", "hi", "hey", "good morning", "good evening", "good afternoon",
		"how are you", "how's it going", "what's up", "yo", "greetings",
	}
	for _, g := range greetings {
		if strings.Contains(lower, g) {
			return true
		}
	}
	return false
}

// isFarewell detects common farewell/closing phrases.
func isFarewell(prompt string) bool {
	lower := strings.ToLower(strings.TrimSpace(prompt))
	farewells := []string{
		"bye", "goodbye", "see you", "farewell", "good night", "goodnight",
		"exit", "quit", "stop", "end conversation", "i'm done", "i am done",
		"that's all", "thats all", "no more", "done for now",
	}
	for _, f := range farewells {
		if strings.Contains(lower, f) {
			return true
		}
	}
	return false
}

// isThanks detects gratitude expressions.
func isThanks(prompt string) bool {
	lower := strings.ToLower(strings.TrimSpace(prompt))
	thanks := []string{
		"thank", "thanks", "appreciate", "grateful", "much obliged",
	}
	for _, t := range thanks {
		if strings.Contains(lower, t) {
			return true
		}
	}
	return false
}

// buildContextAwareResponse generates a response that takes conversation
// history into account.
func functionSnippetFromPrompt(prompt string) string {
	lower := strings.ToLower(strings.TrimSpace(prompt))
	candidates := []string{"add function ", "create function ", "new function ", "insert function ", "define function "}
	for _, prefix := range candidates {
		if idx := strings.Index(lower, prefix); idx >= 0 {
			rest := lower[idx+len(prefix):]
			rest = strings.TrimSpace(rest)
			rest = strings.ReplaceAll(rest, " to file ", " ")
			rest = strings.ReplaceAll(rest, " in file ", " ")
			rest = strings.ReplaceAll(rest, " to /file ", " ")
			rest = strings.ReplaceAll(rest, " in /file ", " ")
			rest = strings.ReplaceAll(rest, " to folder ", " ")
			rest = strings.ReplaceAll(rest, " in folder ", " ")
			rest = strings.ReplaceAll(rest, " to directory ", " ")
			rest = strings.ReplaceAll(rest, " in directory ", " ")
			rest = strings.TrimSpace(rest)
			fields := strings.Fields(rest)
			if len(fields) == 0 {
				continue
			}
			name := fields[0]
			if name == "to" || name == "in" || name == "file" || name == "folder" || name == "directory" {
				continue
			}
			name = strings.Trim(name, "_/.- ")
			if name == "" {
				continue
			}
			name = strings.Map(func(r rune) rune {
				if r == '/' || r == '.' || r == '-' || r == '_' || r == ' ' || r == '\t' || r == '\n' || r == '\r' {
					return -1
				}
				return r
			}, name)
			if name == "" {
				continue
			}
			funcName := strings.ToUpper(name[:1]) + name[1:]
			return fmt.Sprintf("func %s() {\n\t// TODO: implement\n}\n", funcName)
		}
	}
	return ""
}

// extractGoSnippetFromPrompt handles two cases that functionSnippetFromPrompt
// misses:
//
//  1. The user pastes raw Go code directly into the prompt (e.g. a struct
//     literal assignment, a variable declaration, an expression). The code is
//     detected by attempting to parse it as a Go statement inside a synthetic
//     function body.
//
//  2. "add X to file Y" / "add X to jim/jim.go" patterns where X is the Go
//     snippet. The code is extracted as everything between the leading action
//     verb and the trailing " to <file>" clause.
//
// Returns the raw Go snippet string, or "" when nothing suitable is detected.
func extractGoSnippetFromPrompt(prompt string) string {
	trimmed := strings.TrimSpace(prompt)

	// ── Case 1: "add X to file Y" / "add X to Y.go" ────────────────────────
	// Detect a leading action verb followed by Go-like code, then an optional
	// " to <path>" clause.
	actionPrefixes := []string{"add ", "insert ", "append ", "put "}
	for _, prefix := range actionPrefixes {
		if !strings.HasPrefix(strings.ToLower(trimmed), prefix) {
			continue
		}
		rest := strings.TrimSpace(trimmed[len(prefix):])

		// Find the " to <something>" or " in <something>" tail that names the
		// target file. Search from the end so the snippet can itself contain
		// the word "to" (e.g. struct field values).
		for _, sep := range []string{" to file ", " in file ", " to ", " in "} {
			if idx := strings.LastIndex(strings.ToLower(rest), sep); idx > 0 {
				candidate := strings.TrimSpace(rest[:idx])
				if isValidGoSnippet(candidate) {
					return candidate
				}
				// Try stripping a leading backtick fence the user may have added.
				candidate = strings.Trim(candidate, "`")
				if isValidGoSnippet(candidate) {
					return candidate
				}
			}
		}
	}

	// ── Case 2: prompt IS raw Go code (no leading verb) ─────────────────────
	// Strip optional backtick fences.
	candidate := strings.Trim(trimmed, "`")
	if isValidGoSnippet(candidate) {
		// Only treat it as raw code if it looks like Go syntax (contains := or
		// a composite literal or a type-qualified identifier).
		if looksLikeGoCode(candidate) {
			return candidate
		}
	}

	return ""
}

// isValidGoSnippet reports whether the string can be parsed as one or more
// statements inside a synthetic Go function body. It accepts both single-line
// and multi-line snippets.
func isValidGoSnippet(code string) bool {
	code = strings.TrimSpace(code)
	if code == "" {
		return false
	}
	src := fmt.Sprintf("package p\nfunc _(){\n%s\n}", code)
	fset := token.NewFileSet()
	_, err := parser.ParseFile(fset, "", src, 0)
	return err == nil
}

// looksLikeGoCode returns true when the snippet contains syntactic markers
// that strongly suggest it is Go code rather than an English sentence.
func looksLikeGoCode(code string) bool {
	// Short-assignment or regular assignment.
	if strings.Contains(code, ":=") || strings.Contains(code, "= ") {
		return true
	}
	// Composite literal: Foo{ or pkg.Foo{
	if strings.Contains(code, "{") && (strings.Contains(code, ":") || strings.Contains(code, "}")) {
		return true
	}
	// Explicit var / const / type declaration.
	lower := strings.ToLower(strings.TrimSpace(code))
	for _, kw := range []string{"var ", "const ", "type ", "return ", "if ", "for "} {
		if strings.HasPrefix(lower, kw) {
			return true
		}
	}
	return false
}

func buildContextAwareResponse(prompt string, conv *Conversation, model *dense.DenseModel, examples []dense.CommandExample) string {
	cmdType := dense.ClassifyCommandType(prompt)
	if cmdType == "social" {
		input := dense.BagOfWords(prompt, dense.CommandVocab)
		preds := model.Predict([][]float32{input})
		if len(preds) > 0 {
			label := preds[0]
			if label >= 0 && label < len(dense.CommandLabels) {
				candidate := dense.CommandLabels[label]
				if candidate != "social" {
					cmdType = candidate
				}
			}
		}
	}

	if cmdType == "code_update" {
		// Try extracting a Go snippet from the prompt: handles both
		// "add function foo" patterns and raw Go code / "add X to file Y".
		if snippet := functionSnippetFromPrompt(prompt); snippet != "" {
			return "🔧 " + snippet
		}
		if snippet := extractGoSnippetFromPrompt(prompt); snippet != "" {
			return "🔧 " + snippet
		}
	}

	// Match the best command example.
	m := dense.MatchCommandFromExamples(prompt, examples)

	// Handle follow-up questions about previous responses.
	if isFollowUp(prompt, conv) && conv.HasContext() {
		lastType := conv.LastType()
		lastAssistant := conv.LastAssistant()

		lower := strings.ToLower(strings.TrimSpace(prompt))

		// "What did I say?" → recall the user's last message.
		if strings.Contains(lower, "what did i say") || strings.Contains(lower, "what was my last message") {
			lastUser := conv.LastUser()
			if lastUser != "" {
				return fmt.Sprintf("🤖 You said: %q", lastUser)
			}
			return "🤖 I don't have any past conversation history to reference."
		}

		// "What did you say?" → recall the assistant's last response.
		if strings.Contains(lower, "what did you say") || strings.Contains(lower, "what was your last message") {
			if lastAssistant != "" {
				return fmt.Sprintf("🤖 I said: %q", lastAssistant)
			}
			return "🤖 I haven't said anything yet."
		}

		// "Can you repeat / show me again" → repeat the last response.
		if strings.Contains(lower, "repeat") || strings.Contains(lower, "again") || strings.Contains(lower, "show me") {
			if lastAssistant != "" {
				return fmt.Sprintf("🤖 Here it is again: %q", lastAssistant)
			}
		}

		// "Tell me more / elaborate" → expand on the last topic.
		if strings.Contains(lower, "more") || strings.Contains(lower, "elaborate") || strings.Contains(lower, "explain") {
			if lastType == "code_update" {
				return "🤖 That code snippet adds a Go language construct. You can combine it with other commands like adding imports, declaring variables, or wrapping it in a function. What else would you like to add?"
			}
			if lastAssistant != "" {
				return fmt.Sprintf("🤖 To expand on that: %s. Is there anything specific you'd like to know more about?", lastAssistant)
			}
		}

		// "What about / and then / what next" → continue the conversation flow.
		if strings.Contains(lower, "what about") || strings.Contains(lower, "and then") || strings.Contains(lower, "what next") || strings.Contains(lower, "what now") {
			if lastType == "code_update" {
				return "🤖 After that code change, you might want to add error handling, a return statement, or a closing brace. What would you like to do next?"
			}
			return "🤖 What would you like to do next? I can help with Go code updates or just chat."
		}
	}

	// Handle greetings with context awareness.
	if isGreeting(prompt) && conv.HasContext() {
		lastType := conv.LastType()
		if lastType == "code_update" {
			return "🤖 Hello again! Ready to continue with more Go code updates. What would you like to add next?"
		}
		return "🤖 Hello! Good to see you again. How can I help with your Go file today?"
	}

	// Handle farewells.
	if isFarewell(prompt) {
		return "🤖 Goodbye! Feel free to come back anytime you need Go code help."
	}

	// Handle thanks.
	if isThanks(prompt) {
		return "🤖 You're welcome! Let me know if you need any more Go file updates."
	}

	// Standard response based on classification.
	if cmdType == "social" || m.Type == "social" {
		if m.Response == "" {
			return "🤖 I'm here to help with your Go file. What would you like to do?"
		}
		return "🤖 " + m.Response
	}
	if cmdType == "file_create" || cmdType == "file_edit" || cmdType == "file_delete" {
		if m.CodeAfter != "" {
			return "🔧 " + m.CodeAfter
		}
		if cmdType == "file_create" {
			return "🔧 created file"
		}
		if cmdType == "file_edit" {
			return "🔧 updated file"
		}
		return "🔧 deleted file"
	}
	if cmdType == "folder_query" {
		if m.Response != "" {
			return "🤖 " + m.Response
		}
		return "🤖 I can inspect that folder and its contents."
	}
	if cmdType == "folder_create" || cmdType == "folder_delete" {
		if m.CodeAfter != "" {
			return "🔧 " + m.CodeAfter
		}
		if cmdType == "folder_create" {
			return "🔧 created folder"
		}
		return "🔧 deleted folder"
	}

	// Code update response.
	if m.CodeAfter == "" {
		return "⚠️  No valid Go code was generated for this prompt. Try pasting Go code directly (e.g. `j := jake.Jake{FirstName: \"Jake\"}`) or use a pattern like \"add function foo\"."
	}
	return "🔧 " + m.CodeAfter
}

// ─── Go File Editing ──────────────────────────────────────────────────────────

func applyFileOperation(filePath, opType, content string) (string, error) {
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}

	switch opType {
	case "file_create":
		if _, err := os.Stat(absPath); err == nil {
			return "", fmt.Errorf("file already exists: %s", absPath)
		} else if !os.IsNotExist(err) {
			return "", fmt.Errorf("stat file: %w", err)
		}
		if err := os.MkdirAll(filepath.Dir(absPath), 0755); err != nil {
			return "", fmt.Errorf("create parent dir: %w", err)
		}
		if err := os.WriteFile(absPath, []byte(content), 0644); err != nil {
			return "", fmt.Errorf("write file: %w", err)
		}
		return fmt.Sprintf("created %s", absPath), nil
	case "file_edit":
		if _, err := os.Stat(absPath); os.IsNotExist(err) {
			return "", fmt.Errorf("file not found: %s", absPath)
		}
		if err := os.WriteFile(absPath, []byte(content), 0644); err != nil {
			return "", fmt.Errorf("write file: %w", err)
		}
		return fmt.Sprintf("updated %s", absPath), nil
	case "file_delete":
		if err := os.Remove(absPath); err != nil {
			return "", fmt.Errorf("delete file: %w", err)
		}
		return fmt.Sprintf("deleted %s", absPath), nil
	default:
		return "", fmt.Errorf("unsupported file operation: %s", opType)
	}
}

func applyFolderOperation(dirPath, opType string) (string, error) {
	absPath, err := filepath.Abs(dirPath)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}

	switch opType {
	case "folder_create":
		if err := os.MkdirAll(absPath, 0755); err != nil {
			return "", fmt.Errorf("create folder: %w", err)
		}
		return fmt.Sprintf("created folder %s", absPath), nil
	case "folder_delete":
		if err := os.RemoveAll(absPath); err != nil {
			return "", fmt.Errorf("delete folder: %w", err)
		}
		return fmt.Sprintf("deleted folder %s", absPath), nil
	default:
		return "", fmt.Errorf("unsupported folder operation: %s", opType)
	}
}

// applyCodeToFile applies a code snippet to a target Go file. It uses AST-based
// editing for common operations (imports, functions, structs) and falls back to
// appending the snippet for simple statements.
func applyCodeToFile(filePath, code string) (string, error) {
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}
	if !strings.HasSuffix(absPath, ".go") {
		return "", fmt.Errorf("not a .go file: %s", absPath)
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found: %s", absPath)
	}

	// Read the current file content.
	content, err := os.ReadFile(absPath)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}

	// Try AST-based insertion first.
	applied, msg, err := applyCodeViaAST(absPath, string(content), code)
	if err == nil && applied {
		return msg, nil
	}

	// Fallback: append the snippet to the end of the file.
	newContent := string(content)
	if !strings.HasSuffix(newContent, "\n") {
		newContent += "\n"
	}
	newContent += code + "\n"

	// Verify the result is still valid Go.
	fset := token.NewFileSet()
	if _, err := parser.ParseFile(fset, absPath, newContent, parser.ParseComments); err != nil {
		return "", fmt.Errorf("appended code produces invalid Go: %v", err)
	}

	if err := os.WriteFile(absPath, []byte(newContent), 0644); err != nil {
		return "", fmt.Errorf("write file: %w", err)
	}

	// Run gofmt.
	exec.Command("gofmt", "-w", absPath).Run()

	return fmt.Sprintf("appended code to %s", absPath), nil
}

// applyCodeViaAST attempts to apply the code snippet using AST manipulation.
// Returns (applied, message, error).
func applyCodeViaAST(filePath, content, code string) (bool, string, error) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, content, parser.ParseComments)
	if err != nil {
		return false, "", err
	}

	trimmed := strings.TrimSpace(code)

	// Handle import statements.
	if strings.HasPrefix(trimmed, "import ") {
		importPath := strings.Trim(strings.TrimPrefix(trimmed, "import "), `"`)
		importPath = strings.TrimSpace(importPath)
		if importPath == "" {
			return false, "", fmt.Errorf("empty import path")
		}
		// Check if already imported.
		for _, imp := range node.Imports {
			if imp.Path != nil && strings.Trim(imp.Path.Value, `"`) == importPath {
				return true, fmt.Sprintf("import %q already present", importPath), nil
			}
		}
		// Add the import.
		impSpec := &ast.ImportSpec{
			Path: &ast.BasicLit{
				Kind:  token.STRING,
				Value: fmt.Sprintf("%q", importPath),
			},
		}
		// Find or create the import declaration.
		var importDecl *ast.GenDecl
		for _, decl := range node.Decls {
			if gd, ok := decl.(*ast.GenDecl); ok && gd.Tok == token.IMPORT {
				importDecl = gd
				break
			}
		}
		if importDecl == nil {
			importDecl = &ast.GenDecl{Tok: token.IMPORT}
			node.Decls = append([]ast.Decl{importDecl}, node.Decls...)
		}
		importDecl.Specs = append(importDecl.Specs, impSpec)
		if err := writeFormattedFile(filePath, fset, node); err != nil {
			return false, "", err
		}
		return true, fmt.Sprintf("added import %q to %s", importPath, filePath), nil
	}

	// Handle function declarations.
	if strings.HasPrefix(trimmed, "func ") {
		// Parse the function code.
		src := fmt.Sprintf("package main\n\n%s", trimmed)
		funcFset := token.NewFileSet()
		funcNode, err := parser.ParseFile(funcFset, "", src, parser.ParseComments)
		if err != nil {
			return false, "", fmt.Errorf("cannot parse function code: %v", err)
		}
		var newFunc *ast.FuncDecl
		for _, decl := range funcNode.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok {
				newFunc = fn
				break
			}
		}
		if newFunc == nil {
			return false, "", fmt.Errorf("no function declaration found in code")
		}

		// Add imports from the function code.
		for _, imp := range funcNode.Imports {
			if imp.Path != nil {
				path := strings.Trim(imp.Path.Value, `"`)
				// Check if already imported.
				already := false
				for _, existing := range node.Imports {
					if existing.Path != nil && strings.Trim(existing.Path.Value, `"`) == path {
						already = true
						break
					}
				}
				if !already {
					addImportToNode(node, path)
				}
			}
		}

		// Check if function already exists; if so, replace it in place.
		replaced := false
		for i, decl := range node.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == newFunc.Name.Name {
				node.Decls[i] = newFunc
				replaced = true
				break
			}
		}
		if !replaced {
			// Append the function.
			node.Decls = append(node.Decls, newFunc)
		}

		if err := writeFormattedFile(filePath, fset, node); err != nil {
			return false, "", err
		}
		action := "inserted"
		if replaced {
			action = "updated"
		}
		return true, fmt.Sprintf("%s function %q in %s", action, newFunc.Name.Name, filePath), nil
	}

	// Handle struct / type declarations (struct, interface, and any named type).
	if strings.HasPrefix(trimmed, "type ") {
		// Parse the type code.
		src := fmt.Sprintf("package main\n\n%s", trimmed)
		typeFset := token.NewFileSet()
		typeNode, err := parser.ParseFile(typeFset, "", src, parser.ParseComments)
		if err != nil {
			return false, "", fmt.Errorf("cannot parse type code: %v", err)
		}
		var newType *ast.TypeSpec
		for _, decl := range typeNode.Decls {
			if gd, ok := decl.(*ast.GenDecl); ok && gd.Tok == token.TYPE {
				for _, spec := range gd.Specs {
					if ts, ok := spec.(*ast.TypeSpec); ok {
						newType = ts
						break
					}
				}
			}
		}
		if newType == nil {
			return false, "", fmt.Errorf("no type declaration found in code")
		}

		// Check if type already exists; if so, replace it in place.
		replaced := false
		for i := range node.Decls {
			gd, ok := node.Decls[i].(*ast.GenDecl)
			if !ok || gd.Tok != token.TYPE {
				continue
			}
			for j, spec := range gd.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if ok && ts.Name.Name == newType.Name.Name {
					gd.Specs[j] = newType
					replaced = true
					break
				}
			}
			if replaced {
				break
			}
		}
		if !replaced {
			// Append the type declaration.
			newDecl := &ast.GenDecl{
				Tok:   token.TYPE,
				Specs: []ast.Spec{newType},
			}
			node.Decls = append(node.Decls, newDecl)
		}

		if err := writeFormattedFile(filePath, fset, node); err != nil {
			return false, "", err
		}
		action := "inserted"
		if replaced {
			action = "updated"
		}
		return true, fmt.Sprintf("%s type %q in %s", action, newType.Name.Name, filePath), nil
	}

	// Handle package clause.
	if strings.HasPrefix(trimmed, "package ") {
		// Package clause is already present in the file; no-op.
		return true, "package clause already present", nil
	}

	// Handle const/var declaration blocks.
	if strings.HasPrefix(trimmed, "const (") || strings.HasPrefix(trimmed, "var (") {
		// Parse the block.
		src := fmt.Sprintf("package main\n\n%s", trimmed)
		blockFset := token.NewFileSet()
		blockNode, err := parser.ParseFile(blockFset, "", src, parser.ParseComments)
		if err != nil {
			return false, "", fmt.Errorf("cannot parse declaration block: %v", err)
		}
		var newDecl *ast.GenDecl
		for _, decl := range blockNode.Decls {
			if gd, ok := decl.(*ast.GenDecl); ok {
				newDecl = gd
				break
			}
		}
		if newDecl == nil {
			return false, "", fmt.Errorf("no declaration block found in code")
		}
		node.Decls = append(node.Decls, newDecl)
		if err := writeFormattedFile(filePath, fset, node); err != nil {
			return false, "", err
		}
		return true, fmt.Sprintf("inserted declaration block into %s", filePath), nil
	}

	// For simple statements, we can't easily determine where to insert them via
	// AST. Return not-applied so the caller falls back to appending.
	return false, "", nil
}

// addImportToNode adds an import path to the file's import declarations,
// creating the import block if needed.
func addImportToNode(node *ast.File, path string) {
	impSpec := &ast.ImportSpec{
		Path: &ast.BasicLit{
			Kind:  token.STRING,
			Value: fmt.Sprintf("%q", path),
		},
	}
	var importDecl *ast.GenDecl
	for _, decl := range node.Decls {
		if gd, ok := decl.(*ast.GenDecl); ok && gd.Tok == token.IMPORT {
			importDecl = gd
			break
		}
	}
	if importDecl == nil {
		importDecl = &ast.GenDecl{Tok: token.IMPORT}
		node.Decls = append([]ast.Decl{importDecl}, node.Decls...)
	}
	importDecl.Specs = append(importDecl.Specs, impSpec)
}

// writeFormattedFile writes the AST back to disk with gofmt formatting.
func writeFormattedFile(filePath string, fset *token.FileSet, node *ast.File) error {
	var buf strings.Builder
	if err := format.Node(&buf, fset, node); err != nil {
		return fmt.Errorf("format node: %w", err)
	}
	if err := os.WriteFile(filePath, []byte(buf.String()), 0644); err != nil {
		return fmt.Errorf("write file: %w", err)
	}
	return nil
}

// ─── Conversation Commands ────────────────────────────────────────────────────

// handleCommand processes slash-commands in interactive mode.
// Returns (handled, response).
func handleCommand(line string, mgr *ConversationManager) (bool, string) {
	if !strings.HasPrefix(line, "/") {
		return false, ""
	}

	parts := strings.Fields(line)
	cmd := strings.ToLower(parts[0])

	switch cmd {
	case "/new":
		name := ""
		if len(parts) > 1 {
			name = parts[1]
		}
		_, convName := mgr.New(name)
		return true, fmt.Sprintf("🆕 Started new conversation %q", convName)

	case "/list":
		names := mgr.List()
		if len(names) == 0 {
			return true, "No conversations."
		}
		var sb strings.Builder
		sb.WriteString("📋 Conversations:\n")
		for _, n := range names {
			marker := "  "
			if n == mgr.Active() {
				marker = "▶ "
			}
			conv := mgr.GetByName(n)
			turnCount := 0
			target := ""
			if conv != nil {
				turnCount = len(conv.Turns)
				if conv.TargetGoFile() != "" {
					target = " | file: " + conv.TargetGoFile()
				}
			}
			sb.WriteString(fmt.Sprintf("%s%s (%d turns%s)\n", marker, n, turnCount, target))
		}
		return true, strings.TrimRight(sb.String(), "\n")

	case "/switch":
		if len(parts) < 2 {
			return true, "Usage: /switch <conversation-name>"
		}
		name := parts[1]
		if mgr.Switch(name) {
			conv := mgr.Get()
			target := ""
			if conv != nil && conv.TargetGoFile() != "" {
				target = " (" + conv.TargetGoFile() + ")"
			}
			return true, fmt.Sprintf("🔀 Switched to conversation %q%s", name, target)
		}
		return true, fmt.Sprintf("❌ Conversation %q not found. Use /list to see available conversations.", name)

	case "/file":
		if len(parts) < 2 {
			conv := mgr.Get()
			if conv != nil && conv.TargetGoFile() != "" {
				return true, fmt.Sprintf("📄 Active conversation %q targets: %s", mgr.Active(), conv.TargetGoFile())
			}
			return true, "Usage: /file <path-to-.go-file>"
		}
		path := strings.Join(parts[1:], " ")
		conv := mgr.Get()
		if err := conv.SetTargetFile(path); err != nil {
			return true, fmt.Sprintf("❌ %v", err)
		}
		return true, fmt.Sprintf("📄 Conversation %q will now update %s", mgr.Active(), conv.TargetGoFile())

	case "/delete":
		if len(parts) < 2 {
			return true, "Usage: /delete <conversation-name>"
		}
		name := parts[1]
		if mgr.Delete(name) {
			return true, fmt.Sprintf("🗑️ Deleted conversation %q. Active: %q", name, mgr.Active())
		}
		return true, fmt.Sprintf("❌ Cannot delete %q (not found or the last conversation).", name)

	case "/current":
		conv := mgr.Get()
		target := ""
		if conv != nil && conv.TargetGoFile() != "" {
			target = fmt.Sprintf(" -> %s", conv.TargetGoFile())
		}
		return true, fmt.Sprintf("💬 Active conversation: %q%s", mgr.Active(), target)

	case "/help":
		return true, `Available commands:
  /new [name]        start a new conversation (auto-generates a name if omitted)
  /list              list all conversations
  /switch <name>     switch to an existing conversation
  /delete <name>     delete a conversation
  /current           show the active conversation name
  /file <path>       set the target Go file to update for the active conversation
  /help              show this help`

	default:
		return true, fmt.Sprintf("❌ Unknown command: %s. Type /help for available commands.", cmd)
	}
}

func main() {
	modelPath := flag.String("model", "data/models/dense/model.gob", "path to trained gob model file")
	dataPath := flag.String("data", "data/training/command_examples.pb", "path to protobuf training data for response matching")
	oneShot := flag.String("prompt", "", "classify a single prompt and exit (interactive if empty)")
	targetFile := flag.String("file", "", "default target Go file used by the default conversation")
	flag.Parse()

	// Load the trained gob model.
	model, err := dense.LoadGob(*modelPath)
	if err != nil {
		log.Fatalf("load gob model: %v", err)
	}
	fmt.Printf("📦 Loaded model from %s\n", *modelPath)

	// Load the command corpus for response matching (protobuf or CSV).
	var examples []dense.CommandExample
	if strings.HasSuffix(*dataPath, ".pb") {
		examples, err = dense.LoadCommandExamplesFromProto(*dataPath)
	} else {
		examples, err = dense.LoadCommandExamplesFromCSV(*dataPath)
	}
	if err != nil {
		log.Fatalf("load command corpus: %v", err)
	}

	// Initialize the conversation manager (multiconversational).
	mgr := NewConversationManager()

	// Apply the global -file flag to the default conversation so live editing
	// works even before any /file command is issued.
	if *targetFile != "" {
		if err := mgr.Get().SetTargetFile(*targetFile); err != nil {
			log.Fatalf("invalid -file: %v", err)
		}
	}

	respond := func(prompt string) string {
		conv := mgr.Get()
		response := buildContextAwareResponse(prompt, conv, model, examples)

		// Track the turn in conversation history.
		cmdType := dense.ClassifyCommandType(prompt)
		if cmdType == "social" {
			input := dense.BagOfWords(prompt, dense.CommandVocab)
			preds := model.Predict([][]float32{input})
			if len(preds) > 0 {
				label := preds[0]
				if label >= 0 && label < len(dense.CommandLabels) {
					candidate := dense.CommandLabels[label]
					if candidate != "social" {
						cmdType = candidate
					}
				}
			}
		}

		// Strip the emoji prefix for history storage.
		cleanResponse := strings.TrimPrefix(response, "🤖 ")
		cleanResponse = strings.TrimPrefix(cleanResponse, "🔧 ")

		conv.AddTurn(prompt, cleanResponse, cmdType)

		// Apply file or code actions to the active target file when present.
		if cmdType == "code_update" {
			target := conv.TargetGoFile()
			if target == "" {
				target = dense.InferTargetFromPrompt(prompt)
			}
			if target == "" {
				response += "\n📄 No Go file targeted in this conversation. Use /file <path-to-.go-file> to set the file to update."
				return response
			}
			if err := conv.SetTargetFile(target); err != nil {
				response += fmt.Sprintf("\n⚠️  Could not set target file %q: %v", target, err)
				return response
			}

			code := strings.TrimPrefix(response, "🔧 ")
			code = strings.TrimSpace(code)
			if code == "" {
				return response
			}

			msg, err := applyCodeToFile(target, code)
			if err != nil {
				response += fmt.Sprintf("\n⚠️  Could not apply to %s: %v", target, err)
			} else {
				response += fmt.Sprintf("\n✅ Applied to %s: %s", target, msg)
			}
			return response
		}
		if cmdType == "file_create" || cmdType == "file_edit" || cmdType == "file_delete" {
			target := conv.TargetFile
			if target == "" {
				target = dense.InferTargetFromPrompt(prompt)
			}
			if target == "" {
				response += "\n📄 No file targeted in this conversation. Use /file <path> to set the target file."
				return response
			}
			if err := conv.SetTargetFile(target); err != nil {
				response += fmt.Sprintf("\n⚠️  Could not set target file %q: %v", target, err)
				return response
			}

			content := strings.TrimPrefix(response, "🔧 ")
			content = strings.TrimSpace(content)
			if cmdType == "file_delete" {
				msg, err := applyFileOperation(target, "file_delete", "")
				if err != nil {
					response += fmt.Sprintf("\n⚠️  Could not delete %s: %v", target, err)
				} else {
					response += fmt.Sprintf("\n✅ %s", msg)
				}
				return response
			}
			if content == "" {
				if cmdType == "file_create" {
					content = ""
				} else {
					content = "updated"
				}
			}
			msg, err := applyFileOperation(target, cmdType, content)
			if err != nil {
				response += fmt.Sprintf("\n⚠️  Could not apply to %s: %v", target, err)
			} else {
				response += fmt.Sprintf("\n✅ %s", msg)
			}
			return response
		}
		if cmdType == "folder_create" || cmdType == "folder_delete" {
			target := dense.InferTargetFromPrompt(prompt)
			if target == "" {
				target = strings.TrimSpace(strings.Split(prompt, " ")[(len(strings.Fields(prompt)) - 1)])
				if strings.Contains(strings.ToLower(prompt), "folder") || strings.Contains(strings.ToLower(prompt), "directory") {
					parts := strings.Fields(prompt)
					if len(parts) >= 3 {
						target = parts[len(parts)-1]
					}
				}
			}
			if target == "" {
				response += "\n📁 Could not determine which folder to operate on."
				return response
			}
			msg, err := applyFolderOperation(target, cmdType)
			if err != nil {
				response += fmt.Sprintf("\n⚠️  Could not apply folder action to %s: %v", target, err)
			} else {
				response += fmt.Sprintf("\n✅ %s", msg)
			}
			return response
		}

		return response
	}

	if *oneShot != "" {
		fmt.Printf("prompt: %q\n", *oneShot)
		fmt.Println(respond(*oneShot))
		return
	}

	fmt.Println("💬 dense-llm interactive shell  (exit with Ctrl+D)")
	fmt.Println("    Trained on the basic go_edit_agent update-command corpus.")
	fmt.Println("    Multi-conversation mode: use /new, /list, /switch, /delete.")
	fmt.Println("    Go-file editing: use /file <path> to set the target file per conversation.")
	if *targetFile != "" {
		fmt.Printf("    Default target Go file for the default conversation: %s\n", *targetFile)
	}
	fmt.Println()
	sc := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("[%s] > ", mgr.Active())
		if !sc.Scan() {
			break
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}

		// Handle slash commands.
		if handled, resp := handleCommand(line, mgr); handled {
			fmt.Println(resp)
			continue
		}

		fmt.Println(respond(line))
	}
	if err := sc.Err(); err != nil {
		log.Fatalf("read stdin: %v", err)
	}
}
