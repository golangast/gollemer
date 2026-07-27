// Package knowledge provides book ingestion capabilities that parse Go textbooks
// (Markdown, text, or PDF) and automatically extract concepts, idioms, and patterns
// into the ConceptTemplate registry. This enables the system to learn new Go patterns
// from external sources and use them to guide code generation.
package knowledge

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// BookSection represents a parsed section from a Go book.
type BookSection struct {
	Title      string   `json:"title"`
	Level      int      `json:"level"` // 1=chapter, 2=section, 3=subsection
	Content    string   `json:"content"`
	CodeBlocks []string `json:"code_blocks"`
	Terms      []string `json:"terms"` // Key terminology found in this section
}

// IngestionResult captures what was extracted from a book.
type IngestionResult struct {
	BookTitle     string            `json:"book_title"`
	Sections      []BookSection     `json:"sections"`
	Concepts      []ConceptTemplate `json:"concepts"`
	Errors        []string          `json:"errors,omitempty"`
	SourceFile    string            `json:"source_file"`
	TotalCodeRefs int               `json:"total_code_references"`
}

// BookIngester parses Go textbook content and extracts concept templates.
type BookIngester struct {
	registry *Registry
}

// NewBookIngester creates a book ingester that populates the given registry.
func NewBookIngester(reg *Registry) *BookIngester {
	return &BookIngester{
		registry: reg,
	}
}

// IngestFile reads a Go book from a file path (Markdown, TXT, or PDF text dump)
// and extracts concepts into the registry. Returns the ingestion result.
func (bi *BookIngester) IngestFile(path string) (*IngestionResult, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read book file %q: %w", path, err)
	}

	ext := strings.ToLower(filepath.Ext(path))
	var text string

	switch ext {
	case ".md", ".markdown":
		text = string(data)
	case ".txt":
		text = string(data)
	case ".pdf":
		// For PDF, extract text using simple extraction
		text = extractPDFText(string(data))
	default:
		// Try as plain text
		text = string(data)
	}

	title := extractBookTitle(text, filepath.Base(path))
	result := bi.ingestText(text, title, path)
	return result, nil
}

// IngestText processes raw Go book content as a string.
func (bi *BookIngester) IngestText(text string, bookTitle string) *IngestionResult {
	return bi.ingestText(text, bookTitle, "(inline)")
}

// ExportConcepts serializes the registry's concepts to a JSON file for reuse.
func (bi *BookIngester) ExportConcepts(outputPath string) error {
	data, err := bi.registry.MarshalJSON()
	if err != nil {
		return fmt.Errorf("marshal registry: %w", err)
	}
	return os.WriteFile(outputPath, data, 0644)
}

// ImportConcepts loads previously exported concepts from a JSON file
// and merges them into the registry. Existing concepts are not overwritten.
func (bi *BookIngester) ImportConcepts(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read concepts file %q: %w", path, err)
	}

	var imported map[string]ConceptTemplate
	if err := json.Unmarshal(data, &imported); err != nil {
		return fmt.Errorf("unmarshal concepts: %w", err)
	}

	// Merge without overwriting existing concepts
	for key, ct := range imported {
		if _, exists := bi.registry.concepts[key]; !exists {
			bi.registry.concepts[key] = ct
		}
	}

	return nil
}

// ingestText parses the book text and extracts concept templates.
func (bi *BookIngester) ingestText(text string, bookTitle string, sourceFile string) *IngestionResult {
	result := &IngestionResult{
		BookTitle:  bookTitle,
		SourceFile: sourceFile,
		Sections:   make([]BookSection, 0),
		Concepts:   make([]ConceptTemplate, 0),
		Errors:     make([]string, 0),
	}

	// Parse sections from Markdown-style headers
	sections := parseSections(text)
	result.Sections = sections

	// Extract concepts from each section
	for _, section := range sections {
		concept := bi.extractConceptFromSection(section)
		if concept != nil {
			key := generateConceptKey(concept.Term)
			// Only add if not already registered
			if _, exists := bi.registry.concepts[key]; !exists {
				bi.registry.concepts[key] = *concept
			}
			result.Concepts = append(result.Concepts, *concept)
		}
	}

	// Count code references across all sections
	for _, s := range sections {
		result.TotalCodeRefs += len(s.CodeBlocks)
	}

	return result
}

// extractConceptFromSection analyzes a book section and creates a ConceptTemplate
// if the section contains identifiable Go idioms and code patterns.
func (bi *BookIngester) extractConceptFromSection(section BookSection) *ConceptTemplate {
	content := section.Content
	titleLower := strings.ToLower(section.Title)

	// Skip sections that are too short or have no code blocks
	if len(strings.Fields(content)) < 20 && len(section.CodeBlocks) == 0 {
		return nil
	}

	// Extract required Go constructs from the content
	required := extractRequiredConstructs(content)
	if len(required) == 0 && len(section.CodeBlocks) == 0 {
		return nil
	}

	// Extract synonyms from the content (alternative phrasings)
	synonyms := extractSynonyms(titleLower, content)

	// Build mutation rules from code blocks
	mutations := extractMutationsFromCodeBlocks(section.CodeBlocks, section.Title)

	// Determine the primary term from the section title
	term := cleanTerm(section.Title)
	if term == "" {
		return nil
	}

	return &ConceptTemplate{
		Term:               term,
		Synonyms:           synonyms,
		RequiredConstructs: required,
		ASTMutations:       mutations,
	}
}

// parseSections splits Markdown text into hierarchical sections.
func parseSections(text string) []BookSection {
	lines := strings.Split(text, "\n")
	var sections []BookSection
	var current *BookSection
	var codeBlock bool
	var currentCode []string

	headerRegex := regexp.MustCompile(`^(#{1,6})\s+(.+)$`)

	flushCode := func() {
		if current != nil && len(currentCode) > 0 {
			code := strings.Join(currentCode, "\n")
			current.CodeBlocks = append(current.CodeBlocks, code)
			currentCode = nil
		}
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Track code blocks (``` fences or indented code)
		if strings.HasPrefix(trimmed, "```") {
			if codeBlock {
				// End of code block
				codeBlock = false
				flushCode()
			} else {
				codeBlock = true
				currentCode = nil
			}
			continue
		}

		if codeBlock {
			currentCode = append(currentCode, line)
			continue
		}

		// Check for header
		if matches := headerRegex.FindStringSubmatch(trimmed); len(matches) >= 3 {
			flushCode()
			level := len(matches[1])
			title := strings.TrimSpace(matches[2])

			current = &BookSection{
				Title:      title,
				Level:      level,
				Content:    "",
				CodeBlocks: make([]string, 0),
				Terms:      make([]string, 0),
			}
			sections = append(sections, *current)
			continue
		}

		// Track indented code (4 spaces or tab)
		if strings.HasPrefix(line, "    ") || strings.HasPrefix(line, "\t") {
			currentCode = append(currentCode, strings.TrimLeft(line, " \t"))
			continue
		} else if len(currentCode) > 0 && trimmed != "" {
			// Non-empty line after indented code ends the block
			flushCode()
		}

		// Add content to current section
		if current != nil {
			if current.Content != "" {
				current.Content += "\n"
			}
			current.Content += line

			// Extract terms: bold, italic, or code-quoted words
			extractTermsFromLine(trimmed, current)
		}
	}

	flushCode()

	// Update sections in place
	for i := range sections {
		if i < len(sections) {
			// Sections were appended as values, so they're already set
		}
	}

	return sections
}

// extractTermsFromLine finds key terminology in a line of text.
func extractTermsFromLine(line string, section *BookSection) {
	// Match **bold** terms
	boldRegex := regexp.MustCompile(`\*\*(.+?)\*\*`)
	for _, m := range boldRegex.FindAllStringSubmatch(line, -1) {
		term := strings.TrimSpace(m[1])
		if len(term) > 2 {
			section.Terms = append(section.Terms, term)
		}
	}

	// Match `code` terms
	codeRegex := regexp.MustCompile("`(.+?)`")
	for _, m := range codeRegex.FindAllStringSubmatch(line, -1) {
		term := strings.TrimSpace(m[1])
		if len(term) > 2 && !strings.Contains(term, " ") {
			section.Terms = append(section.Terms, term)
		}
	}
}

// extractRequiredConstructs finds Go language primitives mentioned in text.
func extractRequiredConstructs(content string) []string {
	// Known Go primitives to look for
	primitives := []string{
		"chan", "go func", "goroutine", "sync.WaitGroup", "sync.Mutex",
		"sync.RWMutex", "sync.Once", "sync.Map", "context.Context",
		"context.WithCancel", "context.WithDeadline", "context.WithTimeout",
		"interface{}", "struct", "error", "error interface",
		"defer", "panic", "recover", "select", "range",
		"map[", "[]", "make(", "new(", "close(",
		"time.Ticker", "time.Timer", "time.Duration", "time.Sleep",
		"os.Signal", "os/signal", "io.Reader", "io.Writer",
		"string", "int", "float64", "bool", "byte",
		"json.Marshal", "json.Unmarshal", "encoding/json",
		"http.Handler", "http.HandlerFunc", "net/http",
		"io/ioutil", "fmt.Sprintf", "fmt.Errorf",
		"errors.New", "errors.Is", "errors.As",
		"atomic.", "math.", "sort.",
	}

	contentLower := strings.ToLower(content)
	found := make(map[string]bool)
	var result []string

	for _, p := range primitives {
		if strings.Contains(contentLower, strings.ToLower(p)) {
			if !found[p] {
				found[p] = true
				result = append(result, p)
			}
		}
	}

	return result
}

// extractSynonyms finds alternative phrasings for a concept in the content.
func extractSynonyms(titleLower string, content string) []string {
	synonyms := make(map[string]bool)
	contentLower := strings.ToLower(content)

	// Look for "also known as", "also called", "sometimes called" patterns
	akaPatterns := []*regexp.Regexp{
		regexp.MustCompile(`also known as ["'“](.+?)["'”]`),
		regexp.MustCompile(`also called ["'“](.+?)["'”]`),
		regexp.MustCompile(`sometimes called ["'“](.+?)["'”]`),
		regexp.MustCompile(`referred to as ["'“](.+?)["'”]`),
		regexp.MustCompile(`known as ["'“](.+?)["'”]`),
	}

	for _, pat := range akaPatterns {
		matches := pat.FindAllStringSubmatch(contentLower, -1)
		for _, m := range matches {
			if len(m) >= 2 {
				syn := strings.TrimSpace(m[1])
				if len(syn) > 2 && syn != titleLower {
					synonyms[syn] = true
				}
			}
		}
	}

	// Look for parenthetical synonyms: "Worker Pool (goroutine pool)"
	parenPattern := regexp.MustCompile(titleLower + `\s*\((.+?)\)`)
	if matches := parenPattern.FindStringSubmatch(contentLower); len(matches) >= 2 {
		parts := strings.Split(matches[1], ",")
		for _, p := range parts {
			syn := strings.TrimSpace(p)
			if len(syn) > 2 {
				synonyms[syn] = true
			}
		}
	}

	// Filter results
	var result []string
	for s := range synonyms {
		if len(s) > 2 {
			result = append(result, s)
		}
	}

	return result
}

// extractMutationsFromCodeBlocks analyzes Go code blocks and creates MutationRules.
func extractMutationsFromCodeBlocks(blocks []string, sectionTitle string) []MutationRule {
	var mutations []MutationRule

	for _, block := range blocks {
		// Skip non-Go code or very small blocks
		if len(strings.Fields(block)) < 3 {
			continue
		}

		// Detect mutation type based on code patterns
		mType := detectMutationType(block)
		target := detectTargetStruct(block, sectionTitle)

		// Clean and normalize the code template
		template := cleanCodeTemplate(block)
		if template == "" {
			continue
		}

		mutations = append(mutations, MutationRule{
			Type:         mType,
			TargetStruct: target,
			CodeTemplate: template,
		})
	}

	return mutations
}

// detectMutationType determines the type of AST mutation from code patterns.
func detectMutationType(code string) string {
	codeTrimmed := strings.TrimSpace(code)

	// Detect struct field declarations
	if containsGoKeyword(codeTrimmed, "struct") || isFieldDeclaration(code) {
		return "add_field"
	}

	// Detect defer statements
	if strings.HasPrefix(codeTrimmed, "defer ") || strings.Contains(codeTrimmed, "\n\tdefer ") {
		return "add_defer"
	}

	// Detect for loops
	if strings.HasPrefix(codeTrimmed, "for ") || strings.HasPrefix(codeTrimmed, "for ") {
		return "wrap_loop"
	}

	// Detect import blocks or single imports
	if strings.HasPrefix(codeTrimmed, "import") || strings.HasPrefix(codeTrimmed, "\"") {
		return "add_import"
	}

	// Detect function/method definitions
	if strings.HasPrefix(codeTrimmed, "func ") {
		// Check if it's a method with a receiver (struct method)
		if strings.Contains(codeTrimmed, "func (") {
			return "add_method"
		}
		return "wrap_body"
	}

	// Detect type definitions
	if strings.HasPrefix(codeTrimmed, "type ") {
		return "add_type"
	}

	// Default: treat as add_field
	return "add_field"
}

// detectTargetStruct tries to identify which struct a mutation targets.
func detectTargetStruct(code string, sectionTitle string) string {
	// Look for method receivers like `func (s *StructName)`
	receiverRegex := regexp.MustCompile(`func\s+\([a-zA-Z]+\s+(\*?[A-Z][a-zA-Z0-9]*)\)`)
	if matches := receiverRegex.FindStringSubmatch(code); len(matches) >= 2 {
		return strings.TrimPrefix(matches[1], "*")
	}

	// Look for type definitions
	typeRegex := regexp.MustCompile(`type\s+([A-Z][a-zA-Z0-9]*)\s+struct`)
	if matches := typeRegex.FindStringSubmatch(code); len(matches) >= 2 {
		return matches[1]
	}

	// Use the section title as a hint
	titleWords := strings.Fields(sectionTitle)
	for _, w := range titleWords {
		if len(w) > 3 && w[0] >= 'A' && w[0] <= 'Z' {
			return w
		}
	}

	return ""
}

// isFieldDeclaration checks if code looks like struct field declarations.
func isFieldDeclaration(code string) bool {
	lines := strings.Split(strings.TrimSpace(code), "\n")
	if len(lines) == 0 {
		return false
	}

	fieldCount := 0
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		// Match patterns like: "fieldName typeName" or "fieldName typeName `json:"..."`"
		fieldRegex := regexp.MustCompile(`^[a-z][a-zA-Z0-9]*\s+\*?[A-Za-z][A-Za-z0-9.]*`)
		if fieldRegex.MatchString(trimmed) {
			fieldCount++
		}
	}

	return fieldCount > 0
}

// cleanCodeTemplate normalizes a code block into a usable template string.
func cleanCodeTemplate(code string) string {
	lines := strings.Split(code, "\n")
	var cleaned []string

	for _, line := range lines {
		trimmed := strings.TrimRight(line, " \t")
		if trimmed == "" && len(cleaned) == 0 {
			continue // Skip leading empty lines
		}
		// Normalize indentation to tabs
		cleaned = append(cleaned, trimmed)
	}

	// Remove trailing empty lines
	for len(cleaned) > 0 && cleaned[len(cleaned)-1] == "" {
		cleaned = cleaned[:len(cleaned)-1]
	}

	return strings.Join(cleaned, "\n")
}

// cleanTerm normalizes a section title into a concept term.
func cleanTerm(title string) string {
	// Remove trailing colons, periods, numbers like "2.1."
	re := regexp.MustCompile(`^[\d.]+[\s]*`)
	cleaned := re.ReplaceAllString(title, "")

	// Remove trailing punctuation
	cleaned = strings.TrimRight(cleaned, ".:;!?,# ")

	// Capitalize first letter
	if len(cleaned) > 0 {
		cleaned = strings.ToUpper(cleaned[:1]) + cleaned[1:]
	}

	return cleaned
}

// generateConceptKey creates a lowercase key for a concept term.
func generateConceptKey(term string) string {
	key := strings.ToLower(term)
	key = strings.NewReplacer(" ", "_", "-", "_", "/", "_").Replace(key)
	key = regexp.MustCompile(`[^a-z0-9_]`).ReplaceAllString(key, "")
	return key
}

// containsGoKeyword checks if the code contains a specific Go keyword.
func containsGoKeyword(code string, keyword string) bool {
	wordBoundary := regexp.MustCompile(`\b` + regexp.QuoteMeta(keyword) + `\b`)
	return wordBoundary.MatchString(code)
}

// extractBookTitle extracts the book title from the content or uses the filename.
func extractBookTitle(text string, filename string) string {
	// Look for the first H1 header
	headerRegex := regexp.MustCompile(`(?m)^#\s+(.+)$`)
	if matches := headerRegex.FindStringSubmatch(text); len(matches) >= 2 {
		return strings.TrimSpace(matches[1])
	}

	// Try filename without extension
	base := filepath.Base(filename)
	ext := filepath.Ext(base)
	title := strings.TrimSuffix(base, ext)
	title = strings.NewReplacer("_", " ", "-", " ", ".", " ").Replace(title)
	return strings.TrimSpace(title)
}

// extractPDFText performs basic text extraction from PDF-like content.
// This is a simple fallback; full PDF parsing would use a library.
func extractPDFText(content string) string {
	// Remove common PDF artifacts
	lines := strings.Split(content, "\n")
	var textLines []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		// Skip page numbers, PDF metadata artifacts
		if trimmed == "" || isPDFArtifact(trimmed) {
			continue
		}
		textLines = append(textLines, trimmed)
	}
	return strings.Join(textLines, "\n")
}

// isPDFArtifact checks if a line is likely a PDF rendering artifact.
func isPDFArtifact(line string) bool {
	// Page numbers like "--- Page 42 ---"
	if matched, _ := regexp.MatchString(`^-+\s*Page\s+\d+\s*-+$`, line); matched {
		return true
	}
	// All digits (page numbers)
	if matched, _ := regexp.MatchString(`^\d+$`, line); matched && len(line) <= 5 {
		return true
	}
	// Very short lines with special chars
	if len(line) < 3 && strings.ContainsAny(line, "|-_=+") {
		return true
	}
	return false
}
