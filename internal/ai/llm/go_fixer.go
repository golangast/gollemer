package llm

import (
	"fmt"
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"strings"
)

// GoFixer handles detection and fixing of common Go syntax errors
type GoFixer struct {
	FilePath string
	Errors   []string
}

// NewGoFixer creates a new GoFixer instance
func NewGoFixer(filePath string) *GoFixer {
	return &GoFixer{FilePath: filePath}
}

// Fix attempts to automatically fix common Go syntax errors
func (gf *GoFixer) Fix() error {
	// Backup the original file
	backupPath := gf.FilePath + ".bak"
	if err := copyFile(gf.FilePath, backupPath); err != nil {
		return fmt.Errorf("failed to create backup: %v", err)
	}
	defer os.Remove(backupPath)

	// Try to parse the file to identify syntax errors
	content, err := ioutil.ReadFile(gf.FilePath)
	if err != nil {
		return fmt.Errorf("failed to read file: %v", err)
	}

	// Try parsing the file
	_, err = parser.ParseFile(token.NewFileSet(), gf.FilePath, content, parser.ParseComments)
	if err == nil {
		// No syntax errors
		return nil
	}

	// Store the error for analysis
	errStr := err.Error()
	lines := strings.Split(string(content), "\n")

	// Analyze and fix the errors
	fixedContent, fixed := gf.analyzeAndFix(errStr, lines)
	if fixed {
		if err := ioutil.WriteFile(gf.FilePath, []byte(fixedContent), 0644); err != nil {
			return fmt.Errorf("failed to write fixed content: %v", err)
		}

		// Format the file
		if err := exec.Command("gofmt", "-w", gf.FilePath).Run(); err != nil {
			log.Printf("Warning: gofmt failed: %v", err)
		}

		return nil
	}

	// If we couldn't fix it automatically, restore the backup
	if err := copyFile(backupPath, gf.FilePath); err != nil {
		return fmt.Errorf("failed to restore backup: %v", err)
	}

	return fmt.Errorf("could not automatically fix the file: %v", errStr)
}

// analyzeAndFix analyzes the error and applies appropriate fixes
func (gf *GoFixer) analyzeAndFix(errorStr string, lines []string) (string, bool) {
	// Store errors for learning
	gf.Errors = append(gf.Errors, errorStr)

	// Always try missing struct keyword first — catches "type X {" missing 'struct'.
	// Must run BEFORE fixMissingOpeningBrace because "type X {" already has a '{'
	// so structural brace counting alone is ambiguous and the '{' is misinterpreted.
	if content, ok := gf.fixMissingStructKeyword(lines); ok {
		return content, true
	}

	// Always try missing opening brace second — it's safe to check structurally
	// and catches func declarations with params/return types but no '{'
	if content, ok := gf.fixMissingOpeningBrace(lines); ok {
		return content, true
	}

	// Apply fixes based on error patterns
	if strings.Contains(errorStr, "expected '}'") {
		if content, ok := gf.fixMissingClosingBrace(lines); ok {
			return content, true
		}
	}

	if strings.Contains(errorStr, "expected ')'") {
		if content, ok := gf.fixMissingParentheses(lines); ok {
			return content, true
		}
	}

	if strings.Contains(errorStr, "expected '(', found") {
		if content, ok := gf.fixMissingFuncKeyword(lines); ok {
			return content, true
		}
		if content, ok := gf.fixMissingParentheses(lines); ok {
			return content, true
		}
	}

	if strings.Contains(errorStr, "unexpected EOF") {
		if content, ok := gf.fixUnexpectedEOF(lines); ok {
			return content, true
		}
	}

	if strings.Contains(errorStr, "expected declaration") {
		if content, ok := gf.fixExpectedDeclaration(lines); ok {
			return content, true
		}
	}

	// Try common fixes even if error pattern doesn't match
	if fixed, resultLines := gf.tryCommonFixes(lines); fixed {
		return strings.Join(resultLines, "\n"), true
	}

	return strings.Join(lines, "\n"), false
}

// fixMissingStructKeyword fixes "type X {" declarations that are missing the
// 'struct' keyword, converting them to "type X struct {".
func (gf *GoFixer) fixMissingStructKeyword(lines []string) (string, bool) {
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		// Handle: "type X {" missing the 'struct' keyword — e.g. "type jill  {"
		if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, "{") &&
			!strings.Contains(trimmed, " struct ") && !strings.HasPrefix(trimmed, "type struct") {
			// Replace the '{' with 'struct {' to fix "type X {" -> "type X struct {"
			lines[i] = strings.Replace(trimmed, "{", "struct {", 1)
			return strings.Join(lines, "\n"), true
		}
	}
	return strings.Join(lines, "\n"), false
}

// fixMissingOpeningBrace fixes missing opening braces
func (gf *GoFixer) fixMissingOpeningBrace(lines []string) (string, bool) {
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		// Handle: type X struct (no brace) — e.g. "type named struct" followed by a field
		if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, " struct") && !strings.Contains(trimmed, "{") {
			lines[i] = trimmed + " {"
			return strings.Join(lines, "\n"), true
		}
		// Handle: func name(params) returnType (no brace)
		if strings.HasPrefix(trimmed, "func ") && !strings.Contains(trimmed, "{") {
			lines[i] = trimmed + " {"
			return strings.Join(lines, "\n"), true
		}
		// Handle: if/for/switch without brace
		if (strings.HasPrefix(trimmed, "if ") ||
			strings.HasPrefix(trimmed, "for ") ||
			strings.HasPrefix(trimmed, "switch ")) &&
			!strings.HasSuffix(trimmed, "{") && !strings.Contains(trimmed, "{") {
			lines[i] = trimmed + " {"
			return strings.Join(lines, "\n"), true
		}
	}
	return strings.Join(lines, "\n"), false
}

// fixMissingClosingBrace fixes missing closing braces
func (gf *GoFixer) fixMissingClosingBrace(lines []string) (string, bool) {
	// Count opening and closing braces
	openBraces := 0
	closeBraces := 0
	for _, line := range lines {
		openBraces += strings.Count(line, "{")
		closeBraces += strings.Count(line, "}")
	}

	if openBraces > closeBraces {
		lines = append(lines, strings.Repeat("}", openBraces-closeBraces))
		return strings.Join(lines, "\n"), true
	}
	return strings.Join(lines, "\n"), false
}

// fixMissingParentheses fixes missing parentheses
func (gf *GoFixer) fixMissingParentheses(lines []string) (string, bool) {
	for i, line := range lines {
		if strings.Contains(line, "fmt.Println(") && !strings.Contains(line, ")") {
			lines[i] = strings.Replace(line, "fmt.Println(", "fmt.Println(", 1) + ")"
			return strings.Join(lines, "\n"), true
		}
		// Fix missing parentheses in function declarations
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "func ") && !strings.Contains(trimmed, "()") && !strings.Contains(trimmed, "(") {
			// Extract the function name
			parts := strings.Fields(trimmed)
			if len(parts) >= 2 {
				funcName := strings.TrimSuffix(parts[1], "{")
				lines[i] = fmt.Sprintf("func %s() {", funcName)
				return strings.Join(lines, "\n"), true
			}
		}
	}
	return strings.Join(lines, "\n"), false
}

// fixMissingFuncKeyword fixes missing 'func' keyword
func (gf *GoFixer) fixMissingFuncKeyword(lines []string) (string, bool) {
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "fn ") && strings.HasSuffix(trimmed, " {") {
			lines[i] = strings.Replace(trimmed, "fn ", "func ", 1)
			return strings.Join(lines, "\n"), true
		}
	}
	return strings.Join(lines, "\n"), false
}

// fixUnexpectedEOF fixes unexpected end of file
func (gf *GoFixer) fixUnexpectedEOF(lines []string) (string, bool) {
	// Count opening and closing braces
	openBraces := 0
	closeBraces := 0
	for _, line := range lines {
		openBraces += strings.Count(line, "{")
		closeBraces += strings.Count(line, "}")
	}

	if openBraces > closeBraces {
		lines = append(lines, strings.Repeat("}", openBraces-closeBraces))
		return strings.Join(lines, "\n"), true
	}
	return strings.Join(lines, "\n"), false
}

// fixExpectedDeclaration fixes expected declaration errors
func (gf *GoFixer) fixExpectedDeclaration(lines []string) (string, bool) {
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "func ") && !strings.Contains(trimmed, "()") && !strings.Contains(trimmed, "{") {
			// Extract the function name
			parts := strings.Fields(trimmed)
			if len(parts) >= 2 {
				funcName := parts[1]
				lines[i] = fmt.Sprintf("func %s() {", funcName)
				return strings.Join(lines, "\n"), true
			}
		}
	}
	return strings.Join(lines, "\n"), false
}

// tryCommonFixes tries common fixes even if error pattern doesn't match
func (gf *GoFixer) tryCommonFixes(lines []string) (bool, []string) {
	// Fix missing 'func' keyword
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "fn ") && strings.HasSuffix(trimmed, " {") {
			lines[i] = strings.Replace(trimmed, "fn ", "func ", 1)
			return true, lines
		}
	}

	// Fix missing ')' in func init
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "func init") && !strings.Contains(trimmed, "()") {
			lines[i] = strings.Replace(trimmed, "func init", "func init()", 1)
			return true, lines
		}
	}

	// Fix missing '{' after func init()
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "func init()") && !strings.HasSuffix(trimmed, "{") {
			lines[i] = trimmed + " {"
			return true, lines
		}
	}

	// Fix missing ')' in fmt.Println
	for i, line := range lines {
		if strings.Contains(line, "fmt.Println(") && !strings.Contains(line, ")") {
			lines[i] = strings.Replace(line, "fmt.Println(", "fmt.Println(", 1) + ")"
			return true, lines
		}
	}

	// Fix missing '()' and '{' in function declaration
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "func ") && !strings.Contains(trimmed, "()") && !strings.Contains(trimmed, "{") {
			// Extract the function name
			parts := strings.Fields(trimmed)
			if len(parts) >= 2 {
				funcName := parts[1]
				lines[i] = fmt.Sprintf("func %s() {", funcName)
				return true, lines
			}
		}
	}

	// Fix incomplete function declarations
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "func ") && !strings.HasSuffix(trimmed, "{") && !strings.Contains(trimmed, "{") {
			// Check if the line ends with a function name
			if len(strings.Fields(trimmed)) == 2 {
				funcName := strings.Fields(trimmed)[1]
				lines[i] = fmt.Sprintf("func %s() {", funcName)
				return true, lines
			}
		}
	}

	return false, lines
}

// copyFile copies a file from src to dst
func copyFile(src, dst string) error {
	input, err := ioutil.ReadFile(src)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(dst, input, 0644)
}
