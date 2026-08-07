package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// ErrorPattern matches the structure in data/training/go_error_patterns.json
type ErrorPattern struct {
	ID          string   `json:"id"`
	Match       string   `json:"match"`
	Description string   `json:"description"`
	FixType     string   `json:"fix_type"`
	Examples    []string `json:"examples"`
	Confidence  float64  `json:"confidence"`
}

// ErrorPatternsDB matches the training data file structure
type ErrorPatternsDB struct {
	Version  int            `json:"version"`
	Patterns []ErrorPattern `json:"patterns"`
}

func main() {
	filePath := flag.String("file", "", "Go file with errors to analyze")
	trainDataPath := flag.String("data", "data/training/go_error_patterns.json", "Path to training data JSON")
	interactive := flag.Bool("interactive", false, "Interactive mode - prompt to add new patterns")
	flag.Parse()

	if *filePath == "" {
		fmt.Println("Usage: go run cmd/tools/train_error_fixer/main.go -file <path> [-interactive]")
		fmt.Println("\nAnalyzes a Go file for errors and matches them against training patterns.")
		fmt.Println("Use -interactive to teach the agent new error patterns.")
		os.Exit(1)
	}

	// Load existing training data
	db := loadTrainingData(*trainDataPath)

	// Run go vet on the file
	fmt.Printf("🔍 Analyzing %s...\n", *filePath)
	vetOut, _ := exec.Command("go", "vet", *filePath).CombinedOutput()
	vetStr := strings.TrimSpace(string(vetOut))

	if vetStr == "" {
		fmt.Println("✅ No errors found in file.")
		return
	}

	// Try parsing to get syntax errors
	content, _ := os.ReadFile(*filePath)
	_, parseErr := parser.ParseFile(token.NewFileSet(), *filePath, content, parser.ParseComments)
	parseStr := ""
	if parseErr != nil {
		parseStr = parseErr.Error()
	}

	fmt.Printf("⚠️  Found errors:\n")
	if parseStr != "" {
		fmt.Printf("  📝 Parse error: %s\n", parseStr)
	}
	fmt.Printf("  🔧 Go vet: %s\n", vetStr)

	// Match against existing patterns
	allErrors := vetStr
	if parseStr != "" {
		allErrors = parseStr + "\n" + vetStr
	}

	matches := findMatchingPatterns(db, allErrors)
	if len(matches) > 0 {
		fmt.Printf("\n✅ Matched %d existing patterns:\n", len(matches))
		for _, m := range matches {
			fmt.Printf("  - [%s] %s (confidence: %.0f%%)\n", m.ID, m.Description, m.Confidence*100)
		}
	} else {
		fmt.Println("\n❌ No matching patterns found in training data.")
	}

	// Find unmatched error messages
	unmatched := findUnmatchedErrors(allErrors, db)
	if len(unmatched) > 0 {
		fmt.Printf("\n📋 Unmatched error messages (%d):\n", len(unmatched))
		for _, e := range unmatched {
			fmt.Printf("  - %s\n", e)
		}

		if *interactive {
			teachNewPatterns(*trainDataPath, db, unmatched)
		} else {
			fmt.Println("\n💡 Run with -interactive to teach the agent new patterns.")
		}
	}

	// Show training summary
	fmt.Printf("\n📊 Training data: %d patterns in %s\n", len(db.Patterns), *trainDataPath)
}

// loadTrainingData loads the error pattern training data from a JSON file
func loadTrainingData(path string) *ErrorPatternsDB {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("⚠️  Could not load training data from %s: %v\n", path, err)
		return &ErrorPatternsDB{Version: 1, Patterns: []ErrorPattern{}}
	}

	var db ErrorPatternsDB
	if err := json.Unmarshal(data, &db); err != nil {
		fmt.Printf("⚠️  Could not parse training data: %v\n", err)
		return &ErrorPatternsDB{Version: 1, Patterns: []ErrorPattern{}}
	}

	return &db
}

// findMatchingPatterns finds all patterns that match the given error string
func findMatchingPatterns(db *ErrorPatternsDB, errStr string) []ErrorPattern {
	var matches []ErrorPattern
	lower := strings.ToLower(errStr)
	for _, p := range db.Patterns {
		if strings.Contains(lower, strings.ToLower(p.Match)) {
			matches = append(matches, p)
		}
	}
	return matches
}

// findUnmatchedErrors extracts error messages that don't match any pattern
func findUnmatchedErrors(errStr string, db *ErrorPatternsDB) []string {
	lower := strings.ToLower(errStr)
	lines := strings.Split(errStr, "\n")
	var unmatched []string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Skip Go build/vet header lines
		if strings.HasPrefix(line, "# ") || strings.HasPrefix(line, "[") {
			continue
		}

		matched := false
		for _, p := range db.Patterns {
			if strings.Contains(lower, strings.ToLower(p.Match)) {
				matched = true
				break
			}
		}
		if !matched {
			unmatched = append(unmatched, line)
		}
	}

	return unmatched
}

// teachNewPatterns interactively teaches the agent new error patterns
func teachNewPatterns(path string, db *ErrorPatternsDB, unmatched []string) {
	reader := bufio.NewReader(os.Stdin)

	for _, errMsg := range unmatched {
		fmt.Printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
		fmt.Printf("🧠 Teach a new pattern for:\n")
		fmt.Printf("   %s\n", errMsg)
		fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

		fmt.Print("Enter a short ID (e.g., 'missing_brace_custom'): ")
		id, _ := reader.ReadString('\n')
		id = strings.TrimSpace(id)
		if id == "" {
			fmt.Println("  ⏭️  Skipped.")
			continue
		}

		fmt.Print("Enter the match string (substring to match in error): ")
		matchStr, _ := reader.ReadString('\n')
		matchStr = strings.TrimSpace(matchStr)
		if matchStr == "" {
			matchStr = extractMatchKey(errMsg)
			fmt.Printf("  Using auto-extracted: '%s'\n", matchStr)
		}

		fmt.Print("Enter a description: ")
		desc, _ := reader.ReadString('\n')
		desc = strings.TrimSpace(desc)
		if desc == "" {
			desc = "Auto-trained pattern for: " + matchStr
		}

		fmt.Print("Enter fix_type (add_brace_after_func, remove_duplicate_type, add_closing_paren, balance_braces, report_unfixable): ")
		fixType, _ := reader.ReadString('\n')
		fixType = strings.TrimSpace(fixType)
		if fixType == "" {
			fixType = "report_unfixable"
		}

		fmt.Print("Enter confidence (0.0-1.0, default 0.5): ")
		confStr, _ := reader.ReadString('\n')
		confStr = strings.TrimSpace(confStr)
		conf := 0.5
		if confStr != "" {
			fmt.Sscanf(confStr, "%f", &conf)
		}

		// Add the new pattern
		newPattern := ErrorPattern{
			ID:          id,
			Match:       matchStr,
			Description: desc,
			FixType:     fixType,
			Examples:    []string{errMsg},
			Confidence:  conf,
		}

		db.Patterns = append(db.Patterns, newPattern)
		fmt.Printf("  ✅ Added pattern '%s'\n", id)
	}

	// Save the updated training data
	saveTrainingData(path, db)
}

// extractMatchKey extracts a reasonable match key from an error message
func extractMatchKey(errMsg string) string {
	// Try to extract the key part of the error
	// e.g., "ft/jim.go:20:25: missing ',' in parameter list" -> "missing ',' in parameter list"
	if idx := strings.Index(errMsg, ": "); idx >= 0 {
		after := errMsg[idx+2:]
		// Take the first meaningful part
		if idx2 := strings.Index(after, " ("); idx2 >= 0 {
			return after[:idx2]
		}
		return after
	}
	// Take first 60 chars
	if len(errMsg) > 60 {
		return errMsg[:60]
	}
	return errMsg
}

// saveTrainingData saves the updated training data to the JSON file
func saveTrainingData(path string, db *ErrorPatternsDB) {
	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		fmt.Printf("⚠️  Could not create directory %s: %v\n", dir, err)
		return
	}

	data, err := json.MarshalIndent(db, "", "  ")
	if err != nil {
		fmt.Printf("⚠️  Could not marshal training data: %v\n", err)
		return
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		fmt.Printf("⚠️  Could not save training data: %v\n", err)
		return
	}

	fmt.Printf("\n💾 Saved %d patterns to %s\n", len(db.Patterns), path)
}
