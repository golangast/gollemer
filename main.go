package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time" // Added time import

	_ "modernc.org/sqlite" // Pure Go SQLite driver

	"github.com/golangast/gollemer/colors"
	"github.com/golangast/gollemer/internal/sqlite_db"
	"github.com/golangast/gollemer/neural/moe"
	neuralnn "github.com/golangast/gollemer/neural/nn"
	mainvocab "github.com/golangast/gollemer/neural/nnu/vocab"
	"github.com/golangast/gollemer/neural/nnu/word2vec"
	"github.com/golangast/gollemer/tagger/nertagger"
	"github.com/golangast/gollemer/tagger/postagger"
	"github.com/golangast/gollemer/tagger/tag"
)

const kbFilename = "knowledge.json"

// intentIcons maps intents to their visual representation.
var intentIcons = map[string]string{
	"create_webserver": "🌐 [Webserver]",
	"create_handler":   "🔌 [Handler]",
	"create_database":  "🗄️  [Database]",
	"create_file":      "📄 [File]",
	"create_folder":    "📁 [Folder]",
	"create_page":      "🖥️  [Page]",
	"create_form":      "📝 [Form]",
	"create_structure": "🏗️  [Structure]",
	"move_file":        "🚚 [Move]",
	"create_object":    "🔨 [Object]",
}

// paramTriggers maps words like "named" to the key "name".
var paramTriggers = map[string]string{
	"named":   "name",
	"called":  "name",
	"for":     "target",
	"with":    "attribute",
	"in":      "target",
	"into":    "target",
	"to":      "target",
	"using":   "source",
	"usering": "source",
	"from":    "source",
}

// contains is a helper function to check if a string is in a slice of strings.
func contains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

func findProjectRoot() (string, error) {
	currentDir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get current working directory: %v", err)
	}

	for {
		goModPath := filepath.Join(currentDir, "go.mod")
		if _, err := os.Stat(goModPath); err == nil {
			return currentDir, nil // Found go.mod, this is the project root
		}

		parentDir := filepath.Dir(currentDir)
		if parentDir == currentDir {
			// Reached the filesystem root without finding go.mod
			return "", fmt.Errorf("go.mod not found in current directory or any parent directories")
		}
		currentDir = parentDir
	}
}

func findGoModInfo() (modulePath string, projectRoot string, err error) {
	currentDir, err := os.Getwd()
	if err != nil {
		return "", "", fmt.Errorf("failed to get current working directory: %v", err)
	}

	dir := currentDir
	for {
		goModPath := filepath.Join(dir, "go.mod")
		if _, statErr := os.Stat(goModPath); statErr == nil {
			// Found go.mod
			content, readErr := os.ReadFile(goModPath)
			if readErr != nil {
				return "", "", fmt.Errorf("failed to read go.mod file: %v", readErr)
			}
			lines := strings.Split(string(content), "\n")
			for _, line := range lines {
				if strings.HasPrefix(line, "module ") {
					return strings.TrimSpace(strings.TrimPrefix(line, "module ")), dir, nil
				}
			}
			return "", "", fmt.Errorf("module path not found in go.mod")
		}

		parentDir := filepath.Dir(dir)
		if parentDir == dir {
			return "", "", fmt.Errorf("go.mod not found in any parent directory")
		}
		dir = parentDir
	}
}

func main() {
	// Initialize absoluteLastDirConfigPath based on the project root
	projectRoot, err := findProjectRoot()
	if err != nil {
		log.Fatalf("Failed to find project root: %v", err)
	}
	absoluteLastDirConfigPath = filepath.Join(projectRoot, "last_dir.txt")

	trainWord2Vec := flag.Bool("train-word2vec", false, "Train the Word2Vec model")
	trainMoE := flag.Bool("train-moe", false, "Train the MoE model")
	trainIntentClassifier := flag.Bool("train-intent-classifier", false, "Train the intent classification model")
	trainNER := flag.Bool("train-ner", false, "Train the Named Entity Recognition model")
	runLLMFlag := flag.Bool("llm", false, "Run in interactive LLM mode")

	flag.Parse()

	if *runLLMFlag {
		runLLM()
	} else if *trainWord2Vec {
		runModule("cmd/train_word2vec")
	} else if *trainMoE {
		runModule("cmd/train_moe")
	} else if *trainIntentClassifier {
		runModule("cmd/train_intent_classifier")
	} else if *trainNER {
		runModule("cmd/train_ner")
	} else {
		log.Println("No action specified. Use -train-word2vec, -train-moe, -train-intent-classifier, -train-ner, or -llm.")
	}
}

func runModule(path string) {
	cmd := exec.Command("go", "run", "./"+path)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		log.Fatalf("Failed to run module %s: %v", path, err)
	}
}

func buildWasm(wasmDir string) {
	if wasmDir == "" {
		wasmDir = "."
	}
	if _, err := os.Stat(wasmDir); os.IsNotExist(err) {
		return
	}

	// 1. Ensure wasm_exec.js exists
	goroot, err := exec.Command("go", "env", "GOROOT").Output()
	if err == nil {
		gorootPath := strings.TrimSpace(string(goroot))
		// Try multiple locations for wasm_exec.js
		srcs := []string{
			filepath.Join(gorootPath, "misc", "wasm", "wasm_exec.js"),
			filepath.Join(gorootPath, "lib", "wasm", "wasm_exec.js"),
		}
		
		var src string
		for _, s := range srcs {
			if _, err := os.Stat(s); err == nil {
				src = s
				break
			}
		}

		if src != "" {
			dst := filepath.Join(wasmDir, "wasm_exec.js")
			content, err := os.ReadFile(src)
			if err == nil {
				err = os.WriteFile(dst, content, 0644)
				if err == nil {
					fmt.Printf("✅ Copied wasm_exec.js to %s\n", wasmDir)
				} else {
					fmt.Printf("⚠️  Failed to write wasm_exec.js to %s: %v\n", wasmDir, err)
				}
			} else {
				fmt.Printf("⚠️  Failed to read wasm_exec.js from %s: %v\n", src, err)
			}
		} else {
			fmt.Printf("⚠️  Could not find wasm_exec.js in GOROOT (%s)\n", gorootPath)
		}
	}

	fmt.Printf("🏗️  Building WASM in %s...\n", wasmDir)
	// Check for wasm.go or main.go, or in a wasm/ subdirectory
	wasmFile := ""
	candidates := []string{
		"wasm.go",
		"main.go",
		filepath.Join("wasm", "main.go"),
		filepath.Join("wasm", "wasm.go"),
	}

	for _, c := range candidates {
		if _, err := os.Stat(filepath.Join(wasmDir, c)); err == nil {
			wasmFile = c
			break
		}
	}

	if wasmFile == "" {
		fmt.Printf("⚠️  No wasm file found in %s (checked %v), skipping.\n", wasmDir, candidates)
		return
	}

	// Determine if we should use -mod=mod
	args := []string{"build"}
	gowork, _ := exec.Command("go", "env", "GOWORK").Output()
	if strings.TrimSpace(string(gowork)) == "" || strings.TrimSpace(string(gowork)) == "off" {
		args = append(args, "-mod=mod")
	}
	args = append(args, "-o", "main.wasm", wasmFile)

	cmd := exec.Command("go", args...)
	cmd.Dir = wasmDir
	cmd.Env = append(os.Environ(), "GOOS=js", "GOARCH=wasm")
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("❌ WASM build failed in %s: %v\n%s\n", wasmDir, err, string(output))
	} else {
		fmt.Printf("✅ WASM build successful: %s/main.wasm updated.\n", wasmDir)
	}
}

func findName(taggedData tag.Tag, kb *KnowledgeBase) string {

	// First, look for a FILENAME tag

	for i, tag := range taggedData.NerTag {

		if tag == "FILENAME" {

			return taggedData.Tokens[i]

		}

	}

	// Fallback for "named"

	for i, token := range taggedData.Tokens {

		if (token == "named" || token == "called") && i+1 < len(taggedData.Tokens) {

			return taggedData.Tokens[i+1]

		}

	}

	// Fallback for NAME tag
	for i, tag := range taggedData.NerTag {
		if tag == "NAME" {
			return taggedData.Tokens[i]
		}
	}
	
	// Final heuristic fallback: first non-keyword after a known Object Type token
	objectTypeKeywords := map[string]bool{
		"handler": true, "webserver": true, "page": true, "file": true, 
		"folder": true, "database": true, "structure": true, "component": true,
	}
	for i, token := range taggedData.Tokens {
		lower := strings.ToLower(token)
		if (objectTypeKeywords[lower] || (kb != nil && kb.KnownObjects[lower])) && i+1 < len(taggedData.Tokens) {
			candidate := taggedData.Tokens[i+1]
			lowerC := strings.ToLower(candidate)
			// Ensure it's not another keyword or preposition
			if !objectTypeKeywords[lowerC] && lowerC != "with" && lowerC != "the" && lowerC != "named" {
				return candidate
			}
		}
	}

	return ""

}

var absoluteLastDirConfigPath string // Global variable for the absolute path to last_dir.txt

func saveLastDirectory(dirPath string) {

	err := os.WriteFile(absoluteLastDirConfigPath, []byte(dirPath), 0644)

	if err != nil {

		log.Printf("Error saving last directory to %s: %v", absoluteLastDirConfigPath, err)

	}

}

func loadLastDirectory() (string, error) {

	content, err := os.ReadFile(absoluteLastDirConfigPath)

	if err != nil {

		return "", fmt.Errorf("error reading last directory from %s: %v", absoluteLastDirConfigPath, err)

	}

	return strings.TrimSpace(string(content)), nil

}

// findClosestObject uses Levenshtein distance to find the nearest known object.
func findClosestObject(target string, known map[string]bool) (string, int) {
	closest := ""
	minDist := 999

	for obj := range known {
		dist := levenshteinDistance(target, obj)
		if dist < minDist {
			minDist = dist
			closest = obj
		}
	}
	return closest, minDist
}

func levenshteinDistance(s1, s2 string) int {
	s1Raw := []rune(s1)
	s2Raw := []rune(s2)
	len1 := len(s1Raw)
	len2 := len(s2Raw)

	column := make([]int, len1+1)
	for y := 1; y <= len1; y++ {
		column[y] = y
	}

	for x := 1; x <= len2; x++ {
		column[0] = x
		lastkey := x - 1
		for y := 1; y <= len1; y++ {
			oldkey := column[y]
			var incr int
			if s1Raw[y-1] != s2Raw[x-1] {
				incr = 1
			}

			column[y] = min(column[y]+1, min(column[0]+1, lastkey+incr))
			lastkey = oldkey
		}
	}
	return column[len1]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func createTableWithFields(dbFileName, tableName string, fields map[string]string) error {
	db, err := sql.Open("sqlite", dbFileName)
	if err != nil {
		return fmt.Errorf("couldn't open the database file %s: %v", dbFileName, err)
	}
	defer db.Close()

	sqlStatement := fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (\n", tableName)
	columns := []string{"\tid INTEGER PRIMARY KEY AUTOINCREMENT"}
	for fieldName, fieldType := range fields {
		sqlType := ""
		switch strings.ToLower(fieldType) {
		case "string":
			sqlType = "TEXT"
		case "int":
			sqlType = "INTEGER"
		// Add more type mappings as needed
		default:
			sqlType = "TEXT" // Default to TEXT for unknown types
		}
		columns = append(columns, fmt.Sprintf("\t%s %s", strings.ToLower(fieldName), sqlType))
	}
	sqlStatement += strings.Join(columns, ",\n")
	sqlStatement += "\n);"

	_, err = db.Exec(sqlStatement)
	if err != nil {
		return fmt.Errorf("couldn't create the table '%s' in %s: %v", tableName, dbFileName, err)
	}
	return nil
}

func deleteColumnFromTable(dbFileName, tableName, columnToDelete string, remainingFields map[string]string) error {
	db, err := sql.Open("sqlite", dbFileName)
	if err != nil {
		return fmt.Errorf("couldn't open the database file %s: %v", dbFileName, err)
	}
	defer db.Close()

	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("could not begin transaction: %w", err)
	}

	// 1. Create a new temporary table
	tempTableName := tableName + "_temp_gollemer"
	columns := []string{"id INTEGER PRIMARY KEY AUTOINCREMENT"}
	var fieldNames []string
	for fieldName, fieldType := range remainingFields {
		sqlType := "TEXT"
		switch strings.ToLower(fieldType) {
		case "string":
			sqlType = "TEXT"
		case "int":
			sqlType = "INTEGER"
		}
		columns = append(columns, fmt.Sprintf("%s %s", strings.ToLower(fieldName), sqlType))
		fieldNames = append(fieldNames, strings.ToLower(fieldName))
	}
	sort.Strings(fieldNames)

	createSQL := fmt.Sprintf("CREATE TABLE %s (%s)", tempTableName, strings.Join(columns, ", "))
	if _, err := tx.Exec(createSQL); err != nil {
		tx.Rollback()
		return fmt.Errorf("failed to create temp table: %w", err)
	}

	// 2. Copy data from the old table to the new table
	columnList := "id, " + strings.Join(fieldNames, ", ")
	insertSQL := fmt.Sprintf("INSERT INTO %s (%s) SELECT %s FROM %s", tempTableName, columnList, columnList, tableName)
	if _, err := tx.Exec(insertSQL); err != nil {
		tx.Rollback()
		return fmt.Errorf("failed to copy data to temp table: %w", err)
	}

	// 3. Drop the old table
	dropSQL := fmt.Sprintf("DROP TABLE %s", tableName)
	if _, err := tx.Exec(dropSQL); err != nil {
		tx.Rollback()
		return fmt.Errorf("failed to drop old table: %w", err)
	}

	// 4. Rename the new table
	renameSQL := fmt.Sprintf("ALTER TABLE %s RENAME TO %s", tempTableName, tableName)
	if _, err := tx.Exec(renameSQL); err != nil {
		tx.Rollback()
		return fmt.Errorf("failed to rename temp table: %w", err)
	}

	return tx.Commit()
}

func registerHandlerURL(handlerName, handlerURL, mainGoPath string) (string, error) {
	mainGoContent, err := os.ReadFile(mainGoPath)
	if err != nil {
		return "", fmt.Errorf("could not read %s: %w", mainGoPath, err)
	}

	if handlerURL == "" {
		handlerURL = "/" + strings.ToLower(handlerName)
	}

	registration := fmt.Sprintf("http.HandleFunc(\"%s\", %sHandler)", handlerURL, handlerName)
	if !strings.Contains(string(mainGoContent), registration) {
		// Detect the placeholder with any indentation
		placeholder := "// HANDLER_REGISTRATIONS_GO_HERE"
		lines := strings.Split(string(mainGoContent), "\n")
		var updatedLines []string
		found := false
		for _, line := range lines {
			if strings.Contains(line, placeholder) {
				// preserve indentation
				indent := ""
				for _, char := range line {
					if char == ' ' || char == '\t' {
						indent += string(char)
					} else {
						break
					}
				}
				updatedLines = append(updatedLines, fmt.Sprintf("%shttp.HandleFunc(\"%s\", %sHandler)", indent, handlerURL, handlerName))
				updatedLines = append(updatedLines, line) // Keep the placeholder for future registrations
				found = true
			} else {
				updatedLines = append(updatedLines, line)
			}
		}

		if !found {
			return "", fmt.Errorf("placeholder '%s' not found in %s", placeholder, mainGoPath)
		}

		updatedMainGoContent := strings.Join(updatedLines, "\n")

		err = os.WriteFile(mainGoPath, []byte(updatedMainGoContent), 0644)
		if err != nil {
			return "", fmt.Errorf("could not write to %s: %w", mainGoPath, err)
		}
		goImports(mainGoPath)
		return fmt.Sprintf("And registered it to URL '%s' in %s.", handlerURL, mainGoPath), nil
	}

	return fmt.Sprintf("The URL '%s' for handler '%s' is already registered in %s.", handlerURL, handlerName, mainGoPath), nil
}

func registerHandlerWithPackage(packageName, packageImportPath, handlerName, handlerURL, mainGoPath string) (string, error) {
	contentBytes, err := os.ReadFile(mainGoPath)
	if err != nil {
		return "", fmt.Errorf("could not read %s: %w", mainGoPath, err)
	}
	content := string(contentBytes)
	originalContent := content

	// Add import if not present
	importStatement := fmt.Sprintf("\"%s\"", packageImportPath)
	if !strings.Contains(content, importStatement) {
		content = strings.Replace(content, "import (", "import (\n\t"+importStatement, 1)
	}

	// Add handler if not present
	handlerFqn := fmt.Sprintf("%s.%sHandler", packageName, handlerName)
	handlerRegistration := fmt.Sprintf("http.HandleFunc(\"%s\", %s)", handlerURL, handlerFqn)
	if !strings.Contains(content, handlerRegistration) {
		newHandleFunc := fmt.Sprintf("\thttp.HandleFunc(\"%s\", %s)\n\t// HANDLER_REGISTRATIONS_GO_HERE", handlerURL, handlerFqn)
		content = strings.Replace(content, "// HANDLER_REGISTRATIONS_GO_HERE", newHandleFunc, 1) // Expect unindented placeholder
	}

	if content == originalContent {
		return fmt.Sprintf("The URL '%s' for handler '%s' is already registered in %s.", handlerURL, handlerName, mainGoPath), nil
	}

	err = os.WriteFile(mainGoPath, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("could not write to %s: %w", mainGoPath, err)
	}
	goImports(mainGoPath)
	return fmt.Sprintf("And registered it to URL '%s' in %s.", handlerURL, mainGoPath), nil
}

func goImports(filename string) {
	cmd := exec.Command("goimports", "-w", filename)
	err := cmd.Run()
	if err != nil {
		// Log the error but don't fail, as goimports might not be installed.
		if !strings.Contains(err.Error(), "executable file not found") {
			log.Printf("goimports failed for %s: %v", filename, err)
		}
	}
}

// GollemerMoEClient implements the MoEClient interface using the existing NLP pipeline.
type GollemerMoEClient struct {
	KB *KnowledgeBase
}

func (c *GollemerMoEClient) PredictIntent(input string) (string, float64) {
	lowerInput := strings.ToLower(input)
	words := strings.Fields(lowerInput)

	// Heuristic Intent Detection with Semantic Variations
	createVerbs := []string{"create", "make", "add", "generate", "initialize", "init", "new", "setup", "start"}
	isCreating := false
	for _, v := range createVerbs {
		if strings.Contains(lowerInput, v) {
			isCreating = true
			break
		}
	}

	if isCreating {
		targets := map[string]string{
			"webserver": "create_webserver",
			"site":      "create_webserver",
			"project":   "create_webserver",
			"app":       "create_webserver",
			"page":      "create_page",
			"view":      "create_page",
			"homepage":  "create_page",
			"handler":   "create_handler",
			"endpoint":  "create_handler",
			"route":     "create_handler",
			"database":       "create_database",
			"db":             "create_database",
			"file":           "create_file",
			"folder":         "create_folder",
			"directory":      "create_folder",
			"form":           "create_form",
			"structure":      "create_structure",
			"data structure": "create_structure",
		}

		// Proximity search: find which target is closest to the creation verb
		verbIdx := -1
		for i, w := range words {
			for _, v := range createVerbs {
				if w == v {
					verbIdx = i
					break
				}
			}
			if verbIdx != -1 {
				break
			}
		}

		if verbIdx != -1 {
			// Look ahead for the object
			for i := verbIdx + 1; i < len(words); i++ {
				w := words[i]
				// Check for plural variations
				singular := strings.TrimSuffix(w, "s")
				if intent, ok := targets[w]; ok {
					return intent, 0.95
				}
				if intent, ok := targets[singular]; ok {
					return intent, 0.9
				}
			}
			
			// Look behind if ahead failed (e.g. "myapp site initialize")
			for i := verbIdx - 1; i >= 0; i-- {
				w := words[i]
				singular := strings.TrimSuffix(w, "s")
				if intent, ok := targets[w]; ok {
					return intent, 0.85
				}
				if intent, ok := targets[singular]; ok {
					return intent, 0.8
				}
			}
		}

		// Fallback to priority scan if priority nouns are present even if proximity fails
		for _, key := range []string{"webserver", "site", "project", "page", "handler", "database", "file", "folder", "form"} {
			if strings.Contains(lowerInput, key) {
				return targets[key], 0.75
			}
		}

		// --- NEW: Heuristic for Learned Objects ---
		// If we are "creating" but none of our hardcoded nouns matched, check if any word is a KnownObject
		for _, w := range words {
			if c.KB != nil && c.KB.KnownObjects[w] {
				return "create_object", 0.7 // Generic creation intent
			}
		}
	}

	// --- NEW: Command Inference ---
	// If no command verb was found, but the first word is a known major Object, assume "create"
	if lowerInput != "" {
		firstWord := words[0]
		majorObjects := map[string]string{
			"webserver": "create_webserver", "site": "create_webserver",
			"handler": "create_handler", "page": "create_page",
			"database": "create_database", "structure": "create_structure",
		}
		if intent, ok := majorObjects[firstWord]; ok {
			return intent, 0.6 // Lower confidence but gets the job done
		}
	}

	if strings.Contains(lowerInput, "move") && !strings.Contains(lowerInput, "remove") {
		if strings.Contains(lowerInput, "file") {
			return "move_file", 0.9
		}
		return "move_file", 0.8
	}

	return "", 0.0
}

func (c *GollemerMoEClient) ExtractEntities(input string, intent string) map[string]interface{} {
	words := strings.Fields(input)
	posTags := postagger.TagTokens(words)
	taggedData := nertagger.Nertagger(tag.Tag{Tokens: words, PosTag: posTags})

	entities := make(map[string]interface{})

	// Extract Name using existing findName logic
	name := findName(taggedData, c.KB)

	// Improvement for "data structure" name extraction
	if strings.Contains(strings.ToLower(input), "data structure") {
		words := strings.Fields(input)
		for i, w := range words {
			if strings.ToLower(w) == "structure" && i > 0 && strings.ToLower(words[i-1]) == "data" {
				if i+1 < len(words) {
					candidate := words[i+1]
					if strings.ToLower(candidate) != "to" && strings.ToLower(candidate) != "in" && strings.ToLower(candidate) != "with" {
						name = candidate
					}
				}
			}
		}
	}

	// Improvement for "page" name: look for words between "page" and "in/to/for/wasm/webserver"
	if intent == "create_page" {
		lowerInput := strings.ToLower(input)
		pageIdx := strings.Index(lowerInput, "page")
		if pageIdx != -1 {
			remaining := input[pageIdx+4:]
			// Find end marker
			endMarkers := []string{" in ", " to ", " for ", " with ", " wasm ", " webserver "}
			endIdx := len(remaining)
			for _, marker := range endMarkers {
				mIdx := strings.Index(strings.ToLower(remaining), marker)
				if mIdx != -1 && mIdx < endIdx {
					endIdx = mIdx
				}
			}
			candidate := strings.TrimSpace(remaining[:endIdx])
			if candidate != "" {
				name = candidate
			}
		}
	}

	if name != "" {
		entities["name"] = name
	}

	// Extract URL (specific to handlers)
	for i, token := range taggedData.Tokens {
		lower := strings.ToLower(token)
		if lower == "url" && i+1 < len(taggedData.Tokens) {
			// Handle "url /users" or "url is /users"
			val := taggedData.Tokens[i+1]
			if val == "is" && i+2 < len(taggedData.Tokens) {
				val = taggedData.Tokens[i+2]
			}
			if strings.HasPrefix(val, "/") {
				entities["url"] = val
			}
		} else if strings.HasPrefix(token, "/") && !strings.Contains(token, ".") && (intent == "create_handler" || intent == "create_page") {
			// Heuristic: if a token starts with / and it's a handler/page intent, it's likely the URL/Route
			entities["url"] = token
		}
	}

	// Extract Path (for files/folders)
	for i, token := range taggedData.Tokens {
		lowerToken := strings.ToLower(token)
		if (lowerToken == "in" || lowerToken == "into" || lowerToken == "to") && i+1 < len(taggedData.Tokens) {
			// Skip noise words like articles and "folder"/"directory" keywords
			j := i + 1
			for ; j < len(taggedData.Tokens); j++ {
				t := strings.ToLower(taggedData.Tokens[j])
				if t == "the" || t == "a" || t == "an" || t == "folder" || t == "directory" {
					continue
				}
				break
			}
			if j < len(taggedData.Tokens) {
				entities["path"] = taggedData.Tokens[j]
			}
		}
	}

	// Extract Fields (for database)
	if strings.Contains(input, "fields") {
		fields := make(map[string]string)
		parts := strings.Fields(input)
		startIdx := -1

		// Find "fields" keyword
		for i, p := range parts {
			if strings.ToLower(p) == "fields" {
				startIdx = i + 1
				break
			}
		}

		if startIdx != -1 {
			for i := startIdx; i < len(parts)-1; i += 2 {
				if strings.ToLower(parts[i]) == "and" {
					i-- // Adjust for "and"
					continue
				}
				fields[parts[i]] = parts[i+1]
			}
		}
		if len(fields) > 0 {
			entities["tables"] = fields
		}
	}

	return entities
}

// generateDirectoryTree creates a string representation of a directory tree.
func generateDirectoryTree(path string, prefix string, currentDepth int, maxDepth int, highlightPath string) (string, error) {
	var builder strings.Builder

	if currentDepth == 0 {
		if absPath, err := filepath.Abs(path); err == nil {
			builder.WriteString(fmt.Sprintf("📂 \033[36m%s\033[0m\n", absPath))
		} else {
			builder.WriteString(fmt.Sprintf("📂 \033[36m%s\033[0m\n", path))
		}
	}

	if maxDepth != -1 && currentDepth >= maxDepth {
		return builder.String(), nil
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return "", err
	}

	var visibleEntries []os.DirEntry
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), ".") {
			visibleEntries = append(visibleEntries, entry)
		}
	}

	for i, entry := range visibleEntries {
		connector := "├── "
		newPrefix := prefix + "│   "
		if i == len(visibleEntries)-1 {
			connector = "└── "
			newPrefix = prefix + "    "
		}

		entryName := entry.Name()
		fullPath := filepath.Join(path, entryName)

		if highlightPath != "" {
			absEntry, _ := filepath.Abs(fullPath)
			absHighlight, _ := filepath.Abs(highlightPath)
			if absEntry == absHighlight {
				entryName = "\033[32m" + entryName + "\033[0m"
			}
		}

		builder.WriteString(prefix + connector + entryName + "\n")

		if entry.IsDir() {
			subTree, err := generateDirectoryTree(fullPath, newPrefix, currentDepth+1, maxDepth, highlightPath)
			if err != nil {
				builder.WriteString(newPrefix + "└── [error reading dir]\n")
			} else {
				builder.WriteString(subTree)
			}
		}
	}
	return builder.String(), nil
}

type ConversationState struct {
	ActiveIntent    string
	Parameters      map[string]string
	Missing         []string
	IsActive        bool
	SuggestedObject string // Added for smart suggestions
}

type TutorialState struct {
	Active bool
	Step   int
}

func printIntro() {
	fmt.Println("--- Welcome to Gollemer! ʕ◔ϖ◔ʔ ---")
	fmt.Println("It looks like this is your first time running Gollemer.")
	fmt.Println("\n💡 TIP: Type 'tutorial' to start an interactive guide!\n")
	fmt.Println("💡 TIP: Type 'menu' for an easy-to-use options menu!\n")
	fmt.Println("Here is a quick guide to get you started:")
	fmt.Println("")
	fmt.Println("1. Commands:")
	fmt.Println("   You can use natural language to interact with your project.")
	fmt.Println("   - Navigation: 'go to cmd', 'list files', 'tree'")
	fmt.Println("   - File Ops:   'create file main.go', 'delete folder tmp'")
	fmt.Println("   - Web Dev:    'create webserver MyApp', 'create handler Login', 'run webserver'")
	fmt.Println("   - System:     'clear', 'exit', 'history'")
	fmt.Println("")
	fmt.Println("2. The Learning System (How & Why):")
	fmt.Println("   Gollemer learns from your code to automate repetitive tasks.")
	fmt.Println("   - HOW: It scans a 'learningfolder' for templates (files like 'navbar.html', 'auth.go').")
	fmt.Println("          If it finds 'navbar.html', it learns the 'navbar' object.")
	fmt.Println("   - WHY: So you can say 'create navbar' and it generates the code for you instantly,")
	fmt.Println("          using your own preferred style and structure.")
	fmt.Println("")
	fmt.Println("3. Customizing Learning:")
	fmt.Println("   You have full control over what Gollemer learns.")
	fmt.Println("   - Add/Edit files in the 'learningfolder' to teach it new objects.")
	fmt.Println("   - Change the source folder: 'learn from ./my-templates'")
	fmt.Println("   - Teach specific words: 'learn object widget'")
	fmt.Println("")
	fmt.Println("Type 'help' at any time to see this information again.")
	fmt.Println("----------------------------------------")
	fmt.Println("")
}

func runLLM() {
	projectRoot, err := findProjectRoot()
	if err != nil {
		log.Fatalf("Failed to find project root: %v", err)
	}
	log.Printf("DEBUG: Project Root: %s", projectRoot)

	cmd := exec.Command("clear")
	cmd.Stdout = os.Stdout
	cmd.Run()

	// Initialize database once
	dbFileName := "gollemer.db"
	db, err := sqlite_db.InitDB(dbFileName)
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	defer db.Close()

	// Load KnowledgeBase
	kb := LoadKnowledgeBase()

	if kb.FirstRun {
		printIntro()
		kb.FirstRun = false
		kb.Save()
	}

	reader := bufio.NewReader(os.Stdin)

	// Load last directory on startup
	lastDir, err := loadLastDirectory()
	if err == nil {
		err := os.Chdir(lastDir)
		if err != nil {
			log.Printf("Current directory %s: %v", lastDir, err)
		} else {
			currentAbsDir, _ := os.Getwd()
			log.Printf("Current directory: %s", currentAbsDir)
		}
	} else {
		currentAbsDir, _ := os.Getwd()
		log.Printf("DEBUG: No last directory loaded. Current Working Directory: %s", currentAbsDir)
	}

	// Initialize Intent Resolver
	resolver := NewHybridIntentResolver(&GollemerMoEClient{KB: kb})

	var commandHistory []string
	var sessionState ConversationState
	var tutorialState TutorialState

	inMenuMode := false

	for {
		var query string
		if inMenuMode {
			query = "menu"
		} else {
			colors.ColorizeCol("red", "magenta", "/ʕ◔ϖ◔ʔ/> ")
			query, _ = reader.ReadString('\n')
			query = strings.TrimSpace(query)
		}

		if query != "" {
			commandHistory = append(commandHistory, query)
		}

		if query == "exit" {
			break
		} else if query == "clear" {
			cmd := exec.Command("clear")
			cmd.Stdout = os.Stdout
			cmd.Run()
			continue
		}

		if query == "menu" {
			inMenuMode = true
			fmt.Println("\n--- 📋 Main Menu ---")
			fmt.Println("1. 🚀 Start a New Project (Webserver)")
			fmt.Println("2. ➕ Add a Feature (Handler, Page, Database)")
			fmt.Println("3. 📂 Manage Files (Create, Delete, Move)")
			fmt.Println("4. ▶️  Run Project")
			fmt.Println("5. 🧠 Learning & Training")
			fmt.Println("6. 🎓 Tutorial")
			fmt.Println("7. ❓ Help")
			fmt.Println("8. 🚪 Exit")
			fmt.Println("9. 💬 Interactive Mode")
			fmt.Println("10. ⚙️ Model Configuration")
			fmt.Print("\nSelect an option (1-10): ")

			choice, _ := reader.ReadString('\n')
			choice = strings.TrimSpace(choice)

			switch choice {
			case "1":
				fmt.Print("Enter name for your new webserver: ")
				name, _ := reader.ReadString('\n')
				name = strings.TrimSpace(name)
				if name != "" {
					query = "create webserver " + name
				} else {
					continue
				}
			case "2":
				fmt.Println("\nWhat do you want to add?")
				fmt.Println("a. Handler (Backend logic)")
				fmt.Println("b. Page (Frontend view)")
				fmt.Println("c. Database (Storage)")
				fmt.Print("Select (a/b/c): ")
				sub, _ := reader.ReadString('\n')
				sub = strings.TrimSpace(sub)
				if sub == "a" {
					fmt.Print("Handler Name: ")
					n, _ := reader.ReadString('\n')
					query = "create handler " + strings.TrimSpace(n)
				} else if sub == "b" {
					fmt.Print("Page Name: ")
					n, _ := reader.ReadString('\n')
					query = "create page " + strings.TrimSpace(n)
				} else if sub == "c" {
					fmt.Print("Database Name: ")
					n, _ := reader.ReadString('\n')
					query = "create database " + strings.TrimSpace(n)
				} else {
					continue
				}
			case "3":
				fmt.Println("\nWhat file operation?")
				fmt.Println("a. Create File")
				fmt.Println("b. Create Folder")
				fmt.Print("Select (a/b): ")
				sub, _ := reader.ReadString('\n')
				sub = strings.TrimSpace(sub)
				if sub == "a" {
					fmt.Print("File Name: ")
					n, _ := reader.ReadString('\n')
					query = "create file " + strings.TrimSpace(n)
				} else if sub == "b" {
					fmt.Print("Folder Name: ")
					n, _ := reader.ReadString('\n')
					query = "create folder " + strings.TrimSpace(n)
				} else {
					continue
				}
			case "4":
				fmt.Print("Enter webserver name to run (or press enter for current): ")
				n, _ := reader.ReadString('\n')
				n = strings.TrimSpace(n)
				if n != "" {
					query = "run webserver " + n
				} else {
					query = "run webserver"
				}
			case "5":
				fmt.Println("\n--- 🧠 Learning & Training ---")
				if kb.LearningPath != "" {
					fmt.Printf("Current Learning Path: %s\n", kb.LearningPath)
				}
				fmt.Println("1. Show Learning Status (Data & Vocab)")
				fmt.Println("2. Change Learning Source (Folder)")
				fmt.Println("3. Teach New Object Word")
				fmt.Println("4. Run Training Commands")
				fmt.Print("Select (1-4): ")
				sub, _ := reader.ReadString('\n')
				sub = strings.TrimSpace(sub)
				if sub == "1" {
					fmt.Println("\n--- 📊 Learning Status ---")
					fmt.Printf("Knowledge Base: %s\n", kbFilename)
					if kb.LearningPath != "" {
						fmt.Printf("Templates Source: %s\n", kb.LearningPath)
					} else {
						fmt.Println("Templates Source: ./learningfolder (Default)")
					}

					fmt.Println("\n[Training Data & Vocab]")
					checkPath := func(name, path string) {
						fullPath := filepath.Join(projectRoot, path)
						if _, err := os.Stat(fullPath); err == nil {
							fmt.Printf("  ✅ %s: %s\n", name, path)
						} else {
							fmt.Printf("  ❌ %s: %s (Not found)\n", name, path)
						}
					}
					checkPath("Word2Vec", kb.ModelConfig.Word2VecPath)
					checkPath("MoE", kb.ModelConfig.MoEPath)
					checkPath("NER", kb.ModelConfig.NERPath)
					checkPath("Query Vocab", kb.ModelConfig.QueryVocabPath)
					checkPath("Semantic Vocab", kb.ModelConfig.SemanticVocabPath)

					fmt.Println("\n[Metrics]")
					qVocabPath := filepath.Join(projectRoot, kb.ModelConfig.QueryVocabPath)
					if qVocab, err := mainvocab.LoadVocabulary(qVocabPath); err == nil {
						fmt.Printf("  📝 Query Vocabulary: %d words\n", len(qVocab.WordToToken))
					} else {
						fmt.Printf("  ⚠️  Could not load Query Vocabulary: %v\n", err)
					}

					sVocabPath := filepath.Join(projectRoot, kb.ModelConfig.SemanticVocabPath)
					if sVocab, err := mainvocab.LoadVocabulary(sVocabPath); err == nil {
						fmt.Printf("  📝 Semantic Output Vocabulary: %d tokens\n", len(sVocab.WordToToken))
					} else {
						fmt.Printf("  ⚠️  Could not load Semantic Vocabulary: %v\n", err)
					}

					continue
				} else if sub == "2" {
					fmt.Print("Enter path to learning folder (e.g., ./templates): ")
					path, _ := reader.ReadString('\n')
					query = "learn from " + strings.TrimSpace(path)
				} else if sub == "3" {
					fmt.Print("Enter object name to learn: ")
					obj, _ := reader.ReadString('\n')
					query = "learn object " + strings.TrimSpace(obj)
				} else if sub == "4" {
					fmt.Println("\n--- 🏋️ Run Training ---")
					fmt.Println("1. Train Word2Vec")
					fmt.Println("2. Train MoE")
					fmt.Println("3. Train Intent Classifier")
					fmt.Println("4. Train NER")
					fmt.Println("5. Custom Training Module")
					fmt.Println("6. Visualize Neural Network")
					fmt.Println("7. Visualize Word2Vec Model")
					fmt.Println("8. Search Word Neighbors")
					fmt.Println("9. Visualize Word Relationship")
					fmt.Println("10. Visualize Word Distribution (2D Plot)")
					fmt.Println("11. Inspect Model Weights")
					fmt.Println("12. Visualize Attention Mechanism")
					fmt.Println("13. Visualize Word Similarity (One vs List)")
					fmt.Print("Select (1-13): ")
					trainSub, _ := reader.ReadString('\n')
					trainSub = strings.TrimSpace(trainSub)

					var cmdPath string
					switch trainSub {
					case "1":
						cmdPath = "cmd/train_word2vec"
					case "2":
						cmdPath = "cmd/train_moe"
					case "3":
						cmdPath = "cmd/train_intent_classifier"
					case "4":
						cmdPath = "cmd/train_ner"
					case "5":
						cmdPath = "cmd/train_custom" // Ensure this directory exists with a main.go
					case "6":
						modelPath := filepath.Join(projectRoot, kb.ModelConfig.MoEPath)
						if _, err := os.Stat(modelPath); os.IsNotExist(err) {
							fmt.Printf("❌ Model file not found at %s\n", modelPath)
						} else {
							nn, err := moe.LoadIntentMoEModelFromGOB(modelPath)
							if err != nil {
								fmt.Printf("❌ Failed to load model: %v\n", err)
							} else {
								fmt.Println("\n--- 🕸️ Neural Network Architecture ---")
								fmt.Println("")
								fmt.Println("       [ Input Query ]")
								fmt.Println("             ⬇")
								fmt.Println("  ╔═══════════════════════╗")
								if nn.Embedding != nil {
									fmt.Printf("  ║    Embedding Layer    ║  Dimension: %d\n", nn.Embedding.DimModel)
								} else {
									fmt.Println("  ║    Embedding Layer    ║")
								}
								fmt.Println("  ╚═══════════════════════╝")
								fmt.Println("             ⬇")
								fmt.Println("  ╔═══════════════════════╗")
								encoderType := fmt.Sprintf("%T", nn.Encoder)
								if strings.Contains(encoderType, "SimpleRNNEncoder") {
									encoderType = "Simple RNN"
								}
								fmt.Printf("  ║        Encoder        ║  Type: %s\n", encoderType)
								fmt.Println("  ╚═══════════════════════╝")
								fmt.Println("             ⬇")
								fmt.Println("  ╔═══════════════════════╗")
								if nn.Decoder != nil && nn.Decoder.LSTM != nil {
									fmt.Printf("  ║        Decoder        ║  Hidden Size: %d\n", nn.Decoder.LSTM.HiddenSize)
								} else {
									fmt.Println("  ║        Decoder        ║")
								}
								fmt.Println("  ╚═══════════════════════╝")
								fmt.Println("             ⬇")
								fmt.Println("  ╔═══════════════════════╗")
								fmt.Printf("  ║     Output Vocab      ║  Size: %d\n", nn.SentenceVocabSize)
								fmt.Println("  ╚═══════════════════════╝")
								fmt.Println("             ⬇")
								fmt.Println("      [ Predicted Intent ]")
								fmt.Println("")
								fmt.Println("---------------------------------------")
							}
						}
						continue
					case "7":
						modelPath := filepath.Join(projectRoot, kb.ModelConfig.Word2VecPath)
						if _, err := os.Stat(modelPath); os.IsNotExist(err) {
							fmt.Printf("❌ Word2Vec model file not found.\n")
						} else {
							sw2v, err := word2vec.LoadModel(modelPath)
							if err != nil {
								fmt.Printf("❌ Failed to load Word2Vec model: %v\n", err)
							} else {
								fmt.Println("\n--- 🔤 Word2Vec Model Visualization ---")
								fmt.Printf("Vector Size: %d\n", sw2v.VectorSize)
								fmt.Printf("Vocabulary Count: %d words\n", len(sw2v.Vocabulary))
								fmt.Println("---------------------------------------")
							}
						}
						continue
					case "8":
						modelPath := filepath.Join(projectRoot, kb.ModelConfig.Word2VecPath)
						if _, err := os.Stat(modelPath); os.IsNotExist(err) {
							fmt.Printf("❌ Word2Vec model file not found.\n")
						} else {
							sw2v, err := word2vec.LoadModel(modelPath)
							if err != nil {
								fmt.Printf("❌ Failed to load Word2Vec model: %v\n", err)
							} else {
								fmt.Print("Enter word to find neighbors for: ")
								targetWord, _ := reader.ReadString('\n')
								targetWord = strings.TrimSpace(targetWord)

								if targetIdx, ok := sw2v.Vocabulary[targetWord]; !ok {
									fmt.Printf("❌ Word '%s' not found in vocabulary.\n", targetWord)
								} else {
									targetVec := sw2v.WordVectors[targetIdx]
									type result struct {
										Word  string
										Score float64
									}
									var results []result
									for word, idx := range sw2v.Vocabulary {
										if word == targetWord {
											continue
										}
										if vec, ok := sw2v.WordVectors[idx]; ok {
											score := cosineSimilarity(targetVec, vec)
											results = append(results, result{word, score})
										}
									}
									sort.Slice(results, func(i, j int) bool {
										return results[i].Score > results[j].Score
									})

									fmt.Printf("\n--- Neighbors for '%s' ---\n", targetWord)
									limit := 10
									if len(results) < limit {
										limit = len(results)
									}
									for i := 0; i < limit; i++ {
										fmt.Printf("  %d. %s (%.4f)\n", i+1, results[i].Word, results[i].Score)
									}
									fmt.Println("---------------------------")
								}
							}
						}
						continue
					case "9":
						modelPath := filepath.Join(projectRoot, kb.ModelConfig.Word2VecPath)
						if _, err := os.Stat(modelPath); os.IsNotExist(err) {
							fmt.Printf("❌ Word2Vec model file not found.\n")
						} else {
							sw2v, err := word2vec.LoadModel(modelPath)
							if err != nil {
								fmt.Printf("❌ Failed to load Word2Vec model: %v\n", err)
							} else {
								fmt.Print("Enter first word: ")
								word1, _ := reader.ReadString('\n')
								word1 = strings.TrimSpace(word1)

								fmt.Print("Enter second word: ")
								word2, _ := reader.ReadString('\n')
								word2 = strings.TrimSpace(word2)

								idx1, ok1 := sw2v.Vocabulary[word1]
								idx2, ok2 := sw2v.Vocabulary[word2]

								if !ok1 {
									fmt.Printf("❌ Word '%s' not found in vocabulary.\n", word1)
								} else if !ok2 {
									fmt.Printf("❌ Word '%s' not found in vocabulary.\n", word2)
								} else {
									vec1 := sw2v.WordVectors[idx1]
									vec2 := sw2v.WordVectors[idx2]
									similarity := cosineSimilarity(vec1, vec2)
									fmt.Printf("\n--- Relationship: '%s' <-> '%s' ---\n", word1, word2)
									fmt.Printf("Cosine Similarity: %.4f\n", similarity)
									fmt.Println("---------------------------------------")
								}
							}
						}
						continue
					case "10":
						modelPath := filepath.Join(projectRoot, kb.ModelConfig.Word2VecPath)
						if _, err := os.Stat(modelPath); os.IsNotExist(err) {
							fmt.Printf("❌ Word2Vec model file not found.\n")
						} else {
							sw2v, err := word2vec.LoadModel(modelPath)
							if err != nil {
								fmt.Printf("❌ Failed to load Word2Vec model: %v\n", err)
							} else {
								fmt.Println("Generating 2D visualization (HTML)...")

								// Limit to N words for visualization to keep it performant
								limit := 500
								count := 0
								var words []string
								var vectors [][]float64

								for w, idx := range sw2v.Vocabulary {
									if count >= limit {
										break
									}
									if idx < len(sw2v.WordVectors) {
										words = append(words, w)
										vectors = append(vectors, sw2v.WordVectors[idx])
										count++
									}
								}

								// Generate HTML
								htmlContent := generateWordVizHTML(words, vectors)
								outputPath := filepath.Join(projectRoot, "word_distribution.html")
								if err := os.WriteFile(outputPath, []byte(htmlContent), 0644); err != nil {
									fmt.Printf("❌ Failed to write HTML file: %v\n", err)
								} else {
									fmt.Printf("✅ Visualization generated at: %s\n", outputPath)
									fmt.Println("   Open this file in your web browser to view the plot.")
								}
							}
						}
						continue
					case "11":
						modelPath := filepath.Join(projectRoot, kb.ModelConfig.MoEPath)
						if _, err := os.Stat(modelPath); os.IsNotExist(err) {
							fmt.Printf("❌ Model file not found at %s\n", modelPath)
						} else {
							nn, err := moe.LoadIntentMoEModelFromGOB(modelPath)
							if err != nil {
								fmt.Printf("❌ Failed to load model: %v\n", err)
							} else {
								fmt.Println("\n--- ⚖️  Model Weights Inspection ---")
								fmt.Println("1. Embedding Layer")
								fmt.Println("2. Encoder")
								fmt.Println("3. Decoder")
								fmt.Print("Select component (1-3): ")
								compSub, _ := reader.ReadString('\n')
								compSub = strings.TrimSpace(compSub)

								switch compSub {
								case "1":
									if nn.Embedding != nil {
										fmt.Println("--- Embedding Layer ---")
										inspectStruct(nn.Embedding, "  ")
									} else {
										fmt.Println("Embedding layer is nil.")
									}
								case "2":
									fmt.Println("--- Encoder ---")
									inspectStruct(nn.Encoder, "  ")
								case "3":
									fmt.Println("--- Decoder ---")
									inspectStruct(nn.Decoder, "  ")
								}
							}
						}
						continue
					case "12":
						modelPath := filepath.Join(projectRoot, kb.ModelConfig.MoEPath)
						if _, err := os.Stat(modelPath); os.IsNotExist(err) {
							fmt.Printf("❌ Model file not found at %s\n", modelPath)
						} else {
							nnModel, err := moe.LoadIntentMoEModelFromGOB(modelPath)
							if err != nil {
								fmt.Printf("❌ Failed to load model: %v\n", err)
							} else {
								fmt.Println("\n--- 👁️ Attention Mechanism Visualization ---")
								findAndVisualizeAttention(nnModel)
							}
						}
						continue
					case "13":
						modelPath := filepath.Join(projectRoot, kb.ModelConfig.Word2VecPath)
						if _, err := os.Stat(modelPath); os.IsNotExist(err) {
							fmt.Printf("❌ Word2Vec model file not found.\n")
						} else {
							sw2v, err := word2vec.LoadModel(modelPath)
							if err != nil {
								fmt.Printf("❌ Failed to load Word2Vec model: %v\n", err)
							} else {
								fmt.Print("Enter target word: ")
								targetWord, _ := reader.ReadString('\n')
								targetWord = strings.TrimSpace(targetWord)

								fmt.Print("Enter list of words to compare (comma separated): ")
								listStr, _ := reader.ReadString('\n')
								listStr = strings.TrimSpace(listStr)
								compareWords := strings.Split(listStr, ",")

								targetIdx, ok := sw2v.Vocabulary[targetWord]
								if !ok {
									fmt.Printf("❌ Target word '%s' not found in vocabulary.\n", targetWord)
								} else {
									targetVec := sw2v.WordVectors[targetIdx]
									fmt.Printf("\n--- Similarity with '%s' ---\n", targetWord)

									type simResult struct {
										Word  string
										Score float64
									}
									var results []simResult

									for _, w := range compareWords {
										w = strings.TrimSpace(w)
										if w == "" {
											continue
										}
										idx, ok := sw2v.Vocabulary[w]
										if !ok {
											fmt.Printf("  %s: [Not in vocab]\n", w)
											continue
										}
										vec := sw2v.WordVectors[idx]
										score := cosineSimilarity(targetVec, vec)
										results = append(results, simResult{Word: w, Score: score})
									}

									sort.Slice(results, func(i, j int) bool {
										return results[i].Score > results[j].Score
									})

									for _, res := range results {
										fmt.Printf("  %s: %.4f\n", res.Word, res.Score)
									}
									fmt.Println("---------------------------------------")
								}
							}
						}
						continue
					}

					if cmdPath != "" {
						fmt.Printf("Running %s...\n", cmdPath)
						c := exec.Command("go", "run", "./"+cmdPath)
						c.Dir = projectRoot
						c.Stdout = os.Stdout
						c.Stderr = os.Stderr
						if err := c.Run(); err != nil {
							fmt.Printf("Error running training: %v\n", err)
						} else {
							fmt.Println("Training completed.")
						}
					}
					continue
				} else {
					continue
				}
			case "6":
				tutorialState.Active = true
				tutorialState.Step = 1
				inMenuMode = false
				colors.AnimatedOutput("green", "black", "--- Tutorial Mode Started ---\nWelcome to the Gollemer tutorial! I will guide you through the basics.\nStep 1: Let's start by creating a project folder.\nTry typing: 'create folder myproject'", 1*time.Second)
				fmt.Println("\n")
				continue
			case "7":
				query = "help"
			case "8":
				os.Exit(0)
			case "9":
				inMenuMode = false
				fmt.Println("Returning to interactive mode...")
				continue
			case "10":
				fmt.Println("\n--- ⚙️ Model Configuration ---")
				fmt.Printf("1. Word2Vec Model: %s\n", kb.ModelConfig.Word2VecPath)
				fmt.Printf("2. MoE Model: %s\n", kb.ModelConfig.MoEPath)
				fmt.Printf("3. Query Vocab: %s\n", kb.ModelConfig.QueryVocabPath)
				fmt.Printf("4. Semantic Vocab: %s\n", kb.ModelConfig.SemanticVocabPath)
				fmt.Printf("5. NER Model: %s\n", kb.ModelConfig.NERPath)
				fmt.Println("6. Back to Main Menu")
				fmt.Print("Select an item to change (1-6): ")

				confChoice, _ := reader.ReadString('\n')
				confChoice = strings.TrimSpace(confChoice)

				var fieldName string
				switch confChoice {
				case "1":
					fieldName = "Word2Vec Model"
				case "2":
					fieldName = "MoE Model"
				case "3":
					fieldName = "Query Vocab"
				case "4":
					fieldName = "Semantic Vocab"
				case "5":
					fieldName = "NER Model"
				case "6":
					continue
				default:
					fmt.Println("Invalid option.")
					continue
				}

				fmt.Printf("Enter new path for %s: ", fieldName)
				newPath, _ := reader.ReadString('\n')
				newPath = strings.TrimSpace(newPath)
				if newPath != "" {
					switch confChoice {
					case "1":
						kb.ModelConfig.Word2VecPath = newPath
					case "2":
						kb.ModelConfig.MoEPath = newPath
					case "3":
						kb.ModelConfig.QueryVocabPath = newPath
					case "4":
						kb.ModelConfig.SemanticVocabPath = newPath
					case "5":
						kb.ModelConfig.NERPath = newPath
					}
					kb.Save()
					fmt.Println("Configuration saved.")
				}
				continue
			default:
				fmt.Println("Invalid option.")
				continue
			}
			fmt.Printf("Executing: %s\n", query)
		}

		if query == "tutorial" {
			tutorialState.Active = true
			tutorialState.Step = 1
			colors.AnimatedOutput("green", "black", "--- Tutorial Mode Started ---\nWelcome to the Gollemer tutorial! I will guide you through the basics.\nStep 1: Let's start by creating a project folder.\nTry typing: 'create folder myproject'", 1*time.Second)
			fmt.Println("\n")
			continue
		} else if query == "show learning path" {
			if kb.LearningPath != "" {
				fmt.Printf("Current learning path: %s\n", kb.LearningPath)
			} else {
				fmt.Println("No learning path set. Defaulting to 'learningfolder' in project root.")
			}
			continue
		} else if strings.HasPrefix(query, "learn ") {
			parts := strings.Fields(query)
			hasMore := false
			if len(parts) >= 3 && parts[1] == "object" {
				newObject := strings.ToLower(parts[2])
				kb.KnownObjects[newObject] = true
				kb.Save()
				fmt.Printf("Knowledge Base updated: '%s' is now a known object.\n", newObject)
			} else if len(parts) >= 3 && parts[1] == "from" {
				targetFolder := parts[2]
				finalPath := targetFolder

				// Try to resolve path relative to CWD or Project Root
				absPath, err := filepath.Abs(targetFolder)
				if err == nil {
					if _, err := os.Stat(absPath); err == nil {
						finalPath = absPath
					} else if projectRoot != "" {
						projRelPath := filepath.Join(projectRoot, targetFolder)
						if _, err := os.Stat(projRelPath); err == nil {
							finalPath = projRelPath
						}
					}
				}
				kb.LearningPath = finalPath

				count := 0
				totalFound := 0
				err = filepath.WalkDir(finalPath, func(path string, d fs.DirEntry, err error) error {
					if err != nil {
						fmt.Printf("Warning: skipping %s due to error: %v\n", path, err)
						return nil
					}
					if d.IsDir() {
						return nil
					}

					name := d.Name()
					ext := strings.ToLower(filepath.Ext(name))
					validExts := map[string]bool{
						".html": true, ".tpl": true, ".go": true, ".txt": true, ".md": true, ".json": true, ".sql": true,
					}
					if !validExts[ext] {
						return nil
					}

					totalFound++
					baseName := strings.TrimSuffix(name, filepath.Ext(name))
					baseName = strings.ToLower(baseName)
					if baseName != "" && !kb.KnownObjects[baseName] {
						kb.KnownObjects[baseName] = true
						fmt.Printf("DEBUG: Learned object '%s' from file '%s'\n", baseName, path)
						count++
					}
					return nil
				})

				if err != nil {
					fmt.Printf("Error walking directory '%s': %v\n", finalPath, err)
				} else {
					kb.Save()
					fmt.Printf("Found %d matching files. Learned %d new objects from folder '%s'.\n", totalFound, count, finalPath)
				}
			} else {
				fmt.Println("Usage: learn object <word> OR learn from <folder>")
			}

			// Check for chained commands (e.g., "and create header")
			if len(parts) > 3 && strings.ToLower(parts[3]) == "and" {
				query = strings.Join(parts[4:], " ")
				hasMore = true
				fmt.Printf("Continuing with: %s\n", query)
			}

			if !hasMore {
				continue
			}
		}

		// --- Session Logic ---
		isSessionFilled := false
		if sessionState.IsActive {
			if len(sessionState.Missing) > 0 {
				field := sessionState.Missing[0]
				sessionState.Parameters[field] = query
				sessionState.Missing = sessionState.Missing[1:]
				fmt.Printf("DEBUG: Filled slot '%s' with '%s'\n", field, query)
			}

			if len(sessionState.Missing) > 0 {
				fmt.Printf("You need to provide a %s.\n", sessionState.Missing[0])
				continue
			}
			isSessionFilled = true
			sessionState.IsActive = false
		}

		// --- New Intent Layer Logic ---
		// This recursively fills the data layer using the MoE client
		intentData := resolver.Resolve(query, nil)

		if isSessionFilled {
			intentData.Intent = sessionState.ActiveIntent
			// Convert map[string]string to map[string]interface{}
			params := make(map[string]interface{})
			for k, v := range sessionState.Parameters {
				params[k] = v
			}
			intentData.Parameters = params
			sessionState.ActiveIntent = ""
			sessionState.Parameters = nil
		} else if len(intentData.Missing) > 0 {
			sessionState.IsActive = true
			sessionState.ActiveIntent = intentData.Intent
			// Convert map[string]interface{} to map[string]string
			params := make(map[string]string)
			for k, v := range intentData.Parameters {
				if str, ok := v.(string); ok {
					params[k] = str
				}
			}
			sessionState.Parameters = params
			if sessionState.Parameters == nil {
				sessionState.Parameters = make(map[string]string)
			}
			sessionState.Missing = intentData.Missing
			fmt.Printf("You need to provide a %s.\n", sessionState.Missing[0])
			continue
		}

		if intentData.Intent != "" {
			if icon, ok := intentIcons[intentData.Intent]; ok {
				fmt.Printf("   %s\n", icon)
			} else {
				fmt.Printf("   ✨ %s\n", intentData.Intent)
			}
		}

		// --- Tagging ---
		words := strings.Fields(query)
		posTags := postagger.TagTokens(words)
		taggedData := nertagger.Nertagger(tag.Tag{Tokens: words, PosTag: posTags})

		// --- Start of new logic ---

		// 1. Initial Parse with KnowledgeBase
		intent := parse(query, kb)
		intentData = resolver.Resolve(query, nil)

		var (
			hasQuestionWord   bool
			hasPrepositionIn  bool
			hasDirectoryToken bool
			command           string
			objectType        string
			fileName          string
			targetDirectory   string
			predictedSentence string
			handlerURL        string
		)
		objectTypeParts := intent.ObjectTypeParts
		if val, ok := intent.Params["target"]; ok {
			targetDirectory = val
		}
		if targetDirectory == "" && len(intentData.Parameters) > 0 {
			if val, ok := intentData.Parameters["path"]; ok {
				if str, ok := val.(string); ok {
					targetDirectory = str
				}
			}
		}
		// Try to explicitly identify the command using Intent Analysis (MoE) or Tags
		// Only if intent.Command wasn't already found by the KB parser, or if we need more info
		if command == "" || intent.ObjectType == "" {
			// 2. High Confidence MoE Override
			if intentData.Intent != "" && intentData.Confidence > 0.8 {
				parts := strings.Split(intentData.Intent, "_")
				if len(parts) > 0 {
					command = parts[0]
					if intent.Command == "" {
						intent.Command = command
					}
				}
				if len(parts) > 1 && intent.ObjectType == "" {
					intent.ObjectType = parts[1]
					intent.ObjectTypeParts = append(intent.ObjectTypeParts, parts[1])
				}
			}

			// 2. Try POS/NER Tags for a VERB if still empty
			if command == "" {
				for i, tag := range taggedData.NerTag {
					if tag == "VERB" && i < len(taggedData.Tokens) {
						command = strings.ToLower(taggedData.Tokens[i])
						break
					}
				}
			}

			// 3. Fallback to first token heuristic if still empty
			if command == "" && len(taggedData.Tokens) > 0 {
				command = strings.ToLower(taggedData.Tokens[0])
			}

			// Normalize command aliases
			switch command {
			case "add", "put", "make", "generate", "initialize", "init", "setup", "new":
				command = "create"
			case "ls", "show":
				command = "list"
			case "cd", "change":
				command = "go"
			case "remove":
				command = "delete"
			case "start":
				command = "run"
			case "check", "test":
				command = "verify"
			case "search":
				command = "grep"
			}
		}
		for i, token := range taggedData.Tokens {
			if i < len(taggedData.NerTag) {
				switch taggedData.NerTag[i] {
				case "QUESTION_WORD":
					if token == "what" {
						hasQuestionWord = true
					}
				case "VERB":
				case "OBJECT_TYPE":
					if !contains(objectTypeParts, token) {
						objectTypeParts = append(objectTypeParts, token)
					}
				case "PREPOSITION":
					if token == "in" || token == "into" || token == "to" {
						hasPrepositionIn = true
						foundTarget := false
						for j := i + 1; j < len(taggedData.Tokens); j++ {
							if strings.Contains(taggedData.Tokens[j], ".") { // Prioritize file
								foundTarget = true
								break
							}
						}
						if !foundTarget { // If no file, look for directory
							for j := i + 1; j < len(taggedData.Tokens) && targetDirectory == ""; j++ {
								t := strings.ToLower(taggedData.Tokens[j])
								if t == "the" || t == "a" || t == "an" || t == "folder" || t == "directory" {
									continue
								}
								if taggedData.NerTag[j] == "NAME" || strings.Contains(taggedData.Tokens[j], "/") || strings.Contains(taggedData.Tokens[j], "\\") {
									targetDirectory = taggedData.Tokens[j]
									break
								}
							}
							if targetDirectory == "" {
								for k := i + 1; k < len(taggedData.Tokens); k++ {
									t := strings.ToLower(taggedData.Tokens[k])
									if t == "the" || t == "a" || t == "an" || t == "folder" || t == "directory" {
										continue
									}
									if t != "it" {
										targetDirectory = taggedData.Tokens[k]
									}
									break
								}
							}
						}
					}
				}
			}
			// Check for "with url /<path>" pattern
			if strings.ToLower(token) == "url" && i > 0 && strings.ToLower(taggedData.Tokens[i-1]) == "with" && i+1 < len(taggedData.Tokens) && strings.HasPrefix(taggedData.Tokens[i+1], "/") {
				handlerURL = taggedData.Tokens[i+1]
			}
		}

		// 2. Inference & Sync Logic
		if intent.Command == "" || intent.ObjectType == "" {
			intent = resolveIntent(reader, intent, kb)
		}

		command = intent.Command
		objectType = intent.ObjectType
		fileName = intent.Params["name"]
		if fileName == "" {
			fileName = findName(taggedData, kb)
		}

		// Normalize Object Type synonyms
		if objectType == "site" || objectType == "project" || objectType == "app" {
			objectType = "webserver"
		} else if objectType == "view" {
			objectType = "page"
		} else if objectType == "endpoint" || objectType == "route" {
			objectType = "handler"
		} else if objectType == "db" {
			objectType = "database"
		}

		// Fallback logic for specific complex types or if KB missed it
		if strings.Contains(strings.ToLower(query), "handler") && !strings.Contains(objectType, "database") {
			objectType = "handler"
		} else if strings.Contains(strings.ToLower(query), "data structure") && !strings.Contains(objectType, "database") {
			objectType = "data structure"
			objectTypeParts = []string{} // Clear objectTypeParts to prevent interference
		} else if (strings.Contains(strings.ToLower(query), "webserver") || strings.Contains(strings.ToLower(query), "websever")) && !strings.Contains(objectType, "database") {
			objectType = "webserver"
		} else if objectType == "" {
			objectType = strings.Join(objectTypeParts, " ")
		}

		fileName = intent.Params["name"]
		if fileName == "" && len(intentData.Parameters) > 0 {
			if val, ok := intentData.Parameters["name"].(string); ok {
				fileName = val
			}
		}
		if fileName == "" {
			fileName = findName(taggedData, kb)
		}

		// Heuristic: If fileName is still empty, and objectType is "file",
		// check for tokens that look like filenames (e.g., ends with .go)
		if fileName == "" && contains(objectTypeParts, "file") {
			for _, token := range taggedData.Tokens {
				if strings.HasSuffix(token, ".go") || strings.HasSuffix(token, ".txt") || strings.HasSuffix(token, ".md") || strings.HasSuffix(token, ".html") {
					fileName = token
					break
				}
			}
		}

		if hasQuestionWord && (contains(objectTypeParts, "folder") || contains(objectTypeParts, "folders") || contains(objectTypeParts, "file") || contains(objectTypeParts, "files")) {
			command = "list"
		}

		hasDirectoryToken = false
		for _, t := range taggedData.Tokens {
			if t == "directory" {
				hasDirectoryToken = true
				break
			}
		}
		// New logic to find the target directory more robustly
		if command == "go" {
			// Special handling for "go to webserver <name>"
			if contains(taggedData.Tokens, "webserver") {
				for i, token := range taggedData.Tokens {
					if strings.ToLower(token) == "webserver" && i+1 < len(taggedData.Tokens) {
						// Find next non-keyword token
						for j := i + 1; j < len(taggedData.Tokens); j++ {
							nextToken := strings.ToLower(taggedData.Tokens[j])
							if nextToken != "folder" && nextToken != "directory" {
								targetDirectory = filepath.Join("cmd", taggedData.Tokens[j])
								break
							}
						}
						if targetDirectory != "" {
							break
						}
					}
				}
			}

			// Fallback to original logic if no webserver navigation found
			if targetDirectory == "" {
				for i := len(taggedData.Tokens) - 1; i >= 0; i-- {
					token := strings.ToLower(taggedData.Tokens[i])
					// Exclude command words and prepositions
					if token != "go" && token != "to" && token != "project" && token != "folder" && token != "directory" && token != "cd" && token != "change" && token != "move" {
						targetDirectory = taggedData.Tokens[i]
						break
					}
				}
			}
		}

		handled := true
		switch command {
		case "go":
			if targetDirectory != "" {
				if targetDirectory == "root" {
					targetDirectory = "/"
				}
				err := os.Chdir(targetDirectory)
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't change the directory to %s: %v", targetDirectory, err)
				} else {
					predictedSentence = fmt.Sprintf("Changed directory to %s.", targetDirectory)
					currentAbsDir, err := os.Getwd()
					if err != nil {
						log.Printf("Error getting current absolute directory after chdir: %v", err)
					} else {
						saveLastDirectory(currentAbsDir) // Save the absolute path
					}
				}
			} else {
				handled = false
			}
		case "move":
			sourceFile := fileName
			destDir := targetDirectory

			if sourceFile == "" {
				predictedSentence = "Please specify a file to move."
				break
			}
			if destDir == "" {
				predictedSentence = "Please specify a destination directory."
				break
			}

			if _, err := os.Stat(destDir); os.IsNotExist(err) {
				predictedSentence = fmt.Sprintf("Destination directory '%s' does not exist.", destDir)
				break
			}

			destFile := filepath.Join(destDir, filepath.Base(sourceFile))

			err := os.Rename(sourceFile, destFile)
			if err != nil {
				predictedSentence = fmt.Sprintf("I couldn't move the file '%s' to '%s': %v", sourceFile, destDir, err)
			} else {
				predictedSentence = fmt.Sprintf("I have moved the file '%s' to '%s'.", sourceFile, destDir)
				if tree, err := generateDirectoryTree(".", "", 0, 2, destFile); err == nil {
					predictedSentence += "\n\n" + tree
				}
			}
		case "list":
			if strings.Contains(objectType, "handler") {
				targetPath := "main.go"
				if targetDirectory != "" {
					possiblePath := filepath.Join(targetDirectory, "main.go")
					if _, err := os.Stat(possiblePath); err == nil {
						targetPath = possiblePath
					} else {
						baseName := filepath.Base(targetDirectory)
						possiblePath = filepath.Join(targetDirectory, "cmd", baseName, "main.go")
						if _, err := os.Stat(possiblePath); err == nil {
							targetPath = possiblePath
						}
					}
				}

				content, err := os.ReadFile(targetPath)
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't read %s to list handlers. Please ensure you are in a webserver directory or specify one.", targetPath)
				} else {
					lines := strings.Split(string(content), "\n")
					var handlers []string
					for _, line := range lines {
						trimmed := strings.TrimSpace(line)
						if strings.HasPrefix(trimmed, "http.HandleFunc(") {
							args := strings.TrimPrefix(trimmed, "http.HandleFunc(")
							args = strings.TrimSuffix(args, ")")
							parts := strings.SplitN(args, ",", 2)
							if len(parts) == 2 {
								path := strings.TrimSpace(parts[0])
								path = strings.Trim(path, "\"")
								funcName := strings.TrimSpace(parts[1])
								handlers = append(handlers, fmt.Sprintf("%s -> %s", path, funcName))
							}
						}
					}
					if len(handlers) > 0 {
						predictedSentence = fmt.Sprintf("Registered Handlers in %s:\n%s", targetPath, strings.Join(handlers, "\n"))
					} else {
						predictedSentence = fmt.Sprintf("No handlers found registered via http.HandleFunc in %s.", targetPath)
					}
				}
			} else {
				files, err := os.ReadDir(".")
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't list the contents of the directory: %v", err)
				} else {
					var items []string
					showFiles := contains(objectTypeParts, "file") || contains(objectTypeParts, "files")
					showFolders := contains(objectTypeParts, "folder") || contains(objectTypeParts, "folders")

					for _, file := range files {
						isDir := file.IsDir()
						if (!showFiles && !showFolders) || (showFiles && showFolders) {
							items = append(items, file.Name())
						} else if showFiles && !isDir {
							items = append(items, file.Name())
						} else if showFolders && isDir {
							items = append(items, file.Name())
						}
					}
					predictedSentence = "Here are the contents of the directory:\n" + strings.Join(items, "\n")
				}
			}
		case "tree":
			target := "."
			if targetDirectory != "" {
				target = targetDirectory
			}

			maxDepth := -1
			for i, token := range taggedData.Tokens {
				if (token == "-d" || token == "depth" || token == "-L") && i+1 < len(taggedData.Tokens) {
					if d, err := strconv.Atoi(taggedData.Tokens[i+1]); err == nil {
						maxDepth = d
					}
				}
			}

			treeView, err := generateDirectoryTree(target, "", 0, maxDepth, "")
			if err != nil {
				predictedSentence = fmt.Sprintf("I couldn't generate a tree for '%s': %v", target, err)
			} else {
				predictedSentence = fmt.Sprintf("Directory tree for '%s':\n%s", target, treeView)
			}
		case "grep":
			searchTerm := ""
			if val, ok := intent.Params["target"]; ok {
				searchTerm = val
			}

			if searchTerm == "" {
				for i, token := range taggedData.Tokens {
					if (token == "grep" || token == "search") && i+1 < len(taggedData.Tokens) {
						if taggedData.Tokens[i+1] == "for" && i+2 < len(taggedData.Tokens) {
							searchTerm = taggedData.Tokens[i+2]
						} else {
							searchTerm = taggedData.Tokens[i+1]
						}
						break
					}
				}
			}

			if searchTerm == "" {
				predictedSentence = "Please provide text to search for."
			} else {
				target := "."
				if targetDirectory != "" {
					target = targetDirectory
				}

				var results []string
				err := filepath.Walk(target, func(path string, info os.FileInfo, err error) error {
					if err != nil {
						return nil
					}
					if !info.IsDir() && !strings.Contains(path, string(os.PathSeparator)+".") {
						content, err := os.ReadFile(path)
						if err == nil {
							if strings.Contains(string(content), searchTerm) {
								lines := strings.Split(string(content), "\n")
								for i, line := range lines {
									if strings.Contains(line, searchTerm) {
										trimmed := strings.TrimSpace(line)
										if len(trimmed) > 80 {
											trimmed = trimmed[:80] + "..."
										}
										results = append(results, fmt.Sprintf("%s:%d: %s", path, i+1, trimmed))
									}
								}
							}
						}
					}
					return nil
				})

				if err != nil {
					predictedSentence = fmt.Sprintf("Error searching: %v", err)
				} else if len(results) == 0 {
					predictedSentence = fmt.Sprintf("No matches found for '%s' in '%s'.", searchTerm, target)
				} else {
					output := strings.Join(results, "\n")
					if len(results) > 20 {
						output = strings.Join(results[:20], "\n") + fmt.Sprintf("\n... and %d more matches.", len(results)-20)
					}
					predictedSentence = fmt.Sprintf("Found matches for '%s':\n%s", searchTerm, output)
				}
			}
		case "history":
			limit := 10
			for _, token := range taggedData.Tokens {
				if val, err := strconv.Atoi(token); err == nil && val > 0 {
					limit = val
				}
			}
			if limit > len(commandHistory) {
				limit = len(commandHistory)
			}
			var historyLines []string
			for i := len(commandHistory) - limit; i < len(commandHistory); i++ {
				historyLines = append(historyLines, fmt.Sprintf("%d  %s", i+1, commandHistory[i]))
			}
			predictedSentence = strings.Join(historyLines, "\n")
		case "help":
			var sb strings.Builder
			sb.WriteString("--- ʕ◔ϖ◔ʔ Gollemer Help ---\n\n")
			
			sb.WriteString("Categorized Commands:\n")
			sb.WriteString("  [📁 Navigation]  go <dir>, list, tree, pwd\n")
			sb.WriteString("  [🛠️  Files]       create file <name>, create folder <name>, delete <name>, move <file> to <dir>, cat, grep\n")
			sb.WriteString("  [🌐 Web]         create webserver <name>, create handler <name>, create page <name>, run, stop, verify\n")
			sb.WriteString("  [🧠 Learning]    learn from <dir>, learn object <word>, show learning path\n")
			sb.WriteString("  [⚙️  System]      history, clear, exit\n\n")

			sb.WriteString("The Learning System (How & Why):\n")
			sb.WriteString("  Gollemer learns from your code to automate repetitive tasks.\n")
			sb.WriteString("  - HOW: It scans a 'learningfolder' for templates (files like 'navbar.html', 'auth.go').\n")
			sb.WriteString("         If it finds 'navbar.html', it learns the 'navbar' object.\n")
			sb.WriteString("  - WHY: So you can say 'create navbar' and it generates the code for you instantly,\n")
			sb.WriteString("         using your own preferred style and structure.\n")
			sb.WriteString("  - CUSTOMIZE: You have full control. Add/Edit files in 'learningfolder' or change\n")
			sb.WriteString("               the source using 'learn from <dir>'.\n\n")

			sb.WriteString("Examples:\n")
			sb.WriteString("  - \"create webserver myapp\"        : Initialize a new Go web project\n")
			sb.WriteString("  - \"create handler login with url /login\" : Append a new handler to main.go\n")
			sb.WriteString("  - \"create page dashboard\"         : Generate a new WASM frontend page\n")
			sb.WriteString("  - \"learn from learningfolder\"     : Teach Gollemer templates from a directory\n")
			sb.WriteString("  - \"create home\"                   : Uses home.html template if learned\n")
			sb.WriteString("  - \"tree -d 2\"                     : Show directory tree up to 2 levels deep\n\n")

			sb.WriteString("Learning System & AI Features:\n")
			sb.WriteString("  [🔍 Semantic]   Understand synonyms: 'site', 'app', 'view', 'endpoint'\n")
			sb.WriteString("  [📡 Discovery]  Background scanner alerts you to new template patterns\n")
			sb.WriteString("  [💡 Smart]      Suggests close matches if you make a typo (using Levenshtein)\n")
			sb.WriteString("  [📁 Learning]   Use 'learn from <dir>' to teach Gollemer local templates\n\n")

			sb.WriteString("--- Known Objects (Templates) ---\n")
			var objs []string
			for k := range kb.KnownObjects {
				objs = append(objs, k)
			}
			sort.Strings(objs)
			sb.WriteString(strings.Join(objs, ", "))
			sb.WriteString("\n---------------------")
			predictedSentence = sb.String()
		case "create":

			// 1. Template System Check
			// Check if the objectType matches a file in the learning folder
			learningPath := kb.LearningPath
			if learningPath == "" {
				learningPath = filepath.Join(projectRoot, "learningfolder")
			}
			// Fallback to current directory's learningfolder if project root one doesn't exist
			if _, err := os.Stat(learningPath); os.IsNotExist(err) {
				cwd, _ := os.Getwd()
				localLearningPath := filepath.Join(cwd, "learningfolder")
				if _, err := os.Stat(localLearningPath); err == nil {
					learningPath = localLearningPath
				}
			}

			templateFound := false
			// --- Smart Suggestions Logic ---
			if _, ok := kb.KnownObjects[strings.ToLower(objectType)]; !ok && objectType != "" {
				closest, distance := findClosestObject(objectType, kb.KnownObjects)
				if distance > 0 && distance < 4 { // Threshold for "close enough"
					colors.ColorizeCol("yellow", "white", fmt.Sprintf("I don't know how to create '%s', but I found a similar object '%s'. Use that? (y/n) ", objectType, closest))
					resp, _ := reader.ReadString('\n')
					if strings.TrimSpace(strings.ToLower(resp)) == "y" {
						objectType = closest
					}
				}
			}

			if _, err := os.Stat(learningPath); err == nil {
				// 1a. Check for directory-based template first (multi-file)
				templateDir := filepath.Join(learningPath, ".templates", objectType)
				if info, err := os.Stat(templateDir); err == nil && info.IsDir() {
					templateFound = true

					destDir := targetDirectory
					if destDir == "" {
						destDir = "."
					}
					os.MkdirAll(destDir, 0755)

					err = filepath.WalkDir(templateDir, func(path string, d fs.DirEntry, err error) error {
						if err != nil {
							return err
						}
						rel, _ := filepath.Rel(templateDir, path)
						if rel == "." {
							return nil
						}
						targetPath := filepath.Join(destDir, rel)
						if d.IsDir() {
							return os.MkdirAll(targetPath, 0755)
						}
						content, err := os.ReadFile(path)
						if err != nil {
							return err
						}
						return os.WriteFile(targetPath, content, 0644)
					})

					if err != nil {
						predictedSentence = fmt.Sprintf("I found the multi-file template '%s' but failed to copy it: %v", objectType, err)
					} else {
						predictedSentence = fmt.Sprintf("I have created the '%s' object with all its components in '%s'.", objectType, destDir)
						if tree, err := generateDirectoryTree(".", "", 0, 2, destDir); err == nil {
							predictedSentence += "\n\n" + tree
						}
						if strings.Contains(strings.ToLower(destDir), "wasm") || strings.Contains(strings.ToLower(objectType), "wasm") {
							buildWasm(destDir)
						}
					}
				}

				if !templateFound {
					// 1b. Original single-file template logic
					_ = filepath.WalkDir(learningPath, func(path string, d fs.DirEntry, err error) error {
						if templateFound {
							return nil
						}
						if err != nil || d.IsDir() {
							return nil
						}

						name := d.Name()
						ext := filepath.Ext(name)
						baseName := strings.TrimSuffix(name, ext)
						// Exact match on the object name (case-insensitive)
						match := strings.EqualFold(baseName, objectType)
						if !match {
							// Check relative path for "template/head" style matching
							relPath, _ := filepath.Rel(learningPath, path)
							relPathNoExt := strings.TrimSuffix(relPath, ext)
							normalizedObjType := strings.ReplaceAll(objectType, " ", string(os.PathSeparator))
							if strings.EqualFold(relPathNoExt, normalizedObjType) {
								match = true
							}
						}

						if match {
							templateFound = true

							destName := fileName
							if destName == "" {
								destName = baseName // Default to template name if no name provided
							}
							// If dest doesn't have ext, append template's ext
							if filepath.Ext(destName) == "" {
								destName += ext
							}

							destPath := destName
							if targetDirectory != "" {
								destPath = filepath.Join(targetDirectory, destName)
							}

							content, err := os.ReadFile(path)
							if err != nil {
								predictedSentence = fmt.Sprintf("I found the template '%s' but couldn't read it: %v", name, err)
							} else {
								// Ensure the target directory exists
								if targetDirectory != "" {
									os.MkdirAll(targetDirectory, 0755)
								}
								err = os.WriteFile(destPath, content, 0644)
								if err != nil {
									predictedSentence = fmt.Sprintf("I couldn't create the file %s from template: %v", destPath, err)
								} else {
									predictedSentence = fmt.Sprintf("I have created '%s' using the learned template '%s'.", destPath, name)
									if tree, err := generateDirectoryTree(".", "", 0, 2, destPath); err == nil {
										predictedSentence += "\n\n" + tree
									}
									if targetDirectory != "" && (strings.Contains(strings.ToLower(targetDirectory), "wasm") || strings.Contains(strings.ToLower(destPath), "wasm")) {
										buildWasm(targetDirectory)
									}
								}
							}
						}
						return nil
					})
				}
			}

			if templateFound {
				// Template handled, skip other checks
			} else if strings.HasSuffix(strings.ToLower(strings.TrimSpace(targetDirectory)), ".db") {
				// Creating a table in an existing database
				dbFileName := targetDirectory
				tableName := objectType
				structName := tableName

				// Try to find the struct file
				structFile := structName + ".go"
				if _, err := os.Stat(structFile); os.IsNotExist(err) {
					// Try lowercase
					structFile = strings.ToLower(structName) + ".go"
				}

				if _, err := os.Stat(structFile); os.IsNotExist(err) {
					predictedSentence = fmt.Sprintf("I want to create table '%s' in '%s', but I couldn't find a data structure file '%s' to define the fields.", tableName, dbFileName, structFile)
				} else {
					content, err := os.ReadFile(structFile)
					if err != nil {
						predictedSentence = fmt.Sprintf("I found '%s' but couldn't read it: %v", structFile, err)
					} else {
						// Parse fields from struct
						fields := make(map[string]string)
						lines := strings.Split(string(content), "\n")
						inStruct := false
						for _, line := range lines {
							trimmed := strings.TrimSpace(line)
							if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, " struct {") {
								inStruct = true
								continue
							}
							if inStruct {
								if trimmed == "}" {
									break
								}
								parts := strings.Fields(trimmed)
								if len(parts) >= 2 {
									fName := parts[0]
									fType := parts[1]
									// Skip ID if it's auto-increment usually
									if strings.ToLower(fName) != "id" {
										fields[fName] = fType
									}
								}
							}
						}

						if len(fields) > 0 {
							err = createTableWithFields(dbFileName, tableName, fields)
							if err != nil {
								predictedSentence = fmt.Sprintf("Failed to create table '%s' in '%s': %v", tableName, dbFileName, err)
							} else {
								predictedSentence = fmt.Sprintf("Created table '%s' in '%s' using fields from '%s'.", tableName, dbFileName, structFile)
							}
						} else {
							predictedSentence = fmt.Sprintf("I couldn't find any fields in '%s' to create the table.", structFile)
						}
					}
				}
			} else if strings.Contains(objectType, "handler") {
				handlerName := ""
				for i, token := range taggedData.Tokens {
					if strings.ToLower(token) == "handler" && i+1 < len(taggedData.Tokens) {
						handlerName = taggedData.Tokens[i+1]
						break
					}
				}
				if handlerName == "" {
					predictedSentence = "You need to provide a name for the handler."
				} else {
					handlerContent := `package main

import (
	"fmt"
	"net/http"
)

// ` + strings.Title(handlerName) + `Handler is a sample handler function.
func ` + strings.Title(handlerName) + `Handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Executing ` + strings.Title(handlerName) + `Handler! Request URL: %s\n", r.URL.Path)
}
`
					filePath := handlerName + ".go"

					// Check if the handler file already exists
					if _, err := os.Stat(filePath); err == nil {
						// Logic: If it exists, check if it's already a handler or if we should use a different name
						filePath = handlerName + "_handler.go"
					}
					
					err = os.WriteFile(filePath, []byte(handlerContent), 0644)
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't write to the handler file %s: %v", filePath, err)
						goto endOfCreateHandler
					}
					goImports(filePath)
					predictedSentence = fmt.Sprintf("I have created the handler '%s' in %s.", handlerName, filePath)
					if tree, err := generateDirectoryTree(".", "", 0, 2, filePath); err == nil {
						predictedSentence += "\n\n" + tree
					}
					
					// Always attempt to register the handler in the current project's main.go
					currentProjectMainGo := filepath.Join(".", "main.go")
					registrationMsg, err := registerHandlerURL(strings.Title(handlerName), handlerURL, currentProjectMainGo)
					if err != nil {
						log.Printf("Error registering handler URL in %s: %v", currentProjectMainGo, err)
						predictedSentence += fmt.Sprintf(" I tried to register the handler in %s but failed: %v", currentProjectMainGo, err)
					} else {
						predictedSentence += " " + registrationMsg
					}

					// --- Integration: Auto-create WASM Page if requested ---
					if strings.Contains(strings.ToLower(query), "wasm") || strings.Contains(strings.ToLower(query), "page") || strings.Contains(strings.ToLower(query), "frontend") {
						// Copy of the logic from 'create page' for seamless integration
						pageName := handlerName
						cleanName := strings.Title(strings.ToLower(pageName))
						snakeName := strings.ReplaceAll(strings.ToLower(pageName), " ", "_")

						// Find WASM dir (simplified search)
						var wasmDir string
						search, _ := os.Getwd()
						for {
							if strings.EqualFold(filepath.Base(search), "wasm") {
								wasmDir = search
								break
							}
							if info, err := os.Stat(filepath.Join(search, "wasm")); err == nil && info.IsDir() {
								wasmDir = filepath.Join(search, "wasm")
								break
							}
							parent := filepath.Dir(search)
							if parent == search {
								break
							}
							search = parent
						}

						if wasmDir != "" {
							wasmPagesDir := filepath.Join(wasmDir, "pages")
							os.MkdirAll(wasmPagesDir, 0755)

							// Default material import (heuristically found in learningfolder if not local)
							materialImport := "github.com/golangast/gollemer/jim/wasm/ui/material"

							content := fmt.Sprintf(`package pages

import (
	"syscall/js"
	"%s"
)

func Render%s() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("style", "padding: 4rem 2rem; max-width: 800px; margin: 0 auto; min-height: 80vh;")

	heading := document.Call("createElement", "h1")
	heading.Set("innerText", "%s")
	container.Call("appendChild", heading)

	// Add a button to use the material package
	btn := material.NewButton("Interact", "primary", func() {
		js.Global().Call("alert", "WASM Page %s is live!")
	})
	container.Call("appendChild", btn.Render())

	return container
}
`, materialImport, cleanName, cleanName, cleanName)

							pagePath := filepath.Join(wasmPagesDir, snakeName+".go")
							if err := os.WriteFile(pagePath, []byte(content), 0644); err == nil {
								predictedSentence += fmt.Sprintf(" Verified twin WASM page created in %s.", pagePath)

								// Register route in wasm.go
								wasmGoPath := filepath.Join(wasmDir, "wasm.go")
								if wContent, err := os.ReadFile(wasmGoPath); err == nil {
									sWContent := string(wContent)
									routeKey := "#" + snakeName
									routeValue := "pages.Render" + cleanName
									if !strings.Contains(sWContent, routeKey) {
										registration := fmt.Sprintf("\t\t\t\"%s\": %s,\n", routeKey, routeValue)
										if idx := strings.LastIndex(sWContent, "},"); idx != -1 {
											sWContent = sWContent[:idx] + registration + sWContent[idx:]
											os.WriteFile(wasmGoPath, []byte(sWContent), 0644)
											predictedSentence += " Registered WASM route."
										}
									}
								}
								buildWasm(wasmDir)
							}
						}
					}
				}
			endOfCreateHandler:
			} else if strings.Contains(objectType, "component") {
				componentName := fileName
				if componentName == "" {
					componentName = "Component"
				}
				componentName = strings.Title(componentName)

				wasmUIDir := filepath.Join(projectRoot, "learningfolder", "wasm", "ui")
				if _, err := os.Stat(wasmUIDir); os.IsNotExist(err) {
					os.MkdirAll(wasmUIDir, 0755)
				}

				content := fmt.Sprintf(`package ui

import "syscall/js"

type %s struct {
	el js.Value
}

func New%s() *%s {
	return &%s{}
}

func (c *%s) Render() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("className", "%s-component")
	div.Set("innerText", "%s Component")
	return div
}
`, componentName, componentName, componentName, componentName, componentName, strings.ToLower(componentName), componentName)

				filePath := filepath.Join(wasmUIDir, strings.ToLower(componentName)+".go")
				err := os.WriteFile(filePath, []byte(content), 0644)
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't create the component file %s: %v", filePath, err)
				} else {
					predictedSentence = fmt.Sprintf("I have created the WASM component '%s' in %s.", componentName, filePath)
					if tree, err := generateDirectoryTree(".", "", 0, 2, filePath); err == nil {
						predictedSentence += "\n\n" + tree
					}
					buildWasm(filepath.Join(projectRoot, "learningfolder", "wasm"))
				}
			} else if strings.Contains(objectType, "page") {
				pageName := fileName
				if pageName == "" {
					pageName = "NewPage"
				}
				// Sanitize pageName for Go (TitleCase and no spaces)
				cleanName := ""
				for _, part := range strings.Fields(pageName) {
					cleanName += strings.Title(strings.ToLower(part))
				}
				snakeName := strings.ReplaceAll(strings.ToLower(pageName), " ", "_")

				// Dynamic Discovery of WASM Dir
				var wasmDir string

				// 1. Try to find a webserver mentioned in the query
				webserverName := ""
				for i, token := range taggedData.Tokens {
					if (strings.ToLower(token) == "webserver" || strings.ToLower(token) == "websever") && i+1 < len(taggedData.Tokens) {
						webserverName = taggedData.Tokens[i+1]
						break
					}
				}

				if webserverName != "" {
					candidates := []string{
						filepath.Join(projectRoot, webserverName, "cmd", webserverName),
						filepath.Join(projectRoot, "cmd", webserverName),
						filepath.Join(projectRoot, webserverName),
						filepath.Join(projectRoot, webserverName, "cmd"),
					}
					for _, p := range candidates {
						if _, err := os.Stat(filepath.Join(p, "wasm")); err == nil {
							wasmDir = filepath.Join(p, "wasm")
							break
						}
					}
				}

				// 2. Try CWD first if targetDirectory is provided
				if wasmDir == "" && targetDirectory != "" {
					cwd, _ := os.Getwd()

					// Special case: If we are already in the target directory (e.g. 'wasm')
					if strings.EqualFold(filepath.Base(cwd), targetDirectory) {
						wasmDir = cwd
					} else {
						// Try relative to CWD
						relToCwd := filepath.Join(cwd, targetDirectory)
						if _, err := os.Stat(relToCwd); err == nil {
							wasmDir = relToCwd
						} else {
							// Try relative to project root
							relToRoot := filepath.Join(projectRoot, targetDirectory)
							if _, err := os.Stat(relToRoot); err == nil {
								wasmDir = relToRoot
							}
						}
					}
				}

				// 3. Fallback: Search upwards from CWD for any 'wasm' folder
				if wasmDir == "" {
					search, _ := os.Getwd()
					for {
						check := filepath.Join(search, "wasm")
						// If we ARE in a wasm folder, use it
						if strings.EqualFold(filepath.Base(search), "wasm") {
							wasmDir = search
							break
						}
						// Otherwise check if there is a wasm subfolder here
						if info, err := os.Stat(check); err == nil && info.IsDir() {
							wasmDir = check
							break
						}
						parent := filepath.Dir(search)
						if parent == search || strings.Contains(parent, projectRoot) == false {
							break
						}
						search = parent
					}
				}

				// 4. Final Fallback to learningfolder
				if wasmDir == "" {
					wasmDir = filepath.Join(projectRoot, "learningfolder", "wasm")
				}

				wasmPagesDir := filepath.Join(wasmDir, "pages")
				if _, err := os.Stat(wasmPagesDir); os.IsNotExist(err) {
					os.MkdirAll(wasmPagesDir, 0755)
				}

				// Determine module name for imports
				materialImport := "github.com/golangast/gollemer/learningfolder/wasm/ui/material"
				searchDir := wasmDir
				for {
					if _, err := os.Stat(filepath.Join(searchDir, "go.mod")); err == nil {
						if modContent, err := os.ReadFile(filepath.Join(searchDir, "go.mod")); err == nil {
							lines := strings.Split(string(modContent), "\n")
							for _, line := range lines {
								if strings.HasPrefix(line, "module ") {
									moduleRoot := strings.TrimSpace(strings.TrimPrefix(line, "module "))
									rel, _ := filepath.Rel(searchDir, wasmDir)
									materialImport = filepath.Join(moduleRoot, rel, "ui", "material")
									break
								}
							}
						}
						break
					}
					parent := filepath.Dir(searchDir)
					if parent == searchDir {
						break
					}
					searchDir = parent
				}

				content := fmt.Sprintf(`package pages

import (
	"syscall/js"

	"%s"
)

func Render%s() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "%s-page")
	container.Set("style", "padding: 4rem 2rem; max-width: 800px; margin: 0 auto; min-height: 80vh;")

	heading := document.Call("createElement", "h1")
	heading.Set("innerText", "%s")
	heading.Set("className", "mat-h1")
	container.Call("appendChild", heading)

	sub := document.Call("createElement", "p")
	sub.Set("innerText", "Welcome to the %s page.")
	sub.Set("className", "mat-body-1")
	container.Call("appendChild", sub)

	// Add a button to use the material package and avoid 'unused import' error
	btn := material.NewButton("Action", "primary", func() {
		js.Global().Call("alert", "Action triggered on %s page!")
	})
	container.Call("appendChild", btn.Render())

	return container
}
`, materialImport, cleanName, snakeName, pageName, pageName)

				filePath := filepath.Join(wasmPagesDir, snakeName+".go")
				err := os.WriteFile(filePath, []byte(content), 0644)
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't create the page file %s: %v", filePath, err)
				} else {
					// Verify file creation (Recursive Reasoning / Validation)
					if _, checkErr := os.Stat(filePath); checkErr == nil {
						predictedSentence = fmt.Sprintf("I have verified the creation of WASM page '%s' in %s.", pageName, filePath)
						if tree, err := generateDirectoryTree(".", "", 0, 2, filePath); err == nil {
							predictedSentence += "\n\n" + tree
						}
					} else {
						predictedSentence = fmt.Sprintf("File write reported success, but I couldn't find %s on disk. Please check permissions.", filePath)
						goto endOfCreatePage
					}

					// --- Integration 1: Router ---
					wasmGoPath := filepath.Join(wasmDir, "wasm.go")
					if content, err := os.ReadFile(wasmGoPath); err == nil {
						sContent := string(content)
						routeKey := "#" + snakeName
						routeValue := "pages.Render" + cleanName
						if !strings.Contains(sContent, routeKey) {
							registration := fmt.Sprintf("\t\t\t\"%s\": %s,\n", routeKey, routeValue)
							// Smart insertion: Look for the Routes map definition
							if idx := strings.LastIndex(sContent, "},"); idx != -1 {
								sContent = sContent[:idx] + registration + sContent[idx:]
								os.WriteFile(wasmGoPath, []byte(sContent), 0644)
								predictedSentence += " Registered route in wasm.go."
							}
						}
					}

					// --- Integration 2: Navigation ---
					headerPath := filepath.Join(wasmDir, "ui", "header.go")
					if content, err := os.ReadFile(headerPath); err == nil {
						sContent := string(content)
						navLink := fmt.Sprintf("{\"%s\", \"#%s\"}", strings.Title(pageName), snakeName)
						if !strings.Contains(sContent, navLink) {
							// Find the end of the links slice
							sliceMarker := "},"
							if idx := strings.LastIndex(sContent, sliceMarker); idx != -1 {
								// We want to insert it before the closing brace of the slice
								// By finding the LAST '},' which usually ends the last item in a slice literal
								insertIdx := idx + len(sliceMarker)
								sContent = sContent[:insertIdx] + "\n\t\t" + navLink + "," + sContent[insertIdx:]
								os.WriteFile(headerPath, []byte(sContent), 0644)
								predictedSentence += " Updated navigation in header.go."
							}
						}
					}

					buildWasm(wasmDir)
				}
			endOfCreatePage:
			} else if strings.Contains(objectType, "colors") {
				cssPath := filepath.Join(projectRoot, "learningfolder", "assets", "style", "style.css")
				if _, err := os.Stat(cssPath); err == nil {
					// Append some new color variables as an example or update them
					colorsUpdate := `
:root {
    --primary: #6366f1;
    --secondary: #ec4899;
    --accent: #f59e0b;
    --success: #10b981;
    --danger: #ef4444;
}
`
					existing, _ := os.ReadFile(cssPath)
					if !strings.Contains(string(existing), "--secondary") {
						f, _ := os.OpenFile(cssPath, os.O_APPEND|os.O_WRONLY, 0644)
						f.WriteString(colorsUpdate)
						f.Close()
						predictedSentence = "I have updated the colors in style.css."
					} else {
						predictedSentence = "Colors are already defined in style.css."
					}
					buildWasm(filepath.Join(projectRoot, "learningfolder", "wasm")) // Still build WASM to refresh
				} else {
					predictedSentence = "I couldn't find style.css to update colors."
				}
			} else if strings.Contains(objectType, "file") { // New block for generic file creation
				if fileName != "" {
					filePath := fileName
					if targetDirectory != "" {
						filePath = filepath.Join(targetDirectory, fileName)
					}

					var content []byte
					var source string

					// 1. Check if file exists in current directory
					if _, err := os.Stat(fileName); err == nil {
						content, err = os.ReadFile(fileName)
						if err == nil {
							source = "current directory"
						}
					}

					// 2. If not, check learning folder
					if source == "" {
						learningPath := kb.LearningPath
						if learningPath == "" {
							learningPath = filepath.Join(projectRoot, "learningfolder")
						}
						// Fallback to local learningfolder
						if _, err := os.Stat(learningPath); os.IsNotExist(err) {
							cwd, _ := os.Getwd()
							localLearningPath := filepath.Join(cwd, "learningfolder")
							if _, err := os.Stat(localLearningPath); err == nil {
								learningPath = localLearningPath
							}
						}

						if _, err := os.Stat(learningPath); err == nil {
							_ = filepath.WalkDir(learningPath, func(path string, d fs.DirEntry, err error) error {
								if source != "" {
									return nil
								}
								if err != nil || d.IsDir() {
									return nil
								}
								if strings.EqualFold(d.Name(), fileName) {
									content, err = os.ReadFile(path)
									if err == nil {
										source = "learning folder"
									}
								}
								return nil
							})
						}
					}

					// Ensure the target directory exists
					if targetDirectory != "" {
						os.MkdirAll(targetDirectory, 0755)
					}
					err := os.WriteFile(filePath, content, 0644)
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't create the file %s: %v", filePath, err)
					} else {
						if source != "" {
							predictedSentence = fmt.Sprintf("I have created the file %s using content from %s.", filePath, source)
						} else {
							predictedSentence = fmt.Sprintf("I have created the empty file %s.", filePath)
						}
						if tree, err := generateDirectoryTree(".", "", 0, 2, filePath); err == nil {
							predictedSentence += "\n\n" + tree
						}
						if targetDirectory != "" && (strings.Contains(strings.ToLower(targetDirectory), "wasm") || strings.Contains(strings.ToLower(filePath), "wasm")) {
							buildWasm(targetDirectory)
						}
					}
				} else {
					predictedSentence = "You need to provide a name for the file."
				}
			} else if strings.Contains(objectType, "webserver") {
				if fileName == "" {
					predictedSentence = "You need to provide a name for the webserver."
				} else {
					serverDir := filepath.Join("cmd", fileName)
					if targetDirectory != "" {
						serverDir = filepath.Join(targetDirectory, "cmd", fileName)
					}
					err := os.MkdirAll(serverDir, 0755)
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't create the webserver directory %s: %v", serverDir, err)
					} else {
						serverContent := `package main

import (
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	_ "modernc.org/sqlite"
)

var db *sql.DB

func InitDB(filepath string) *sql.DB {
	d, err := sql.Open("sqlite", filepath)
	if err != nil {
		log.Fatalf("Error opening database: %v", err)
	}
	if err = d.Ping(); err != nil {
		log.Fatalf("Error connecting to database: %v", err)
	}

	createTableSQL := ` + "`" + `
	CREATE TABLE IF NOT EXISTS webservers (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT,
		status TEXT,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);` + "`" + `

	_, err = d.Exec(createTableSQL)
	if err != nil {
		log.Fatalf("Error creating table 'webservers': %v", err)
	}

	// Seed data if empty
	var count int
	err = d.QueryRow("SELECT COUNT(*) FROM webservers WHERE name = ?", "` + fileName + `").Scan(&count)
	if err == nil && count == 0 {
		_, _ = d.Exec("INSERT INTO webservers (name, status) VALUES (?, ?)", "` + fileName + `", "running")
	}

	return d
}

func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from the %s webserver!", "` + fileName + `")
}

func main() {
	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "` + fileName + `.db")
	db = InitDB(dbPath)
	defer db.Close()

	http.HandleFunc("/", handler)
	// HANDLER_REGISTRATIONS_GO_HERE
	log.Println("Starting webserver on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
`
						mainGoPath := filepath.Join(serverDir, "main.go")
						err = os.WriteFile(mainGoPath, []byte(serverContent), 0644)
						if err != nil {
							predictedSentence = fmt.Sprintf("I couldn't create the webserver file %s: %v", mainGoPath, err)
						} else {
							goImports(mainGoPath)
							predictedSentence = fmt.Sprintf("I have created the webserver '%s' in %s.", fileName, mainGoPath)
							if tree, err := generateDirectoryTree(".", "", 0, 2, serverDir); err == nil {
								predictedSentence += "\n\n" + tree
							}

							// --- Add go mod init ---
							modCmd := exec.Command("go", "mod", "init", fileName)
							modCmd.Dir = serverDir
							modErr := modCmd.Run()
							if modErr != nil {
								predictedSentence += fmt.Sprintf(" However, I failed to initialize go.mod: %v", modErr)
							} else {
								predictedSentence += " I also created a go.mod file."
							}
						}
					}
				}
			} else if strings.Contains(objectType, "structure") || strings.Contains(objectType, "data structure") {
				structName := fileName
				if structName == "" {
					// Try to find name after "structure" or "named"
					for i, token := range taggedData.Tokens {
						if (strings.ToLower(token) == "structure" || strings.ToLower(token) == "named") && i+1 < len(taggedData.Tokens) {
							structName = taggedData.Tokens[i+1]
							break
						}
					}
				}

				if structName == "" {
					predictedSentence = "You need to provide a name for the data structure."
				} else {
					// capitalize for Go exported struct
					goStructName := ""
					for _, part := range strings.Split(structName, "_") {
						if part != "" {
							goStructName += strings.ToUpper(part[:1]) + part[1:]
						}
					}

					var structBody strings.Builder
					structBody.WriteString(fmt.Sprintf("package main\n\n// %s represents a data structure.\ntype %s struct {\n", goStructName, goStructName))

					// Parse fields: "with the fields name string age int"
					if strings.Contains(strings.ToLower(query), "with the fields") {
						queryParts := strings.Fields(query)
						fieldStartIndex := -1
						for i, part := range queryParts {
							if strings.ToLower(part) == "fields" && i > 0 && strings.ToLower(queryParts[i-1]) == "the" && i > 1 && strings.ToLower(queryParts[i-2]) == "with" {
								fieldStartIndex = i
								break
							}
						}

						if fieldStartIndex != -1 {
							for i := fieldStartIndex + 1; i < len(queryParts); {
								if strings.ToLower(queryParts[i]) == "and" || strings.ToLower(queryParts[i]) == "with" {
									i++
									continue
								}
								if i+1 < len(queryParts) {
									fieldName := queryParts[i]
									fieldType := queryParts[i+1]
									
									// capitalize field name
									goFieldName := strings.ToUpper(fieldName[:1]) + fieldName[1:]
									
									structBody.WriteString(fmt.Sprintf("\t%s %s `json:\"%s\"` \n", goFieldName, fieldType, fieldName))
									i += 2
								} else {
									break
								}
							}
						}
					}
					structBody.WriteString("}\n")

					filePath := structName + ".go"
					if targetDirectory != "" {
						filePath = filepath.Join(targetDirectory, filePath)
						os.MkdirAll(targetDirectory, 0755)
					}

					err := os.WriteFile(filePath, []byte(structBody.String()), 0644)
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't create the structure file %s: %v", filePath, err)
					} else {
						goImports(filePath)
						predictedSentence = fmt.Sprintf("I have created the data structure '%s' in %s.", goStructName, filePath)
						if tree, err := generateDirectoryTree(".", "", 0, 2, filePath); err == nil {
							predictedSentence += "\n\n" + tree
						}
					}
				}
			} else if strings.Contains(objectType, "folder") || strings.Contains(objectType, "directory") { // New block for folder creation
				folderName := fileName

				// Prefer explicit name after "folder" to avoid picking up filenames elsewhere in sentence
				for i, token := range taggedData.Tokens {
					if (strings.ToLower(token) == "folder" || strings.ToLower(token) == "directory") && i+1 < len(taggedData.Tokens) {
						candidate := taggedData.Tokens[i+1]
						if !contains([]string{"with", "in", "named", "and", "that", "which", "containing"}, strings.ToLower(candidate)) {
							folderName = candidate
						}
						break
					}
				}

				if folderName != "" {
					folderPath := folderName
					if targetDirectory != "" {
						folderPath = filepath.Join(targetDirectory, folderName)
					}
					err := os.MkdirAll(folderPath, 0755) // Use MkdirAll to create parent directories if needed
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't create the folder %s: %v", folderPath, err)
					} else {
						predictedSentence = fmt.Sprintf("I have created the folder %s.", folderPath)
						if tree, err := generateDirectoryTree(".", "", 0, 2, folderPath); err == nil {
							predictedSentence += "\n\n" + tree
						}
						if strings.Contains(strings.ToLower(folderName), "wasm") || strings.Contains(strings.ToLower(folderPath), "wasm") {
							buildWasm(folderPath)
						}
					}
				} else {
					predictedSentence = "You need to provide a name for the folder."
				}
			} else if strings.Contains(objectType, "database") { // New block for database creation
				if fileName == "" { // If findName didn't catch it, try to find it directly after "database"
					for i, token := range taggedData.Tokens {
						if strings.ToLower(token) == "database" && i+1 < len(taggedData.Tokens) {
							fileName = taggedData.Tokens[i+1]
							break
						}
					}
				}
				var db *sql.DB
				var err error
				if fileName != "" {
					dbFileName := fileName + ".db"
					db, err = sql.Open("sqlite", dbFileName)
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't open the database file %s: %v", dbFileName, err)
					} else {
						err = db.Ping() // This should force file creation
						if err != nil {
							predictedSentence = fmt.Sprintf("I couldn't connect to the database file %s: %v", dbFileName, err)
						} else {
							db.Close()
							predictedSentence = fmt.Sprintf("I have created the database file %s using the program's command.", dbFileName)
							if tree, err := generateDirectoryTree(".", "", 0, 2, dbFileName); err == nil {
								predictedSentence += "\n\n" + tree
							}

							// Check if "with the fields" is present to create a table
							if strings.Contains(strings.ToLower(query), "with the fields") {
								queryParts := strings.Fields(query)
								fieldStartIndex := -1
								for i, part := range queryParts {
									if strings.ToLower(part) == "fields" && i > 0 && strings.ToLower(queryParts[i-1]) == "the" && i > 1 && strings.ToLower(queryParts[i-2]) == "with" {
										fieldStartIndex = i
										break
									}
								}

								if fieldStartIndex != -1 {
									fields := make(map[string]string) // fieldName -> fieldType
									for i := fieldStartIndex + 1; i < len(queryParts); {
										if strings.ToLower(queryParts[i]) == "and" { // Skip "and"
											i++
											continue
										}
										if i+1 < len(queryParts) {
											fieldName := queryParts[i]
											fieldType := queryParts[i+1]
											fields[fieldName] = fieldType
											i += 2 // Move past fieldName and fieldType
										} else {
											log.Printf("Incomplete field definition found in query: %s", query)
											break
										}
									}

									if len(fields) > 0 {
										err = createTableWithFields(dbFileName, fileName, fields) // Use fileName as tableName
										if err != nil {
											predictedSentence += fmt.Sprintf(" But couldn't create the table '%s' in %s: %v", fileName, dbFileName, err)
										} else {
											predictedSentence += fmt.Sprintf(" And created table '%s' with the specified fields.", fileName)
										}
									} else {
										predictedSentence += " But no valid fields were provided to create a table."
									}
								}
							} else if strings.Contains(strings.ToLower(query), "using the data structure") || strings.Contains(strings.ToLower(query), "using data structure") {
								// Extract struct name
								parts := strings.Fields(query)
								var structName string
								for i, p := range parts {
									if (strings.ToLower(p) == "structure") && i+1 < len(parts) {
										structName = parts[i+1]
										break
									}
								}

								if structName != "" {
									// Try to find the struct file
									structFile := structName + ".go"
									if _, err := os.Stat(structFile); os.IsNotExist(err) {
										// Try lowercase
										structFile = strings.ToLower(structName) + ".go"
									}

									content, err := os.ReadFile(structFile)
									if err != nil {
										predictedSentence += fmt.Sprintf(" But I couldn't read the data structure file '%s': %v", structFile, err)
									} else {
										// Parse fields from struct
										fields := make(map[string]string)
										lines := strings.Split(string(content), "\n")
										inStruct := false
										for _, line := range lines {
											trimmed := strings.TrimSpace(line)
											if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, " struct {") {
												inStruct = true
												continue
											}
											if inStruct {
												if trimmed == "}" {
													break
												}
												parts := strings.Fields(trimmed)
												if len(parts) >= 2 {
													fName := parts[0]
													fType := parts[1]
													// Skip ID if it's auto-increment usually, but createTableWithFields handles ID separately
													if strings.ToLower(fName) != "id" {
														fields[fName] = fType
													}
												}
											}
										}

										if len(fields) > 0 {
											err = createTableWithFields(dbFileName, fileName, fields)
											if err != nil {
												predictedSentence += fmt.Sprintf(" But couldn't create the table '%s' in %s using struct '%s': %v", fileName, dbFileName, structName, err)
											} else {
												predictedSentence += fmt.Sprintf(" And created table '%s' using fields from data structure '%s'.", fileName, structName)
											}
										} else {
											predictedSentence += fmt.Sprintf(" But I couldn't find any fields in data structure '%s'.", structName)
										}
									}
								}
							}
						}
					}
				} else {
					predictedSentence = "You need to provide a name for the database."
				}

			} else if objectType == "data structure" {
				var err error
				var dbFileName string
				var tableName string
				var structFileName string
				var fieldKeywordFound bool
				var fieldStartIndex int
				var withTheFieldsIndex int = -1
				var dirName string
				var updateRegMsg string
				var err1 error
				var deleteRegMsg string
				var err2 error
				var showRegMsg string
				var err3 error
				var mainGoPath string
				var packageFileContent string
				var modulePath, projectRoot, cwd, relativeDir, packageImportPath string
				var lowercaseName string
				var packageName string

				queryParts := strings.Fields(query)
				structName := ""
				fields := make(map[string]string)

				for i, part := range queryParts {
					if part == "structure" && i+1 < len(queryParts) {
						if strings.ToLower(queryParts[i+1]) == "named" && i+2 < len(queryParts) {
							structName = strings.Title(queryParts[i+2])
						} else {
							structName = strings.Title(queryParts[i+1])
						}
						break
					}
				}

				if structName == "" {
					predictedSentence = "You need to provide a name for the data structure."
					goto endOfDataStructureCreation
				}

				// Create a directory for the data structure's database
				if targetDirectory != "" {
					dirName = targetDirectory
					packageName = filepath.Base(dirName)
				} else {
					dirName = strings.ToLower(structName)
					packageName = dirName
				}
				if err := os.MkdirAll(dirName, 0755); err != nil {
					predictedSentence = fmt.Sprintf("I couldn't create the directory %s: %v", dirName, err)
					goto endOfDataStructureCreation
				}

				// Look for "with fields" or "with the fields"
				for i := 0; i < len(queryParts)-1; i++ {
					if queryParts[i] == "with" && queryParts[i+1] == "fields" {
						withTheFieldsIndex = i + 2
						break
					}
					if i+2 < len(queryParts) && queryParts[i] == "with" && queryParts[i+1] == "the" && queryParts[i+2] == "fields" {
						withTheFieldsIndex = i + 3
						break
					}
				}

				if withTheFieldsIndex != -1 {
					fieldKeywordFound = true
					for i := withTheFieldsIndex; i < len(queryParts); {
						if queryParts[i] == "and" {
							i++
							continue
						}
						if i+1 < len(queryParts) {
							fieldName := queryParts[i]
							fieldType := queryParts[i+1]
							fields[fieldName] = fieldType
							i += 2
						} else {
							predictedSentence = "Incomplete field definition found."
							goto endOfDataStructureCreation
						}
					}
				} else {
					// Look for "field"
					for i, part := range queryParts {
						if part == "field" {
							fieldStartIndex = i
							break
						}
					}

					if fieldStartIndex != -1 {
						fieldKeywordFound = true
						for i := fieldStartIndex + 1; i < len(queryParts); {
							if queryParts[i] == "and" || queryParts[i] == "field" {
								i++
								continue
							}
							if i+1 < len(queryParts) {
								fieldName := queryParts[i]
								fieldType := queryParts[i+1]
								fields[fieldName] = fieldType
								i += 2
							} else {
								predictedSentence = "Incomplete field definition found."
								goto endOfDataStructureCreation
							}
						}
					}
				}

				if !fieldKeywordFound || len(fields) == 0 {
					fmt.Println("Please provide the fields for the data structure (e.g., 'name string age int'):")
					fieldQuery, _ := reader.ReadString('\n')
					fieldQuery = strings.TrimSpace(fieldQuery)
					fieldParts := strings.Fields(fieldQuery)
					for i := 0; i < len(fieldParts); {
						if i+1 < len(fieldParts) {
							fieldName := fieldParts[i]
							fieldType := fieldParts[i+1]
							fields[fieldName] = fieldType
							i += 2
						} else {
							predictedSentence = "Incomplete field definition found."
							goto endOfDataStructureCreation
						}
					}
				}

				// --- Start of new generation logic ---
				packageFileContent = generateDataStructurePackageContent(structName, packageName, dirName, fields)
				lowercaseName = strings.ToLower(structName)

				// Write the package file
				structFileName = filepath.Join(dirName, lowercaseName+".go")
				err = os.WriteFile(structFileName, []byte(packageFileContent), 0644)
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't create the Go package file %s: %v", structFileName, err)
					goto endOfDataStructureCreation
				}
				goImports(structFileName)
				predictedSentence = fmt.Sprintf("I have created the Go package '%s' in %s.", packageName, structFileName)
				if tree, err := generateDirectoryTree(".", "", 0, 2, structFileName); err == nil {
					predictedSentence += "\n\n" + tree
				}

				// Create database table
				tableName = lowercaseName
				dbFileName = filepath.Join(dirName, tableName+".db")
				err = createTableWithFields(dbFileName, tableName, fields)
				if err != nil {
					predictedSentence += fmt.Sprintf(" But couldn't create the table '%s' in %s: %v", tableName, dbFileName, err)
					goto endOfDataStructureCreation
				}
				predictedSentence += fmt.Sprintf(" And created the database '%s' with table '%s'.", dbFileName, tableName)

				// --- Find module path and project root for import path calculation ---
				modulePath, projectRoot, err = findGoModInfo()
				if err != nil {
					predictedSentence = fmt.Sprintf("Could not find go.mod info: %v", err)
					goto endOfDataStructureCreation
				}

				cwd, err = os.Getwd()
				if err != nil {
					predictedSentence = fmt.Sprintf("Could not get current working directory: %v", err)
					goto endOfDataStructureCreation
				}

				relativeDir, err = filepath.Rel(projectRoot, cwd)
				if err != nil {
					predictedSentence = fmt.Sprintf("Could not calculate relative path: %v", err)
					goto endOfDataStructureCreation
				}

				// The new package is in a subdirectory named dirName
				packageImportPath = filepath.Join(modulePath, relativeDir, dirName)
				packageImportPath = filepath.ToSlash(packageImportPath)

				// Register Handlers in main.go
				mainGoPath = "main.go"
				showRegMsg, err3 = registerHandlerWithPackage(packageName, packageImportPath, "Show"+structName, "/show/"+lowercaseName+"/", mainGoPath)
				if err3 != nil {
					predictedSentence += " " + err3.Error()
				} else {
					predictedSentence += " " + showRegMsg
				}
				updateRegMsg, err1 = registerHandlerWithPackage(packageName, packageImportPath, "Update"+structName, "/update/"+lowercaseName+"/", mainGoPath)
				if err1 != nil {
					predictedSentence += " " + err1.Error()
				} else {
					predictedSentence += " " + updateRegMsg
				}
				deleteRegMsg, err2 = registerHandlerWithPackage(packageName, packageImportPath, "Delete"+structName, "/delete/"+lowercaseName+"/", mainGoPath)
				if err2 != nil {
					predictedSentence += " " + err2.Error()
				} else {
					predictedSentence += " " + deleteRegMsg
				}

			endOfDataStructureCreation:
			} else if strings.Contains(objectType, "form") {
				log.Println("DEBUG: Starting form creation logic")
				sourceParam := intent.Params["source"]
				learningPath := kb.LearningPath
				if learningPath == "" {
					learningPath = filepath.Join(projectRoot, "learningfolder")
				}
				// Fallback to current directory's learningfolder if project root one doesn't exist
				if _, err := os.Stat(learningPath); os.IsNotExist(err) {
					cwd, _ := os.Getwd()
					localLearningPath := filepath.Join(cwd, "learningfolder")
					if _, err := os.Stat(localLearningPath); err == nil {
						learningPath = localLearningPath
					}
				}
				var htmlContent string
				var goHandlerContent string
				handlerName := "FormHandler"
				var learnedFilesList string
				useLearning := false

				if _, err := os.Stat(learningPath); err == nil {
					files, _ := os.ReadDir(learningPath)
					if len(files) > 0 {
						useLearning = true
					}
				}

				if sourceParam != "" {
					log.Printf("DEBUG: Attempting to generate form from source: %s", sourceParam)
					lowerName := strings.ToLower(sourceParam)
					candidates := []string{
						filepath.Join(projectRoot, lowerName, lowerName+".go"),
						filepath.Join(projectRoot, "cmd", lowerName, lowerName+".go"),
						filepath.Join(projectRoot, "internal", lowerName, lowerName+".go"),
						lowerName + ".go",
					}
					// Add singular variations if plural
					var singular string
					if strings.HasSuffix(lowerName, "s") {
						singular = strings.TrimSuffix(lowerName, "s")
						candidates = append(candidates,
							filepath.Join(projectRoot, singular, singular+".go"),
							filepath.Join(projectRoot, "cmd", singular, singular+".go"),
							filepath.Join(projectRoot, "internal", singular, singular+".go"),
						)
					}
					// Add search in subdirectories (depth 1) for project/app/pkg/struct.go style
					entries, _ := os.ReadDir(projectRoot)
					for _, entry := range entries {
						if entry.IsDir() && !strings.HasPrefix(entry.Name(), ".") {
							candidates = append(candidates, filepath.Join(projectRoot, entry.Name(), lowerName, lowerName+".go"))
							if singular != "" {
								candidates = append(candidates, filepath.Join(projectRoot, entry.Name(), lowerName, singular+".go"))
							}
							candidates = append(candidates, filepath.Join(projectRoot, entry.Name(), "pkg", lowerName, lowerName+".go"))
						}
					}

					var structContent string
					for _, path := range candidates {
						if content, err := os.ReadFile(path); err == nil {
							structContent = string(content)
							log.Printf("DEBUG: Found struct file at %s", path)
							break
						}
					}

					if structContent != "" {
						fields := make(map[string]string)
						lines := strings.Split(structContent, "\n")
						inStruct := false
						for _, line := range lines {
							trimmed := strings.TrimSpace(line)
							if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, " struct {") {
								inStruct = true
								continue
							}
							if inStruct {
								if trimmed == "}" {
									break
								}
								parts := strings.Fields(trimmed)
								if len(parts) >= 2 {
									fName := parts[0]
									if fName != "ID" {
										fields[fName] = parts[1]
									}
								}
							}
						}
						if len(fields) > 0 {
							htmlContent = fmt.Sprintf("<h1>Create %s</h1><form method='POST'>", strings.Title(sourceParam))
							for fName, fType := range fields {
								inputType := "text"
								if fType == "int" || fType == "int64" || fType == "float64" {
									inputType = "number"
								}
								htmlContent += fmt.Sprintf("<label>%s</label><input name='%s' type='%s' /><br/>", fName, strings.ToLower(fName), inputType)
							}
							htmlContent += "<button>Submit</button></form>"
						}
					}
				}

				if htmlContent == "" && goHandlerContent == "" && useLearning {
					log.Printf("DEBUG: Learning from files in %s", learningPath)
					files, _ := os.ReadDir(learningPath)
					var learnedFiles []string
					for _, file := range files {
						if !file.IsDir() {
							learnedFiles = append(learnedFiles, file.Name())
							content, err := os.ReadFile(filepath.Join(learningPath, file.Name()))
							if err == nil {
								if strings.HasSuffix(file.Name(), ".html") {
									htmlContent = string(content)
								} else if strings.HasSuffix(file.Name(), ".go") {
									fileContent := string(content)
									lines := strings.Split(fileContent, "\n")
									capture := false
									braceCount := 0
									for _, line := range lines {
										if goHandlerContent != "" && !capture {
											break
										}
										if strings.HasPrefix(strings.TrimSpace(line), "func ") && strings.Contains(line, "Handler") {
											capture = true
											parts := strings.Fields(line)
											if len(parts) >= 2 {
												namePart := parts[1]
												if idx := strings.Index(namePart, "("); idx != -1 {
													handlerName = namePart[:idx]
												}
											}
										}
										if capture {
											goHandlerContent += line + "\n"
											braceCount += strings.Count(line, "{")
											braceCount -= strings.Count(line, "}")
											if braceCount == 0 && strings.Contains(line, "}") {
												capture = false
											}
										}
									}
								}
							}
						}
					}
					if len(learnedFiles) > 0 {
						learnedFilesList = strings.Join(learnedFiles, ", ")
						fmt.Printf("Learning from files: %s\n", learnedFilesList)
						if htmlContent == "" && goHandlerContent == "" {
							fmt.Println("Warning: Learning folder found, but no suitable .html or .go handler content was extracted.")
						}
					}
				} else if htmlContent == "" && goHandlerContent == "" && strings.Contains(query, "database") {
					goHandlerContent = `
func FormHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		err := r.ParseForm()
		if err != nil {
			http.Error(w, "Error parsing form", http.StatusBadRequest)
			return
		}
		data := r.Form.Get("data")
		if db != nil {
			_, err := db.Exec("CREATE TABLE IF NOT EXISTS form_data (id INTEGER PRIMARY KEY, content TEXT)")
			if err == nil {
				_, err = db.Exec("INSERT INTO form_data (content) VALUES (?)", data)
				if err != nil {
					fmt.Fprintf(w, "Error saving to DB: %v", err)
					return
				}
				fmt.Fprintf(w, "Data saved successfully!")
				return
			}
		}
		fmt.Fprintf(w, "Database not available or error creating table.")
	} else {
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, "<form method='POST'><input name='data' type='text'/><button>Submit</button></form>")
	}
}
`
				}

				targetWebserverPath := ""
				// 0. Check targetDirectory if specified
				if targetDirectory != "" {
					// Check direct path
					path := filepath.Join(targetDirectory, "main.go")
					if _, err := os.Stat(path); err == nil {
						targetWebserverPath = path
					} else {
						// Check for cmd/ inside targetDirectory
						cmdDir := filepath.Join(targetDirectory, "cmd")
						if _, err := os.Stat(cmdDir); err == nil {
							entries, _ := os.ReadDir(cmdDir)
							for _, entry := range entries {
								if entry.IsDir() {
									path := filepath.Join(cmdDir, entry.Name(), "main.go")
									if _, err := os.Stat(path); err == nil {
										targetWebserverPath = path
										break
									}
								}
							}
						}
					}
				}

				// 1. Check current directory for a webserver main.go
				if targetWebserverPath == "" {
					if targetDirectory != "" {
						log.Printf("DEBUG: Webserver not found in target directory '%s'. Falling back to search.", targetDirectory)
					}
					log.Println("DEBUG: Checking current directory for main.go")
					if _, err := os.Stat("main.go"); err == nil {
						content, _ := os.ReadFile("main.go")
						sContent := string(content)
						// Avoid modifying the assistant itself
						if !strings.Contains(sContent, "github.com/golangast/gollemer") {
							if strings.Contains(sContent, "http.ListenAndServe") || strings.Contains(sContent, "\"net/http\"") {
								targetWebserverPath = "main.go"
								log.Println("DEBUG: Found main.go in current directory")
							}
						}
					}
				}

				// 2. If not found, check cmd/ directory for any webserver
				if targetWebserverPath == "" {
					cmdDir := filepath.Join(projectRoot, "cmd")
					entries, _ := os.ReadDir(cmdDir)
					for _, entry := range entries {
						if entry.IsDir() {
							path := filepath.Join(cmdDir, entry.Name(), "main.go")
							if _, err := os.Stat(path); err == nil {
								content, _ := os.ReadFile(path)
								if strings.Contains(string(content), "http.ListenAndServe") || strings.Contains(string(content), "\"net/http\"") {
									targetWebserverPath = path
									break
								}
							}
						}
					}
				}

				// 3. If still not found, check for nested structures like project/jim/cmd/jim/main.go
				if targetWebserverPath == "" {
					entries, _ := os.ReadDir(projectRoot)
					for _, entry := range entries {
						log.Printf("DEBUG: Checking project root entry: %s", entry.Name())
						if entry.IsDir() && entry.Name() != "cmd" && entry.Name() != "bin" && !strings.HasPrefix(entry.Name(), ".") {
							// Check project/jim/main.go
							path := filepath.Join(projectRoot, entry.Name(), "main.go")
							if _, err := os.Stat(path); err == nil {
								content, _ := os.ReadFile(path)
								if strings.Contains(string(content), "http.ListenAndServe") || strings.Contains(string(content), "\"net/http\"") {
									targetWebserverPath = path
									break
								}
							}

							if targetWebserverPath == "" {
								nestedCmdDir := filepath.Join(projectRoot, entry.Name(), "cmd")
								if _, err := os.Stat(nestedCmdDir); err == nil {
									log.Printf("DEBUG: Checking nested cmd dir: %s", nestedCmdDir)
									// Check project/jim/cmd/main.go
									path := filepath.Join(nestedCmdDir, "main.go")
									if _, err := os.Stat(path); err == nil {
										content, _ := os.ReadFile(path)
										if strings.Contains(string(content), "http.ListenAndServe") || strings.Contains(string(content), "\"net/http\"") {
											targetWebserverPath = path
											break
										}
									}

									// Check project/jim/cmd/*/main.go
									if targetWebserverPath == "" {
										nestedEntries, _ := os.ReadDir(nestedCmdDir)
										for _, nestedEntry := range nestedEntries {
											if nestedEntry.IsDir() {
												log.Printf("DEBUG: Checking nested entry in cmd: %s", nestedEntry.Name())
												path := filepath.Join(nestedCmdDir, nestedEntry.Name(), "main.go")
												if _, err := os.Stat(path); err == nil {
													content, _ := os.ReadFile(path)
													if strings.Contains(string(content), "http.ListenAndServe") || strings.Contains(string(content), "\"net/http\"") {
														targetWebserverPath = path
														break
													}
												}
											}
										}
									}
								}
							}
						}
						if targetWebserverPath != "" {
							break
						}
					}
				}

				// 4. Hard fallback for common structure if still not found
				if targetWebserverPath == "" {
					fallback := filepath.Join(projectRoot, "jim", "cmd", "jim", "main.go")
					if _, err := os.Stat(fallback); err == nil {
						targetWebserverPath = fallback
					}
				}

				if targetWebserverPath == "" {
					log.Printf("DEBUG: Could not find webserver. targetDirectory='%s', projectRoot='%s'", targetDirectory, projectRoot)
					predictedSentence = "I couldn't find a target webserver (main.go with net/http) in the current directory or in cmd/."
				} else {
					log.Printf("DEBUG: Found target webserver at %s", targetWebserverPath)
					newHandlerCode := ""
					if goHandlerContent != "" {
						newHandlerCode = "\n" + goHandlerContent
					} else if htmlContent != "" {
						newHandlerCode = fmt.Sprintf(`
func FormHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	fmt.Fprint(w, `+"`"+`%s`+"`"+`)
}
`, htmlContent)
					} else {
						newHandlerCode = `
func FormHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "<h1>Generated Form</h1><form><input type='text'/><button>Submit</button></form>")
}
`
					}

					// Try to find module root to create forms/ package
					absTargetWebserverPath, absErr := filepath.Abs(targetWebserverPath)
					if absErr != nil {
						absTargetWebserverPath = targetWebserverPath
					}
					webserverDir := filepath.Dir(absTargetWebserverPath)
					moduleRoot := ""
					moduleName := ""
					searchDir := webserverDir
					for {
						if _, err := os.Stat(filepath.Join(searchDir, "go.mod")); err == nil {
							moduleRoot = searchDir
							modContent, _ := os.ReadFile(filepath.Join(searchDir, "go.mod"))
							lines := strings.Split(string(modContent), "\n")
							for _, line := range lines {
								if strings.HasPrefix(line, "module ") {
									moduleName = strings.TrimSpace(strings.TrimPrefix(line, "module "))
									break
								}
							}
							break
						}
						parent := filepath.Dir(searchDir)
						if parent == searchDir {
							break
						}
						searchDir = parent
					}

					if moduleRoot != "" && moduleName != "" {
						// Create forms directory
						formsDir := filepath.Join(moduleRoot, "forms")
						if err := os.MkdirAll(formsDir, 0755); err != nil {
							predictedSentence = fmt.Sprintf("Failed to create forms directory: %v", err)
						} else {
							formGoPath := filepath.Join(formsDir, "form.go")
							pkgContent := "package forms\n\nimport (\n\t\"fmt\"\n\t\"net/http\"\n)\n"
							fullContent := pkgContent + newHandlerCode

							if err := os.WriteFile(formGoPath, []byte(fullContent), 0644); err != nil {
								predictedSentence = fmt.Sprintf("Failed to write form.go: %v", err)
							} else {
								goImports(formGoPath)
								importPath := moduleName + "/forms"
								regName := strings.TrimSuffix(handlerName, "Handler")
								regMsg, err := registerHandlerWithPackage("forms", importPath, regName, "/form", targetWebserverPath)

								verifyMsg := ""
								if _, err := os.Stat(formsDir); err == nil {
									if _, err := os.Stat(formGoPath); err == nil {
										verifyMsg = " Verified forms folder and form.go exist."
									} else {
										verifyMsg = " Verified forms folder exists, but form.go is missing."
									}

									// Verify forms package builds
									buildCmd := exec.Command("go", "build", ".")
									buildCmd.Dir = formsDir
									if out, err := buildCmd.CombinedOutput(); err != nil {
										log.Printf("DEBUG: Forms package build failed: %s", strings.TrimSpace(string(out)))
										verifyMsg += fmt.Sprintf(" Warning: forms package failed to build: %s", strings.TrimSpace(string(out)))
									} else {
										log.Printf("DEBUG: Forms package built successfully.")
										verifyMsg += " Verified forms package builds."
									}

									// Verify main package builds (integration check)
									mainBuildCmd := exec.Command("go", "build", ".")
									mainBuildCmd.Dir = webserverDir
									if out, err := mainBuildCmd.CombinedOutput(); err != nil {
										log.Printf("DEBUG: Webserver build failed: %s", strings.TrimSpace(string(out)))
										verifyMsg += fmt.Sprintf(" Warning: Webserver failed to build after update: %s", strings.TrimSpace(string(out)))
									} else {
										log.Printf("DEBUG: Webserver built successfully.")
										verifyMsg += " Verified webserver builds."
									}
								} else {
									verifyMsg = fmt.Sprintf(" Warning: Could not verify forms folder exists: %v", err)
								}

								if err != nil {
									predictedSentence = fmt.Sprintf("Created form in %s but failed to register: %v%s", formGoPath, err, verifyMsg)
								} else {
									predictedSentence = fmt.Sprintf("Created form in %s. %s%s", formGoPath, regMsg, verifyMsg)
									if tree, err := generateDirectoryTree(".", "", 0, 2, formGoPath); err == nil {
										predictedSentence += "\n\n" + tree
									}
								}
							}
						}
					} else {
						// Fallback: Append to main.go if module root not found
						mainContentBytes, err := os.ReadFile(targetWebserverPath)
						if err != nil {
							predictedSentence = fmt.Sprintf("Could not read target main.go: %v", err)
						} else {
							mainContent := string(mainContentBytes)
							if strings.Contains(mainContent, "func "+handlerName) {
								predictedSentence = fmt.Sprintf("Handler %s already exists in %s.", handlerName, targetWebserverPath)
							} else {
								mainContent += newHandlerCode
								regLine := fmt.Sprintf("\thttp.HandleFunc(\"/form\", %s)", handlerName)
								if strings.Contains(mainContent, "// HANDLER_REGISTRATIONS_GO_HERE") {
									lines := strings.Split(mainContent, "\n")
									for i, line := range lines {
										if strings.Contains(line, "// HANDLER_REGISTRATIONS_GO_HERE") {
											indent := line[:strings.Index(line, "// HANDLER_REGISTRATIONS_GO_HERE")]
											lines[i] = indent + regLine + "\n" + line
											break
										}
									}
									mainContent = strings.Join(lines, "\n")
								} else if idx := strings.LastIndex(mainContent, "http.ListenAndServe"); idx != -1 {
									mainContent = mainContent[:idx] + regLine + "\n\t" + mainContent[idx:]
								} else if idx := strings.Index(mainContent, "func main() {"); idx != -1 {
									// Fallback: Insert at the beginning of main function
									insertionPoint := idx + len("func main() {")
									mainContent = mainContent[:insertionPoint] + "\n" + regLine + mainContent[insertionPoint:]
								}

								err = os.WriteFile(targetWebserverPath, []byte(mainContent), 0644)
								if err != nil {
									predictedSentence = fmt.Sprintf("Failed to update main.go: %v", err)
								} else {
									log.Printf("DEBUG: Successfully wrote form handler to %s", targetWebserverPath)
									goImports(targetWebserverPath)
									sourceMsg := "learningfolder"
									if learnedFilesList != "" {
										sourceMsg = fmt.Sprintf("files (%s)", learnedFilesList)
									}
									predictedSentence = fmt.Sprintf("I have added the form handler to %s based on %s.", targetWebserverPath, sourceMsg)
									if tree, err := generateDirectoryTree(".", "", 0, 2, targetWebserverPath); err == nil {
										predictedSentence += "\n\n" + tree
									}

									// Verify the file content
									checkContent, err := os.ReadFile(targetWebserverPath)
									if err == nil && strings.Contains(string(checkContent), "func "+handlerName) {
										predictedSentence += " I checked the file and confirmed the handler is there."
									}
								}
							}
						}
					}
				}
			} else {
				handled = false
			}

		case "run", "start":
			if strings.Contains(objectType, "webserver") || contains(objectTypeParts, "file") {
				webserverName := ""
				for i, token := range taggedData.Tokens {
					tokenLower := strings.ToLower(token)
					if (tokenLower == "webserver" || tokenLower == "websever" || tokenLower == "file") && i+1 < len(taggedData.Tokens) {
						webserverName = taggedData.Tokens[i+1]
						break
					}
				}
				if webserverName == "" {
					webserverName = fileName
				}

				// --- Intent Guessing: Handle runs targeting a specific main.go file ---
				var jimSourcePath string
				if strings.HasSuffix(webserverName, ".go") {
					absPath, err := filepath.Abs(webserverName)
					if err == nil {
						if info, err := os.Stat(absPath); err == nil && !info.IsDir() {
							jimSourcePath = filepath.Dir(absPath)
							// If the file is main.go, infer the webserver name from the directory
							if strings.ToLower(filepath.Base(absPath)) == "main.go" {
								webserverName = filepath.Base(jimSourcePath)
							} else {
								webserverName = strings.TrimSuffix(filepath.Base(absPath), ".go")
							}
							log.Printf("   [INTENT] Guessed webserver '%s' from file path.", webserverName)
						}
					}
				}

				if webserverName == "" {
					predictedSentence = "You need to provide a name for the webserver to run."
				} else {
					// Path to the jim webserver's main package
					// (Removed redundant declaration as it's now handled by the intent guessing above)

						// Prioritized Search: 
						// 1. Current Working Directory (if name matches)
						// 2. Project root directory with name (e.g. ./jim)
						// 3. cmd/ subdirectory (e.g. ./cmd/jim)
						cwd, _ := os.Getwd()
						candidates := []string{}
						
						if strings.EqualFold(filepath.Base(cwd), webserverName) {
							candidates = append(candidates, cwd)
						}
						
						candidates = append(candidates, []string{
							filepath.Join(projectRoot, webserverName),
							filepath.Join(projectRoot, "cmd", webserverName),
							filepath.Join(projectRoot, webserverName, "cmd", webserverName),
							filepath.Join(projectRoot, webserverName, "cmd"),
						}...)

						for _, p := range candidates {
							if _, err := os.Stat(filepath.Join(p, "main.go")); err == nil {
								// If we find one with a wasm or template folder, it's a strong candidate
								if _, errW := os.Stat(filepath.Join(p, "wasm")); errW == nil {
									jimSourcePath = p
									break
								}
								if _, errT := os.Stat(filepath.Join(p, "template")); errT == nil {
									jimSourcePath = p
									break
								}
								if jimSourcePath == "" {
									jimSourcePath = p
								}
							}
						}

					log.Printf("DEBUG: Jim Webserver Source Path: %s", jimSourcePath)

					// Check if jimSourcePath exists
					if _, err := os.Stat(jimSourcePath); err != nil {
						if os.IsNotExist(err) {
							predictedSentence = fmt.Sprintf("I couldn't find a webserver named '%s' at path '%s'.", webserverName, jimSourcePath)
						} else {
							predictedSentence = fmt.Sprintf("Error checking webserver directory '%s': %v", jimSourcePath, err)
						}
					} else {
						// Define the output path for the built executable
						buildOutputDir := filepath.Join(projectRoot, "bin")
						if err := os.MkdirAll(buildOutputDir, 0755); err != nil {
							predictedSentence = fmt.Sprintf("Failed to create build directory %s: %v", buildOutputDir, err)
							goto endOfRunWebserver
						}
						jimExecutablePath := filepath.Join(buildOutputDir, webserverName)

						// Build the jim webserver executable
						log.Printf("DEBUG: Building webserver %s...", webserverName)

						// Add missing sqlite dependency
						getCmd := exec.Command("go", "get", "modernc.org/sqlite")
						getCmd.Dir = jimSourcePath
						getOutput, getErr := getCmd.CombinedOutput()
						if getErr != nil {
							predictedSentence = fmt.Sprintf("Failed to get sqlite dependency for webserver %s: %v\nOutput:\n%s", webserverName, getErr, string(getOutput))
							goto endOfRunWebserver
						}
						log.Printf("DEBUG: Successfully got sqlite dependency for webserver %s", webserverName)

						buildCmd := exec.Command("go", "build", "-o", jimExecutablePath, ".")
						buildCmd.Dir = jimSourcePath // Build from the webserver's source directory
						buildOutput, buildErr := buildCmd.CombinedOutput()
						if buildErr != nil {
							predictedSentence = fmt.Sprintf("Failed to build webserver %s: %v\nBuild Output:\n%s", webserverName, buildErr, string(buildOutput))
							goto endOfRunWebserver
						}
						log.Printf("DEBUG: Webserver %s built successfully to %s", webserverName, jimExecutablePath)

						// --- Build WASM if relevant ---
						buildWasm(filepath.Join(jimSourcePath, "wasm"))
						buildWasm(filepath.Join(projectRoot, "learningfolder", "wasm"))

						// --- Categorize: Webserver vs CLI Script ---
						isWebserver := false
						mainFile := filepath.Join(jimSourcePath, "main.go")
						if content, err := os.ReadFile(mainFile); err == nil {
							sContent := string(content)
							if strings.Contains(sContent, "http.ListenAndServe") || strings.Contains(sContent, "\"net/http\"") {
								isWebserver = true
							}
						}

						if isWebserver {
							// Run as Webserver (Asynchronous + Verification)
							runCmd := exec.Command(jimExecutablePath, "-llm")
							runCmd.Dir = jimSourcePath
							runCmd.Stdout = os.Stdout
							runCmd.Stderr = os.Stderr

							err := runCmd.Start()
							if err != nil {
								predictedSentence = fmt.Sprintf("I couldn't run the webserver %s: %v", webserverName, err)
							} else {
								pidFile := filepath.Join(buildOutputDir, webserverName+".pid")
								if err := SavePid(runCmd.Process.Pid, pidFile); err != nil {
									log.Printf("Failed to save PID file: %v", err)
								}
								predictedSentence = fmt.Sprintf("I have started the webserver %s. PID: %d", webserverName, runCmd.Process.Pid)

								// --- Verification step ---
								log.Printf("DEBUG: Waiting for webserver to start...")
								time.Sleep(2 * time.Second)

								resp, err := http.Get("http://localhost:8080/")
								if err != nil {
									log.Printf("WARNING: Webserver verification failed: %v", err)
									predictedSentence += " However, I could not verify that the webserver is running."
								} else {
									defer resp.Body.Close()
									if resp.StatusCode == http.StatusOK {
										predictedSentence += " And I have verified that the webserver is running."
										respForm, errForm := http.Get("http://localhost:8080/form")
										if errForm == nil {
											if respForm.StatusCode == http.StatusOK {
												predictedSentence += " The /form endpoint is also accessible."
											}
											respForm.Body.Close()
										}
									} else {
										predictedSentence += fmt.Sprintf(" However, the webserver returned status code %d during verification.", resp.StatusCode)
									}
								}
							}
						} else {
							// Run as CLI Script (Synchronous)
							log.Printf("   [INTENT] Detected CLI script, running synchronously...")
							runCmd := exec.Command(jimExecutablePath)
							runCmd.Dir = jimSourcePath
							runCmd.Stdout = os.Stdout
							runCmd.Stderr = os.Stderr
							err := runCmd.Run()
							if err != nil {
								predictedSentence = fmt.Sprintf("Program '%s' finished with error: %v", webserverName, err)
							} else {
								predictedSentence = fmt.Sprintf("Program '%s' executed successfully.", webserverName)
							}
						}
					}
				endOfRunWebserver: // Label for goto
				}
			} else {
				handled = false
			}
		case "update":
			if strings.Contains(objectType, "form") {
				learningPath := kb.LearningPath
				if learningPath == "" {
					learningPath = filepath.Join(projectRoot, "learningfolder")
				}
				// Fallback to current directory's learningfolder if project root one doesn't exist
				if _, err := os.Stat(learningPath); os.IsNotExist(err) {
					cwd, _ := os.Getwd()
					localLearningPath := filepath.Join(cwd, "learningfolder")
					if _, err := os.Stat(localLearningPath); err == nil {
						learningPath = localLearningPath
					}
				}
				var htmlContent string
				var goHandlerContent string
				handlerName := "FormHandler"
				var learnedFilesList string
				useLearning := false

				if _, err := os.Stat(learningPath); err == nil {
					files, _ := os.ReadDir(learningPath)
					if len(files) > 0 {
						useLearning = true
					}
				}

				if useLearning {
					files, _ := os.ReadDir(learningPath)
					var learnedFiles []string
					for _, file := range files {
						if !file.IsDir() {
							learnedFiles = append(learnedFiles, file.Name())
							content, err := os.ReadFile(filepath.Join(learningPath, file.Name()))
							if err == nil {
								if strings.HasSuffix(file.Name(), ".html") {
									htmlContent = string(content)
								} else if strings.HasSuffix(file.Name(), ".go") {
									fileContent := string(content)
									lines := strings.Split(fileContent, "\n")
									capture := false
									braceCount := 0
									for _, line := range lines {
										if goHandlerContent != "" && !capture {
											break
										}
										if strings.HasPrefix(strings.TrimSpace(line), "func ") && strings.Contains(line, "Handler") {
											capture = true
											parts := strings.Fields(line)
											if len(parts) >= 2 {
												namePart := parts[1]
												if idx := strings.Index(namePart, "("); idx != -1 {
													handlerName = namePart[:idx]
												}
											}
										}
										if capture {
											goHandlerContent += line + "\n"
											braceCount += strings.Count(line, "{")
											braceCount -= strings.Count(line, "}")
											if braceCount == 0 && strings.Contains(line, "}") {
												capture = false
											}
										}
									}
								}
							}
						}
					}
					if len(learnedFiles) > 0 {
						learnedFilesList = strings.Join(learnedFiles, ", ")
						fmt.Printf("Learning from files: %s\n", learnedFilesList)
						if htmlContent == "" && goHandlerContent == "" {
							fmt.Println("Warning: Learning folder found, but no suitable .html or .go handler content was extracted.")
						}
					}
				} else if strings.Contains(query, "database") {
					goHandlerContent = `
func FormHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		err := r.ParseForm()
		if err != nil {
			http.Error(w, "Error parsing form", http.StatusBadRequest)
			return
		}
		data := r.Form.Get("data")
		if db != nil {
			_, err := db.Exec("CREATE TABLE IF NOT EXISTS form_data (id INTEGER PRIMARY KEY, content TEXT)")
			if err == nil {
				_, err = db.Exec("INSERT INTO form_data (content) VALUES (?)", data)
				if err != nil {
					fmt.Fprintf(w, "Error saving to DB: %v", err)
					return
				}
				fmt.Fprintf(w, "Data saved successfully!")
				return
			}
		}
		fmt.Fprintf(w, "Database not available or error creating table.")
	} else {
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, "<form method='POST'><input name='data' type='text'/><button>Submit</button></form>")
	}
}
`
				}

				targetWebserverPath := ""
				// 0. Check targetDirectory if specified
				if targetDirectory != "" {
					// Check direct path
					path := filepath.Join(targetDirectory, "main.go")
					if _, err := os.Stat(path); err == nil {
						targetWebserverPath = path
					} else {
						// Check for cmd/ inside targetDirectory
						cmdDir := filepath.Join(targetDirectory, "cmd")
						if _, err := os.Stat(cmdDir); err == nil {
							entries, _ := os.ReadDir(cmdDir)
							for _, entry := range entries {
								if entry.IsDir() {
									path := filepath.Join(cmdDir, entry.Name(), "main.go")
									if _, err := os.Stat(path); err == nil {
										targetWebserverPath = path
										break
									}
								}
							}
						}
					}
				}

				// 1. Check current directory for a webserver main.go
				if targetWebserverPath == "" {
					if _, err := os.Stat("main.go"); err == nil {
						content, _ := os.ReadFile("main.go")
						sContent := string(content)
						// Avoid modifying the assistant itself
						if !strings.Contains(sContent, "github.com/golangast/gollemer") {
							if strings.Contains(sContent, "http.ListenAndServe") || strings.Contains(sContent, "\"net/http\"") {
								targetWebserverPath = "main.go"
							}
						}
					}
				}

				// 2. If not found, check cmd/ directory for any webserver
				if targetWebserverPath == "" {
					cmdDir := filepath.Join(projectRoot, "cmd")
					entries, _ := os.ReadDir(cmdDir)
					for _, entry := range entries {
						if entry.IsDir() {
							path := filepath.Join(cmdDir, entry.Name(), "main.go")
							if _, err := os.Stat(path); err == nil {
								content, _ := os.ReadFile(path)
								if strings.Contains(string(content), "http.ListenAndServe") || strings.Contains(string(content), "\"net/http\"") {
									targetWebserverPath = path
									break
								}
							}
						}
					}
				}

				// 3. If still not found, check for nested structures like project/jim/cmd/jim/main.go
				if targetWebserverPath == "" {
					entries, _ := os.ReadDir(projectRoot)
					for _, entry := range entries {
						if entry.IsDir() && entry.Name() != "cmd" && entry.Name() != "bin" && !strings.HasPrefix(entry.Name(), ".") {
							// Check project/jim/main.go
							path := filepath.Join(projectRoot, entry.Name(), "main.go")
							if _, err := os.Stat(path); err == nil {
								content, _ := os.ReadFile(path)
								if strings.Contains(string(content), "http.ListenAndServe") || strings.Contains(string(content), "\"net/http\"") {
									targetWebserverPath = path
									break
								}
							}

							if targetWebserverPath == "" {
								nestedCmdDir := filepath.Join(projectRoot, entry.Name(), "cmd")
								if _, err := os.Stat(nestedCmdDir); err == nil {
									// Check project/jim/cmd/main.go
									path := filepath.Join(nestedCmdDir, "main.go")
									if _, err := os.Stat(path); err == nil {
										content, _ := os.ReadFile(path)
										if strings.Contains(string(content), "http.ListenAndServe") || strings.Contains(string(content), "\"net/http\"") {
											targetWebserverPath = path
											break
										}
									}

									// Check project/jim/cmd/*/main.go
									if targetWebserverPath == "" {
										nestedEntries, _ := os.ReadDir(nestedCmdDir)
										for _, nestedEntry := range nestedEntries {
											if nestedEntry.IsDir() {
												path := filepath.Join(nestedCmdDir, nestedEntry.Name(), "main.go")
												if _, err := os.Stat(path); err == nil {
													content, _ := os.ReadFile(path)
													if strings.Contains(string(content), "http.ListenAndServe") || strings.Contains(string(content), "\"net/http\"") {
														targetWebserverPath = path
														break
													}
												}
											}
										}
									}
								}
							}
						}
						if targetWebserverPath != "" {
							break
						}
					}
				}

				if targetWebserverPath == "" {
					predictedSentence = "I couldn't find a target webserver (main.go) in the current directory or in cmd/."
				} else {
					newHandlerCode := ""
					if goHandlerContent != "" {
						newHandlerCode = "\n" + goHandlerContent
					} else if htmlContent != "" {
						newHandlerCode = fmt.Sprintf(`
func FormHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	fmt.Fprint(w, `+"`"+`%s`+"`"+`)
}
`, htmlContent)
					} else {
						newHandlerCode = `
func FormHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "<h1>Generated Form</h1><form><input type='text'/><button>Submit</button></form>")
}
`
					}

					mainContentBytes, err := os.ReadFile(targetWebserverPath)
					if err != nil {
						predictedSentence = fmt.Sprintf("Could not read target main.go: %v", err)
					} else {
						mainContent := string(mainContentBytes)
						funcStart := "func " + handlerName
						startIdx := strings.Index(mainContent, funcStart)

						if startIdx != -1 {
							braceCount := 0
							endIdx := -1
							started := false
							for i := startIdx; i < len(mainContent); i++ {
								if mainContent[i] == '{' {
									braceCount++
									started = true
								} else if mainContent[i] == '}' {
									braceCount--
								}
								if started && braceCount == 0 {
									endIdx = i + 1
									break
								}
							}

							if endIdx != -1 {
								mainContent = mainContent[:startIdx] + strings.TrimSpace(newHandlerCode) + mainContent[endIdx:]
								sourceMsg := "learningfolder"
								if learnedFilesList != "" {
									sourceMsg = fmt.Sprintf("files (%s)", learnedFilesList)
								}
								predictedSentence = fmt.Sprintf("I have updated the handler %s in %s based on %s.", handlerName, targetWebserverPath, sourceMsg)
							} else {
								predictedSentence = fmt.Sprintf("Could not parse existing handler %s in %s to update it.", handlerName, targetWebserverPath)
							}
						} else {
							mainContent += newHandlerCode
							regLine := fmt.Sprintf("\thttp.HandleFunc(\"/form\", %s)", handlerName)
							if strings.Contains(mainContent, "// HANDLER_REGISTRATIONS_GO_HERE") {
								mainContent = strings.Replace(mainContent, "// HANDLER_REGISTRATIONS_GO_HERE", regLine+"\n\t// HANDLER_REGISTRATIONS_GO_HERE", 1)
							} else if idx := strings.LastIndex(mainContent, "http.ListenAndServe"); idx != -1 {
								mainContent = mainContent[:idx] + regLine + "\n\t" + mainContent[idx:]
							} else if idx := strings.Index(mainContent, "func main() {"); idx != -1 {
								// Fallback: Insert at the beginning of main function
								insertionPoint := idx + len("func main() {")
								mainContent = mainContent[:insertionPoint] + "\n" + regLine + mainContent[insertionPoint:]
							}
							predictedSentence = fmt.Sprintf("Handler %s did not exist, so I added it to %s.", handlerName, targetWebserverPath)
						}

						err = os.WriteFile(targetWebserverPath, []byte(mainContent), 0644)
						if err != nil {
							predictedSentence = fmt.Sprintf("Failed to update main.go: %v", err)
						} else {
							goImports(targetWebserverPath)
						}
					}
				}
			} else if objectType != "" {
				msg, ok := handleGenericCreate(objectType, fileName, targetDirectory, handlerURL, kb)
				if ok {
					predictedSentence = msg
					handled = true
				} else {
					handled = false
				}
			} else {
				handled = false
			}
		case "stop":
			if strings.Contains(objectType, "webserver") {
				webserverName := ""
				for i, token := range taggedData.Tokens {
					if strings.ToLower(token) == "webserver" && i+1 < len(taggedData.Tokens) {
						webserverName = taggedData.Tokens[i+1]
						break
					}
				}
				if webserverName == "" {
					webserverName = fileName
				}

				if webserverName == "" {
					predictedSentence = "You need to provide a name for the webserver to stop."
				} else {
					buildOutputDir := filepath.Join(projectRoot, "bin")
					pidFile := filepath.Join(buildOutputDir, webserverName+".pid")
					err := StopWebserver(pidFile)
					if err != nil {
						predictedSentence = fmt.Sprintf("Failed to stop webserver %s: %v", webserverName, err)
					} else {
						predictedSentence = fmt.Sprintf("Stopped webserver %s.", webserverName)
					}
				}
			} else {
				handled = false
			}
		case "delete":
			if objectType == "data structure" {
				queryParts := strings.Fields(query)
				structName := ""
				for i, part := range queryParts {
					if part == "structure" && i+1 < len(queryParts) {
						if strings.ToLower(queryParts[i+1]) == "named" && i+2 < len(queryParts) {
							structName = strings.Title(queryParts[i+2])
						} else {
							structName = strings.Title(queryParts[i+1])
						}
						break
					}
				}

				if structName == "" {
					predictedSentence = "You need to provide the name of the data structure to delete."
				} else {
					var fieldToDelete string
					for i, part := range queryParts {
						if part == "field" && i+1 < len(queryParts) {
							fieldToDelete = queryParts[i+1]
							break
						}
					}

					if fieldToDelete != "" {
						lowercaseName := strings.ToLower(structName)
						dirName := lowercaseName
						structFileName := filepath.Join(dirName, lowercaseName+".go")

						content, err := os.ReadFile(structFileName)
						if err != nil {
							predictedSentence = fmt.Sprintf("Could not read file %s: %v", structFileName, err)
						} else {
							fields := make(map[string]string)
							lines := strings.Split(string(content), "\n")
							inStruct := false
							for _, line := range lines {
								trimmed := strings.TrimSpace(line)
								if strings.HasPrefix(trimmed, fmt.Sprintf("type %s struct {", structName)) {
									inStruct = true
									continue
								}
								if inStruct {
									if trimmed == "}" {
										break
									}
									parts := strings.Fields(trimmed)
									if len(parts) >= 2 {
										fName := parts[0]
										if fName == "ID" {
											continue
										}
										fType := parts[1]
										fields[strings.ToLower(fName)] = fType
									}
								}
							}

							if _, exists := fields[strings.ToLower(fieldToDelete)]; exists {
								// Field exists, proceed with deletion.
								delete(fields, strings.ToLower(fieldToDelete))

								// 1. Update the database table
								dbFileName := filepath.Join(dirName, lowercaseName+".db")
								tableName := lowercaseName
								err = deleteColumnFromTable(dbFileName, tableName, fieldToDelete, fields)
								if err != nil {
									predictedSentence = fmt.Sprintf("I failed to delete column '%s' from database table '%s': %v. The Go struct was not modified.", fieldToDelete, tableName, err)
								} else {
									// 2. Update the Go source file
									newContent := generateDataStructurePackageContent(structName, lowercaseName, dirName, fields)
									err = os.WriteFile(structFileName, []byte(newContent), 0644)
									if err != nil {
										predictedSentence = fmt.Sprintf("I updated the database, but failed to update the Go file %s: %v. Please check for inconsistencies.", structFileName, err)
									} else {
										goImports(structFileName)
										predictedSentence = fmt.Sprintf("I have deleted the field '%s' from data structure '%s' and updated the database.", fieldToDelete, structName)
									}
								}
							} else {
								predictedSentence = fmt.Sprintf("Field '%s' not found in data structure '%s'.", fieldToDelete, structName)
							}
						}
					} else {
						lowercaseName := strings.ToLower(structName)
						dirName := lowercaseName

						// Remove the directory
						err := os.RemoveAll(dirName)
						if err != nil {
							predictedSentence = fmt.Sprintf("I couldn't delete the directory %s: %v", dirName, err)
						} else {
							predictedSentence = fmt.Sprintf("I have deleted the data structure '%s' (directory '%s').", structName, dirName)
							if tree, err := generateDirectoryTree(".", "", 0, 2, ""); err == nil {
								predictedSentence += "\n\n" + tree
							}

							// Remove handlers from main.go
							mainGoPath := "main.go"
							contentBytes, err := os.ReadFile(mainGoPath)
							if err != nil {
								predictedSentence += fmt.Sprintf(" However, I couldn't read %s to remove handlers: %v", mainGoPath, err)
							} else {
								lines := strings.Split(string(contentBytes), "\n")
								var newLines []string
								urlsToRemove := []string{
									fmt.Sprintf("\"/show/%s/\"", lowercaseName),
									fmt.Sprintf("\"/update/%s/\"", lowercaseName),
									fmt.Sprintf("\"/delete/%s/\"", lowercaseName),
								}

								importSuffix := fmt.Sprintf("/%s\"", lowercaseName)
								handlersRemoved := 0
								importsRemoved := 0
								for _, line := range lines {
									keep := true
									for _, url := range urlsToRemove {
										if strings.Contains(line, url) {
											keep = false
											handlersRemoved++
											break
										}
									}
									if keep {
										if strings.Contains(line, importSuffix) {
											keep = false
											importsRemoved++
										}
									}
									if keep {
										newLines = append(newLines, line)
									}
								}

								if handlersRemoved > 0 || importsRemoved > 0 {
									err = os.WriteFile(mainGoPath, []byte(strings.Join(newLines, "\n")), 0644)
									if err != nil {
										predictedSentence += fmt.Sprintf(" But I failed to update %s: %v", mainGoPath, err)
									} else {
										goImports(mainGoPath) // This will remove the unused import
										predictedSentence += fmt.Sprintf(" And removed %d handler registration(s) and %d import(s) from %s.", handlersRemoved, importsRemoved, mainGoPath)
									}
								} else {
									predictedSentence += " I didn't find any handler registrations to remove in main.go."
								}
							}
						}
					}
				}
			} else if contains(objectTypeParts, "folder") || contains(objectTypeParts, "directory") {
				folderName := findName(taggedData, kb)
				if folderName != "" {
					err := os.RemoveAll(folderName)
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't delete the folder %s: %v", folderName, err)
					} else {
						predictedSentence = fmt.Sprintf("I have deleted the folder %s.", folderName)
						if tree, err := generateDirectoryTree(".", "", 0, 2, ""); err == nil {
							predictedSentence += "\n\n" + tree
						}
					}
				} else {
					predictedSentence = "You need to provide a name for the folder."
				}
			} else if contains(objectTypeParts, "file") {
				if fileName != "" {
					err := os.Remove(fileName)
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't delete the file %s: %v", fileName, err)
					} else {
						predictedSentence = fmt.Sprintf("I have deleted the file %s.", fileName)
						if tree, err := generateDirectoryTree(".", "", 0, 2, ""); err == nil {
							predictedSentence += "\n\n" + tree
						}
					}
				} else {
					predictedSentence = "You need to provide a name for the file."
				}
			} else {
				handled = false
			}
		case "verify":
			if strings.Contains(objectType, "form") {
				targetPath := "main.go"
				if targetDirectory != "" {
					// Try to find main.go in target directory or subdirectories
					possiblePath := filepath.Join(targetDirectory, "main.go")
					if _, err := os.Stat(possiblePath); err == nil {
						targetPath = possiblePath
					} else {
						// Check cmd/ inside targetDirectory
						cmdDir := filepath.Join(targetDirectory, "cmd")
						if _, err := os.Stat(cmdDir); err == nil {
							entries, _ := os.ReadDir(cmdDir)
							for _, entry := range entries {
								if entry.IsDir() {
									path := filepath.Join(cmdDir, entry.Name(), "main.go")
									if _, err := os.Stat(path); err == nil {
										targetPath = path
										break
									}
								}
							}
						}
					}
				}

				content, err := os.ReadFile(targetPath)
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't read %s to verify the form. Please ensure you are in the correct directory or specify one with 'in <folder>'.", targetPath)
				} else {
					sContent := string(content)
					foundHandler := strings.Contains(sContent, "func FormHandler")
					foundReg := strings.Contains(sContent, "\"/form\"")

					if foundHandler && foundReg {
						predictedSentence = fmt.Sprintf("Verification Successful: Found 'FormHandler' and '/form' registration in %s.", targetPath)
					} else if foundHandler {
						predictedSentence = fmt.Sprintf("Partial Verification: Found 'FormHandler' in %s, but could not find the '/form' registration.", targetPath)
					} else {
						predictedSentence = fmt.Sprintf("Verification Failed: Could not find 'FormHandler' in %s.", targetPath)
					}
				}
			} else {
				port := "8080" // Default port
				baseURL := fmt.Sprintf("http://localhost:%s", port)

				predictedSentence = fmt.Sprintf("Verifying webserver at %s...", baseURL)

				client := http.Client{
					Timeout: 2 * time.Second,
				}

				resp, err := client.Get(baseURL)
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't connect to the webserver at %s. Is it running? Error: %v", baseURL, err)
				} else {
					defer resp.Body.Close()
					predictedSentence = fmt.Sprintf("The webserver is running (Status: %s).", resp.Status)

					// Check /form endpoint
					formURL := baseURL + "/form"
					respForm, errForm := client.Get(formURL)
					if errForm == nil {
						defer respForm.Body.Close()
						if respForm.StatusCode == http.StatusOK {
							body, errRead := io.ReadAll(respForm.Body)
							if errRead == nil && strings.Contains(string(body), "<form") {
								predictedSentence += " The /form endpoint is accessible and contains a valid HTML form."
							} else {
								predictedSentence += " The /form endpoint is accessible (200 OK)."
							}
						} else {
							predictedSentence += fmt.Sprintf(" The /form endpoint returned status: %s.", respForm.Status)
						}
					} else {
						predictedSentence += fmt.Sprintf(" However, I couldn't connect to the form endpoint at %s: %v", formURL, errForm)
					}
				}
			}
		case "cat", "read":
			targetFile := fileName
			if targetFile == "" {
				// Fallback: try to find a token with a dot
				for _, token := range taggedData.Tokens {
					if strings.Contains(token, ".") {
						targetFile = token
						break
					}
				}
			}
			if targetFile != "" {
				content, err := os.ReadFile(targetFile)
				if err != nil {
					predictedSentence = fmt.Sprintf("I couldn't read the file %s: %v", targetFile, err)
				} else {
					predictedSentence = fmt.Sprintf("Content of %s:\n%s", targetFile, string(content))
				}
			} else {
				predictedSentence = "Please specify a file to read."
			}
		default:
			if query == "pwd" || (hasDirectoryToken && command == "") {
				cwd, err := os.Getwd()
				if err != nil {
					predictedSentence = "I'm sorry, I couldn't determine the current directory."
				} else {
					predictedSentence = fmt.Sprintf("The current directory is: %s", cwd)
				}
			} else {
				handled = false
			}
		}

		if !handled {
			if hasQuestionWord && hasDirectoryToken {
				cwd, err := os.Getwd()
				if err != nil {
					predictedSentence = "I'm sorry, I couldn't determine the current directory."
				} else {
					predictedSentence = fmt.Sprintf("The current directory is: %s", cwd)
				}
			} else {
				predictedSentence = "|ʕ>ϖ<ʔ|I'm sorry, I couldn't understand your request."
				fmt.Printf("ObjectTypeParts: %v\n", objectTypeParts)
				fmt.Printf("ObjectType: %s\n", objectType)
				fmt.Printf("HasQuestionWord: %t\n", hasQuestionWord)
				fmt.Printf("HasPrepositionIn: %t\n", hasPrepositionIn)
				fmt.Printf("Command: %s\n", command)
				fmt.Printf("FileName: %s\n", fileName)
				fmt.Printf("HasDirectoryToken: %t\n", hasDirectoryToken)
				fmt.Printf("TargetDirectory: %s\n", targetDirectory) // New debug info
				fmt.Println("--------------------")
			}
		}

		fgColor := "green"
		bgColor := "black"

		if tutorialState.Active {
			if tutorialState.Step == 1 {
				if command == "create" && (strings.Contains(objectType, "folder") || strings.Contains(objectType, "directory")) {
					tutorialState.Step = 2
					predictedSentence += "\n\n[Tutorial] Great job! You created a folder. Now, let's create a file inside it.\nStep 2: Create a file. Try typing: 'create file hello.txt'"
				} else {
					predictedSentence += "\n\n[Tutorial] Hint: We are on Step 1. Try creating a folder using 'create folder <name>'."
				}
			} else if tutorialState.Step == 2 {
				if command == "create" && strings.Contains(objectType, "file") {
					tutorialState.Step = 3
					predictedSentence += "\n\n[Tutorial] Excellent! You've created a file. Now for the fun part.\nStep 3: Create a webserver. Try typing: 'create webserver myserver'"
				} else {
					predictedSentence += "\n\n[Tutorial] Hint: We are on Step 2. Try creating a file using 'create file <name>'."
				}
			} else if tutorialState.Step == 3 {
				if command == "create" && strings.Contains(objectType, "webserver") {
					tutorialState.Step = 4
					predictedSentence += "\n\n[Tutorial] Fantastic! You've created a webserver. Now, let's run it.\nStep 4: Run the webserver. Try typing: 'run webserver <name>'"
				} else {
					predictedSentence += "\n\n[Tutorial] Hint: We are on Step 3. Try creating a webserver using 'create webserver <name>'."
				}
			} else if tutorialState.Step == 4 {
				if (command == "run" || command == "start") && strings.Contains(objectType, "webserver") {
					tutorialState.Active = false
					predictedSentence += "\n\n[Tutorial] Awesome! Your webserver is running. You have completed the basic tutorial!\nYou can now explore other commands like 'create handler', 'stop webserver', or 'help'."
				} else {
					predictedSentence += "\n\n[Tutorial] Hint: We are on Step 4. Try running the webserver using 'run webserver <name>'."
				}
			}
		}

		colors.AnimatedOutput(fgColor, bgColor, predictedSentence, 1*time.Second)
		fmt.Println("\n")
	}
}

// --- KnowledgeBase and Inference Logic ---

// Intent represents the state of the command understanding.

func generateDataStructurePackageContent(structName, packageName, dirName string, fields map[string]string) string {
	lowercaseName := strings.ToLower(structName)

	// Struct Definition
	structDef := fmt.Sprintf("type %s struct {\n", structName)
	structDef += "\tID int `json:\"id\"`\n"

	sortedFieldNames := make([]string, 0, len(fields))
	for k := range fields {
		sortedFieldNames = append(sortedFieldNames, k)
	}
	sort.Strings(sortedFieldNames)

	for _, fieldName := range sortedFieldNames {
		structDef += fmt.Sprintf("\t%s %s `json:\"%s\"`\n", strings.Title(fieldName), fields[fieldName], fieldName)
	}
	structDef += "}\n\n"

	// Show Handler construction
	selectColumns := []string{"id"}
	scanFields := []string{"&u.ID"}
	for _, fieldName := range sortedFieldNames {
		selectColumns = append(selectColumns, strings.ToLower(fieldName))
		scanFields = append(scanFields, "&u."+strings.Title(fieldName))
	}

	showHandlerContent := fmt.Sprintf(`
func Show%sHandler(w http.ResponseWriter, r *http.Request) {
	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "%s", "%s.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	rows, err := db.Query("SELECT %s FROM %s")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	results := make([]%s, 0)
	for rows.Next() {
		var u %s
		if err := rows.Scan(%s); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		results = append(results, u)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}`, structName, dirName, lowercaseName, strings.Join(selectColumns, ", "), lowercaseName, structName, structName, strings.Join(scanFields, ", "))

	var structFields []string
	var structFieldExecs []string
	for _, fieldName := range sortedFieldNames {
		structFields = append(structFields, fmt.Sprintf("%s = ?", strings.ToLower(fieldName)))
		structFieldExecs = append(structFieldExecs, "u."+strings.Title(fieldName))
	}

	updateHandlerContent := fmt.Sprintf(`
func Update%sHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 { // e.g. /update/user/123
		http.Error(w, "Invalid URL, expecting /update/%s/{id}", http.StatusBadRequest)
		return
	}
	id := parts[len(parts)-1]

	var u %s
	err := json.NewDecoder(r.Body).Decode(&u)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "%s", "%s.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	stmt, err := db.Prepare("UPDATE %s SET %s WHERE id = ?")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	_, err = stmt.Exec(%s, id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintf(w, "%s with ID %%s updated successfully", id)
}`, structName, lowercaseName, structName, dirName, lowercaseName, lowercaseName, strings.Join(structFields, ", "), strings.Join(structFieldExecs, ", "), structName)

	deleteHandlerContent := fmt.Sprintf(`
func Delete%sHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 { // e.g. /delete/user/123
		http.Error(w, "Invalid URL, expecting /delete/%s/{id}", http.StatusBadRequest)
		return
	}
	id := parts[len(parts)-1]

	cwd, _ := os.Getwd()
	dbPath := filepath.Join(cwd, "%s", "%s.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	stmt, err := db.Prepare("DELETE FROM %s WHERE id = ?")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	_, err = stmt.Exec(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintf(w, "%s with ID %%s deleted successfully", id)
}`, structName, lowercaseName, dirName, lowercaseName, lowercaseName, structName)

	packageFileContent := fmt.Sprintf(`package %s
import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	_ "modernc.org/sqlite"
)

%s
%s
%s
%s
`, packageName, structDef, showHandlerContent, updateHandlerContent, deleteHandlerContent)

	return packageFileContent
}

// --- KnowledgeBase and Inference Logic ---

// Intent represents the state of the command understanding.
type Intent struct {
	RawInput        string
	Command         string
	ObjectType      string
	ObjectTypeParts []string
	Params          map[string]string
}

type ModelConfig struct {
	Word2VecPath      string `json:"word2vec_path"`
	MoEPath           string `json:"moe_path"`
	QueryVocabPath    string `json:"query_vocab_path"`
	SemanticVocabPath string `json:"semantic_vocab_path"`
	NERPath           string `json:"ner_path"`
}

// KnowledgeBase acts as the memory for the session.
type KnowledgeBase struct {
	KnownCommands map[string]bool `json:"known_commands"`
	KnownObjects  map[string]bool `json:"known_objects"`
	StopWords     map[string]bool `json:"stop_words"`
	LearningPath  string          `json:"learning_path"`
	FirstRun      bool            `json:"first_run"`
	ModelConfig   ModelConfig     `json:"model_config"`
}

func NewKnowledgeBase() *KnowledgeBase {
	return &KnowledgeBase{
		KnownCommands: map[string]bool{
			"create": true, "make": true, "generate": true, "add": true, "put": true, "copy": true,
			"delete": true, "remove": true,
			"list": true, "ls": true, "show": true,
			"go": true, "cd": true, "change": true, "move": true,
			"run": true, "start": true,
			"stop":   true,
			"update": true,
			"verify": true, "check": true, "test": true,
			"cat": true, "read": true,
			"tree": true,
			"grep": true, "search": true,
			"history": true,
			"help":    true,
		},
		KnownObjects: map[string]bool{
			"user": true, "file": true, "database": true, "folder": true, "directory": true,
			"webserver": true, "handler": true, "structure": true, "form": true,
		},
		StopWords: map[string]bool{
			"a": true, "an": true, "the": true, "please": true, "this": true,
			"me": true, "my": true, "i": true, "new": true, "to": true, "for": true, "and": true, "it": true,
		},
		FirstRun: true,
		ModelConfig: ModelConfig{
			Word2VecPath:      "gob_models/word2vec.model",
			MoEPath:           "gob_models/moe_classification_model.gob",
			QueryVocabPath:    "gob_models/query_vocabulary.gob",
			SemanticVocabPath: "gob_models/semantic_output_vocabulary.gob",
			NERPath:           "gob_models/ner_model.gob",
		},
	}
}

func LoadKnowledgeBase() *KnowledgeBase {
	data, err := os.ReadFile(kbFilename)
	if os.IsNotExist(err) {
		return NewKnowledgeBase()
	}
	var kb KnowledgeBase
	if err := json.Unmarshal(data, &kb); err != nil {
		return NewKnowledgeBase()
	}

	// Ensure built-in commands and stop words are always present
	defaults := NewKnowledgeBase()
	if kb.KnownCommands == nil {
		kb.KnownCommands = make(map[string]bool)
	}
	for k := range defaults.KnownCommands {
		kb.KnownCommands[k] = true
	}
	if kb.StopWords == nil {
		kb.StopWords = make(map[string]bool)
	}
	for k := range defaults.StopWords {
		kb.StopWords[k] = true
	}

	if kb.ModelConfig.Word2VecPath == "" {
		kb.ModelConfig = defaults.ModelConfig
	}

	return &kb
}

func (kb *KnowledgeBase) Save() {
	data, _ := json.MarshalIndent(kb, "", "  ")
	_ = os.WriteFile(kbFilename, data, 0644)
}

// parse identifies commands, known objects, and parameters.
func parse(input string, kb *KnowledgeBase) Intent {
	parts := strings.Fields(input)
	intent := Intent{
		RawInput:        input,
		ObjectTypeParts: []string{},
		Params:          make(map[string]string),
	}

	consumed := make(map[int]bool)

	// 1. Extract Parameters first (e.g., "named login")
	for i := 0; i < len(parts); i++ {
		word := strings.ToLower(parts[i])
		if paramKey, isTrigger := paramTriggers[word]; isTrigger {
			if i+1 < len(parts) {
				value := parts[i+1]
				nextIndex := i + 1

				// Skip noise words like "the", "a", "an", "folder", "directory"
				for nextIndex < len(parts) {
					v := strings.ToLower(parts[nextIndex])
					if v == "the" || v == "a" || v == "an" || v == "folder" || v == "directory" {
						consumed[nextIndex] = true
						nextIndex++
						if nextIndex < len(parts) {
							value = parts[nextIndex]
						}
						continue
					}
					break
				}

				if strings.ToLower(value) == "it" {
					continue
				}
				intent.Params[paramKey] = value
				consumed[i] = true
				consumed[nextIndex] = true
				i = nextIndex
			}
		}
	}

	// 2. Extract Command and ObjectType from remaining words
	for i, word := range parts {
		if consumed[i] {
			continue
		}
		lower := strings.ToLower(word)

		if intent.Command == "" && kb.KnownCommands[lower] {
			if lower == "make" || lower == "generate" || lower == "add" || lower == "put" || lower == "copy" {
				lower = "create"
			} else if lower == "ls" || lower == "show" {
				lower = "list"
			} else if lower == "cd" || lower == "change" {
				lower = "go"
			}
			intent.Command = lower
			continue
		}

		if intent.ObjectType == "" && kb.KnownObjects[lower] {
			intent.ObjectType = lower
			intent.ObjectTypeParts = append(intent.ObjectTypeParts, lower)
			continue
		}
	}
	return intent
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, magA, magB float64
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		magA += a[i] * a[i]
		magB += b[i] * b[i]
	}
	if magA == 0 || magB == 0 {
		return 0
	}
	return dot / (math.Sqrt(magA) * math.Sqrt(magB))
}

func generateWordVizHTML(words []string, vectors [][]float64) string {
	wordsJSON, _ := json.Marshal(words)
	vectorsJSON, _ := json.Marshal(vectors)

	return fmt.Sprintf(`<!DOCTYPE html>
<html>
<head>
    <title>Word2Vec 2D Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/pca-js@1.0.0/pca.min.js"></script>
    <style>body { margin: 0; font-family: sans-serif; }</style>
</head>
<body>
    <div id="myDiv" style="width: 100%%; height: 100vh;"></div>
    <script>
        var words = %s;
        var rawVectors = %s;

        var x = [];
        var y = [];

        if (rawVectors.length > 0 && rawVectors[0].length > 2) {
            var vectors = PCA.getEigenVectors(rawVectors);
            var adData = PCA.computeAdjustedData(rawVectors, vectors[0], vectors[1]);
            x = adData.formattedAdjustedData[0];
            y = adData.formattedAdjustedData[1];
        } else {
             for (var i = 0; i < rawVectors.length; i++) {
                 x.push(rawVectors[i][0]);
                 y.push(rawVectors[i][1]);
             }
        }

        var trace = {
            x: x, y: y, mode: 'markers+text', type: 'scatter',
            text: words, textposition: 'top center', marker: { size: 8 }
        };
        var layout = { title: 'Word Vector Distribution (PCA Reduced)', hovermode: 'closest' };
        Plotly.newPlot('myDiv', [trace], layout);
    </script>
</body>
</html>`, string(wordsJSON), string(vectorsJSON))
}

func inspectStruct(v interface{}, indent string) {
	val := reflect.ValueOf(v)
	if !val.IsValid() {
		fmt.Println(indent + "<nil>")
		return
	}
	if val.Kind() == reflect.Ptr || val.Kind() == reflect.Interface {
		if val.IsNil() {
			fmt.Println(indent + "<nil>")
			return
		}
		val = val.Elem()
	}
	if val.Kind() != reflect.Struct {
		fmt.Printf("%s%v\n", indent, val)
		return
	}

	typ := val.Type()
	for i := 0; i < val.NumField(); i++ {
		field := val.Field(i)
		fieldType := typ.Field(i)

		if fieldType.PkgPath != "" {
			continue // Skip unexported fields
		}

		fmt.Printf("%s%s (%s): ", indent, fieldType.Name, fieldType.Type)

		if field.Kind() == reflect.Slice {
			fmt.Printf("Slice with %d elements\n", field.Len())
			if field.Len() > 0 && field.Type().Elem().Kind() == reflect.Float64 {
				count := 5
				if field.Len() < 5 {
					count = field.Len()
				}
				fmt.Printf("%s  Sample: %v...\n", indent, field.Slice(0, count).Interface())
			}
		} else if field.Kind() == reflect.Struct || (field.Kind() == reflect.Ptr && field.Elem().Kind() == reflect.Struct) {
			fmt.Println("")
			if len(indent) < 10 {
				inspectStruct(field.Interface(), indent+"  ")
			} else {
				fmt.Println(indent + "  ...")
			}
		} else {
			fmt.Printf("%v\n", field)
		}
	}
}

func findAndVisualizeAttention(v interface{}) {
	val := reflect.ValueOf(v)
	if !val.IsValid() {
		return
	}
	if val.Kind() == reflect.Ptr || val.Kind() == reflect.Interface {
		if val.IsNil() {
			return
		}
		val = val.Elem()
	}

	if val.Type() == reflect.TypeOf(neuralnn.MultiHeadAttention{}) {
		mha := val.Interface().(neuralnn.MultiHeadAttention)
		fmt.Printf("\nFound MultiHeadAttention Layer:\n")
		fmt.Printf("  Heads: %d, Model Dim: %d\n", mha.NumHeads, mha.DimModel)
		if mha.Wq != nil {
			fmt.Printf("  Query Weights: %v\n", mha.Wq.Shape)
		}
		f := val.FieldByName("attentionWeights")
		if f.IsValid() && !f.IsNil() {
			fmt.Println("  Last Attention Weights: [Present in memory]")
		} else {
			fmt.Println("  Last Attention Weights: [Not present]")
		}
		return
	}

	if val.Type() == reflect.TypeOf(neuralnn.MultiHeadCrossAttention{}) {
		mhca := val.Interface().(neuralnn.MultiHeadCrossAttention)
		fmt.Printf("\nFound MultiHeadCrossAttention Layer:\n")
		fmt.Printf("  Q Heads: %d, KV Heads: %d, Model Dim: %d\n", mhca.NumQHeads, mhca.NumKVHeads, mhca.DimModel)
		f := val.FieldByName("attentionWeights")
		if f.IsValid() && !f.IsNil() {
			fmt.Println("  Last Attention Weights: [Present in memory]")
		} else {
			fmt.Println("  Last Attention Weights: [Not present]")
		}
		return
	}

	if val.Kind() == reflect.Struct {
		for i := 0; i < val.NumField(); i++ {
			field := val.Field(i)
			// Only traverse exported fields to avoid panic
			if field.CanInterface() {
				findAndVisualizeAttention(field.Interface())
			}
		}
	} else if val.Kind() == reflect.Slice {
		for i := 0; i < val.Len(); i++ {
			findAndVisualizeAttention(val.Index(i).Interface())
		}
	}
}

// resolveIntent attempts to find the missing object type in the remaining words.
func resolveIntent(r *bufio.Reader, intent Intent, kb *KnowledgeBase) Intent {
	parts := strings.Fields(intent.RawInput)
	var candidate string

	consumed := make(map[int]bool)
	for i := 0; i < len(parts); i++ {
		if _, isTrigger := paramTriggers[strings.ToLower(parts[i])]; isTrigger {
			consumed[i] = true
			if i+1 < len(parts) {
				consumed[i+1] = true
			}
		}
	}

	for i, word := range parts {
		if consumed[i] {
			continue
		}
		lower := strings.ToLower(word)
		if lower == intent.Command || kb.KnownCommands[lower] {
			continue
		}
		if kb.StopWords[lower] || lower == "named" || lower == "called" {
			continue
		}
		candidate = lower
		break
	}

	if candidate != "" {
		fmt.Println("   ... Attempting recursive inference ...")
		fmt.Printf("   [INFERENCE] I detected the unknown token '%s'.\n", candidate)
		fmt.Printf("   [CONFIRMATION] Did you mean to create a '%s'? (y/n): ", candidate)
		resp, _ := r.ReadString('\n')
		resp = strings.TrimSpace(strings.ToLower(resp))
		if resp == "y" || resp == "yes" {
			intent.ObjectType = candidate
			intent.ObjectTypeParts = append(intent.ObjectTypeParts, candidate)
			kb.KnownObjects[candidate] = true
			fmt.Printf("   [LEARNING] Knowledge updated: '%s' is now a known object type.\n", candidate)
			kb.Save()
			
			// If we haven't identified a command yet, assume "create"
			if intent.Command == "" {
				intent.Command = "create"
			}
			return intent
		}
	}
	return intent
}

func handleGenericCreate(objectType, fileName, targetDirectory, handlerURL string, kb *KnowledgeBase) (string, bool) {

	// 1. If it looks like a handler (has a URL)
	if handlerURL != "" {
		handlerName := fileName
		if handlerName == "" {
			handlerName = objectType
		}

		handlerContent := fmt.Sprintf("package main\n\nimport \"net/http\"\n\n// %s is a generic handler for %s\nfunc %s(w http.ResponseWriter, r *http.Request) {\n\tw.Write([]byte(\"Generic implementation for %s\"))\n}\n", strings.Title(handlerName), objectType, strings.Title(handlerName), objectType)
		filePath := handlerName + ".go"
		if targetDirectory != "" {
			filePath = filepath.Join(targetDirectory, filePath)
			os.MkdirAll(targetDirectory, 0755)
		}
		os.WriteFile(filePath, []byte(handlerContent), 0644)
		goImports(filePath)

		mainPath := "main.go"
		if targetDirectory != "" {
			if _, err := os.Stat(filepath.Join(targetDirectory, "main.go")); err == nil {
				mainPath = filepath.Join(targetDirectory, "main.go")
			}
		}
		regMsg, _ := registerHandlerURL(strings.Title(handlerName), handlerURL, mainPath)
		return fmt.Sprintf("I don't have a specialized handler for '%s', but since you provided a URL, I've generated a backend handler for it. %s", objectType, regMsg), true
	}

	// 2. If a name is provided, create a Go skeleton
	if fileName != "" {
		content := fmt.Sprintf("// %s implementation for the %s object\npackage main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"Executing %s (%s logic)\")\n}\n", strings.Title(fileName), objectType, fileName, objectType)
		filePath := fileName + ".go"
		if targetDirectory != "" {
			filePath = filepath.Join(targetDirectory, filePath)
			os.MkdirAll(targetDirectory, 0755)
		}
		os.WriteFile(filePath, []byte(content), 0644)
		goImports(filePath)
		return fmt.Sprintf("I identified '%s' as a new object type. I've created a basic Go skeleton '%s.go' for you.", objectType, fileName), true
	}

	// 3. Just a folder
	folderPath := objectType
	if targetDirectory != "" {
		folderPath = filepath.Join(targetDirectory, folderPath)
	}
	os.MkdirAll(folderPath, 0755)
	return fmt.Sprintf("I don't know how to implement '%s' yet, so I've created a work directory for it in /%s.", objectType, folderPath), true
}
