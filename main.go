package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time" // Added time import

	_ "modernc.org/sqlite" // Pure Go SQLite driver

	"github.com/golangast/gollemer/colors"
	"github.com/golangast/gollemer/internal/sqlite_db"
	"github.com/golangast/gollemer/tagger/nertagger"
	"github.com/golangast/gollemer/tagger/postagger"
	"github.com/golangast/gollemer/tagger/tag"
)

const kbFilename = "knowledge.json"

// paramTriggers maps words like "named" to the key "name".
var paramTriggers = map[string]string{
	"named":  "name",
	"called": "name",
	"for":    "target",
	"with":   "attribute",
	"in":     "target",
	"into":   "target",
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
	} else {
		log.Println("No action specified. Use -train-word2vec, -train-moe, -train-intent-classifier, or -llm.")
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

func findName(taggedData tag.Tag) string {

	// First, look for a FILENAME tag

	for i, tag := range taggedData.NerTag {

		if tag == "FILENAME" {

			return taggedData.Tokens[i]

		}

	}

	// Fallback for "named"

	for i, token := range taggedData.Tokens {

		if token == "named" && i+1 < len(taggedData.Tokens) {

			return taggedData.Tokens[i+1]

		}

	}

	// Fallback for NAME tag

	for i, tag := range taggedData.NerTag {

		if tag == "NAME" {

			return taggedData.Tokens[i]

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

	registration := fmt.Sprintf("http.HandleFunc(\"%s\", %sHandler)", handlerURL, handlerName)
	if !strings.Contains(string(mainGoContent), registration) {
		newHandleFunc := fmt.Sprintf("\thttp.HandleFunc(\"%s\", %sHandler)\n\t// HANDLER_REGISTRATIONS_GO_HERE", handlerURL, handlerName)
		updatedMainGoContent := strings.Replace(string(mainGoContent), "// HANDLER_REGISTRATIONS_GO_HERE", newHandleFunc, 1) // Expect unindented placeholder

		if updatedMainGoContent == string(mainGoContent) {
			return "", fmt.Errorf("placeholder '\\t// HANDLER_REGISTRATIONS_GO_HERE' not found in %s", mainGoPath)
		}

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
		log.Printf("goimports failed for %s: %v. Please ensure goimports is installed and in your PATH. You can install it by running: 'go install golang.org/x/tools/cmd/goimports@latest'", filename, err)
	}
}

// GollemerMoEClient implements the MoEClient interface using the existing NLP pipeline.
type GollemerMoEClient struct{}

func (c *GollemerMoEClient) PredictIntent(input string) (string, float64) {
	lowerInput := strings.ToLower(input)

	// Heuristic Intent Detection based on existing keywords
	if strings.Contains(lowerInput, "create") {
		if strings.Contains(lowerInput, "webserver") {
			return "create_webserver", 0.9
		}
		if strings.Contains(lowerInput, "handler") {
			return "create_handler", 0.9
		}
		if strings.Contains(lowerInput, "database") {
			return "create_database", 0.9
		}
		if strings.Contains(lowerInput, "file") {
			return "create_file", 0.8
		}
		if strings.Contains(lowerInput, "folder") || strings.Contains(lowerInput, "directory") {
			return "create_folder", 0.8
		}
	}
	return "", 0.0
}

func (c *GollemerMoEClient) ExtractEntities(input string, intent string) map[string]interface{} {
	words := strings.Fields(input)
	posTags := postagger.TagTokens(words)
	taggedData := nertagger.Nertagger(tag.Tag{Tokens: words, PosTag: posTags})

	entities := make(map[string]interface{})

	// Extract Name using existing findName logic
	name := findName(taggedData)
	if name != "" {
		entities["name"] = name
	}

	// Extract URL (specific to handlers)
	for i, token := range taggedData.Tokens {
		if strings.ToLower(token) == "url" && i > 0 && strings.ToLower(taggedData.Tokens[i-1]) == "with" && i+1 < len(taggedData.Tokens) {
			entities["url"] = taggedData.Tokens[i+1]
		}
	}

	// Extract Path (for files/folders)
	for i, token := range taggedData.Tokens {
		if (strings.ToLower(token) == "in" || strings.ToLower(token) == "into") && i+1 < len(taggedData.Tokens) {
			entities["path"] = taggedData.Tokens[i+1]
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

	reader := bufio.NewReader(os.Stdin)

	// Load last directory on startup
	lastDir, err := loadLastDirectory()
	if err == nil {
		err := os.Chdir(lastDir)
		if err != nil {
			log.Printf("DEBUG: Error changing to last directory %s: %v", lastDir, err)
		} else {
			currentAbsDir, _ := os.Getwd()
			log.Printf("DEBUG: Changed to last directory: %s", currentAbsDir)
		}
	} else {
		currentAbsDir, _ := os.Getwd()
		log.Printf("DEBUG: No last directory loaded. Current Working Directory: %s", currentAbsDir)
	}

	// Initialize Hybrid Intent Resolver with our local MoE client
	resolver := NewHybridIntentResolver(&GollemerMoEClient{})

	for {
		colors.ColorizeCol("red", "magenta", "/ʕ◔ϖ◔ʔ/> ")

		query, _ := reader.ReadString('\n')
		query = strings.TrimSpace(query)

		if query == "exit" {
			break
		} else if query == "clear" {
			cmd := exec.Command("clear")
			cmd.Stdout = os.Stdout
			cmd.Run()
			continue
		} else if query == "show learning path" {
			if kb.LearningPath != "" {
				fmt.Printf("Current learning path: %s\n", kb.LearningPath)
			} else {
				fmt.Println("No learning path set. Defaulting to 'learningfolder' in project root.")
			}
			continue
		} else if query == "help" {
			fmt.Println("--- Knowledge Base ---")
			fmt.Println("Known Commands:")
			var cmds []string
			for k := range kb.KnownCommands {
				cmds = append(cmds, k)
			}
			sort.Strings(cmds)
			fmt.Println(strings.Join(cmds, ", "))

			fmt.Println("\nKnown Objects:")
			var objs []string
			for k := range kb.KnownObjects {
				objs = append(objs, k)
			}
			sort.Strings(objs)
			fmt.Println(strings.Join(objs, ", "))
			fmt.Println("----------------------")
			continue
		} else if strings.HasPrefix(query, "learn ") {
			parts := strings.Fields(query)
			if len(parts) >= 3 && parts[1] == "object" {
				newObject := strings.ToLower(parts[2])
				kb.KnownObjects[newObject] = true
				kb.Save()
				fmt.Printf("Knowledge Base updated: '%s' is now a known object.\n", newObject)
			} else if len(parts) >= 3 && parts[1] == "from" {
				targetFolder := parts[2]
				entries, err := os.ReadDir(targetFolder)
				if err != nil {
					fmt.Printf("Error reading directory '%s': %v\n", targetFolder, err)
				} else {
					absPath, err := filepath.Abs(targetFolder)
					if err == nil {
						kb.LearningPath = absPath
					}
					count := 0
					for _, entry := range entries {
						if !entry.IsDir() {
							name := entry.Name()
							ext := filepath.Ext(name)
							baseName := strings.TrimSuffix(name, ext)
							baseName = strings.ToLower(baseName)
							if baseName != "" && !kb.KnownObjects[baseName] {
								kb.KnownObjects[baseName] = true
								count++
							}
						}
					}
					kb.Save()
					fmt.Printf("Learned %d new objects from folder '%s'.\n", count, targetFolder)
				}
			} else {
				fmt.Println("Usage: learn object <word> OR learn from <folder>")
			}
			continue
		}

		// --- New Intent Layer Logic ---
		// This recursively fills the data layer using the MoE client
		intentData := resolver.Resolve(query, nil)
		if intentData.Intent != "" {
			jsonOutput, _ := json.MarshalIndent(intentData, "", "  ")
			fmt.Println(string(jsonOutput))
		}

		// --- Tagging ---
		words := strings.Fields(query)
		posTags := postagger.TagTokens(words)
		taggedData := nertagger.Nertagger(tag.Tag{Tokens: words, PosTag: posTags})

		// --- Start of new logic ---
		
		// 1. Parse with KnowledgeBase
		intent := parse(query, kb)

		// 2. Inference if needed (Command understood, but ObjectType missing)
		if intent.Command != "" && intent.ObjectType == "" {
			intent = resolveIntent(reader, intent, kb)
		}

		hasQuestionWord := false
		var objectTypeParts []string = intent.ObjectTypeParts
		hasPrepositionIn := false
		var command string = intent.Command
		var targetDirectory string
		var predictedSentence string
		var handlerURL string // New variable to store the handler URL
		
		// Initialize variables from Intent Params if available
		if val, ok := intent.Params["target"]; ok {
			targetDirectory = val
		}
		
			// Try to explicitly identify the command if it's the first token
			// Only if intent.Command wasn't already found by the KB parser
			if command == "" && len(taggedData.Tokens) > 0 {
				token := strings.ToLower(taggedData.Tokens[0])
				if token == "create" || token == "add" || token == "put" {
					command = "create"
				} else if token == "list" || token == "ls" || token == "show" {
					command = "list"
				} else if token == "go" || token == "cd" || token == "change" || token == "move" {
					command = "go"
				} else if token == "delete" || token == "remove" {
					command = "delete"
				} else if token == "run" || token == "start" {
					command = "run"
				} else if token == "stop" {
					command = "stop"
				} else if token == "update" {
					command = "update"
				} else if token == "verify" || token == "check" || token == "test" {
					command = "verify"
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
						if token == "in" || token == "into" {
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
									if taggedData.NerTag[j] == "NAME" || strings.Contains(taggedData.Tokens[j], "/") || strings.Contains(taggedData.Tokens[j], "\\") {
										targetDirectory = taggedData.Tokens[j]
										break
									}
								}
								if targetDirectory == "" && i+1 < len(taggedData.Tokens) {
									candidate := taggedData.Tokens[i+1]
									if candidate != "the" && candidate != "a" && candidate != "an" {
										targetDirectory = candidate
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

			var objectType string = intent.ObjectType
			
			// Fallback logic for specific complex types or if KB missed it
			if strings.Contains(strings.ToLower(query), "handler") {
				objectType = "handler"
			} else if strings.Contains(strings.ToLower(query), "data structure") {
				objectType = "data structure"
				objectTypeParts = []string{} // Clear objectTypeParts to prevent interference
			} else if strings.Contains(strings.ToLower(query), "webserver") || strings.Contains(strings.ToLower(query), "websever") {
				objectType = "webserver"
			} else if objectType == "" {
				objectType = strings.Join(objectTypeParts, " ")
			}

			fileName := intent.Params["name"]
			if fileName == "" {
				fileName = findName(taggedData)
			}

			// Heuristic: If fileName is still empty, and objectType is "file",
			// check for tokens that look like filenames (e.g., ends with .go)
			if fileName == "" && contains(objectTypeParts, "file") {
				for _, token := range taggedData.Tokens {
					if strings.HasSuffix(token, ".go") || strings.HasSuffix(token, ".txt") || strings.HasSuffix(token, ".md") {
						fileName = token
						break
					}
				}
			}

			if hasQuestionWord && (contains(objectTypeParts, "folder") || contains(objectTypeParts, "folders") || contains(objectTypeParts, "file") || contains(objectTypeParts, "files")) {
				command = "list"
			}

			hasDirectoryToken := false
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
			case "create":
				log.Printf("DEBUG: Entering create command. ObjectType: '%s', TargetDirectory: '%s'", objectType, targetDirectory)
				if strings.Contains(objectType, "handler") {
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
							predictedSentence = fmt.Sprintf("The handler file '%s' already exists.", filePath)
						} else {
							err = os.WriteFile(filePath, []byte(handlerContent), 0644)
							if err != nil {
								predictedSentence = fmt.Sprintf("I couldn't write to the handler file %s: %v", filePath, err)
								goto endOfCreateHandler
							}
							goImports(filePath)
							predictedSentence = fmt.Sprintf("I have created the handler '%s' in %s.", handlerName, filePath)
							// Always attempt to register the handler in the current project's main.go
							currentProjectMainGo := filepath.Join(".", "main.go")
							registrationMsg, err := registerHandlerURL(strings.Title(handlerName), handlerURL, currentProjectMainGo)
							if err != nil {
								log.Printf("Error registering handler URL in %s: %v", currentProjectMainGo, err)
								predictedSentence += fmt.Sprintf(" I tried to register the handler in %s but failed: %v", currentProjectMainGo, err)
							} else {
								predictedSentence += " " + registrationMsg
							}
						}
					}
				endOfCreateHandler:
				} else if strings.Contains(objectType, "file") { // New block for generic file creation
					if fileName != "" {
						filePath := fileName
						if targetDirectory != "" {
							filePath = filepath.Join(targetDirectory, fileName)
						}
						err := os.WriteFile(filePath, []byte(""), 0644)
						if err != nil {
							predictedSentence = fmt.Sprintf("I couldn't create the file %s: %v", filePath, err)
						} else {
							predictedSentence = fmt.Sprintf("I have created the file %s.", filePath)
						}
					} else {
						predictedSentence = "You need to provide a name for the file."
					}
				} else if strings.Contains(objectType, "webserver") {
					if fileName != "" {
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
				} else if strings.Contains(objectType, "folder") { // New block for folder creation
					folderName := findName(taggedData)
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
						}
					} else {
						predictedSentence = "You need to provide a name for the folder."
					}
				} else if objectType == "database" { // New block for database creation
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
					dirName = strings.ToLower(structName)
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
					packageFileContent = generateDataStructurePackageContent(structName, fields)
					lowercaseName = strings.ToLower(structName)
					packageName = lowercaseName

					// Write the package file
					structFileName = filepath.Join(dirName, lowercaseName+".go")
					err = os.WriteFile(structFileName, []byte(packageFileContent), 0644)
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't create the Go package file %s: %v", structFileName, err)
						goto endOfDataStructureCreation
					}
					goImports(structFileName)
					predictedSentence = fmt.Sprintf("I have created the Go package '%s' in %s.", packageName, structFileName)

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

					// The new package is in a subdirectory named lowercaseName
					packageImportPath = filepath.Join(modulePath, relativeDir, lowercaseName)
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
					;
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
			}
			case "run", "start":
				if strings.Contains(objectType, "webserver") {
					webserverName := ""
					for i, token := range taggedData.Tokens {
						if (strings.ToLower(token) == "webserver" || strings.ToLower(token) == "websever") && i+1 < len(taggedData.Tokens) {
							webserverName = taggedData.Tokens[i+1]
							break
						}
					}
					if webserverName == "" {
						webserverName = fileName
					}

					if webserverName == "" {
						predictedSentence = "You need to provide a name for the webserver to run."
					} else {
						// Path to the jim webserver's main package
						jimSourcePath := filepath.Join(projectRoot, "cmd", webserverName)

						// If not found in cmd/, check root or nested structure
						if _, err := os.Stat(jimSourcePath); os.IsNotExist(err) {
							altPath := filepath.Join(projectRoot, webserverName)
							if _, err := os.Stat(altPath); err == nil {
								// Check nested: project/jill/cmd/jill
								nestedPath := filepath.Join(altPath, "cmd", webserverName)
								if _, err := os.Stat(nestedPath); err == nil {
									jimSourcePath = nestedPath
								} else {
									nestedCmd := filepath.Join(altPath, "cmd")
									if _, err := os.Stat(filepath.Join(nestedCmd, "main.go")); err == nil {
										jimSourcePath = nestedCmd
									} else if _, err := os.Stat(filepath.Join(altPath, "main.go")); err == nil {
										jimSourcePath = altPath
									}
								}
							}

							// If still not found, check current working directory
							if _, err := os.Stat(jimSourcePath); os.IsNotExist(err) {
								cwd, _ := os.Getwd()
								// Check if we are currently IN the webserver directory
								if strings.EqualFold(filepath.Base(cwd), webserverName) {
									if _, err := os.Stat(filepath.Join(cwd, "main.go")); err == nil {
										jimSourcePath = cwd
									}
								}
								if jimSourcePath != cwd {
									localPath := filepath.Join(cwd, webserverName)
									if _, err := os.Stat(filepath.Join(localPath, "main.go")); err == nil {
										jimSourcePath = localPath
									} else if _, err := os.Stat(filepath.Join(cwd, "cmd", webserverName)); err == nil {
										jimSourcePath = filepath.Join(cwd, "cmd", webserverName)
									}
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

							// Run the built executable
							runCmd := exec.Command(jimExecutablePath, "-llm") // No "run webserver jim" arguments needed now
							runCmd.Dir = projectRoot  // Running from project root
							runCmd.Stdout = os.Stdout // Redirect stdout
							runCmd.Stderr = os.Stderr // Redirect stderr

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
								time.Sleep(2 * time.Second) // Give the server a moment to start

								resp, err := http.Get("http://localhost:8080/")
								if err != nil {
									log.Printf("WARNING: Webserver verification failed: %v", err)
									predictedSentence += " However, I could not verify that the webserver is running."
								} else {
									defer resp.Body.Close()
									if resp.StatusCode == http.StatusOK {
										predictedSentence += " And I have verified that the webserver is running."

										// Check /form endpoint
										respForm, errForm := http.Get("http://localhost:8080/form")
										if errForm == nil {
											if respForm.StatusCode == http.StatusOK {
												predictedSentence += " The /form endpoint is also accessible."
											}
											respForm.Body.Close()
										}
									} else {
										log.Printf("WARNING: Webserver returned status code %d during verification.", resp.StatusCode)
										predictedSentence += fmt.Sprintf(" However, the webserver returned status code %d during verification.", resp.StatusCode)
									}
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
										newContent := generateDataStructurePackageContent(structName, fields)
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
					folderName := findName(taggedData)
					if folderName != "" {
						err := os.RemoveAll(folderName)
						if err != nil {
							predictedSentence = fmt.Sprintf("I couldn't delete the folder %s: %v", folderName, err)
						} else {
							predictedSentence = fmt.Sprintf("I have deleted the folder %s.", folderName)
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

			colors.AnimatedOutput("blue", "red", predictedSentence, 1*time.Second)
			fmt.Println("\n")
		}
	}


	// --- KnowledgeBase and Inference Logic ---

	// Intent represents the state of the command understanding.
	
func generateDataStructurePackageContent(structName string, fields map[string]string) string {
	lowercaseName := strings.ToLower(structName)
	packageName := lowercaseName
	dirName := lowercaseName

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
}`, structName, lowercaseName, structName, dirName, lowercaseName, lowercaseName, strings.Join(structFields, ", "), strings.Join(structFieldExecs, ", "))

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
}`, structName, lowercaseName, dirName, lowercaseName, lowercaseName)

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

// KnowledgeBase acts as the memory for the session.
type KnowledgeBase struct {
	KnownCommands map[string]bool `json:"known_commands"`
	KnownObjects  map[string]bool `json:"known_objects"`
	StopWords     map[string]bool `json:"stop_words"`
	LearningPath  string          `json:"learning_path"`
}

func NewKnowledgeBase() *KnowledgeBase {
	return &KnowledgeBase{
		KnownCommands: map[string]bool{
			"create": true, "make": true, "generate": true, "add": true, "put": true,
			"delete": true, "remove": true,
			"list": true, "ls": true, "show": true,
			"go": true, "cd": true, "change": true, "move": true,
			"run": true, "start": true,
			"stop": true,
			"update": true,
			"verify": true, "check": true, "test": true,
			"cat": true, "read": true,
		},
		KnownObjects: map[string]bool{
			"user": true, "file": true, "database": true, "folder": true, "directory": true,
			"webserver": true, "handler": true, "structure": true, "form": true,
		},
		StopWords: map[string]bool{
			"a": true, "an": true, "the": true, "please": true, "this": true,
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
				intent.Params[paramKey] = value
				consumed[i] = true
				consumed[i+1] = true
				i++
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
			if lower == "make" || lower == "generate" || lower == "add" || lower == "put" {
				lower = "create"
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

// resolveIntent attempts to find the missing object type in the remaining words.
func resolveIntent(r *bufio.Reader, intent Intent, kb *KnowledgeBase) Intent {
	fmt.Println("   ... Attempting recursive inference ...")

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
		if kb.StopWords[lower] {
			continue
		}
		candidate = lower
		break
	}

	if candidate != "" {
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
			return intent
		}
	}
	// If inference fails, we return the intent as-is and let the legacy logic or error handler deal with it.
	return intent
}