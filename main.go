package main

import (
	"bufio"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time" // Added time import

	_ "modernc.org/sqlite" // Pure Go SQLite driver

	"github.com/golangast/gollemer/colors"
	"github.com/golangast/gollemer/internal/sqlite_db"
	"github.com/golangast/gollemer/tagger/nertagger"
	"github.com/golangast/gollemer/tagger/postagger"
	"github.com/golangast/gollemer/tagger/tag"
)

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
	serveFlag := flag.Bool("serve", false, "Run in web server mode")

	flag.Parse()

	if *runLLMFlag {
		runLLM()
	} else if *serveFlag {
		startWebServer()
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

func startWebServer() {
	// HANDLER_REGISTRATIONS_GO_HERE
	log.Println("Starting web server on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
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
	columns := []string{}
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

func registerHandlerURL(handlerName, handlerURL string) (string, error) {
	log.Printf("Attempting to register handler '%s' with URL '%s'", handlerName, handlerURL)

	mainGoContent, err := os.ReadFile("main.go")
	if err != nil {
		log.Printf("Error reading main.go: %v", err)
		return "", fmt.Errorf("could not read main.go: %w", err)
	}
	log.Println("Successfully read main.go")

	newHandleFunc := fmt.Sprintf("\thttp.HandleFunc(\"%s\", %sHandler)\n\t// HANDLER_REGISTRATIONS_GO_HERE", handlerURL, strings.Title(handlerName))
	log.Printf("New HandleFunc string: %s", newHandleFunc)

	if !strings.Contains(string(mainGoContent), fmt.Sprintf("http.HandleFunc(\"%s\", %sHandler)", handlerURL, strings.Title(handlerName))) {
		log.Println("Handler not already registered. Proceeding with replacement.")
		updatedMainGoContent := strings.Replace(string(mainGoContent), "// HANDLER_REGISTRATIONS_GO_HERE", newHandleFunc, 1)

		if updatedMainGoContent == string(mainGoContent) {
			log.Println("Warning: Replacement did not change the content of main.go. The placeholder might be missing.")
		} else {
			log.Println("Replacement successful. Content of main.go has been updated in memory.")
		}

		err = os.WriteFile("main.go", []byte(updatedMainGoContent), 0644)
		if err != nil {
			log.Printf("Error writing to main.go: %v", err)
			return "", fmt.Errorf("could not write to main.go: %w", err)
		}
		log.Println("Successfully wrote updated content to main.go")
		return fmt.Sprintf("And registered it to URL '%s'.", handlerURL), nil
	}

	log.Println("Handler already registered.")
	return fmt.Sprintf("The URL '%s' for handler '%s' is already registered.", handlerURL, handlerName), nil
}

func runLLM() {
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

	reader := bufio.NewReader(os.Stdin)

	// Load last directory on startup
	lastDir, err := loadLastDirectory()
	if err == nil {
		err := os.Chdir(lastDir)
		if err != nil {
		}
	}

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
		}

		// --- Tagging ---
		words := strings.Fields(query)
		posTags := postagger.TagTokens(words)
		taggedData := nertagger.Nertagger(tag.Tag{Tokens: words, PosTag: posTags})

		// --- Start of new logic ---

		hasQuestionWord := false
		var objectTypeParts []string
		hasPrepositionIn := false
		var command string
		var targetDirectory string // Declare targetDirectory here
		var targetFile string      // New variable
		var predictedSentence string
		var handlerURL string // New variable to store the handler URL

		// Try to explicitly identify the command if it's the first token
		if len(taggedData.Tokens) > 0 {
			token := strings.ToLower(taggedData.Tokens[0])
			if token == "create" || token == "add" || token == "put" {
				command = "create"
			} else if token == "list" || token == "ls" || token == "show" {
				command = "list"
			} else if token == "go" || token == "cd" {
				command = "go"
			} else if token == "delete" || token == "remove" {
				command = "delete"
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
					objectTypeParts = append(objectTypeParts, token)
				case "PREPOSITION":
					if token == "in" || token == "into" {
						hasPrepositionIn = true
						foundTarget := false
						for j := i + 1; j < len(taggedData.Tokens); j++ {
							if strings.Contains(taggedData.Tokens[j], ".") { // Prioritize file
								targetFile = taggedData.Tokens[j]
								foundTarget = true
								break
							}
						}
						if !foundTarget { // If no file, look for directory
							for j := i + 1; j < len(taggedData.Tokens); j++ {
								if taggedData.NerTag[j] == "NAME" {
									targetDirectory = taggedData.Tokens[j]
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

		var objectType string
		if strings.Contains(strings.ToLower(query), "data structure") {
			objectType = "data structure"
			objectTypeParts = []string{} // Clear objectTypeParts to prevent interference
		} else {
			objectType = strings.Join(objectTypeParts, " ")
			if objectType == "" && strings.Contains(strings.ToLower(query), "handler") {
				objectType = "handler"
			}
		}

		fileName := findName(taggedData)

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
			// Find the last token that is not a common command word or preposition
			// Iterate backwards from the end of the tokens
			for i := len(taggedData.Tokens) - 1; i >= 0; i-- {
				token := strings.ToLower(taggedData.Tokens[i])
				// Exclude command words and prepositions
				if token != "go" && token != "to" && token != "project" && token != "folder" && token != "directory" && token != "cd" {
					targetDirectory = taggedData.Tokens[i]
					break
				}
			}
		}

		if command == "go" && targetDirectory != "" {
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
		} else if query == "pwd" || (hasDirectoryToken && command == "") {
			cwd, err := os.Getwd()
			if err != nil {
				predictedSentence = "I'm sorry, I couldn't determine the current directory."
			} else {
				predictedSentence = fmt.Sprintf("The current directory is: %s", cwd)
			}
		} else if command == "list" {
			files, err := os.ReadDir(".")
			if err != nil {
				predictedSentence = fmt.Sprintf("I couldn't list the contents of the directory: %v", err)
			} else {
				var items []string
				showFiles := contains(objectTypeParts, "file") || contains(objectTypeParts, "files")
				showFolders := contains(objectTypeParts, "folder") || contains(objectTypeParts, "folders")

				for _, file := range files {
					isDir := file.IsDir()
					// If no specific type is requested, or if both are requested, show everything.
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
		} else if command == "create" && strings.Contains(objectType, "handler") {
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
					predictedSentence = fmt.Sprintf("I have created the handler '%s' in %s.", handlerName, filePath)
				}

				if targetFile == "main.go" && handlerURL != "" {
					registrationMsg, err := registerHandlerURL(handlerName, handlerURL)
					if err != nil {
						log.Printf("Error registering handler URL: %v", err)
					} else {
						predictedSentence += " " + registrationMsg
					}
				}
			}
		endOfCreateHandler:
		} else if command == "create" && strings.Contains(objectType, "file") { // New block for generic file creation
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
		} else if command == "create" && strings.Contains(objectType, "webserver") {
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
	"fmt"
	"log"
	"net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from the %s webserver!", "` + fileName + `")
}

func main() {
	http.HandleFunc("/", handler)
	log.Println("Starting webserver on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
`
					err = os.WriteFile(filepath.Join(serverDir, "main.go"), []byte(serverContent), 0644)
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't create the webserver file %s: %v", filepath.Join(serverDir, "main.go"), err)
					} else {
						predictedSentence = fmt.Sprintf("I have created the webserver '%s' in %s/main.go.", fileName, serverDir)
					}
				}
			} else {
				predictedSentence = "You need to provide a name for the webserver."
			}
		} else if command == "create" && strings.Contains(objectType, "folder") { // New block for folder creation
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
		} else if command == "create" && objectType == "database" { // New block for database creation
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
									predictedSentence += fmt.Sprintf(" But no valid fields were provided to create a table.")
								}
							}
						}
					}
				}
			} else {
				predictedSentence = "You need to provide a name for the database."
			}

		} else if command == "create" && objectType == "data structure" {
			var err error
			var dbFileName string
			var tableName string
			var structFileName string
			var structContent string
			var fieldKeywordFound bool
			var fieldStartIndex int
			var withTheFieldsIndex int = -1

			queryParts := strings.Fields(query)
			structName := ""
			fields := make(map[string]string)

			for i, part := range queryParts {
				if part == "structure" && i+1 < len(queryParts) {
					structName = strings.Title(queryParts[i+1])
					break
				}
			}

			if structName == "" {
				predictedSentence = "You need to provide a name for the data structure."
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

			structContent = fmt.Sprintf("package main\n\ntype %s struct {\n", structName)
			for fieldName, fieldType := range fields {
				structContent += fmt.Sprintf("\t%s %s\n", strings.Title(fieldName), fieldType)
			}
			structContent += "}\n"

			structFileName = strings.ToLower(structName) + ".go"
			err = os.WriteFile(structFileName, []byte(structContent), 0644)
			if err != nil {
				predictedSentence = fmt.Sprintf("I couldn't create the Go struct file %s: %v", structFileName, err)
				goto endOfDataStructureCreation
			}
			predictedSentence = fmt.Sprintf("I have created the Go struct '%s' in %s.", structName, structFileName)

			tableName = strings.ToLower(structName)
			dbFileName = tableName + ".db"
			err = createTableWithFields(dbFileName, tableName, fields)
			if err != nil {
				predictedSentence += fmt.Sprintf(" But couldn't create the table '%s' in %s: %v", tableName, dbFileName, err)
				goto endOfDataStructureCreation
			}
			predictedSentence += fmt.Sprintf(" And updated the database '%s' with table '%s'.", dbFileName, tableName)

		endOfDataStructureCreation:
		} else if command == "delete" && (contains(objectTypeParts, "folder") || contains(objectTypeParts, "directory")) {
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
		} else if command == "delete" && contains(objectTypeParts, "file") {
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
		} else if hasQuestionWord && hasDirectoryToken {
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
		colors.AnimatedOutput("blue", "red", predictedSentence, 1*time.Second)
		fmt.Println("\n")

	}
}
