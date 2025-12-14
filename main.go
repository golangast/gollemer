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

	_ "modernc.org/sqlite" // Pure Go SQLite driver

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

func main() {
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

func runLLM() {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("/ʕ◔ϖ◔ʔ/> ")
		query, _ := reader.ReadString('\n')
		query = strings.TrimSpace(query)

		if query == "exit" {
			break
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
			} else if token == "list" || token == "ls" {
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
			}
		} else if query == "pwd" {
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
				for _, file := range files {
					isDir := file.IsDir()
					if contains(objectTypeParts, "folder") || contains(objectTypeParts, "folders") {
						if isDir {
							items = append(items, file.Name())
						}
					} else if contains(objectTypeParts, "file") || contains(objectTypeParts, "files") {
						if !isDir {
							items = append(items, file.Name())
						}
					} else {
						items = append(items, file.Name())
					}
				}
				predictedSentence = "Here are the contents of the directory:\n" + strings.Join(items, "\n")
			}
		} else if command == "create" && objectType == "handler" {
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
				handlerContent := `
// ` + strings.Title(handlerName) + `Handler is a sample handler function.
func ` + strings.Title(handlerName) + `Handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Executing ` + strings.Title(handlerName) + `Handler! Request URL: %s\n", r.URL.Path)
}
`
				filePath := handlerName + ".go" // Default file path if targetFile is not specified.
				if targetFile != "" {
					filePath = targetFile
				}

				existingContentBytes, err := os.ReadFile(filePath)
				handlerExists := false
				if err == nil { // File exists
					handlerExists = strings.Contains(string(existingContentBytes), "func "+strings.Title(handlerName)+"Handler(w http.ResponseWriter, r *http.Request)")
				}

				if !handlerExists {
					var fileContentToAppend string
					if err == nil { // File exists, append to it
						fileContentToAppend = string(existingContentBytes) + handlerContent
					} else { // File does not exist, create new
						fileContentToAppend = handlerContent
					}

					err = os.WriteFile(filePath, []byte(fileContentToAppend), 0644)
					if err != nil {
						predictedSentence = fmt.Sprintf("I couldn't write to the target file %s: %v", filePath, err)
						goto endOfCreateHandler
					}
					predictedSentence = fmt.Sprintf("I have added the handler '%s' to %s.", handlerName, filePath)
				} else {
					predictedSentence = fmt.Sprintf("The handler '%s' already exists in %s.", handlerName, filePath)
				}

				if filePath == "main.go" && handlerURL != "" {
					mainGoContent, err := os.ReadFile("main.go")
					if err != nil {
						log.Printf("Error reading main.go to update startWebServer: %v", err)
						goto endOfCreateHandler
					}
					newHandleFunc := fmt.Sprintf("\thttp.HandleFunc(\"%s\", %sHandler)\n\t// HANDLER_REGISTRATIONS_GO_HERE", handlerURL, strings.Title(handlerName))
					if !strings.Contains(string(mainGoContent), fmt.Sprintf("http.HandleFunc(\"%s\", %sHandler)", handlerURL, strings.Title(handlerName))) {
						updatedMainGoContent := strings.Replace(string(mainGoContent), "// HANDLER_REGISTRATIONS_GO_HERE", newHandleFunc, 1)
						err = os.WriteFile("main.go", []byte(updatedMainGoContent), 0644)
						if err != nil {
							log.Printf("Error writing main.go to update startWebServer: %v", err)
						} else {
							predictedSentence += fmt.Sprintf(" And registered it to URL '%s'.", handlerURL)
						}
					} else {
						predictedSentence += fmt.Sprintf(" The URL '%s' for handler '%s' is already registered.", handlerURL, handlerName)
					}
				}
			}
		endOfCreateHandler:
		} else if command == "create" && objectType == "file" { // New block for generic file creation
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
		} else if command == "create" && objectType == "webserver" {
			if fileName != "" {
				serverDir := filepath.Join("cmd", fileName)
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
						predictedSentence = fmt.Sprintf("I have created the webserver '%s' in cmd/%s/main.go.", fileName, fileName)
					}
				}
			} else {
				predictedSentence = "You need to provide a name for the webserver."
			}
		} else if command == "create" && objectType == "webserver" {
			if fileName != "" {
				serverDir := filepath.Join("cmd", fileName)
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
						predictedSentence = fmt.Sprintf("I have created the webserver '%s' in cmd/%s/main.go.", fileName, fileName)
					}
				}
			} else {
				predictedSentence = "You need to provide a name for the webserver."
			}
		} else if command == "create" && objectType == "folder" { // New block for folder creation
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
					}
				}
			} else {
				predictedSentence = "You need to provide a name for the database."
			}

		} else if command == "create" && objectType == "data structure" {
			var db *sql.DB
			var err error
			var dbFileName string
			var columns []string
			var sqlStatement string
			var tableName string
			var structFileName string
			var structContent string
			var fieldKeywordFound bool
			var fieldStartIndex int // New declaration

			// Extract struct name and fields from the query
			queryParts := strings.Fields(query)
			structName := ""
			fields := make(map[string]string) // fieldName -> fieldType

			// Find struct name (e.g., "jim" from "add a data structure jim")
			for i, part := range queryParts {
				if part == "structure" && i+1 < len(queryParts) {
					structName = strings.Title(queryParts[i+1]) // Capitalize for Go struct name
					break
				}
			}

			if structName == "" {
				predictedSentence = "You need to provide a name for the data structure."
				goto endOfDataStructureCreation
			}
			fileName = structName // Set fileName for data structure

			// Parse fields (e.g., "field name string field age int" or "field name string and age int")
			fieldKeywordFound = false
			fieldStartIndex = -1
			for i, part := range queryParts {
				if part == "field" {
					fieldStartIndex = i
					break
				}
			}

			if fieldStartIndex != -1 {
				fieldKeywordFound = true
				for i := fieldStartIndex + 1; i < len(queryParts); {
					if queryParts[i] == "and" || queryParts[i] == "field" { // CRITICAL FIX
						i++
						continue
					}
					if i+1 < len(queryParts) {
						fieldName := queryParts[i]
						fieldType := queryParts[i+1]
						fields[fieldName] = fieldType
						i += 2 // Move past fieldName and fieldType
					} else {
						predictedSentence = "Incomplete field definition found."
						goto endOfDataStructureCreation
					}
				}
			}

			if !fieldKeywordFound || len(fields) == 0 {
				predictedSentence = "You need to provide fields for the data structure."
				goto endOfDataStructureCreation
			}

			// Generate Go Struct
			structContent = fmt.Sprintf("package main\n\ntype %s struct {\n", structName)
			for fieldName, fieldType := range fields {
				structContent += fmt.Sprintf("\t%s %s\n", strings.Title(fieldName), fieldType)
			}
			structContent += "}\n"

			// Write Go file (jim.go)
			structFileName = strings.ToLower(structName) + ".go"
			err = os.WriteFile(structFileName, []byte(structContent), 0644)
			if err != nil {
				predictedSentence = fmt.Sprintf("I couldn't create the Go struct file %s: %v", structFileName, err)
				goto endOfDataStructureCreation
			}
			predictedSentence = fmt.Sprintf("I have created the Go struct '%s' in %s.", structName, structFileName)

			// Generate SQL CREATE TABLE statement
			tableName = strings.ToLower(structName) // Lowercase for table name
			sqlStatement = fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (\n", tableName)
			columns = []string{}
			for fieldName, fieldType := range fields {
				sqlType := ""
				switch fieldType {
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

			// Update jim.db with the new table
			dbFileName = "jim.db" // Assuming jim.db is the target database
			db, err = sql.Open("sqlite", dbFileName)
			if err != nil {
				predictedSentence += fmt.Sprintf(" And couldn't open the database file %s to create the table: %v", dbFileName, err)
				goto endOfDataStructureCreation
			}
			defer db.Close()

			_, err = db.Exec(sqlStatement)
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

		// Print the output
		fmt.Println(predictedSentence)
	}
}
