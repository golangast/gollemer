package main

import (
	"bufio"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time" // Added time import
	"net/http"

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

func registerHandlerURL(handlerName, handlerURL, mainGoPath string) (string, error) {
	fmt.Printf("Attempting to register handler '%s' with URL '%s' in file '%s'\n", handlerName, handlerURL, mainGoPath)

	mainGoContent, err := os.ReadFile(mainGoPath)
	if err != nil {
		fmt.Printf("Error reading %s: %v\n", mainGoPath, err)
		return "", fmt.Errorf("could not read %s: %w", mainGoPath, err)
	}
	fmt.Printf("Successfully read %s\n", mainGoPath)

	newHandleFunc := fmt.Sprintf("\thttp.HandleFunc(\"%s\", %sHandler)\n\t// HANDLER_REGISTRATIONS_GO_HERE", handlerURL, handlerName)
	fmt.Printf("New HandleFunc string: %s\n", newHandleFunc)

	if !strings.Contains(string(mainGoContent), fmt.Sprintf("http.HandleFunc(\"%s\", %sHandler)", handlerURL, handlerName)) {
		fmt.Println("Handler not already registered. Proceeding with replacement.")
		updatedMainGoContent := strings.Replace(string(mainGoContent), "// HANDLER_REGISTRATIONS_GO_HERE", newHandleFunc, 1)

		if updatedMainGoContent == string(mainGoContent) {
			fmt.Printf("Warning: Replacement did not change the content of %s. The placeholder might be missing.\n", mainGoPath)
		} else {
			fmt.Printf("Replacement successful. Content of %s has been updated in memory.\n", mainGoPath)
		}

		err = os.WriteFile(mainGoPath, []byte(updatedMainGoContent), 0644)
		if err != nil {
			fmt.Printf("Error writing to %s: %v\n", mainGoPath, err)
			return "", fmt.Errorf("could not write to %s: %w", mainGoPath, err)
		}
		goImports(mainGoPath)
		fmt.Printf("Successfully wrote updated content to %s\n", mainGoPath)
		return fmt.Sprintf("And registered it to URL '%s' in %s.", handlerURL, mainGoPath), nil
	}

	fmt.Println("Handler already registered.")
	return fmt.Sprintf("The URL '%s' for handler '%s' is already registered in %s.", handlerURL, handlerName, mainGoPath), nil
}

func registerHandlerWithPackage(packageName, packageImportPath, handlerName, handlerURL, mainGoPath string) (string, error) {
	fmt.Printf("Attempting to register handler '%s' with URL '%s' in file '%s'\n", handlerName, handlerURL, mainGoPath)

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
		content = strings.Replace(content, "// HANDLER_REGISTRATIONS_GO_HERE", newHandleFunc, 1)
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
				} else if token == "run" {
					command = "run"
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
			if strings.Contains(strings.ToLower(query), "handler") {
				objectType = "handler"
			} else if strings.Contains(strings.ToLower(query), "data structure") {
				objectType = "data structure"
				objectTypeParts = []string{} // Clear objectTypeParts to prevent interference
			} else {
				objectType = strings.Join(objectTypeParts, " ")
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
						if token != "go" && token != "to" && token != "project" && token != "folder" && token != "directory" && token != "cd" {
							targetDirectory = taggedData.Tokens[i]
							break
						}
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
						goImports(filePath)
						predictedSentence = fmt.Sprintf("I have created the handler '%s' in %s.", handlerName, filePath)
					}

					if targetFile != "" && strings.HasSuffix(targetFile, ".go") && handlerURL != "" {
						registrationMsg, err := registerHandlerURL(strings.Title(handlerName), handlerURL, targetFile)
						if err != nil {
							log.Printf("Error registering handler URL in %s: %v", targetFile, err)
							predictedSentence += fmt.Sprintf(" I tried to register the handler in %s but failed: %v", targetFile, err)
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
				} else {
					predictedSentence = "You need to provide a name for the webserver."
				}
			            } else if command == "run" && strings.Contains(objectType, "webserver") {
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
								predictedSentence = "You need to provide a name for the webserver to run."
							} else {
								// Path to the jim webserver's main package
								jimSourcePath := filepath.Join(projectRoot, "jim", "cmd", webserverName)
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
									runCmd.Dir = projectRoot // Running from project root
									runCmd.Stdout = os.Stdout // Redirect stdout
									runCmd.Stderr = os.Stderr // Redirect stderr
									
									err := runCmd.Start()
									if err != nil {
										predictedSentence = fmt.Sprintf("I couldn't run the webserver %s: %v", webserverName, err)
									} else {
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
											} else {
												log.Printf("WARNING: Webserver returned status code %d during verification.", resp.StatusCode)
												predictedSentence += fmt.Sprintf(" However, the webserver returned status code %d during verification.", resp.StatusCode)
											}
										}
									}
								}
								endOfRunWebserver: // Label for goto
							}
						} else if command == "stop" && strings.Contains(objectType, "webserver") {
							// Existing stop logic will also need to be updated to stop the built executable if needed,
							// or simply removed if the /stop endpoint is the only way to stop it.
							// For now, I will comment it out as the jim server only stops with its /stop endpoint
							// and this orchestrator should not directly try to stop it.
							predictedSentence = "The webserver can only be stopped by navigating to its /stop endpoint (e.g., http://localhost:8080/stop)."
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
										predictedSentence += " But no valid fields were provided to create a table."
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
				var fieldKeywordFound bool
				var fieldStartIndex int
				var withTheFieldsIndex int = -1
				var dirName string
				var updateRegMsg string
				var err1 error
				var deleteRegMsg string
				var err2 error
				var mainGoPath string
				var packageFileContent string
				var deleteHandlerContent string
				var updateHandlerContent string
				var lowercaseName string
				var packageName string
				var structDef string
				var structFields []string
				var structFieldExecs []string
				var sortedFieldNames []string
				var dbPathForHandler string
				var modulePath, projectRoot, cwd, relativeDir, packageImportPath string

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
				lowercaseName = strings.ToLower(structName)
				packageName = lowercaseName

				// Struct Definition
			structDef = fmt.Sprintf("type %s struct {\n", structName)
			structDef += "\tID int `json:\"id\"`\n"
			for fieldName, fieldType := range fields {
				structDef += fmt.Sprintf("\t%s %s `json:\"%s\"`\n", strings.Title(fieldName), fieldType, fieldName)
			}
			structDef += "}\n\n"

			// Handler field construction
			sortedFieldNames = make([]string, 0, len(fields))
			for k := range fields {
				sortedFieldNames = append(sortedFieldNames, k)
			}
			sort.Strings(sortedFieldNames)

			for _, fieldName := range sortedFieldNames {
				structFields = append(structFields, fmt.Sprintf("%s = ?", strings.ToLower(fieldName)))
				structFieldExecs = append(structFieldExecs, "u."+strings.Title(fieldName))
			}

			dbPathForHandler = lowercaseName + ".db"

			// Update Handler
			updateHandlerContent = fmt.Sprintf(`
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

	db, err := sql.Open("sqlite", "%s")
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
}`, structName, lowercaseName, structName, dbPathForHandler, lowercaseName, strings.Join(structFields, ", "), strings.Join(structFieldExecs, ", "))

			// Delete Handler
			deleteHandlerContent = fmt.Sprintf(`
func Delete%sHandler(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 { // e.g. /delete/user/123
		http.Error(w, "Invalid URL, expecting /delete/%s/{id}", http.StatusBadRequest)
		return
	}
	id := parts[len(parts)-1]

	db, err := sql.Open("sqlite", "%s")
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
}`, structName, lowercaseName, dbPathForHandler, lowercaseName)

			// Combine all parts into one file
							packageFileContent = fmt.Sprintf(`package %s
import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	_ "modernc.org/sqlite"
)

%s
%s
%s
`, packageName, structDef, updateHandlerContent, deleteHandlerContent)

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

			// Register Handlers in main.go
			mainGoPath = "main.go"
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