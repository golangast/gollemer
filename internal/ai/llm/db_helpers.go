package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

func createTableWithFields(dbFileName, tableName string, fields map[string]string) error {
	// If it ends in .db, change to .json
	if strings.HasSuffix(dbFileName, ".db") {
		dbFileName = strings.TrimSuffix(dbFileName, ".db") + ".json"
	}

	// In a JSON-based system, a "table" is just a JSON file containing a list of objects.
	// We ensure the file exists.
	if _, err := os.Stat(dbFileName); os.IsNotExist(err) {
		err := os.MkdirAll(filepath.Dir(dbFileName), 0755)
		if err != nil {
			return err
		}
		// Create an empty list
		return os.WriteFile(dbFileName, []byte("[]"), 0644)
	}

	return nil
}

func deleteColumnFromTable(dbFileName, tableName, columnToDelete string, remainingFields map[string]string) error {
	// If it ends in .db, change to .json
	if strings.HasSuffix(dbFileName, ".db") {
		dbFileName = strings.TrimSuffix(dbFileName, ".db") + ".json"
	}

	data, err := os.ReadFile(dbFileName)
	if err != nil {
		return fmt.Errorf("failed to read JSON file: %w", err)
	}

	var items []map[string]interface{}
	if err := json.Unmarshal(data, &items); err != nil {
		return fmt.Errorf("failed to parse JSON file: %w", err)
	}

	// Remove the column from each item
	for _, item := range items {
		delete(item, strings.ToLower(columnToDelete))
	}

	updatedData, err := json.MarshalIndent(items, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal updated data: %w", err)
	}

	return os.WriteFile(dbFileName, updatedData, 0644)
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
				var indent strings.Builder
				for _, char := range line {
					if char == ' ' || char == '\t' {
						indent.WriteString(string(char))
					} else {
						break
					}
				}
				updatedLines = append(updatedLines, fmt.Sprintf("%shttp.HandleFunc(\"%s\", %sHandler)", indent.String(), handlerURL, handlerName))
				updatedLines = append(updatedLines, line) // Keep the placeholder for future registrations
				found = true
			} else {
				updatedLines = append(updatedLines, line)
			}
		}

		if !found {
			return "", fmt.Errorf("placeholder '%s' not found in %s. Tip: Place it inside func main() to enable auto-registration.", placeholder, mainGoPath)
		}

		updatedMainGoContent := strings.Join(updatedLines, "\n")

		err = os.WriteFile(mainGoPath, []byte(updatedMainGoContent), 0644)
		if err != nil {
			return "", fmt.Errorf("could not write to %s: %w", mainGoPath, err)
		}
		goImports(mainGoPath)
		return fmt.Sprintf("And registered it to URL '%s' in %s.", handlerURL, mainGoPath), nil
	}

	return "Handler already registered.", nil
}

// InjectPlaceholder looks for the main function and injects the handler registration tag
func InjectPlaceholder(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	// 1. Check if it's already there to avoid duplicates
	if strings.Contains(string(content), "// HANDLER_REGISTRATIONS_GO_HERE") {
		return nil
	}

	// 2. Look for the main function signature
	re := regexp.MustCompile(`func main\(\) \{`)
	if !re.Match(content) {
		return fmt.Errorf("could not find func main in %s", path)
	}

	insertion := "func main() {\n\t// HANDLER_REGISTRATIONS_GO_HERE"
	newContent := re.ReplaceAllString(string(content), insertion)

	return os.WriteFile(path, []byte(newContent), 0644)
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

