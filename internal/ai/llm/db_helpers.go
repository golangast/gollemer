package llm

import (
	"database/sql"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	_ "modernc.org/sqlite"
)

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
