package csvgen

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
)

// CSVSchema represents the structure of a CSV file
type CSVSchema struct {
	FileName   string
	StructName string
	Fields     []CSVField
}

// CSVField represents a field in the CSV
type CSVField struct {
	Name    string
	GoType  string
	CSVName string
	JSONTag string
}

// ParseCSVFile reads a CSV file and extracts its schema
func ParseCSVFile(filePath string) (*CSVSchema, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open CSV file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Read header row
	headers, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV headers: %w", err)
	}

	// Read first data row to infer types
	firstRow, err := reader.Read()
	if err != nil {
		// If there's no data, default to string types
		firstRow = make([]string, len(headers))
	}

	// Generate struct name from filename
	baseName := filepath.Base(filePath)
	baseName = strings.TrimSuffix(baseName, filepath.Ext(baseName))
	structName := toStructName(baseName)

	schema := &CSVSchema{
		FileName:   filePath,
		StructName: structName,
		Fields:     make([]CSVField, 0, len(headers)),
	}

	for i, header := range headers {
		fieldName := toFieldName(header)
		goType := inferGoType(firstRow[i])

		schema.Fields = append(schema.Fields, CSVField{
			Name:    fieldName,
			GoType:  goType,
			CSVName: header,
			JSONTag: toJSONTag(header),
		})
	}

	return schema, nil
}

// toStructName converts a filename to a Go struct name
func toStructName(name string) string {
	// Remove numbers and special characters, capitalize
	name = strings.ReplaceAll(name, "-", " ")
	name = strings.ReplaceAll(name, "_", " ")

	// Remove numbers at the end (e.g., "customers-100" -> "customers")
	parts := strings.Fields(name)
	if len(parts) > 0 {
		// Check if last part is a number
		lastPart := parts[len(parts)-1]
		isNumber := true
		for _, r := range lastPart {
			if !unicode.IsDigit(r) {
				isNumber = false
				break
			}
		}
		if isNumber {
			parts = parts[:len(parts)-1]
		}
	}

	name = strings.Join(parts, " ")
	name = strings.Title(name)
	name = strings.ReplaceAll(name, " ", "")

	// Ensure singular form (simple approach)
	if strings.HasSuffix(name, "s") && len(name) > 1 {
		name = name[:len(name)-1]
	}

	return name
}

// toFieldName converts a CSV header to a Go field name
func toFieldName(header string) string {
	// Replace spaces and special characters with underscores
	header = strings.ReplaceAll(header, " ", "_")
	header = strings.ReplaceAll(header, "-", "_")

	// Split by underscore and capitalize each part
	parts := strings.Split(header, "_")
	for i, part := range parts {
		if len(part) > 0 {
			parts[i] = strings.Title(strings.ToLower(part))
		}
	}

	return strings.Join(parts, "")
}

// toJSONTag converts a CSV header to a JSON tag
func toJSONTag(header string) string {
	return strings.ToLower(strings.ReplaceAll(header, " ", "_"))
}

// inferGoType infers the Go type from a sample value
func inferGoType(value string) string {
	value = strings.TrimSpace(value)

	if value == "" {
		return "string"
	}

	// Check if it's a number
	isInt := true
	isFloat := false
	for i, r := range value {
		if r == '.' {
			isFloat = true
			isInt = false
		} else if r == '-' && i == 0 {
			// Negative number
			continue
		} else if !unicode.IsDigit(r) {
			isInt = false
			isFloat = false
			break
		}
	}

	if isInt {
		return "int"
	}
	if isFloat {
		return "float64"
	}

	return "string"
}

// GenerateStruct generates Go struct code from CSV schema
func GenerateStruct(schema *CSVSchema) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("// %s represents a row from %s\n", schema.StructName, filepath.Base(schema.FileName)))
	sb.WriteString(fmt.Sprintf("type %s struct {\n", schema.StructName))

	for _, field := range schema.Fields {
		sb.WriteString(fmt.Sprintf("\t%s %s `json:\"%s\" csv:\"%s\"`\n",
			field.Name, field.GoType, field.JSONTag, field.CSVName))
	}

	sb.WriteString("}\n")

	return sb.String()
}

// goTypeToSQLType converts Go types to SQLite types
func goTypeToSQLType(goType string) string {
	switch goType {
	case "int":
		return "INTEGER"
	case "float64":
		return "REAL"
	case "bool":
		return "INTEGER"
	default:
		return "TEXT"
	}
}

// GenerateTableSchema generates SQLite CREATE TABLE statement
func GenerateTableSchema(schema *CSVSchema) string {
	var sb strings.Builder

	tableName := strings.ToLower(schema.StructName) + "s"

	sb.WriteString(fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (\n", tableName))

	for i, field := range schema.Fields {
		sqlType := goTypeToSQLType(field.GoType)
		columnName := strings.ToLower(field.Name)

		// First field is typically ID/primary key
		if i == 0 {
			sb.WriteString(fmt.Sprintf("    %s %s PRIMARY KEY", columnName, sqlType))
		} else {
			sb.WriteString(fmt.Sprintf("    %s %s", columnName, sqlType))
		}

		if i < len(schema.Fields)-1 {
			sb.WriteString(",\n")
		} else {
			sb.WriteString("\n")
		}
	}

	sb.WriteString(");")

	return sb.String()
}

// GenerateDatabaseFunctions generates database initialization and helper functions
func GenerateDatabaseFunctions(schema *CSVSchema) string {
	var sb strings.Builder

	structName := schema.StructName
	tableName := strings.ToLower(structName) + "s"

	// Database initialization
	sb.WriteString("// Database connection\n")
	sb.WriteString("var db *sql.DB\n")
	sb.WriteString("var dbOnce sync.Once\n\n")

	sb.WriteString("// InitDB initializes the database connection\n")
	sb.WriteString("func InitDB(dbPath string) error {\n")
	sb.WriteString("\tvar err error\n")
	sb.WriteString("\tdbOnce.Do(func() {\n")
	sb.WriteString("\t\tdb, err = sql.Open(\"sqlite3\", dbPath)\n")
	sb.WriteString("\t\tif err != nil {\n")
	sb.WriteString("\t\t\treturn\n")
	sb.WriteString("\t\t}\n")
	sb.WriteString("\t\terr = db.Ping()\n")
	sb.WriteString("\t})\n")
	sb.WriteString("\treturn err\n")
	sb.WriteString("}\n\n")

	// Create table function
	sb.WriteString(fmt.Sprintf("// Create%sTable creates the %s table\n", structName, tableName))
	sb.WriteString(fmt.Sprintf("func Create%sTable() error {\n", structName))
	sb.WriteString("\tschema := `\n")
	sb.WriteString(GenerateTableSchema(schema))
	sb.WriteString("\n\t`\n")
	sb.WriteString("\t_, err := db.Exec(schema)\n")
	sb.WriteString("\treturn err\n")
	sb.WriteString("}\n\n")

	// Insert function
	sb.WriteString(fmt.Sprintf("// Insert%s inserts a %s into the database\n", structName, strings.ToLower(structName)))
	sb.WriteString(fmt.Sprintf("func Insert%s(item %s) error {\n", structName, structName))

	// Build INSERT statement
	var columns []string
	var placeholders []string
	for _, field := range schema.Fields {
		columns = append(columns, strings.ToLower(field.Name))
		placeholders = append(placeholders, "?")
	}

	sb.WriteString(fmt.Sprintf("\tquery := `INSERT INTO %s (%s) VALUES (%s)`\n",
		tableName,
		strings.Join(columns, ", "),
		strings.Join(placeholders, ", ")))

	sb.WriteString("\t_, err := db.Exec(query")
	for _, field := range schema.Fields {
		sb.WriteString(fmt.Sprintf(", item.%s", field.Name))
	}
	sb.WriteString(")\n")
	sb.WriteString("\treturn err\n")
	sb.WriteString("}\n\n")

	// Get all function
	sb.WriteString(fmt.Sprintf("// GetAll%s retrieves all %s from the database\n", structName+"s", strings.ToLower(structName)+"s"))
	sb.WriteString(fmt.Sprintf("func GetAll%s() ([]%s, error) {\n", structName+"s", structName))
	sb.WriteString(fmt.Sprintf("\trows, err := db.Query(\"SELECT * FROM %s\")\n", tableName))
	sb.WriteString("\tif err != nil {\n")
	sb.WriteString("\t\treturn nil, err\n")
	sb.WriteString("\t}\n")
	sb.WriteString("\tdefer rows.Close()\n\n")
	sb.WriteString(fmt.Sprintf("\tvar items []%s\n", structName))
	sb.WriteString("\tfor rows.Next() {\n")
	sb.WriteString(fmt.Sprintf("\t\tvar item %s\n", structName))
	sb.WriteString("\t\terr := rows.Scan(")

	for i, field := range schema.Fields {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(fmt.Sprintf("&item.%s", field.Name))
	}

	sb.WriteString(")\n")
	sb.WriteString("\t\tif err != nil {\n")
	sb.WriteString("\t\t\treturn nil, err\n")
	sb.WriteString("\t\t}\n")
	sb.WriteString("\t\titems = append(items, item)\n")
	sb.WriteString("\t}\n")
	sb.WriteString("\treturn items, rows.Err()\n")
	sb.WriteString("}\n\n")

	// Get by ID function (using first field)
	if len(schema.Fields) > 0 {
		idField := schema.Fields[0]
		sb.WriteString(fmt.Sprintf("// Get%sBy%s retrieves a %s by %s\n",
			structName, idField.Name, strings.ToLower(structName), strings.ToLower(idField.Name)))
		sb.WriteString(fmt.Sprintf("func Get%sBy%s(id %s) (*%s, error) {\n",
			structName, idField.Name, idField.GoType, structName))
		sb.WriteString(fmt.Sprintf("\trow := db.QueryRow(\"SELECT * FROM %s WHERE %s = ?\", id)\n",
			tableName, strings.ToLower(idField.Name)))
		sb.WriteString(fmt.Sprintf("\tvar item %s\n", structName))
		sb.WriteString("\terr := row.Scan(")

		for i, field := range schema.Fields {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("&item.%s", field.Name))
		}

		sb.WriteString(")\n")
		sb.WriteString("\tif err == sql.ErrNoRows {\n")
		sb.WriteString("\t\treturn nil, nil\n")
		sb.WriteString("\t}\n")
		sb.WriteString("\tif err != nil {\n")
		sb.WriteString("\t\treturn nil, err\n")
		sb.WriteString("\t}\n")
		sb.WriteString("\treturn &item, nil\n")
		sb.WriteString("}\n\n")
	}

	return sb.String()
}

// GenerateHandlers generates CRUD handlers for the struct
func GenerateHandlers(schema *CSVSchema) string {
	var sb strings.Builder

	structName := schema.StructName
	varName := strings.ToLower(string(structName[0])) + structName[1:]
	pluralName := structName + "s"
	varPluralName := strings.ToLower(string(pluralName[0])) + pluralName[1:]

	// Load from CSV function
	sb.WriteString(fmt.Sprintf("// Load%sFromCSV loads %s from CSV file\n", pluralName, varPluralName))
	sb.WriteString(fmt.Sprintf("func Load%sFromCSV(filePath string) error {\n", pluralName))
	sb.WriteString("\tfile, err := os.Open(filePath)\n")
	sb.WriteString("\tif err != nil {\n")
	sb.WriteString("\t\treturn fmt.Errorf(\"failed to open CSV file: %%w\", err)\n")
	sb.WriteString("\t}\n")
	sb.WriteString("\tdefer file.Close()\n\n")
	sb.WriteString("\treader := csv.NewReader(file)\n")
	sb.WriteString("\trecords, err := reader.ReadAll()\n")
	sb.WriteString("\tif err != nil {\n")
	sb.WriteString("\t\treturn fmt.Errorf(\"failed to read CSV: %%w\", err)\n")
	sb.WriteString("\t}\n\n")
	sb.WriteString("\tif len(records) < 2 {\n")
	sb.WriteString("\t\treturn fmt.Errorf(\"CSV file is empty or has no data rows\")\n")
	sb.WriteString("\t}\n\n")

	// Create table
	sb.WriteString(fmt.Sprintf("\tif err := Create%sTable(); err != nil {\n", structName))
	sb.WriteString("\t\treturn fmt.Errorf(\"failed to create table: %%w\", err)\n")
	sb.WriteString("\t}\n\n")

	sb.WriteString("\t// Skip header row\n")
	sb.WriteString("\tfor _, record := range records[1:] {\n")
	sb.WriteString(fmt.Sprintf("\t\tif len(record) < %d {\n", len(schema.Fields)))
	sb.WriteString("\t\t\tcontinue // Skip incomplete rows\n")
	sb.WriteString("\t\t}\n\n")
	sb.WriteString(fmt.Sprintf("\t\t%s := %s{\n", varName, structName))

	for i, field := range schema.Fields {
		switch field.GoType {
		case "int":
			sb.WriteString(fmt.Sprintf("\t\t\t%s: parseInt(record[%d]),\n", field.Name, i))
		case "float64":
			sb.WriteString(fmt.Sprintf("\t\t\t%s: parseFloat(record[%d]),\n", field.Name, i))
		default:
			sb.WriteString(fmt.Sprintf("\t\t\t%s: record[%d],\n", field.Name, i))
		}
	}

	sb.WriteString("\t\t}\n")
	sb.WriteString(fmt.Sprintf("\t\tif err := Insert%s(%s); err != nil {\n", structName, varName))
	sb.WriteString(fmt.Sprintf("\t\t\tfmt.Printf(\"Warning: failed to insert record: %%v\\n\", err)\n"))
	sb.WriteString("\t\t}\n")
	sb.WriteString("\t}\n\n")
	sb.WriteString("\treturn nil\n")
	sb.WriteString("}\n\n")

	// Helper functions for parsing
	sb.WriteString("// Helper functions for type conversion\n")
	sb.WriteString("func parseInt(s string) int {\n")
	sb.WriteString("\ts = strings.TrimSpace(s)\n")
	sb.WriteString("\tval, _ := strconv.Atoi(s)\n")
	sb.WriteString("\treturn val\n")
	sb.WriteString("}\n\n")
	sb.WriteString("func parseFloat(s string) float64 {\n")
	sb.WriteString("\ts = strings.TrimSpace(s)\n")
	sb.WriteString("\tval, _ := strconv.ParseFloat(s, 64)\n")
	sb.WriteString("\treturn val\n")
	sb.WriteString("}\n\n")

	// GET all handler
	sb.WriteString(fmt.Sprintf("// Handle%sList returns all %s\n", structName, varPluralName))
	sb.WriteString(fmt.Sprintf("func Handle%sList(w http.ResponseWriter, r *http.Request) {\n", structName))
	sb.WriteString(fmt.Sprintf("\titems, err := GetAll%s()\n", pluralName))
	sb.WriteString("\tif err != nil {\n")
	sb.WriteString("\t\thttp.Error(w, err.Error(), http.StatusInternalServerError)\n")
	sb.WriteString("\t\treturn\n")
	sb.WriteString("\t}\n\n")
	sb.WriteString("\tw.Header().Set(\"Content-Type\", \"application/json\")\n")
	sb.WriteString("\tjson.NewEncoder(w).Encode(items)\n")
	sb.WriteString("}\n\n")

	// GET by ID handler (using first field as ID)
	if len(schema.Fields) > 0 {
		idField := schema.Fields[0]
		sb.WriteString(fmt.Sprintf("// Handle%sGet returns a single %s by %s\n", structName, varName, idField.Name))
		sb.WriteString(fmt.Sprintf("func Handle%sGet(w http.ResponseWriter, r *http.Request) {\n", structName))
		sb.WriteString(fmt.Sprintf("\t%s := r.URL.Query().Get(\"%s\")\n", idField.JSONTag, idField.JSONTag))
		sb.WriteString(fmt.Sprintf("\tif %s == \"\" {\n", idField.JSONTag))
		sb.WriteString("\t\thttp.Error(w, \"ID parameter required\", http.StatusBadRequest)\n")
		sb.WriteString("\t\treturn\n")
		sb.WriteString("\t}\n\n")

		// Convert ID if necessary
		var idVar string
		if idField.GoType == "int" {
			sb.WriteString(fmt.Sprintf("\tidVal, err := strconv.Atoi(%s)\n", idField.JSONTag))
			sb.WriteString("\tif err != nil {\n")
			sb.WriteString("\t\thttp.Error(w, \"Invalid ID format\", http.StatusBadRequest)\n")
			sb.WriteString("\t\treturn\n")
			sb.WriteString("\t}\n")
			idVar = "idVal"
		} else if idField.GoType == "float64" {
			sb.WriteString(fmt.Sprintf("\tidVal, err := strconv.ParseFloat(%s, 64)\n", idField.JSONTag))
			sb.WriteString("\tif err != nil {\n")
			sb.WriteString("\t\thttp.Error(w, \"Invalid ID format\", http.StatusBadRequest)\n")
			sb.WriteString("\t\treturn\n")
			sb.WriteString("\t}\n")
			idVar = "idVal"
		} else {
			idVar = idField.JSONTag
		}

		sb.WriteString(fmt.Sprintf("\titem, err := Get%sBy%s(%s)\n", structName, idField.Name, idVar))
		sb.WriteString("\tif err != nil {\n")
		sb.WriteString("\t\thttp.Error(w, err.Error(), http.StatusInternalServerError)\n")
		sb.WriteString("\t\treturn\n")
		sb.WriteString("\t}\n")
		sb.WriteString("\tif item == nil {\n")
		sb.WriteString("\t\thttp.Error(w, \"Not found\", http.StatusNotFound)\n")
		sb.WriteString("\t\treturn\n")
		sb.WriteString("\t}\n\n")

		sb.WriteString("\t\tw.Header().Set(\"Content-Type\", \"application/json\")\n")
		sb.WriteString("\t\tjson.NewEncoder(w).Encode(item)\n")
		sb.WriteString("}\n\n")
	}

	// POST create handler
	sb.WriteString(fmt.Sprintf("// Handle%sCreate creates a new %s\n", structName, varName))
	sb.WriteString(fmt.Sprintf("func Handle%sCreate(w http.ResponseWriter, r *http.Request) {\n", structName))
	sb.WriteString(fmt.Sprintf("\tvar %s %s\n", varName, structName))
	sb.WriteString(fmt.Sprintf("\tif err := json.NewDecoder(r.Body).Decode(&%s); err != nil {\n", varName))
	sb.WriteString("\t\thttp.Error(w, err.Error(), http.StatusBadRequest)\n")
	sb.WriteString("\t\treturn\n")
	sb.WriteString("\t}\n\n")

	sb.WriteString(fmt.Sprintf("\tif err := Insert%s(%s); err != nil {\n", structName, varName))
	sb.WriteString("\t\thttp.Error(w, err.Error(), http.StatusInternalServerError)\n")
	sb.WriteString("\t\treturn\n")
	sb.WriteString("\t}\n\n")

	sb.WriteString("\tw.Header().Set(\"Content-Type\", \"application/json\")\n")
	sb.WriteString("\tw.WriteHeader(http.StatusCreated)\n")
	sb.WriteString(fmt.Sprintf("\tjson.NewEncoder(w).Encode(%s)\n", varName))
	sb.WriteString("}\n\n")

	return sb.String()
}

// GenerateRoutes generates HTTP route registration code
func GenerateRoutes(schema *CSVSchema) string {
	var sb strings.Builder

	structName := schema.StructName
	basePath := "/" + strings.ToLower(structName) + "s"

	sb.WriteString(fmt.Sprintf("\t// %s routes\n", structName))
	sb.WriteString(fmt.Sprintf("\thttp.HandleFunc(\"%s\", Handle%sList)\n", basePath, structName))
	sb.WriteString(fmt.Sprintf("\thttp.HandleFunc(\"%s/get\", Handle%sGet)\n", basePath, structName))
	sb.WriteString(fmt.Sprintf("\thttp.HandleFunc(\"%s/create\", Handle%sCreate)\n", basePath, structName))

	return sb.String()
}

// GenerateCompleteFile generates a complete Go file with all components
func GenerateCompleteFile(schema *CSVSchema, outputPath string) error {
	var sb strings.Builder

	// Package and imports
	sb.WriteString("package main\n\n")
	sb.WriteString("import (\n")
	sb.WriteString("\t\"database/sql\"\n")
	sb.WriteString("\t\"encoding/csv\"\n")
	sb.WriteString("\t\"encoding/json\"\n")
	sb.WriteString("\t\"fmt\"\n")
	sb.WriteString("\t\"net/http\"\n")
	sb.WriteString("\t\"os\"\n")
	sb.WriteString("\t\"strconv\"\n")
	sb.WriteString("\t\"strings\"\n")
	sb.WriteString("\t\"sync\"\n")
	sb.WriteString("\n\t_ \"github.com/mattn/go-sqlite3\"\n")
	sb.WriteString(")\n\n")

	// Struct definition
	sb.WriteString(GenerateStruct(schema))
	sb.WriteString("\n")

	// Database functions
	sb.WriteString(GenerateDatabaseFunctions(schema))
	sb.WriteString("\n")

	// Handlers
	sb.WriteString(GenerateHandlers(schema))

	// Write to file
	return os.WriteFile(outputPath, []byte(sb.String()), 0644)
}

// ProcessFeedsFolder processes all CSV files in the feeds folder
func ProcessFeedsFolder(feedsPath, outputDir string) error {
	// Ensure output directory exists
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Read all files in feeds folder
	entries, err := os.ReadDir(feedsPath)
	if err != nil {
		return fmt.Errorf("failed to read feeds directory: %w", err)
	}

	schemas := make([]*CSVSchema, 0)

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		if !strings.HasSuffix(entry.Name(), ".csv") {
			continue
		}

		filePath := filepath.Join(feedsPath, entry.Name())
		fmt.Printf("📄 Processing: %s\n", entry.Name())

		schema, err := ParseCSVFile(filePath)
		if err != nil {
			fmt.Printf("  ⚠️ Failed to parse %s: %v\n", entry.Name(), err)
			continue
		}

		schemas = append(schemas, schema)

		// Generate individual file
		outputFile := filepath.Join(outputDir, strings.ToLower(schema.StructName)+"_handlers.go")
		if err := GenerateCompleteFile(schema, outputFile); err != nil {
			fmt.Printf("  ⚠️ Failed to generate file for %s: %v\n", entry.Name(), err)
			continue
		}

		fmt.Printf("  ✅ Generated: %s\n", outputFile)
		fmt.Printf("  📦 Struct: %s\n", schema.StructName)
		fmt.Printf("  🔧 Fields: %d\n", len(schema.Fields))
	}

	// Generate main server file with all routes
	if len(schemas) > 0 {
		if err := GenerateMainServer(schemas, outputDir, ""); err != nil {
			return fmt.Errorf("failed to generate main server: %w", err)
		}
	}

	return nil
}

// GenerateMainServer generates the main server file with all routes
func GenerateMainServer(schemas []*CSVSchema, outputDir string, dbPath string) error {
	var sb strings.Builder

	if dbPath == "" {
		dbPath = "orchestrator.db"
	}

	sb.WriteString("package main\n\n")
	sb.WriteString("import (\n")
	sb.WriteString("\t\"fmt\"\n")
	sb.WriteString("\t\"log\"\n")
	sb.WriteString("\t\"net/http\"\n")
	sb.WriteString(")\n\n")

	sb.WriteString("func main() {\n")
	sb.WriteString("\tfmt.Println(\"🚀 Starting CSV-based API server...\")\n\n")

	// Initialize Database
	sb.WriteString("\t// Initialize Database\n")
	sb.WriteString(fmt.Sprintf("\tif err := InitDB(\"%s\"); err != nil {\n", dbPath))
	sb.WriteString("\t\tlog.Fatalf(\"Failed to initialize database: %v\", err)\n")
	sb.WriteString("\t}\n\n")

	// Load CSV data for each schema
	for _, schema := range schemas {
		csvPath := schema.FileName
		sb.WriteString(fmt.Sprintf("\t// Load %s data\n", schema.StructName))
		sb.WriteString(fmt.Sprintf("\tif err := Load%ssFromCSV(\"%s\"); err != nil {\n", schema.StructName, csvPath))
		sb.WriteString(fmt.Sprintf("\t\tlog.Printf(\"Warning: Failed to load %s: %%v\", err)\n", schema.StructName))
		sb.WriteString("\t} else {\n")
		sb.WriteString(fmt.Sprintf("\t\tfmt.Printf(\"✅ Loaded %s data\\n\")\n", schema.StructName))
		sb.WriteString("\t}\n\n")
	}

	// Register routes
	sb.WriteString("\t// Register routes\n")
	for _, schema := range schemas {
		sb.WriteString(GenerateRoutes(schema))
		sb.WriteString("\n")
	}

	// Root handler
	sb.WriteString("\t// Root handler\n")
	sb.WriteString("\thttp.HandleFunc(\"/\", func(w http.ResponseWriter, r *http.Request) {\n")
	sb.WriteString("\t\tw.Header().Set(\"Content-Type\", \"application/json\")\n")
	sb.WriteString("\t\tfmt.Fprintf(w, `{\"status\":\"ok\",\"endpoints\":[\n")

	for i, schema := range schemas {
		basePath := "/" + strings.ToLower(schema.StructName) + "s"
		if i > 0 {
			sb.WriteString(",\n")
		}
		sb.WriteString(fmt.Sprintf("\t\t\"%s\",\"%s/get\",\"%s/create\"", basePath, basePath, basePath))
	}

	sb.WriteString("\n\t\t]}`)\n")
	sb.WriteString("\t})\n\n")

	// Start server
	sb.WriteString("\tport := \":8080\"\n")
	sb.WriteString("\tfmt.Printf(\"🌐 Server listening on http://localhost%%s\\n\", port)\n")
	sb.WriteString("\tlog.Fatal(http.ListenAndServe(port, nil))\n")
	sb.WriteString("}\n")

	outputPath := filepath.Join(outputDir, "main.go")
	return os.WriteFile(outputPath, []byte(sb.String()), 0644)
}
