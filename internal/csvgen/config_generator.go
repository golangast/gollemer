package csvgen

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/config"
)

// ApplyConfigOverrides updates the schema based on the configuration
func ApplyConfigOverrides(schema *CSVSchema, modelConfig config.DataModel) {
	if modelConfig.StructName != "" {
		schema.StructName = modelConfig.StructName
	}

	for i, field := range schema.Fields {
		if override, ok := modelConfig.FieldOverrides[field.Name]; ok {
			if override.Type != "" {
				schema.Fields[i].GoType = override.Type
			}
			// Note: PrimaryKey handling would require updating CSVField struct or handling it in generation
		}
		// Check by CSV header name as well if different
		if override, ok := modelConfig.FieldOverrides[field.CSVName]; ok {
			if override.Type != "" {
				schema.Fields[i].GoType = override.Type
			}
		}
	}
}

// GenerateFromConfig processes the project configuration
func GenerateFromConfig(cfg *config.ProjectConfig, outputDir string) error {
	// Ensure output directory exists
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	var schemas []*CSVSchema

	for _, model := range cfg.DataModels {
		fmt.Printf("📄 Processing from config: %s\n", model.Source)

		// Parse the CSV file
		schema, err := ParseCSVFile(model.Source)
		if err != nil {
			return fmt.Errorf("failed to parse %s: %w", model.Source, err)
		}

		// Apply overrides
		ApplyConfigOverrides(schema, model)

		// Generate the complete file for this model
		outputFile := filepath.Join(outputDir, strings.ToLower(schema.StructName)+"_handlers.go")
		if err := GenerateCompleteFile(schema, outputFile); err != nil {
			return fmt.Errorf("failed to generate code for %s: %w", model.Source, err)
		}

		fmt.Printf("  ✅ Generated: %s\n", outputFile)
		fmt.Printf("  📦 Struct: %s\n", schema.StructName)

		schemas = append(schemas, schema)
	}

	// Generate main.go
	if len(schemas) > 0 {
		// Use configured database connection string or default
		dbPath := cfg.Project.Database.ConnectionString
		if dbPath == "" {
			dbPath = "orchestrator.db"
		}

		if err := GenerateMainServer(schemas, outputDir, dbPath); err != nil {
			return fmt.Errorf("failed to generate main server: %w", err)
		}
	}

	return nil
}
