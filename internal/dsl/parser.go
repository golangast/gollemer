package dsl

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/golangast/gollemer/internal/config"
)

// ParseDSL parses a series of DSL commands and returns a ProjectConfig
func ParseDSL(commands []string) (*config.ProjectConfig, error) {
	cfg := &config.ProjectConfig{
		Project: config.ProjectDetails{
			Name: "GeneratedProject",
			Database: config.DatabaseConfig{
				Type:             "sqlite",
				ConnectionString: "orchestrator.db",
			},
		},
		DataModels: []config.DataModel{},
	}

	modelMap := make(map[string]*config.DataModel)

	// Regex patterns
	modelRegex := regexp.MustCompile(`(?i)^MODEL\s+(\w+)\s+FROM\s+([^\s]+)(?:\s+AS\s+(\w+))?`)
	endpointRegex := regexp.MustCompile(`(?i)^ENDPOINT\s+([^\s]+)\s+ADD\s+CRUD(?:\s+WITH\s+AUTH)?`)

	for _, cmd := range commands {
		cmd = strings.TrimSpace(cmd)
		if cmd == "" || strings.HasPrefix(cmd, "#") {
			continue
		}

		if matches := modelRegex.FindStringSubmatch(cmd); matches != nil {
			structName := matches[1]
			source := matches[2]
			// tableName := matches[3] // Not used in current config, but parsed

			model := config.DataModel{
				Source:         source,
				StructName:     structName,
				Endpoints:      []string{},
				FieldOverrides: make(map[string]config.FieldOverride),
			}
			cfg.DataModels = append(cfg.DataModels, model)
			// Keep pointer to last added model for subsequent commands if needed,
			// but ENDPOINT command might need to infer which model it applies to based on path
			// For simplicity, we'll assume ENDPOINT applies to the last defined model or we need to match path
			modelMap[structName] = &cfg.DataModels[len(cfg.DataModels)-1]

		} else if matches := endpointRegex.FindStringSubmatch(cmd); matches != nil {
			path := matches[1]
			// Try to guess which model this endpoint belongs to
			// e.g., /api/v1/customer -> Customer
			var targetModel *config.DataModel
			for i := range cfg.DataModels {
				model := &cfg.DataModels[i]
				if strings.Contains(strings.ToLower(path), strings.ToLower(model.StructName)) {
					targetModel = model
					break
				}
			}

			if targetModel != nil {
				targetModel.Endpoints = append(targetModel.Endpoints, "GET "+path)
				targetModel.Endpoints = append(targetModel.Endpoints, "POST "+path+"/create")
				// Add other CRUD endpoints as needed
			} else {
				// If no model found, maybe just add to the last one?
				if len(cfg.DataModels) > 0 {
					model := &cfg.DataModels[len(cfg.DataModels)-1]
					model.Endpoints = append(model.Endpoints, "GET "+path)
				}
			}
		} else {
			return nil, fmt.Errorf("unknown command: %s", cmd)
		}
	}

	return cfg, nil
}
