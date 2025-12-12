package config

// ProjectConfig represents the top-level configuration for the project
type ProjectConfig struct {
	Project    ProjectDetails `json:"project"`
	DataModels []DataModel    `json:"data_models"`
}

// ProjectDetails holds general project information
type ProjectDetails struct {
	Name     string         `json:"name"`
	Database DatabaseConfig `json:"database"`
}

// DatabaseConfig holds database connection details
type DatabaseConfig struct {
	Type             string `json:"type"` // e.g., "sqlite", "postgres"
	ConnectionString string `json:"connection_string"`
}

// DataModel represents a single CSV source and its generation rules
type DataModel struct {
	Source         string                   `json:"source"`      // e.g., "customers.csv"
	StructName     string                   `json:"struct_name"` // e.g., "Customer"
	Endpoints      []string                 `json:"endpoints"`   // e.g., "GET /api/v1/customers"
	FieldOverrides map[string]FieldOverride `json:"field_overrides"`
}

// FieldOverride allows customizing specific fields
type FieldOverride struct {
	Type       string `json:"type"`        // e.g., "time.Time", "int64"
	PrimaryKey bool   `json:"primary_key"` // e.g., true
	Ignored    bool   `json:"ignored"`     // e.g., true to skip generation
}
