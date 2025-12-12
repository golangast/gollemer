package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// LoadConfig reads and parses a JSON configuration file
func LoadConfig(filePath string) (*ProjectConfig, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	var config ProjectConfig
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&config); err != nil {
		return nil, fmt.Errorf("failed to decode config file: %w", err)
	}

	return &config, nil
}
