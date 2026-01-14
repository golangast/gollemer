package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
)

// Using map[string]interface{} for flexibility, to mirror the Python script's dynamic nature.
type Example map[string]interface{}

func flattenSemanticOutput(semanticOutput map[string]interface{}) (string, error) {
	var parts []string

	// Add operation
	if op, ok := semanticOutput["operation"].(string); ok {
		parts = append(parts, "operation:"+op)
	}

	// Add target_resource fields
	if resource, ok := semanticOutput["target_resource"].(map[string]interface{}); ok {
		if resType, ok := resource["type"].(string); ok {
			// Simplify type (remove :: separators)
			resType = strings.ReplaceAll(resType, "::", "_")
			parts = append(parts, "type:"+resType)
		}
		if name, ok := resource["name"].(string); ok {
			parts = append(parts, "name:"+name)
		}

		// Add properties
		if properties, ok := resource["properties"].(map[string]interface{}); ok {
			for key, value := range properties {
				parts = append(parts, fmt.Sprintf("%s:%v", key, value))
			}
		}
	}

	// Add context fields
	if context, ok := semanticOutput["context"].(map[string]interface{}); ok {
		for key, value := range context {
			parts = append(parts, fmt.Sprintf("%s:%v", key, value))
		}
	}

	return strings.Join(parts, " "), nil
}

func convertFile(inputPath, outputPath string) {
	fmt.Printf("Reading %s...\n", inputPath)
	inputFile, err := ioutil.ReadFile(inputPath)
	if err != nil {
		fmt.Printf("Error: Failed to read input file: %v\n", err)
		os.Exit(1)
	}

	var data []Example
	if err := json.Unmarshal(inputFile, &data); err != nil {
		fmt.Printf("Error: Failed to parse JSON: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Converting %d examples...\n", len(data))
	converted := 0
	for _, example := range data {
		if semanticOutput, ok := example["semantic_output"].(map[string]interface{}); ok {
			flatOutput, err := flattenSemanticOutput(semanticOutput)
			if err != nil {
				query, _ := example["query"].(string)
				fmt.Printf("Warning: Failed to convert example: %s\n", query)
				fmt.Printf("  Error: %v\n", err)
				continue
			}
			example["flat_output"] = flatOutput
			converted++
		}
	}

	fmt.Printf("Successfully converted %d/%d examples\n", converted, len(data))

	fmt.Printf("Writing to %s...\n", outputPath)
	outputFile, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		fmt.Printf("Error: Failed to marshal output JSON: %v\n", err)
		os.Exit(1)
	}

	if err := ioutil.WriteFile(outputPath, outputFile, 0644); err != nil {
		fmt.Printf("Error: Failed to write output file: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Done!")

	// Show example
	fmt.Println("\n--- Example conversion ---")
	if len(data) > 0 {
		example := data[0]
		query, _ := example["query"].(string)
		flatOutput, _ := example["flat_output"].(string)
		fmt.Printf("Query: %s\n", query)
		fmt.Printf("Flat output: %s\n", flatOutput)
	}
}

func main() {
	// Get the directory of the executable to build the paths relative to the project root.
	// This mimics the behavior of the Python script's Path(__file__).parent.
	// For a `go run` command, this is less straightforward. We will assume the script
	// is run from the project root.
	projectDir, err := os.Getwd()
	if err != nil {
		fmt.Printf("Error getting current directory: %v\n", err)
		os.Exit(1)
	}

	inputFile := filepath.Join(projectDir, "trainingdata", "semantic_output_data.json")
	outputFile := filepath.Join(projectDir, "trainingdata", "semantic_output_data_flat.json")

	if _, err := os.Stat(inputFile); os.IsNotExist(err) {
		fmt.Printf("Error: Input file not found: %s\n", inputFile)
		os.Exit(1)
	}

	convertFile(inputFile, outputFile)
}
