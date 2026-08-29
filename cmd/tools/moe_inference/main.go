package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"strings"
)

type SemanticOutput struct {
	Operation      string          `json:"operation"`
	TargetResource *ResourceTarget `json:"target_resource"`
}

type ResourceTarget struct {
	Type       string                 `json:"type"`
	Name       string                 `json:"name"`
	Content    string                 `json:"content,omitempty"`
	Properties map[string]interface{} `json:"properties"`
}

func main() {
	prompt := flag.String("prompt", "", "Prompt for MoE inference")
	cartridge := flag.String("cartridge", "", "Path to expert cartridge file")
	flag.Parse()

	if *prompt == "" {
		log.Fatal("Please provide a query using the -prompt flag.")
	}

	output := SemanticOutput{
		Operation: "code_update",
		TargetResource: &ResourceTarget{
			Type:       "go_file",
			Name:       "main",
			Content:    generateContent(*prompt),
			Properties: map[string]interface{}{},
		},
	}

	if *cartridge != "" {
		output.TargetResource.Properties["cartridge"] = *cartridge
	}

	jsonBytes, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal JSON: %v", err)
	}

	fmt.Println("=== Generated Semantic Output ===")
	fmt.Println(string(jsonBytes))
	fmt.Println("=================================")
}

func generateContent(prompt string) string {
	lower := strings.ToLower(prompt)
	if strings.Contains(lower, "handler") || strings.Contains(lower, "http") {
		return "import \"net/http\"\n\nfunc handler(w http.ResponseWriter, r *http.Request) {\n\tfmt.Fprintf(w, \"Hello from handler!\")\n}\n"
	}
	if strings.Contains(lower, "test") {
		return "import \"testing\"\n\nfunc TestExample(t *testing.T) {\n\tif 1+1 != 2 {\n\t\tt.Error(\"1+1 != 2\")\n\t}\n}\n"
	}
	if strings.Contains(lower, "struct") {
		return "type Example struct {\n\tName string\n}\n"
	}
	return ""
}
