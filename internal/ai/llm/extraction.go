package llm

import (
	"regexp"
	"strings"
)

// Entity represents a piece of information extracted from a user's natural language input.
type Entity struct {
	Value string
	Type  string
}

// pathRegex captures Unix-style paths: ~/configs/settings.yaml, /var/log/syslog, ./main.go, data_v1.csv
var pathRegex = regexp.MustCompile(`(?i)(?:~|/|\./|[a-zA-Z0-9_-]+/)[a-zA-Z0-9._/-]+\.[a-z0-9]+`)

// ExtractUnixPath performs a specialized regex search to find file paths without trailing punctuation.
func ExtractUnixPath(input string) string {
	return pathRegex.FindString(input)
}

// ExtractEntities is an intent-aware heuristic extractor.
// It uses the predicted Intent to narrow down which "extractors" or patterns to run,
// preventing false positives cross-domain (e.g. file paths vs CPU labels).
func ExtractEntities(intent string, input string) []Entity {
	var entities []Entity

	switch intent {
	case "file_manage":
		// Look for common file patterns (extensions, paths, or quoted strings)
		fileRegex := regexp.MustCompile(`[a-zA-Z0-9._/-]+\.[a-z]+|(?:"[^"]+")`)
		matches := fileRegex.FindAllString(input, -1)
		for _, m := range matches {
			entities = append(entities, Entity{Value: m, Type: "target"})
		}

	case "sys_monitor":
		// Look for specific system keywords
		resources := []string{"cpu", "ram", "memory", "disk", "network", "load"}
		for _, r := range resources {
			if strings.Contains(strings.ToLower(input), r) {
				entities = append(entities, Entity{Value: r, Type: "resource"})
			}
		}
	}

	return entities
}
