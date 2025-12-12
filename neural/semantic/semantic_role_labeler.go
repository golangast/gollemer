package semantic

import (
	"fmt"
	"strings"
)

// ExtractedRole represents a single semantic role
type ExtractedRole struct {
	RoleType   string
	Value      string
	Confidence float64
}

// SemanticRoleLabeler performs semantic role labeling on queries
type SemanticRoleLabeler struct {
	OperationKeywords map[string]bool
	ResourceTypes     map[string][]string
	ArgumentPatterns  map[string][]string
	StopWords         map[string]bool
}

// NewSemanticRoleLabeler creates a new SRL instance
func NewSemanticRoleLabeler() *SemanticRoleLabeler {
	return &SemanticRoleLabeler{
		OperationKeywords: map[string]bool{
			"create": true, "make": true, "build": true, "generate": true,
			"delete": true, "remove": true, "rm": true, "copy": true, "move": true, "mv": true,
			"modify": true, "update": true, "change": true, "list": true, "show": true,
			"run": true, "execute": true, "start": true, "stop": true, "read": true, "write": true,
		},
		ResourceTypes: map[string][]string{
			"FILESYSTEM": {"file", "folder", "directory", "dir", "path"},
			"USER":       {"user", "admin", "guest"},
			"PROCESS":    {"process", "job", "task", "application", "app"},
			"DATABASE":   {"database", "db", "table", "schema"},
		},
		ArgumentPatterns: map[string][]string{
			"PATH":     {"path", "at", "in", "to", "from"},
			"NAME":     {"named", "called", "as"},
			"MODE":     {"with", "using", "mode"},
			"OWNER":    {"for", "by", "owner"},
			"PROPERTY": {"properties"},
		},
		StopWords: map[string]bool{
			"a": true, "an": true, "the": true, "and": true, "or": true, "is": true, "are": true,
			"to": true, "in": true, "at": true, "for": true, "of": true, "with": true,
		},
	}
}

// ExtractRoles performs SRL on a query string
func (srl *SemanticRoleLabeler) ExtractRoles(queryText string) (map[string]interface{}, error) {
	if queryText == "" {
		return nil, fmt.Errorf("empty query text")
	}

	tokens := strings.Fields(strings.ToLower(queryText))
	result := make(map[string]interface{})

	// Extract operation
	operation := srl.extractOperation(tokens)
	if operation != "" {
		result["operation"] = operation
	}

	// Extract resources
	resources := srl.extractResources(tokens, queryText)
	if len(resources) > 0 {
		result["resources"] = resources
	}

	// Extract arguments
	arguments := srl.extractArguments(tokens, queryText)
	if len(arguments) > 0 {
		result["arguments"] = arguments
	}

	// Extract modifiers
	modifiers := srl.extractModifiers(tokens)
	if len(modifiers) > 0 {
		result["modifiers"] = modifiers
	}

	return result, nil
}

// extractOperation finds the main verb/operation
func (srl *SemanticRoleLabeler) extractOperation(tokens []string) string {
	for _, token := range tokens {
		if srl.OperationKeywords[token] {
			return token
		}
	}
	if len(tokens) > 0 {
		return tokens[0]
	}
	return ""
}

// extractResources identifies entities
func (srl *SemanticRoleLabeler) extractResources(tokens []string, queryText string) []map[string]string {
	resources := make([]map[string]string, 0)

	for i, token := range tokens {
		for resType, keywords := range srl.ResourceTypes {
			for _, keyword := range keywords {
				if token == keyword {
					// Next token is likely the resource name
					var name string
					if i+1 < len(tokens) && !srl.StopWords[tokens[i+1]] {
						name = tokens[i+1]
					} else {
						name = token
					}

					resources = append(resources, map[string]string{
						"type": resType,
						"name": name,
						"path": "./",
					})
					break
				}
			}
		}
	}

	return resources
}

// extractArguments identifies properties and parameters
func (srl *SemanticRoleLabeler) extractArguments(tokens []string, queryText string) []map[string]string {
	arguments := make([]map[string]string, 0)

	for i, token := range tokens {
		for argRole, keywords := range srl.ArgumentPatterns {
			for _, keyword := range keywords {
				if token == keyword && i+1 < len(tokens) {
					value := tokens[i+1]
					arguments = append(arguments, map[string]string{
						"role":  argRole,
						"value": value,
					})
					break
				}
			}
		}
	}

	return arguments
}

// extractModifiers identifies how/when aspects
func (srl *SemanticRoleLabeler) extractModifiers(tokens []string) []string {
	modifiers := make([]string, 0)
	modifierKeywords := map[string]bool{
		"recursively": true, "recursive": true, "forcefully": true, "force": true,
		"quietly": true, "verbose": true, "immediately": true, "later": true,
	}

	for _, token := range tokens {
		if modifierKeywords[token] {
			modifiers = append(modifiers, token)
		}
	}

	return modifiers
}
