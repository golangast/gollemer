package llm

import (
	"encoding/json"
)

// IntentDataLayer represents the structured state of the user's intent and data.
// It tracks the intent, parameters, and what is currently missing.
type IntentDataLayer struct {
	Intent     string         `json:"intent"`
	Confidence float64        `json:"confidence"`
	Parameters map[string]any `json:"parameters"`
	Missing    []string       `json:"missing"`
	IsComplete bool           `json:"is_complete"`
}

// ToJSON returns the JSON representation of the data layer.
func (d *IntentDataLayer) ToJSON() (string, error) {
	b, err := json.MarshalIndent(d, "", "  ")
	return string(b), err
}

// IntentSchema defines the expected structure for a specific intent.
type IntentSchema struct {
	Name     string
	Fields   map[string]string // Field name -> Type (e.g., "string", "int")
	Required []string
}

// MoEClient defines the interface for the existing MoE and NLP model capabilities.
// This allows the resolver to use the "nlp that is there" to predict data.
type MoEClient interface {
	// PredictIntent uses the MoE model to classify the intent from input.
	PredictIntent(input string) (string, float64)
	// ExtractEntities uses the NER/NLP pipeline to find parameters for the given intent.
	ExtractEntities(input string, intent string) map[string]any
}

// HybridIntentResolver implements the recursive intent resolution using MoE.
type HybridIntentResolver struct {
	Schemas  map[string]IntentSchema
	MoE      MoEClient
	MaxDepth int
}

// NewHybridIntentResolver initializes the resolver with supported schemas and the MoE client.
func NewHybridIntentResolver(moe MoEClient) *HybridIntentResolver {
	return &HybridIntentResolver{
		MoE:      moe,
		MaxDepth: 10,
		Schemas: map[string]IntentSchema{
			"create_webserver": {
				Name:     "create_webserver",
				Fields:   map[string]string{"name": "string", "port": "int"},
				Required: []string{"name"},
			},
			"create_handler": {
				Name:     "create_handler",
				Fields:   map[string]string{"name": "string", "url": "string", "method": "string"},
				Required: []string{"name", "url"},
			},
			"create_database": {
				Name:     "create_database",
				Fields:   map[string]string{"name": "string", "tables": "list"},
				Required: []string{"name"},
			},
			"create_file": {
				Name:     "create_file",
				Fields:   map[string]string{"name": "string", "path": "string"},
				Required: []string{"name"},
			},
			"create_folder": {
				Name:     "create_folder",
				Fields:   map[string]string{"name": "string", "path": "string"},
				Required: []string{"name"},
			},
			"create_page": {
				Name:     "create_page",
				Fields:   map[string]string{"name": "string", "path": "string"},
				Required: []string{"name"},
			},
			"create_form": {
				Name:     "create_form",
				Fields:   map[string]string{"name": "string", "source": "string"},
				Required: []string{"name"},
			},
			"create_structure": {
				Name:     "create_structure",
				Fields:   map[string]string{"name": "string", "fields": "list"},
				Required: []string{"name"},
			},
			"status_query": {
				Name:   "status_query",
				Fields: map[string]string{},
			},
			"identity_query": {
				Name:   "identity_query",
				Fields: map[string]string{},
			},
			"greeting": {
				Name:   "greeting",
				Fields: map[string]string{},
			},
			"help_command": {
				Name:     "help_command",
				Fields:   map[string]string{"command": "string"},
				Required: []string{"command"},
			},
			"chat_response": {
				Name:   "chat_response",
				Fields: map[string]string{"response": "string"},
			},
		},
	}
}

// Resolve recursively fills the data layer using the MoE model until complete or stable.
func (r *HybridIntentResolver) Resolve(input string, current *IntentDataLayer) *IntentDataLayer {
	if current == nil {
		current = &IntentDataLayer{
			Parameters: make(map[string]any),
		}
	}
	return r.resolveRecursive(input, current, 0)
}

func (r *HybridIntentResolver) resolveRecursive(input string, layer *IntentDataLayer, depth int) *IntentDataLayer {
	if depth >= r.MaxDepth {
		return layer
	}

	// 1. Predict Intent using MoE if not already known
	if layer.Intent == "" {
		intent, conf := r.MoE.PredictIntent(input)
		if intent != "" {
			layer.Intent = intent
			layer.Confidence = conf
		}
	}

	// If intent is unknown, we cannot validate schema or extract specific entities effectively
	if layer.Intent == "" {
		return layer
	}

	// 2. Extract Entities using MoE/NLP
	// We pass the intent to the MoE client to allow for context-aware extraction
	extracted := r.MoE.ExtractEntities(input, layer.Intent)
	stateChanged := false

	// Update parameters
	for k, v := range extracted {
		if _, exists := layer.Parameters[k]; !exists {
			// Verify field exists in schema
			if schema, ok := r.Schemas[layer.Intent]; ok {
				if _, valid := schema.Fields[k]; valid {
					layer.Parameters[k] = v
					stateChanged = true
				}
			}
		}
	}

	// 3. Validate against Schema to determine what is missing
	if schema, ok := r.Schemas[layer.Intent]; ok {
		layer.Missing = []string{}
		allRequiredPresent := true
		for _, req := range schema.Required {
			if _, exists := layer.Parameters[req]; !exists {
				layer.Missing = append(layer.Missing, req)
				allRequiredPresent = false
			}
		}
		layer.IsComplete = allRequiredPresent
	}

	// 4. Recursion
	// If state changed (new data found), recurse to ensure stability or trigger dependent extractions.
	if stateChanged && !layer.IsComplete {
		return r.resolveRecursive(input, layer, depth+1)
	}

	return layer
}
