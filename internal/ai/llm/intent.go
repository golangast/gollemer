package llm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"

	neuralnn "github.com/golangast/gollemer/internal/ai/neural/nn"
)

const kbFilename = "data/knowledge.json"

// Intent represents the state of the command understanding.
type Intent struct {
	RawInput        string
	Command         string
	ObjectType      string
	ObjectTypeParts []string
	Params          map[string]string
}

type ModelConfig struct {
	Word2VecPath      string `json:"word2vec_path"`
	MoEPath           string `json:"moe_path"`
	QueryVocabPath    string `json:"query_vocab_path"`
	SemanticVocabPath string `json:"semantic_vocab_path"`
	NERPath           string `json:"ner_path"`
}

// KnowledgeBase acts as the memory for the session.
type KnowledgeBase struct {
	KnownCommands map[string]bool `json:"known_commands"`
	KnownObjects  map[string]bool `json:"known_objects"`
	StopWords     map[string]bool `json:"stop_words"`
	LearningPath  string          `json:"learning_path"`
	FirstRun      bool            `json:"first_run"`
	ModelConfig   ModelConfig     `json:"model_config"`
}

var paramTriggers = map[string]string{
	"named": "name", "called": "name", "port": "port", "at": "port", "url": "url", "on": "url",
	"method": "method", "using": "method", "source": "source", "from": "source", "path": "path",
	"in": "path", "into": "path", "tables": "tables", "fields": "fields", "columns": "fields",
}

func NewKnowledgeBase() *KnowledgeBase {
	return &KnowledgeBase{
		KnownCommands: map[string]bool{
			"create": true, "make": true, "generate": true, "add": true, "put": true, "copy": true,
			"delete": true, "remove": true,
			"list": true, "ls": true, "show": true,
			"go": true, "cd": true, "change": true, "move": true,
			"run": true, "start": true,
			"stop":   true,
			"update": true,
			"verify": true, "check": true, "test": true,
			"cat": true, "read": true,
			"tree": true,
			"grep": true, "search": true,
			"history": true,
			"help":    true,
			"pwd":     true,
		},
		KnownObjects: map[string]bool{
			"user": true, "file": true, "database": true, "folder": true, "directory": true,
			"webserver": true, "handler": true, "structure": true, "form": true,
		},
		StopWords: map[string]bool{
			"a": true, "an": true, "the": true, "please": true, "this": true,
			"me": true, "my": true, "i": true, "new": true, "to": true, "for": true, "and": true, "it": true,
			"how": true, "are": true, "is": true, "was": true, "were": true, "you": true, "your": true,
			"in": true, "into": true,
			"what": true, "when": true, "where": true, "why": true, "who": true, "which": true,
		},
		FirstRun: true,
		ModelConfig: ModelConfig{
			Word2VecPath:      "data/models/gob_models/word2vec_model.gob",
			MoEPath:           "data/models/gob_models/moe_classification_model_best.gob",
			QueryVocabPath:    "data/models/gob_models/query_vocabulary.gob",
			SemanticVocabPath: "data/models/gob_models/semantic_output_vocabulary.gob",
			NERPath:           "data/models/gob_models/ner_model.gob",
		},
	}
}

func LoadKnowledgeBase() *KnowledgeBase {
	data, err := os.ReadFile(kbFilename)
	if os.IsNotExist(err) {
		return NewKnowledgeBase()
	}
	var kb KnowledgeBase
	if err := json.Unmarshal(data, &kb); err != nil {
		return NewKnowledgeBase()
	}

	// Ensure built-in commands and stop words are always present
	defaults := NewKnowledgeBase()
	if kb.KnownCommands == nil {
		kb.KnownCommands = make(map[string]bool)
	}
	for k := range defaults.KnownCommands {
		kb.KnownCommands[k] = true
	}
	if kb.StopWords == nil {
		kb.StopWords = make(map[string]bool)
	}
	for k := range defaults.StopWords {
		kb.StopWords[k] = true
	}

	if kb.ModelConfig.Word2VecPath == "" {
		kb.ModelConfig = defaults.ModelConfig
	}

	return &kb
}

func (kb *KnowledgeBase) Save() {
	data, _ := json.MarshalIndent(kb, "", "  ")
	_ = os.WriteFile(kbFilename, data, 0644)
}

// resolveIntent attempts to find the missing object type in the remaining words.
func resolveIntent(r *bufio.Reader, intent Intent, kb *KnowledgeBase) Intent {
	parts := strings.Fields(intent.RawInput)
	var candidate string

	consumed := make(map[int]bool)
	for i := range parts {
		if _, isTrigger := paramTriggers[strings.ToLower(parts[i])]; isTrigger {
			consumed[i] = true
			if i+1 < len(parts) {
				consumed[i+1] = true
			}
		}
	}

	for i, word := range parts {
		if consumed[i] {
			continue
		}
		lower := strings.ToLower(word)
		if lower == intent.Command || kb.KnownCommands[lower] {
			continue
		}
		if kb.StopWords[lower] || lower == "named" || lower == "called" {
			continue
		}
		candidate = lower
		break
	}

	if candidate != "" {
		fmt.Println("   ... Attempting recursive inference ...")
		fmt.Printf("   [INFERENCE] I detected the unknown token '%s'.\n", candidate)
		fmt.Printf("   [CONFIRMATION] Did you mean to create a '%s'? (y/n): ", candidate)
		resp, _ := r.ReadString('\n')
		resp = strings.TrimSpace(strings.ToLower(resp))
		if resp == "y" || resp == "yes" {
			intent.ObjectType = candidate
			intent.ObjectTypeParts = append(intent.ObjectTypeParts, candidate)
			kb.KnownObjects[candidate] = true
			fmt.Printf("   [LEARNING] Knowledge updated: '%s' is now a known object type.\n", candidate)
			kb.Save()

			// If we haven't identified a command yet, assume "create"
			if intent.Command == "" {
				intent.Command = "create"
			}
			return intent
		}
	}
	return intent
}

// parse identifies commands, known objects, and parameters.
func parse(input string, kb *KnowledgeBase) Intent {
	parts := strings.Fields(input)
	intent := Intent{
		RawInput:        input,
		ObjectTypeParts: []string{},
		Params:          make(map[string]string),
	}

	consumed := make(map[int]bool)

	// 1. Extract Parameters first (e.g., "named login")
	for i := 0; i < len(parts); i++ {
		word := strings.ToLower(parts[i])
		if paramKey, isTrigger := paramTriggers[word]; isTrigger {
			if i+1 < len(parts) {
				value := parts[i+1]
				nextIndex := i + 1

				// Skip noise words like "the", "a", "an", "folder", "directory"
				for nextIndex < len(parts) {
					v := strings.ToLower(parts[nextIndex])
					if v == "the" || v == "a" || v == "an" || v == "folder" || v == "directory" {
						consumed[nextIndex] = true
						nextIndex++
						if nextIndex < len(parts) {
							value = parts[nextIndex]
						}
						continue
					}
					break
				}

				if strings.ToLower(value) == "it" {
					continue
				}
				intent.Params[paramKey] = value
				consumed[i] = true
				consumed[nextIndex] = true
				i = nextIndex
			}
		}
	}

	// 2. Extract Command and ObjectType from remaining words
	for i, word := range parts {
		if consumed[i] {
			continue
		}
		lower := strings.ToLower(word)

		if intent.Command == "" && kb.KnownCommands[lower] {
			switch lower {
			case "make", "generate", "add", "put", "copy":
				lower = "create"
			case "ls", "show":
				lower = "list"
			case "cd", "change":
				lower = "go"
			}
			intent.Command = lower
			continue
		}

		if intent.ObjectType == "" && kb.KnownObjects[lower] {
			intent.ObjectType = lower
			intent.ObjectTypeParts = append(intent.ObjectTypeParts, lower)
			continue
		}

		// Capture the first unrecognised, non-stopword token as the "name" param
		// (e.g. "create folder news" -> name="news").
		if intent.Params["name"] == "" && intent.ObjectType != "" &&
			!kb.StopWords[lower] && !kb.KnownCommands[lower] && !kb.KnownObjects[lower] {
			intent.Params["name"] = word
		}
	}
	return intent
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, magA, magB float64
	for i := range a {
		dot += a[i] * b[i]
		magA += a[i] * a[i]
		magB += b[i] * b[i]
	}
	if magA == 0 || magB == 0 {
		return 0
	}
	return dot / (math.Sqrt(magA) * math.Sqrt(magB))
}

func inspectStruct(v any, indent string) {
	val := reflect.ValueOf(v)
	if !val.IsValid() {
		fmt.Println(indent + "<nil>")
		return
	}
	if val.Kind() == reflect.Pointer || val.Kind() == reflect.Interface {
		if val.IsNil() {
			fmt.Println(indent + "<nil>")
			return
		}
		val = val.Elem()
	}
	if val.Kind() != reflect.Struct {
		fmt.Printf("%s%v\n", indent, val)
		return
	}

	typ := val.Type()
	for i := 0; i < val.NumField(); i++ {
		field := val.Field(i)
		fieldType := typ.Field(i)

		if fieldType.PkgPath != "" {
			continue // Skip unexported fields
		}

		fmt.Printf("%s%s (%s): ", indent, fieldType.Name, fieldType.Type)

		if field.Kind() == reflect.Slice {
			fmt.Printf("Slice with %d elements\n", field.Len())
			if field.Len() > 0 && field.Type().Elem().Kind() == reflect.Float64 {
				count := min(field.Len(), 5)
				fmt.Printf("%s  Sample: %v...\n", indent, field.Slice(0, count).Interface())
			}
		} else if field.Kind() == reflect.Struct || (field.Kind() == reflect.Pointer && field.Elem().Kind() == reflect.Struct) {
			fmt.Println("")
			if len(indent) < 10 {
				inspectStruct(field.Interface(), indent+"  ")
			} else {
				fmt.Println(indent + "  ...")
			}
		} else {
			fmt.Printf("%v\n", field)
		}
	}
}

func findAndVisualizeAttention(v any) {
	val := reflect.ValueOf(v)
	if !val.IsValid() {
		return
	}
	if val.Kind() == reflect.Pointer || val.Kind() == reflect.Interface {
		if val.IsNil() {
			return
		}
		val = val.Elem()
	}

	if val.Type() == reflect.TypeOf(neuralnn.MultiHeadAttention{}) {
		mha := val.Interface().(neuralnn.MultiHeadAttention)
		fmt.Printf("\nFound MultiHeadAttention Layer:\n")
		fmt.Printf("  Heads: %d, Model Dim: %d\n", mha.NumHeads, mha.DimModel)
		if mha.QueryLinear != nil {
			fmt.Printf("  Query Weights Shape: %v\n", mha.QueryLinear.Weights.Shape)
		}
		return
	}

	if val.Type() == reflect.TypeOf(neuralnn.MultiHeadCrossAttention{}) {
		mhca := val.Interface().(neuralnn.MultiHeadCrossAttention)
		fmt.Printf("\nFound MultiHeadCrossAttention Layer:\n")
		fmt.Printf("  Q Heads: %d, KV Heads: %d, Model Dim: %d\n", mhca.NumQHeads, mhca.NumKVHeads, mhca.DimModel)
		return
	}

	if val.Kind() == reflect.Struct {
		for i := 0; i < val.NumField(); i++ {
			field := val.Field(i)
			if field.CanInterface() {
				findAndVisualizeAttention(field.Interface())
			}
		}
	} else if val.Kind() == reflect.Slice {
		for i := 0; i < val.Len(); i++ {
			findAndVisualizeAttention(val.Index(i).Interface())
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
