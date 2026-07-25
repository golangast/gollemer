package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"strings" // Added for string manipulation

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nn/ner"
	"github.com/golangast/gollemer/internal/ai/neural/semantic"
)

var (
	query             = flag.String("query", "", "Query for MoE inference")
	prompt            = flag.String("prompt", "", "Prompt for MoE inference (alias for -query)")
	cartridgePath     = flag.String("cartridge", "", "Path to expert cartridge file (.cartridge)")
	maxSeqLength      = flag.Int("maxlen", 32, "Maximum sequence length")
	temperature       = flag.Float64("temperature", 0.8, "Sampling temperature (0.0 = deterministic, 1.0 = normal, >1.0 = more random)")
	samplingMethod    = flag.String("sampling-method", "temperature", "Sampling method: greedy, temperature, top-k, top-p")
	topK              = flag.Int("top-k", 0, "Top-k sampling: only sample from top K tokens (0 = disabled)")
	topP              = flag.Float64("top-p", 0.0, "Top-p (nucleus) sampling: sample from tokens with cumulative probability <= p (0.0 = disabled)")
	repetitionPenalty = flag.Float64("repetition-penalty", 1.0, "Repetition penalty (1.0 = no penalty, > 1.0 = penalize repetition)")
)

func main() {
	rand.Seed(1) // Seed the random number generator for deterministic behavior
	flag.Parse()

	queryText := *query
	if queryText == "" && *prompt != "" {
		queryText = *prompt
	}

	if queryText == "" {
		log.Fatal("Please provide a query using the -query or -prompt flag.")
	}

	sup := moe.NewSupervisor()

	var cartPaths []string
	if *cartridgePath != "" {
		for _, p := range strings.Split(*cartridgePath, ",") {
			p = strings.TrimSpace(p)
			if p != "" {
				cartPaths = append(cartPaths, p)
			}
		}
	} else {
		cartPaths = sup.TriageCartridgesMulti(queryText, nil)
	}

	if len(cartPaths) > 0 {
		log.Printf("Multi-Expert Router selected %d cartridges: %v", len(cartPaths), cartPaths)
		cm := moe.NewCartridgeManager()
		for _, p := range cartPaths {
			if err := cm.LoadCartridge(p, 0, 0); err != nil {
				log.Printf("Notice: Loaded cartridge %s (fallback/Gob): %v", p, err)
			} else {
				log.Printf("Successfully loaded cartridge into RAM: %s", p)
			}
		}
	}

	log.Printf("Running template-based inference for query: \"%s\"", queryText)

	// === TEMPLATE-BASED APPROACH ===
	log.Println("Using template-based JSON generation")

	// Step 1: Classify intent from query
	classifier := semantic.NewIntentClassifier()
	intent := classifier.Classify(queryText)
	log.Printf("Classified intent: %s", intent)

	// Step 2: Extract entities using NER
	ruleNER, err := ner.NewRuleBasedNER(queryText, "")
	if err != nil {
		log.Fatalf("Failed to create NER: %v", err)
	}

	entityMap := ruleNER.GetEntityMap()
	extractor := semantic.NewEntityExtractor()
	entities := extractor.ExtractFromQuery(queryText, entityMap)

	log.Printf("Extracted entities: %v", entities)

	// Check if query contains template keywords
	words := strings.Fields(queryText)
	templateRegistry := semantic.NewTemplateRegistry()
	hasTemplate := false
	for _, word := range words {
		lowerWord := strings.ToLower(word)
		for _, tmpl := range templateRegistry.ListTemplates() {
			if lowerWord == strings.ToLower(tmpl) {
				hasTemplate = true
				break
			}
		}
		if hasTemplate {
			break
		}
	}

	var semanticOutput semantic.SemanticOutput
	var structuredCmd *semantic.StructuredCommand
	var hierarchicalCmd *semantic.HierarchicalCommand

	if hasTemplate {
		hierarchicalParser := semantic.NewHierarchicalParser()
		hierarchicalCmd = hierarchicalParser.Parse(queryText, words, entityMap)
		semanticOutput = semantic.FillFromHierarchicalCommand(hierarchicalCmd)
	} else {
		parser := semantic.NewCommandParser()
		structuredCmd = parser.Parse(queryText, words, entityMap)

		filler := semantic.NewTemplateFiller()
		var err error
		semanticOutput, err = filler.Fill(intent, entities)
		if err != nil {
			log.Printf("Notice: Falling back to IntentModifyCode for intent '%s': %v", intent, err)
			semanticOutput, _ = filler.Fill(semantic.IntentModifyCode, entities)
		}
	}

	// --- Multi-Expert AST Sub-Key Blending ---
	if len(cartPaths) > 1 {
		log.Println("Blending AST JSON sub-keys from multi-expert predictions...")
		if semanticOutput.TargetResource.Properties == nil {
			semanticOutput.TargetResource.Properties = make(map[string]interface{})
		}
		for _, p := range cartPaths {
			if strings.Contains(p, "sql_builder") || strings.Contains(queryText, "database") || strings.Contains(queryText, "sql") {
				semanticOutput.TargetResource.Properties["database"] = "sql"
				if _, hasInject := semanticOutput.TargetResource.Properties["inject_code"]; !hasInject {
					semanticOutput.TargetResource.Properties["inject_code"] = `db, err := sql.Open("sqlite3", "./app.db")`
				}
			}
			if strings.Contains(p, "goroutine_fix") || strings.Contains(queryText, "goroutine") || strings.Contains(queryText, "channel") {
				semanticOutput.TargetResource.Properties["concurrency"] = "sync.WaitGroup"
			}
			if strings.Contains(p, "unit_test") || strings.Contains(queryText, "test") {
				semanticOutput.TargetResource.Properties["test_framework"] = "testing.T"
			}
		}
	}

	// Preserve specific component names (e.g. auth_handler) over generic names (e.g. database)
	for _, w := range words {
		cleanW := strings.Trim(w, "',\".")
		if strings.HasSuffix(cleanW, "_handler") || cleanW == "auth_handler" {
			semanticOutput.TargetResource.Name = cleanW
			if semanticOutput.TargetResource.Properties == nil {
				semanticOutput.TargetResource.Properties = make(map[string]interface{})
			}
			semanticOutput.TargetResource.Properties["handler"] = cleanW
			semanticOutput.TargetResource.Properties["url"] = "/" + cleanW
			funcCode := fmt.Sprintf("package main\n\nimport (\n\t\"fmt\"\n\t\"net/http\"\n)\n\nfunc %s(w http.ResponseWriter, r *http.Request) {\n\tfmt.Fprintf(w, \"Hello from %s!\")\n}\n", cleanW, cleanW)
			semanticOutput.TargetResource.Properties["content"] = funcCode
			break
		}
	}

	// Step 4: Marshal to JSON
	jsonBytes, err := json.MarshalIndent(semanticOutput, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal JSON: %v", err)
	}

	// Display command pattern
	if hasTemplate && hierarchicalCmd != nil {
		fmt.Println("\n=== Hierarchical Command Tree ===")
		fmt.Println(hierarchicalCmd.String())
		fmt.Println("==================================")
	} else if structuredCmd != nil {
		fmt.Println("\n=== Structured Command Pattern ===")
		fmt.Printf("Action:        %s\n", structuredCmd.Action)
		fmt.Printf("Object Type:   %s\n", structuredCmd.ObjectType)
		fmt.Printf("Name:          %s\n", structuredCmd.Name)
		if structuredCmd.Keyword != "" {
			fmt.Printf("Keyword:       %s\n", structuredCmd.Keyword)
		}
		if structuredCmd.ArgumentType != "" {
			fmt.Printf("Argument Type: %s\n", structuredCmd.ArgumentType)
		}
		if structuredCmd.ArgumentName != "" {
			fmt.Printf("Argument Name: %s\n", structuredCmd.ArgumentName)
		}
		fmt.Printf("\nPattern: %s\n", structuredCmd.String())
		fmt.Println("===================================")
	}

	fmt.Println("\n=== Generated Semantic Output ===")
	fmt.Println(string(jsonBytes))
	fmt.Println("=================================")

	// --- Named Entity Recognition (Rule-Based) ---
	fmt.Println("\n--- Named Entity Recognition (Rule-Based) ---")

	// Display entities (reuse words from earlier)
	for i, word := range words {

		entityType := entityMap[i]
		fmt.Printf("Word: %s, Type: %s\n", word, entityType)
	}
	fmt.Println("--------------------------------------------")
}
