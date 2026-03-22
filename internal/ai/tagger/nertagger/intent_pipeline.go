package nertagger

import (
	"log"
	"strings"
)

// ========== INTEGRATED INTENT PIPELINE ==========

// IntentPipeline orchestrates the full intent processing workflow
type IntentPipeline struct {
	text               string
	segmenter          *IntentSegmenter
	verbCounter        *VerbCounter
	postProcessor      *EntityPostProcessor
	converter          *SegmentToTaskConverter
	prepositionMatcher *PrepositionMatcher
}

// NewIntentPipeline creates a new intent processing pipeline
func NewIntentPipeline(text string, tokens []string, posTags []string) *IntentPipeline {
	return &IntentPipeline{
		text:               text,
		segmenter:          NewIntentSegmenter(text),
		verbCounter:        NewVerbCounter(),
		postProcessor:      NewEntityPostProcessor(tokens, posTags),
		prepositionMatcher: NewPrepositionMatcher(),
	}
}

// Process orchestrates the full intent analysis pipeline
func (ip *IntentPipeline) Process() (*TaskGraph, error) {
	// Step 1: Segment multi-intent text
	segments := ip.segmenter.Segment()
	log.Printf("DEBUG: Segmented %d intent(s)", len(segments))

	// Step 2: Enrich segments with entities and parameters
	enrichedSegments := ip.enrichSegments(segments)

	// Step 3: Create converter and convert to task graph
	ip.converter = NewSegmentToTaskConverter(enrichedSegments)
	taskGraph := ip.converter.Convert()

	// Step 4: Sort tasks topologically
	if err := taskGraph.SortTopologically(); err != nil {
		log.Printf("ERROR: Failed to sort tasks: %v", err)
		return nil, err
	}

	log.Printf("DEBUG: Generated task graph with %d task(s)", len(taskGraph.Tasks))
	return taskGraph, nil
}

// enrichSegments enhances segments with extracted entities
func (ip *IntentPipeline) enrichSegments(segments []IntentSegment) []IntentSegment {
	registry := NewActionRegistry()

	for i := range segments {
		// Extract primary entity
		primary := ip.postProcessor.ExtractPrimaryEntity()
		if primary != "" {
			segments[i].Entities[primary] = "IDENTIFIER"
		}

		// Extract preposition-based relationships
		prepPairs := ip.prepositionMatcher.ExtractPrepositionPairs(segments[i].RawText)

		// Assign to parameters based on action type
		for _, action := range segments[i].Actions {
			actionType := registry.GetActionType(action)
			segments[i].Parameters["action_type"] = actionType

			switch actionType {
			case "CREATE":
				if primary != "" {
					segments[i].Parameters["name"] = primary
				}
				if destinations, ok := prepPairs["destination"]; ok && len(destinations) > 0 {
					segments[i].Parameters["path"] = destinations[0]
				}

			case "MOVE":
				if sources, ok := prepPairs["source"]; ok && len(sources) > 0 {
					segments[i].Parameters["source"] = sources[0]
				}
				if destinations, ok := prepPairs["destination"]; ok && len(destinations) > 0 {
					segments[i].Parameters["destination"] = destinations[0]
				} else if primary != "" {
					segments[i].Parameters["destination"] = primary
				}

			case "DELETE":
				if primary != "" {
					segments[i].Parameters["target"] = primary
				}

			case "UPDATE":
				if primary != "" {
					segments[i].Parameters["target"] = primary
				}
				if attributes, ok := prepPairs["attribute"]; ok && len(attributes) > 0 {
					segments[i].Parameters["attribute"] = attributes[0]
				}
			}
		}

		// Clean parameters with stop-word filter
		segments[i].Parameters = ip.postProcessor.ProcessParameters(segments[i].Parameters)
	}

	return segments
}

// ========== QUICK FIX: VERB COUNTER MIDDLEWARE ==========

// VerbCounterMiddleware provides a quick fix for duplicate parameter detection
// Use this to prevent "jill" being assigned as both filename and destination
type VerbCounterMiddleware struct {
	verbCounter *VerbCounter
}

// NewVerbCounterMiddleware creates the middleware
func NewVerbCounterMiddleware() *VerbCounterMiddleware {
	return &VerbCounterMiddleware{
		verbCounter: NewVerbCounter(),
	}
}

// ValidateParameters checks if parameters might be echoed/duplicated
func (vcm *VerbCounterMiddleware) ValidateParameters(params map[string]string) map[string]string {
	validated := make(map[string]string)
	seenValues := make(map[string][]string) // value -> [keys that used it]

	// Track which parameters share the same value
	for key, value := range params {
		seenValues[value] = append(seenValues[value], key)
	}

	// Resolve conflicts: keep first occurrence, remove duplicates
	for key, value := range params {
		if keys, ok := seenValues[value]; ok && len(keys) > 1 {
			// Multiple keys point to same value - keep only first
			if keys[0] == key {
				validated[key] = value
			}
			// Skip duplicates
		} else {
			validated[key] = value
		}
	}

	return validated
}

// ========== EXECUTION PLAN GENERATOR ==========

// ExecutionPlan represents an ordered sequence of tasks to execute
type ExecutionPlan struct {
	Tasks        []*Task
	ErrorHandler func(*Task, error) bool // Returns true if should continue
}

// NewExecutionPlan creates a plan from a task graph
func NewExecutionPlan(graph *TaskGraph) *ExecutionPlan {
	plan := &ExecutionPlan{
		Tasks: make([]*Task, 0, len(graph.Tasks)),
		ErrorHandler: func(t *Task, e error) bool {
			log.Printf("Task %d failed: %v", t.ID, e)
			return false // Stop on first error
		},
	}

	// Add tasks in topologically sorted order
	for _, taskID := range graph.Order {
		if task, ok := graph.Tasks[taskID]; ok {
			plan.Tasks = append(plan.Tasks, task)
		}
	}

	return plan
}

// ========== CONVENIENCE FUNCTIONS ==========

// ProcessMultiIntentInput is a high-level function for the main program
// It takes raw user input and returns a ready-to-execute task graph
func ProcessMultiIntentInput(text string, tokens []string, posTags []string) (*TaskGraph, error) {
	pipeline := NewIntentPipeline(text, tokens, posTags)
	return pipeline.Process()
}

// CountActions provides a quick action verb count (for debugging/routing)
func CountActions(tokens []string) map[string]int {
	counter := NewVerbCounter()
	return counter.CountVerbs(tokens)
}

// HasMultipleActions checks if input contains multiple action verbs
func HasMultipleActions(tokens []string) bool {
	counter := NewVerbCounter()
	counter.CountVerbs(tokens)
	return counter.HasMultipleActions()
}

// SegmentedIntentAnalysis provides detailed breakdown of intent segmentation
func SegmentedIntentAnalysis(text string) map[string]any {
	segmenter := NewIntentSegmenter(text)
	segments := segmenter.Segment()

	analysis := map[string]any{
		"original_text": text,
		"segment_count": len(segments),
		"segments":      make([]map[string]any, 0),
	}

	for i, seg := range segments {
		segData := map[string]any{
			"index":      i,
			"text":       seg.RawText,
			"tokens":     seg.Tokens,
			"actions":    seg.Actions,
			"priority":   seg.Priority,
			"depends_on": seg.Dependencies,
		}
		analysis["segments"] = append(analysis["segments"].([]map[string]any), segData)
	}

	return analysis
}

// ========== VALIDATION HELPERS ==========

// ValidateTaskDependencies checks if all dependencies can be resolved
func ValidateTaskDependencies(graph *TaskGraph) (bool, []string) {
	var errors []string

	for id, task := range graph.Tasks {
		for _, depID := range task.DependsOn {
			if _, ok := graph.Tasks[depID]; !ok {
				errors = append(errors,
					"Task "+string(rune(id))+" depends on non-existent task "+string(rune(depID)))
			}
		}
	}

	return len(errors) == 0, errors
}

// GetTasksByAction filters tasks by action type
func GetTasksByAction(graph *TaskGraph, actionType string) []*Task {
	var matching []*Task
	for _, task := range graph.Tasks {
		if task.Action == actionType {
			matching = append(matching, task)
		}
	}
	return matching
}

// GetTaskDependencyChain returns the full chain of dependencies for a task
func GetTaskDependencyChain(graph *TaskGraph, taskID int) []*Task {
	var chain []*Task
	visited := make(map[int]bool)

	var traverse func(int)
	traverse = func(id int) {
		if visited[id] {
			return
		}
		visited[id] = true

		if task, ok := graph.Tasks[id]; ok {
			chain = append(chain, task)
			for _, depID := range task.DependsOn {
				traverse(depID)
			}
		}
	}

	traverse(taskID)
	return chain
}

// ========== DEBUG OUTPUT ==========

// FormatTaskGraph returns a human-readable representation of the task graph
func FormatTaskGraph(graph *TaskGraph) string {
	var sb strings.Builder

	sb.WriteString("=== TASK GRAPH ===\n")
	sb.WriteString("Topological Order: ")

	for _, taskID := range graph.Order {
		sb.WriteString(string(rune(taskID)) + " ")
	}
	sb.WriteString("\n\n")

	for _, taskID := range graph.Order {
		task := graph.Tasks[taskID]
		sb.WriteString("Task " + string(rune(taskID)) + ":\n")
		sb.WriteString("  Action: " + task.Action + "\n")
		sb.WriteString("  Target: " + task.Target + "\n")
		sb.WriteString("  Destination: " + task.Destination + "\n")

		if len(task.DependsOn) > 0 {
			sb.WriteString("  Depends On: ")
			for _, depID := range task.DependsOn {
				sb.WriteString(string(rune(depID)) + " ")
			}
			sb.WriteString("\n")
		}

		sb.WriteString("  Parameters:\n")
		for key, value := range task.Parameters {
			sb.WriteString("    " + key + ": " + value + "\n")
		}
		sb.WriteString("\n")
	}

	return sb.String()
}
