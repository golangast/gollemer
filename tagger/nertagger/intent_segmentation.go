package nertagger

import (
	"regexp"
	"slices"
	"strings"
)

// ========== SEQUENTIAL INTENT SEGMENTATION ==========

// IntentSegment represents a single intent extracted from a multi-intent sentence
type IntentSegment struct {
	RawText      string            // Original text of this segment
	Tokens       []string          // Tokenized version
	Actions      []string          // All verbs identified in this segment
	Entities     map[string]string // Entity name -> type mapping
	Parameters   map[string]string // Parameter key -> value
	Priority     int               // Execution order
	Dependencies []int             // Indices of IntentSegments this depends on
}

// IntentSegmenter splits multi-intent sentences into individual intents
type IntentSegmenter struct {
	originalText string
	segments     []IntentSegment
}

// ========== ACTION WORD DETECTION ==========

// ActionRegistry tracks action verbs and their types
type ActionRegistry struct {
	CreationActions     map[string]bool
	DeletionActions     map[string]bool
	MovementActions     map[string]bool
	ModificationActions map[string]bool
	QueryActions        map[string]bool
}

// NewActionRegistry creates a registry of known action verbs
func NewActionRegistry() *ActionRegistry {
	return &ActionRegistry{
		CreationActions: map[string]bool{
			"create": true, "make": true, "build": true, "generate": true,
			"add": true, "insert": true, "append": true, "write": true,
			"initialize": true, "init": true, "setup": true, "new": true,
		},
		DeletionActions: map[string]bool{
			"delete": true, "remove": true, "drop": true, "erase": true,
			"destroy": true, "unlink": true,
		},
		MovementActions: map[string]bool{
			"move": true, "copy": true, "rename": true, "mv": true,
			"cp": true, "put": true, "place": true, "transfer": true,
		},
		ModificationActions: map[string]bool{
			"update": true, "modify": true, "change": true, "alter": true,
			"edit": true, "patch": true,
		},
		QueryActions: map[string]bool{
			"list": true, "show": true, "display": true, "print": true,
			"read": true, "cat": true, "grep": true, "search": true,
		},
	}
}

// GetActionType returns the type of action
func (ar *ActionRegistry) GetActionType(verb string) string {
	lower := strings.ToLower(verb)
	if ar.CreationActions[lower] {
		return "CREATE"
	}
	if ar.DeletionActions[lower] {
		return "DELETE"
	}
	if ar.MovementActions[lower] {
		return "MOVE"
	}
	if ar.ModificationActions[lower] {
		return "UPDATE"
	}
	if ar.QueryActions[lower] {
		return "QUERY"
	}
	return "UNKNOWN"
}

// IsActionVerb checks if a word is an action verb
func (ar *ActionRegistry) IsActionVerb(verb string) bool {
	lower := strings.ToLower(verb)
	return ar.CreationActions[lower] ||
		ar.DeletionActions[lower] ||
		ar.MovementActions[lower] ||
		ar.ModificationActions[lower] ||
		ar.QueryActions[lower]
}

// ========== CONJUNCTION AND TRANSITION DETECTION ==========

// ConjunctionSplitter identifies conjunction points in text
type ConjunctionSplitter struct {
	conjunctions map[string]bool
	transitions  map[string]bool
}

// NewConjunctionSplitter creates a splitter for multi-intent sentences
func NewConjunctionSplitter() *ConjunctionSplitter {
	return &ConjunctionSplitter{
		conjunctions: map[string]bool{
			"and": true, "or": true, "then": true, "afterwards": true,
			"next": true, "also": true, "additionally": true, "furthermore": true,
		},
		transitions: map[string]bool{
			"followed by": true, "after that": true, "subsequently": true,
			"then": true, "next": true, "finally": true,
		},
	}
}

// FindSplitPoints identifies where to split multi-intent sentences
func (cs *ConjunctionSplitter) FindSplitPoints(text string) []int {
	tokens := strings.Fields(text)
	var splitPoints []int

	for i, token := range tokens {
		lower := strings.ToLower(token)
		// Check if token is a conjunction
		if cs.conjunctions[lower] {
			splitPoints = append(splitPoints, i)
		}
		// Check for multi-word transitions
		if i+1 < len(tokens) {
			twoWord := lower + " " + strings.ToLower(tokens[i+1])
			if cs.transitions[twoWord] {
				splitPoints = append(splitPoints, i)
			}
		}
	}

	return splitPoints
}

// ========== INTENT SEGMENTER IMPLEMENTATION ==========

// NewIntentSegmenter creates a segmenter for multi-intent text
func NewIntentSegmenter(text string) *IntentSegmenter {
	return &IntentSegmenter{
		originalText: text,
		segments:     []IntentSegment{},
	}
}

// Segment splits text into individual intent segments
func (is *IntentSegmenter) Segment() []IntentSegment {
	splitter := NewConjunctionSplitter()
	splitPoints := splitter.FindSplitPoints(is.originalText)

	if len(splitPoints) == 0 {
		// Single intent
		segment := is.createSegment(is.originalText, 0)
		is.segments = []IntentSegment{segment}
		return is.segments
	}

	// Multiple intents - split at conjunction points
	tokens := strings.Fields(is.originalText)
	var segmentTexts []string
	lastIdx := 0

	for _, splitIdx := range splitPoints {
		// Collect tokens before the conjunction
		if splitIdx > lastIdx {
			segment := strings.Join(tokens[lastIdx:splitIdx], " ")
			if strings.TrimSpace(segment) != "" {
				segmentTexts = append(segmentTexts, segment)
			}
		}
		lastIdx = splitIdx + 1
	}

	// Add remaining tokens
	if lastIdx < len(tokens) {
		segment := strings.Join(tokens[lastIdx:], " ")
		if strings.TrimSpace(segment) != "" {
			segmentTexts = append(segmentTexts, segment)
		}
	}

	// Create segments and track dependencies
	for i, text := range segmentTexts {
		segment := is.createSegment(text, i)
		is.segments = append(is.segments, segment)
	}

	// Identify dependencies between segments
	is.identifyDependencies()

	return is.segments
}

// createSegment creates a single IntentSegment from text
func (is *IntentSegmenter) createSegment(text string, priority int) IntentSegment {
	segment := IntentSegment{
		RawText:      text,
		Tokens:       strings.Fields(text),
		Actions:      []string{},
		Entities:     make(map[string]string),
		Parameters:   make(map[string]string),
		Priority:     priority,
		Dependencies: []int{},
	}

	// Extract actions from tokens
	registry := NewActionRegistry()
	for _, token := range segment.Tokens {
		if registry.IsActionVerb(token) {
			segment.Actions = append(segment.Actions, token)
		}
	}

	return segment
}

// identifyDependencies tracks which segments depend on which
func (is *IntentSegmenter) identifyDependencies() {
	registry := NewActionRegistry()

	for i := range is.segments {
		// Movement actions depend on creation actions
		for _, action := range is.segments[i].Actions {
			if registry.GetActionType(action) == "MOVE" {
				// Look for prior CREATE actions
				for j := range i {
					for _, priorAction := range is.segments[j].Actions {
						if registry.GetActionType(priorAction) == "CREATE" {
							is.segments[i].Dependencies = append(is.segments[i].Dependencies, j)
						}
					}
				}
			}
		}
	}
}

// ========== SLOT CONSTRAINTS ==========

// SlotConstraint defines rules for parameter extraction
type SlotConstraint struct {
	SlotName        string
	IsRequired      bool
	AllowedTypes    []string // e.g., "FILENAME", "DIRNAME", "PATH"
	ValidationRules []func(string) bool
	PriorityRank    int // Lower = higher priority
}

// SlotConstraintValidator validates and prioritizes entity assignments
type SlotConstraintValidator struct {
	constraints map[string][]SlotConstraint
}

// NewSlotConstraintValidator creates a validator with default constraints
func NewSlotConstraintValidator() *SlotConstraintValidator {
	return &SlotConstraintValidator{
		constraints: map[string][]SlotConstraint{
			"CREATE": {
				{
					SlotName:     "name",
					IsRequired:   true,
					AllowedTypes: []string{"FILENAME", "DIRNAME", "IDENTIFIER"},
					PriorityRank: 1, // First noun is the name
				},
				{
					SlotName:     "path",
					IsRequired:   false,
					AllowedTypes: []string{"PATH"},
					PriorityRank: 2,
				},
			},
			"MOVE": {
				{
					SlotName:     "source",
					IsRequired:   true,
					AllowedTypes: []string{"FILENAME", "DIRNAME"},
					PriorityRank: 1, // First entity is source
				},
				{
					SlotName:     "destination",
					IsRequired:   true,
					AllowedTypes: []string{"PATH", "DIRNAME"},
					PriorityRank: 2, // After preposition "to" or "in"
				},
			},
			"DELETE": {
				{
					SlotName:     "target",
					IsRequired:   true,
					AllowedTypes: []string{"FILENAME", "DIRNAME"},
					PriorityRank: 1,
				},
			},
		},
	}
}

// AssignSlots assigns entities to slots based on constraints and priority
func (scv *SlotConstraintValidator) AssignSlots(actionType string, entities []string, entityTypes []string) map[string]string {
	slots := make(map[string]string)

	constraints, ok := scv.constraints[actionType]
	if !ok {
		// Unknown action, assign generically
		for i, entity := range entities {
			if i == 0 {
				slots["primary"] = entity
			} else {
				slots["secondary"] = entity
			}
		}
		return slots
	}

	// Sort constraints by priority
	// Assign entities to slots in order of constraint priority
	entityIdx := 0
	for _, constraint := range constraints {
		if entityIdx < len(entities) && constraint.IsRequired {
			// Check if entity type matches
			if entityIdx < len(entityTypes) {
				if is_allowed_type(entityTypes[entityIdx], constraint.AllowedTypes) {
					slots[constraint.SlotName] = entities[entityIdx]
					entityIdx++
				}
			}
		}
	}

	return slots
}

// is_allowed_type checks if an entity type is in the allowed list
func is_allowed_type(entityType string, allowedTypes []string) bool {
	if slices.Contains(allowedTypes, entityType) {
		return true
	}
	return len(allowedTypes) == 0 // If no restrictions, allow any
}

// ========== MULTI-ACTION VERB COUNTER ==========

// VerbCounter counts and tracks actions in a sentence
type VerbCounter struct {
	registry     *ActionRegistry
	actionCounts map[string]int // Action type -> count
}

// NewVerbCounter creates a new verb counter
func NewVerbCounter() *VerbCounter {
	return &VerbCounter{
		registry:     NewActionRegistry(),
		actionCounts: make(map[string]int),
	}
}

// CountVerbs analyzes verbs in tokens and returns counts
func (vc *VerbCounter) CountVerbs(tokens []string) map[string]int {
	for _, token := range tokens {
		if vc.registry.IsActionVerb(token) {
			actionType := vc.registry.GetActionType(token)
			vc.actionCounts[actionType]++
		}
	}
	return vc.actionCounts
}

// HasMultipleActions checks if multiple different actions are present
func (vc *VerbCounter) HasMultipleActions() bool {
	actionTypeCount := 0
	for _, count := range vc.actionCounts {
		if count > 0 {
			actionTypeCount++
		}
	}
	return actionTypeCount > 1
}

// ========== TASK AND TASK GRAPH ==========

// Task represents a single executable action
type Task struct {
	ID          int
	Action      string // "CREATE", "MOVE", "DELETE", etc.
	Target      string // Primary entity
	Destination string // Where/what it's going
	Parameters  map[string]string
	DependsOn   []int  // IDs of tasks that must complete first
	Status      string // "PENDING", "RUNNING", "COMPLETED", "FAILED"
}

// TaskGraph represents a DAG of dependent tasks
type TaskGraph struct {
	Tasks map[int]*Task
	Order []int // Topologically sorted task IDs
}

// NewTaskGraph creates an empty task graph
func NewTaskGraph() *TaskGraph {
	return &TaskGraph{
		Tasks: make(map[int]*Task),
		Order: []int{},
	}
}

// AddTask adds a task to the graph
func (tg *TaskGraph) AddTask(task *Task) {
	tg.Tasks[task.ID] = task
}

// SortTopologically sorts tasks by dependencies using Kahn's algorithm
func (tg *TaskGraph) SortTopologically() error {
	// Count in-degrees
	inDegree := make(map[int]int)
	adjList := make(map[int][]int)

	for id := range tg.Tasks {
		if _, ok := inDegree[id]; !ok {
			inDegree[id] = 0
		}
	}

	for _, task := range tg.Tasks {
		for _, depID := range task.DependsOn {
			inDegree[task.ID]++
			adjList[depID] = append(adjList[depID], task.ID)
		}
	}

	// Find all nodes with in-degree 0
	var queue []int
	for id, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, id)
		}
	}

	// Process queue
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		tg.Order = append(tg.Order, current)

		for _, neighbor := range adjList[current] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}

	// Check for cycles
	if len(tg.Order) != len(tg.Tasks) {
		return &CycleError{Message: "Circular dependency detected in task graph"}
	}

	return nil
}

// CycleError represents a cycle in the dependency graph
type CycleError struct {
	Message string
}

func (e *CycleError) Error() string {
	return e.Message
}

// ========== SEGMENT TO TASK CONVERSION ==========

// SegmentToTaskConverter converts intent segments into a task graph
type SegmentToTaskConverter struct {
	segments  []IntentSegment
	actionReg *ActionRegistry
	validator *SlotConstraintValidator
}

// NewSegmentToTaskConverter creates a converter
func NewSegmentToTaskConverter(segments []IntentSegment) *SegmentToTaskConverter {
	return &SegmentToTaskConverter{
		segments:  segments,
		actionReg: NewActionRegistry(),
		validator: NewSlotConstraintValidator(),
	}
}

// Convert transforms segments into a task graph
func (stc *SegmentToTaskConverter) Convert() *TaskGraph {
	graph := NewTaskGraph()
	taskID := 0

	for segIdx, segment := range stc.segments {
		// Each segment with actions becomes one or more tasks
		for _, action := range segment.Actions {
			actionType := stc.actionReg.GetActionType(action)

			// Extract entities and assign to slots
			entities := stc.extractEntities(segment)
			slots := stc.validator.AssignSlots(actionType, entities, stc.getEntityTypes(segment, entities))

			task := &Task{
				ID:          taskID,
				Action:      actionType,
				Target:      slots["source"] + slots["name"] + slots["primary"], // Combine possible names
				Destination: slots["destination"] + slots["secondary"],
				Parameters:  slots,
				Status:      "PENDING",
			}

			// Add dependencies from segment dependencies
			for _, depIdx := range segment.Dependencies {
				if depIdx < segIdx {
					// Find tasks from that segment
					for _, otherTask := range graph.Tasks {
						if len(otherTask.DependsOn) == 0 && depIdx < segIdx {
							task.DependsOn = append(task.DependsOn, otherTask.ID)
						}
					}
				}
			}

			graph.AddTask(task)
			taskID++
		}
	}

	_ = graph.SortTopologically()
	return graph
}

// extractEntities pulls entity names from segment
func (stc *SegmentToTaskConverter) extractEntities(segment IntentSegment) []string {
	var entities []string
	for entity := range segment.Entities {
		entities = append(entities, entity)
	}
	return entities
}

// getEntityTypes returns types for extracted entities
func (stc *SegmentToTaskConverter) getEntityTypes(segment IntentSegment, entities []string) []string {
	var types []string
	for _, entity := range entities {
		if etype, ok := segment.Entities[entity]; ok {
			types = append(types, etype)
		} else {
			types = append(types, "IDENTIFIER")
		}
	}
	return types
}

// ========== REGEX-BASED PREPOSITION DETECTION ==========

// PrepositionMatcher identifies relationship-bearing prepositions
type PrepositionMatcher struct {
	patterns map[string]*regexp.Regexp
}

// NewPrepositionMatcher creates a matcher for prepositions
func NewPrepositionMatcher() *PrepositionMatcher {
	return &PrepositionMatcher{
		patterns: map[string]*regexp.Regexp{
			"destination": regexp.MustCompile(`\b(in|into|to|within|inside|under)\s+(\w+)`),
			"source":      regexp.MustCompile(`\b(from|out of|off)\s+(\w+)`),
			"attribute":   regexp.MustCompile(`\b(with|using|via)\s+(\w+)`),
			"location":    regexp.MustCompile(`\b(at|on|near|beside)\s+(\w+)`),
		},
	}
}

// ExtractPrepositionPairs extracts entity-preposition-entity relationships
func (pm *PrepositionMatcher) ExtractPrepositionPairs(text string) map[string][]string {
	pairs := make(map[string][]string)

	for relation, pattern := range pm.patterns {
		matches := pattern.FindAllStringSubmatch(text, -1)
		for _, match := range matches {
			if len(match) > 2 {
				pairs[relation] = append(pairs[relation], match[2])
			}
		}
	}

	return pairs
}
