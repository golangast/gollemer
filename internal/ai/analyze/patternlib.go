package analyze

import (
	"encoding/json"
	"os"
	"strings"
)

// This file implements the cross-project pattern library: structural
// patterns learned from analyzing many Go repos. When the analyzer sees
// totally new code, it matches against the library — not just the current
// project's patterns. This is how it generalizes to unseen code.

// StructuralPattern describes what a kind of function looks like,
// structurally — not the code, but the shape.
type StructuralPattern struct {
	Kind string `json:"kind"` // "validator", "handler", "getter", "constructor", etc.

	// Structural features.
	HasReceiver   bool     `json:"has_receiver"`
	ParamCount    []int    `json:"param_count"`    // typical param counts
	ReturnsError  bool     `json:"returns_error"`
	ReturnsBool   bool     `json:"returns_bool"`
	MinBranches   int      `json:"min_branches"`
	HasLoop       bool     `json:"has_loop"`
	TypicalVerbs  []string `json:"typical_verbs"`  // "validate", "check", "verify"
	TypicalNouns  []string `json:"typical_nouns"`  // "email", "input", "user"

	// How many examples were seen.
	Count int `json:"count"`

	// Example intents (for human reading, not matching).
	ExampleIntents []string `json:"example_intents"`
}

// PatternLibrary is the cross-project learned patterns.
type PatternLibrary struct {
	Patterns map[string]*StructuralPattern `json:"patterns"`
	Version  int                           `json:"version"`
}

// NewPatternLibrary creates an empty library.
func NewPatternLibrary() *PatternLibrary {
	return &PatternLibrary{
		Patterns: map[string]*StructuralPattern{},
		Version:  1,
	}
}

// LearnFromProject extracts structural patterns from an analyzed project.
func (lib *PatternLibrary) LearnFromProject(p *Project) {
	for _, fn := range p.AllFuncs() {
		kind := classifyFunc(fn)
		if kind == "" {
			continue
		}
		pat, ok := lib.Patterns[kind]
		if !ok {
			pat = &StructuralPattern{Kind: kind}
			lib.Patterns[kind] = pat
		}
		// Update structural features.
		pat.Count++
		if fn.Receiver != "" {
			pat.HasReceiver = true
		}
		// Track param count.
		found := false
		for _, c := range pat.ParamCount {
			if c == fn.Params {
				found = true
				break
			}
		}
		if !found && len(pat.ParamCount) < 5 {
			pat.ParamCount = append(pat.ParamCount, fn.Params)
		}
		if strings.Contains(fn.Sig, "error") {
			pat.ReturnsError = true
		}
		if strings.Contains(fn.Sig, ") bool") {
			pat.ReturnsBool = true
		}
		if len(fn.Guards) > pat.MinBranches {
			// Track max, not min, for now.
			pat.MinBranches = len(fn.Guards)
		}
		if len(fn.LoopCalls) > 0 {
			pat.HasLoop = true
		}
		// Track verbs/nouns from name.
		words := splitName(fn.Name)
		if len(words) > 0 {
			verb := strings.ToLower(words[0])
			if !containsStr(pat.TypicalVerbs, verb) && len(pat.TypicalVerbs) < 10 {
				pat.TypicalVerbs = append(pat.TypicalVerbs, verb)
			}
		}
		if len(pat.ExampleIntents) < 3 {
			pat.ExampleIntents = append(pat.ExampleIntents, fn.Intent())
		}
	}
}

// classifyFunc determines the pattern kind from structure.
func classifyFunc(fn *Func) string {
	// Use the existing patternIntent to classify.
	if pat := patternIntent(fn); pat != "" {
		if strings.Contains(pat, "HTTP requests") {
			return "handler"
		}
		if strings.Contains(pat, "Returns the") {
			return "getter"
		}
		if strings.Contains(pat, "Sets the") {
			return "setter"
		}
		if strings.Contains(pat, "Reports whether") {
			return "predicate"
		}
		if strings.Contains(pat, "Releases resources") {
			return "closer"
		}
		if strings.Contains(pat, "Retries") {
			return "retry"
		}
		if strings.Contains(pat, "string representation") {
			return "stringer"
		}
	}
	// Validator: returns error + has guards.
	if strings.Contains(fn.Sig, "error") && len(fn.Guards) > 0 {
		return "validator"
	}
	// Constructor: NewX.
	if isConstructor(fn.Name) {
		return "constructor"
	}
	return ""
}

// MatchAgainstLibrary checks if a function matches a library pattern.
// Returns the kind and confidence (0-1).
func (lib *PatternLibrary) MatchAgainstLibrary(fn *Func) (string, float64) {
	bestKind := ""
	bestScore := 0.0

	nameLower := strings.ToLower(fn.Name)

	for kind, pat := range lib.Patterns {
		score := 0.0
		total := 0.0

		// Structural features.
		total++
		if (fn.Receiver != "") == pat.HasReceiver {
			score++
		}
		total++
		for _, c := range pat.ParamCount {
			if c == fn.Params {
				score++
				break
			}
		}
		total++
		if strings.Contains(fn.Sig, "error") == pat.ReturnsError {
			score++
		}
		total++
		if (len(fn.Guards) > 0) == (pat.MinBranches > 0) {
			score++
		}

		// Name features: critical for disambiguation.
		// For some kinds, the name is definitional — if it doesn't match,
		// the kind is impossible (confidence 0).
		total++
		nameMatches := false
		hardGate := false
		switch kind {
		case "constructor":
			nameMatches = strings.HasPrefix(fn.Name, "New")
			hardGate = true // Must start with New, no exceptions.
		case "validator":
			nameMatches = strings.Contains(nameLower, "valid") || strings.Contains(nameLower, "check") ||
				strings.Contains(nameLower, "verify")
		case "getter":
			nameMatches = strings.HasPrefix(fn.Name, "Get") || len(splitName(fn.Name)) == 1
		case "setter":
			nameMatches = strings.HasPrefix(fn.Name, "Set")
			hardGate = true
		case "predicate":
			nameMatches = strings.HasPrefix(fn.Name, "Is") || strings.HasPrefix(fn.Name, "Has")
			hardGate = true
		case "handler":
			nameMatches = strings.Contains(nameLower, "handler")
		default:
			nameMatches = true // No name constraint.
		}
		if hardGate && !nameMatches {
			// Definitional mismatch: skip this kind entirely.
			continue
		}
		if nameMatches {
			score++
		}

		confidence := score / total
		if confidence > bestScore && confidence >= 0.8 {
			bestScore = confidence
			bestKind = kind
		}
	}

	return bestKind, bestScore
}

// Save writes the library to disk.
func (lib *PatternLibrary) Save(path string) error {
	data, err := json.MarshalIndent(lib, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// Load reads the library from disk.
func LoadPatternLibrary(path string) (*PatternLibrary, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var lib PatternLibrary
	if err := json.Unmarshal(data, &lib); err != nil {
		return nil, err
	}
	return &lib, nil
}

func containsStr(slice []string, s string) bool {
	for _, v := range slice {
		if v == s {
			return true
		}
	}
	return false
}

// Global pattern library, loaded lazily from disk.
var globalPatternLib *PatternLibrary
var globalPatternLibLoaded bool

func loadGlobalPatternLib() *PatternLibrary {
	if globalPatternLibLoaded {
		return globalPatternLib
	}
	globalPatternLibLoaded = true
	// Try standard locations.
	for _, path := range []string{
		"data/patterns.json",
		"/home/hatch/workspace/gollemer/data/patterns.json",
	} {
		if lib, err := LoadPatternLibrary(path); err == nil {
			globalPatternLib = lib
			return lib
		}
	}
	return nil
}

func describeKind(kind string) string {
	switch kind {
	case "validator":
		return "Validates input"
	case "handler":
		return "Handles requests"
	case "getter":
		return "Returns a field"
	case "setter":
		return "Sets a field"
	case "predicate":
		return "Checks a condition"
	case "constructor":
		return "Creates a new value"
	case "stringer":
		return "Returns string representation"
	case "closer":
		return "Releases resources"
	case "retry":
		return "Retries with backoff"
	default:
		return kind
	}
}
