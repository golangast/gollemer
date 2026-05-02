package moe

import (
	"strings"
)

// IntentRule defines the structural and vocabulary expectations for a specific intent.
type IntentRule struct {
	ParentIntent string
	ChildIntent  string
	
	// GrammarSkeleton defines the expected sequence of POS types (simplified)
	// Example: ["PRON", "VERB", "ADJ"] -> "I am happy"
	GrammarSkeleton []string
	
	// RequiredKeywords are words that SHOULD be present for this intent to be valid.
	RequiredKeywords []string
	
	// ForbiddenPatterns are sequences that make the sentence incoherent for this intent.
	ForbiddenPatterns []string
}

// RuleBook is the sophisticated collection of linguistic rules for the MoE model.
type RuleBook struct {
	Rules map[string]IntentRule
}

// NewRuleBook initializes the rule system with standard conversational grammar.
func NewRuleBook() *RuleBook {
	rb := &RuleBook{
		Rules: make(map[string]IntentRule),
	}

	// Rule: Greeting
	rb.Rules["social:greeting"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "greeting",
		GrammarSkeleton: []string{"GREET", "PRON", "VERB"}, // "Hello how are..."
		RequiredKeywords: []string{"hello", "hi", "hey", "morning", "evening"},
	}

	// Rule: Identity
	rb.Rules["social:identity"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "identity",
		GrammarSkeleton: []string{"PRON", "VERB", "NAME"}, // "My name is..."
		RequiredKeywords: []string{"name", "gollemer", "ai", "bot", "assistant"},
	}

	// Rule: Status Check
	rb.Rules["social:status_check"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "status_check",
		GrammarSkeleton: []string{"PRON", "VERB", "ADJ"}, // "I am doing well"
		RequiredKeywords: []string{"doing", "well", "fine", "good", "great"},
	}

	// Rule: General Social
	rb.Rules["social:social_chat"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "social_chat",
		GrammarSkeleton: []string{"PRON", "VERB", "OTHER"}, // Simple subject-verb structure
		RequiredKeywords: []string{},
	}

	return rb
}

// GetRuleByIntent retrieves the rule for a specific intent pair.
func (rb *RuleBook) GetRuleByIntent(parent, child string) (IntentRule, bool) {
	key := parent + ":" + child
	r, ok := rb.Rules[key]
	return r, ok
}

// MapWordToGrammarType returns a coarse-grained tag for a word.
// This is used for structural training.
func MapWordToGrammarType(w string) string {
	w = strings.ToLower(w)
	
	// GREETINGS
	switch w {
	case "hello", "hi", "hey", "greetings": return "GREET"
	}

	// PRONOUNS
	switch w {
	case "i", "me", "my", "you", "your", "it", "we", "they": return "PRON"
	}

	// VERBS (Common)
	switch w {
	case "am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had": return "VERB"
	}

	// ARTICLES
	switch w {
	case "the", "a", "an": return "ART"
	}

	// IDENTITY
	switch w {
	case "name", "gollemer", "bot", "assistant", "system": return "NAME"
	}

	// ADJECTIVES
	switch w {
	case "good", "fine", "well", "great", "excellent", "bad", "okay": return "ADJ"
	}

	return "OTHER"
}
