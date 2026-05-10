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

	// Rule: Greeting — "hi there! how can i help you today?"
	// CSV grammar: GREET OTHER OTHER AUX PRON VERB PRON OTHER
	rb.Rules["social:greeting"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "greeting",
		GrammarSkeleton: []string{"GREET", "OTHER", "OTHER", "AUX", "PRON", "VERB", "PRON", "OTHER"},
		RequiredKeywords: []string{"hi", "hello", "help"},
	}

	// Rule: Identity — "i am gollemer, your ai assistant."
	// CSV grammar: PRON VERB NOUN PRON NOUN NOUN
	rb.Rules["social:identity"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "identity",
		GrammarSkeleton: []string{"PRON", "VERB", "NOUN", "PRON", "NOUN", "NOUN"},
		RequiredKeywords: []string{"gollemer", "ai", "assistant"},
	}

	// Rule: Status Check — "i am doing well, thank you for asking!"
	// CSV grammar: PRON VERB OTHER ADJ OTHER PRON PREP OTHER
	rb.Rules["social:status_check"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "status_check",
		GrammarSkeleton: []string{"PRON", "VERB", "OTHER", "ADJ", "OTHER", "PRON", "PREP", "OTHER"},
		RequiredKeywords: []string{"doing", "well"},
	}

	// Rule: Polite — "you are very welcome! i am happy to help."
	// CSV grammar: PRON VERB OTHER OTHER PRON VERB ADJ PREP VERB
	rb.Rules["social:polite"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "polite",
		GrammarSkeleton: []string{"PRON", "VERB", "OTHER", "OTHER", "PRON", "VERB", "ADJ", "PREP", "VERB"},
		RequiredKeywords: []string{"welcome", "happy", "help"},
	}

	// Rule: Farewell — "goodbye! it was nice talking to you today."
	// CSV grammar: GREET PRON VERB ADJ OTHER PREP PRON OTHER
	rb.Rules["social:farewell"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "farewell",
		GrammarSkeleton: []string{"GREET", "PRON", "VERB", "ADJ", "OTHER", "PREP", "PRON", "OTHER"},
		RequiredKeywords: []string{"goodbye"},
	}

	// Rule: Capabilities — "i can answer questions, tell jokes, and help you with your code."
	// CSV grammar: PRON AUX OTHER NOUN OTHER OTHER PREP VERB PRON PREP PRON NOUN
	rb.Rules["social:capabilities"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "capabilities",
		GrammarSkeleton: []string{"PRON", "AUX", "OTHER", "NOUN", "OTHER", "VERB", "PRON", "PREP", "PRON", "NOUN"},
		RequiredKeywords: []string{"can", "help"},
	}

	// Rule: Emotional Support — "i hope you can get some rest soon."
	// CSV grammar: PRON OTHER PRON AUX VERB OTHER OTHER OTHER
	rb.Rules["social:emotional_support"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "emotional_support",
		GrammarSkeleton: []string{"PRON", "OTHER", "PRON", "AUX", "VERB", "OTHER", "OTHER"},
		RequiredKeywords: []string{},
	}

	// Rule: Support/Help — "i would be happy to help!"
	// CSV grammar: PRON AUX VERB ADJ PREP VERB
	rb.Rules["social:support"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "support",
		GrammarSkeleton: []string{"PRON", "AUX", "VERB", "ADJ", "PREP", "VERB"},
		RequiredKeywords: []string{"help"},
	}

	// Rule: General Social — simple subject-verb structure
	rb.Rules["social:social_chat"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "social_chat",
		GrammarSkeleton: []string{"PRON", "VERB", "OTHER"},
		RequiredKeywords: []string{},
	}

	// Rule: Trivia / Knowledge
	rb.Rules["social:trivia"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "trivia",
		GrammarSkeleton: []string{"OTHER", "PRON", "OTHER", "OTHER", "OTHER", "VERB"},
		RequiredKeywords: []string{},
	}

	// Rule: Small Talk
	rb.Rules["social:small_talk"] = IntentRule{
		ParentIntent: "social",
		ChildIntent:  "small_talk",
		GrammarSkeleton: []string{"PRON", "OTHER", "OTHER", "PRON", "VERB"},
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
	case "hello", "hi", "hey", "greetings", "morning", "evening": return "GREET"
	}

	// PRONOUNS
	switch w {
	case "i", "me", "my", "you", "your", "it", "we", "they", "them", "us", "our": return "PRON"
	}

	// VERBS (Copula / State)
	switch w {
	case "am", "is", "are", "was", "were", "be", "been", "being": return "VERB"
	}

	// AUXILIARY VERBS
	switch w {
	case "will", "can", "should", "must", "might", "do", "does", "did", "have", "has", "had": return "AUX"
	}

	// ADJECTIVES / ADVERBS
	switch w {
	case "good", "fine", "well", "great", "excellent", "bad", "okay", "happy", "sad", "very", "really": return "ADJ"
	}

	// NOUNS / NAMES
	switch w {
	case "name", "gollemer", "bot", "assistant", "system", "ai", "machine", "human", "person", "thing": return "NOUN"
	}

	// PREPOSITIONS / ARTICLES / CONJUNCTIONS
	switch w {
	case "the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "and", "but", "or", "of": return "PREP"
	}

	// INTERROGATIVES
	switch w {
	case "how", "what", "where", "why", "when", "who": return "INTERROGATIVE"
	}
	
	return "OTHER"
}
