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
		GrammarSkeleton: []string{"GREET", "OTHER", "OTHER", "INTERROGATIVE", "AUX", "PRON", "VERB", "PRON", "OTHER", "OTHER"},
		RequiredKeywords: []string{"hi", "hello", "help", "how", "today"},
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
// This is used for structural training and grammar scoring.
// Vocabulary is tuned to match the human_chat.txt training corpus.
func MapWordToGrammarType(w string) string {
	w = strings.ToLower(strings.Trim(w, ".,!?;:\"'()[]"))

	// GREETINGS / Discourse markers
	switch w {
	case "hello", "hi", "hey", "greetings", "afternoon",
		"hiya", "howdy", "yo", "sup", "hii", "heya", "ohh", "oh", "wow",
		"haha", "lol", "hehe", "yep", "yup", "yeah", "yea", "yes",
		"nope", "nah", "cool", "awesome", "amazing",
		"interesting", "true", "right", "exactly", "indeed", "absolutely", "there", "here", "now":
		return "GREET"
	}

	// PRONOUNS
	switch w {
	case "i", "me", "my", "mine", "myself",
		"you", "your", "yours", "yourself",
		"he", "him", "his", "himself",
		"she", "her", "hers", "herself",
		"it", "its", "itself",
		"we", "us", "our", "ours", "ourselves",
		"they", "them", "their", "theirs", "themselves",
		"this", "that", "these", "those",
		"someone", "anyone", "everyone", "nobody", "somebody":
		return "PRON"
	}

	// VERBS — Copula / State verbs
	switch w {
	case "am", "is", "are", "was", "were", "be", "been", "being",
		"seem", "seems", "seemed", "feel", "feels", "felt",
		"looks", "sounded", "become", "became", "stay", "stayed",
		"remain", "remains":
		return "VERB"
	}

	// AUXILIARY VERBS
	switch w {
	case "will", "would", "can", "could", "should", "shall", "may", "might",
		"must", "ought", "dare",
		"do", "does", "did", "doing",
		"have", "has", "had", "having",
		"get", "got", "gotten", "getting",
		"going", "gonna", "gotta", "wanna":
		return "AUX"
	}

	// ACTION VERBS — common in casual conversation
	switch w {
	case "think", "thinking", "thought", "know", "knowing", "knew",
		"want", "wanted", "wanting", "like", "liked", "liking", "love", "loved", "loving",
		"make", "made", "making", "go", "went", "gone", "start", "started", "starting",
		"use", "used", "using", "work", "worked", "working",
		"try", "tried", "trying", "learn", "learned", "learning",
		"take", "took", "taken", "taking", "put", "keep", "kept",
		"come", "came", "coming", "run", "ran", "running",
		"see", "saw", "seen", "seeing", "look", "looked", "looking",
		"find", "found", "finding", "give", "gave", "given", "giving",
		"tell", "told", "telling", "ask", "asked", "asking",
		"talk", "talked", "talking", "say", "said", "saying",
		"help", "helped", "helping", "plan", "planned", "planning",
		"buy", "bought", "buying", "eat", "ate", "eaten", "eating",
		"drink", "drank", "drunk", "drinking", "read", "reading",
		"write", "wrote", "written", "writing", "play", "played", "playing",
		"cook", "cooked", "cooking", "build", "built", "building",
		"enjoy", "enjoyed", "enjoying", "listen", "listened", "listening",
		"practice", "practiced", "practicing", "need", "needed", "needing",
		"check", "checked", "checking", "share", "shared", "sharing",
		"hear", "heard", "hearing", "remember", "remembered", "remembering",
		"spend", "spent", "spending", "live", "lived", "living",
		"grow", "grew", "grown", "growing", "move", "moved", "moving",
		"done", "finished", "completed":
		return "VERB"
	}

	// ADJECTIVES / ADVERBS
	switch w {
	case "good", "fine", "well", "great", "excellent", "bad", "okay", "ok",
		"happy", "sad", "excited", "bored", "tired", "busy", "free", "ready",
		"easy", "hard", "difficult", "complex", "fun", "funny",
		"nice", "beautiful", "lovely", "wonderful",
		"big", "small", "large", "little", "long", "short", "new", "old",
		"hot", "cold", "warm", "clean", "fresh", "quick", "slow",
		"better", "worse", "best", "worst", "more", "most", "less", "least",
		"very", "really", "quite", "just", "still", "already", "again",
		"always", "never", "often", "sometimes", "usually", "actually",
		"definitely", "probably", "maybe", "perhaps", "sure", "only",
		"also", "too", "even", "much", "many", "few", "lot", "lots",
		"pretty", "fairly", "kind", "sort", "bit", "enough",
		"so", "such", "both", "each", "every", "all", "any", "some",
		"local", "different", "same", "next", "last", "other", "own",
		"full", "whole", "main", "basic", "simple", "special", "first",
		"second", "several", "single", "important", "natural",
		"classic", "healthy", "perfect":
		return "ADJ"
	}

	// NOUNS — common content words in casual conversation
	switch w {
	case "name", "gollemer", "bot", "assistant", "system", "ai", "machine",
		"human", "person", "people", "friend", "friends", "family",
		"thing", "things", "stuff", "way", "ways", "time", "day", "days",
		"week", "month", "year", "place", "home", "house", "room",
		"job", "school", "class", "project", "idea", "ideas",
		"food", "water", "coffee", "tea", "book", "books", "music",
		"life", "world", "city", "town", "country", "morning", "night",
		"habit", "hobby", "skill", "goal", "goals",
		"garden", "kitchen", "dog", "cat", "recipe", "hiking",
		"weekend", "vacation", "holiday", "trip", "weather", "sleep",
		"rest", "exercise", "health", "mind", "body", "heart", "energy",
		"money", "phone", "computer", "app", "game", "show",
		"movie", "podcast", "video", "photo", "message", "email",
		"question", "answer", "story", "point", "reason", "moment", "today", "yesterday", "tomorrow":
		return "NOUN"
	}

	// PREPOSITIONS / ARTICLES / CONJUNCTIONS
	switch w {
	case "the", "a", "an",
		"in", "on", "at", "to", "for", "with", "by", "from", "of",
		"about", "above", "after", "before", "between", "during",
		"into", "near", "off", "out", "over", "through",
		"under", "until", "up", "upon", "within", "without",
		"and", "but", "or", "nor", "yet", "because", "if",
		"when", "while", "although", "though", "since", "unless",
		"as", "than", "then", "not", "no":
		return "PREP"
	}

	// INTERROGATIVES
	switch w {
	case "how", "what", "where", "why", "who", "which", "whose", "whom":
		return "INTERROGATIVE"
	}

	return "OTHER"
}

