package context

import (
	"fmt"
	"strings"
)

// Entity represents a recognized entity with its type and value.
type Entity struct {
	Type  string
	Value string
}

// ConversationContext stores the history of a conversation for context and co-reference resolution.
type ConversationContext struct {
	MaxTurns        int
	TurnHistory     []string   // Raw text of past user inputs
	ResponseHistory []string   // Raw text of past system responses
	LastIntents     []string   // Last N predicted intents
	LastEntities    [][]Entity // Last N sets of recognized entities
	CurrentIntent   string     // The most recently predicted intent
	CurrentEntities []Entity   // The most recently recognized entities
}

// NewConversationContext initializes a new ConversationContext with a specified history size.
func NewConversationContext(maxTurns int) *ConversationContext {
	return &ConversationContext{
		MaxTurns:        maxTurns,
		TurnHistory:     make([]string, 0, maxTurns),
		ResponseHistory: make([]string, 0, maxTurns),
		LastIntents:     make([]string, 0, maxTurns),
		LastEntities:    make([][]Entity, 0, maxTurns),
	}
}

// AddTurn adds the current turn's information to the conversation context.
func (cc *ConversationContext) AddTurn(intent string, entities []Entity, rawText string) {
	// Update current intent and entities
	cc.CurrentIntent = intent
	cc.CurrentEntities = entities

	// Add to history and manage size
	cc.TurnHistory = append(cc.TurnHistory, rawText)
	cc.LastIntents = append(cc.LastIntents, intent)
	cc.LastEntities = append(cc.LastEntities, entities)

	if len(cc.TurnHistory) > cc.MaxTurns {
		cc.TurnHistory = cc.TurnHistory[1:]
		cc.LastIntents = cc.LastIntents[1:]
		cc.LastEntities = cc.LastEntities[1:]
	}
}

// AddResponse adds the system's response to the conversation context.
func (cc *ConversationContext) AddResponse(response string) {
	cc.ResponseHistory = append(cc.ResponseHistory, response)
	if len(cc.ResponseHistory) > cc.MaxTurns {
		cc.ResponseHistory = cc.ResponseHistory[1:]
	}
}

// GetConversationHistory returns the full conversation history formatted as a string.
func (cc *ConversationContext) GetConversationHistory() string {
	var sb strings.Builder
	turns := len(cc.TurnHistory)
	responses := len(cc.ResponseHistory)
	maxLen := turns
	if responses > maxLen {
		maxLen = responses
	}

	for i := 0; i < maxLen; i++ {
		if i < turns {
			sb.WriteString("Human: " + cc.TurnHistory[i] + "\n")
		}
		if i < responses {
			sb.WriteString("AI: " + cc.ResponseHistory[i] + "\n")
		}
	}
	return sb.String()
}

// GetLastIntent retrieves the most recent intent from the history.
func (cc *ConversationContext) GetLastIntent() string {
	if len(cc.LastIntents) > 0 {
		return cc.LastIntents[len(cc.LastIntents)-1]
	}
	return ""
}

// GetLastEntities retrieves the most recent set of entities from the history.
func (cc *ConversationContext) GetLastEntities() []Entity {
	if len(cc.LastEntities) > 0 {
		return cc.LastEntities[len(cc.LastEntities)-1]
	}
	return nil
}

// GetLastResponse retrieves the most recent system response from the history.
func (cc *ConversationContext) GetLastResponse() string {
	if len(cc.ResponseHistory) > 0 {
		return cc.ResponseHistory[len(cc.ResponseHistory)-1]
	}
	return ""
}

// ResolveCoReference attempts to resolve simple pronominal co-references in the input text.
// For example, if the last command was "create handler login" and the current command is "delete it",
// this function should resolve "it" to "login".
func (cc *ConversationContext) ResolveCoReference(inputText string) string {
	tokens := strings.Fields(inputText)
	resolvedTokens := make([]string, len(tokens))
	copy(resolvedTokens, tokens)

	// Simple pronoun resolution for now
	pronouns := map[string]bool{
		"it":   true,
		"that": true,
		"this": true,
		"them": true,
	}

	for i, token := range tokens {
		lowerToken := strings.ToLower(token)
		if pronouns[lowerToken] {
			// Found a pronoun, now look for an antecedent in previous turns
			antecedent := cc.findAntecedent()
			if antecedent != "" {
				fmt.Printf("Context: Resolved pronoun '%s' to '%s'\n", token, antecedent)
				resolvedTokens[i] = antecedent
				// Once resolved, we can stop for this pronoun to avoid multiple replacements.
				// A more advanced system would handle multiple pronouns.
			}
		}
	}

	return strings.Join(resolvedTokens, " ")
}

// findAntecedent searches backwards in the context for the most likely entity.
func (cc *ConversationContext) findAntecedent() string {
	// Search backwards through the history of entities
	for i := len(cc.LastEntities) - 1; i >= 0; i-- {
		entities := cc.LastEntities[i]

		// Prioritize specific, named entities over generic component types.
		var bestCandidate string

		for _, entity := range entities {
			// Prefer entities that are likely to be nouns/names.
			switch entity.Type {
			case "handler_name", "database_name", "project_name", "file", "folder", "name", "FILENAME", "NAME":
				return entity.Value // Return the first high-priority entity found
			case "component", "feature":
				if bestCandidate == "" {
					bestCandidate = entity.Value
				}
			}
		}

		if bestCandidate != "" {
			return bestCandidate
		}
	}
	return "" // No suitable antecedent found
}
