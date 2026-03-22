package nertagger

import (
	"strings"

	"github.com/golangast/gollemer/internal/ai/tagger/tag"
)

// EntityPostProcessor applies semantic filtering and validation to extracted entities
type EntityPostProcessor struct {
	tokens         []string
	posTags        []string
	stopWordFilter *StopWordFilter
	semanticParser *SemanticParser
	tree           *DependencyTree
	srl            *SRLAnalysis
}

// NewEntityPostProcessor creates a new entity post-processor
func NewEntityPostProcessor(tokens []string, posTags []string) *EntityPostProcessor {
	parser := NewSemanticParser(tokens, posTags)
	tree, srl := parser.Parse()

	return &EntityPostProcessor{
		tokens:         tokens,
		posTags:        posTags,
		stopWordFilter: NewStopWordFilter(),
		semanticParser: parser,
		tree:           tree,
		srl:            srl,
	}
}

// ProcessParameters applies semantic filtering to command parameters
func (ep *EntityPostProcessor) ProcessParameters(params map[string]string) map[string]string {
	processed := make(map[string]string)

	for key, value := range params {
		// Filter out intent/object keywords
		filtered := ep.stopWordFilter.FilterParameter(value)

		// Clean up whitespace
		filtered = strings.TrimSpace(filtered)

		// Only keep non-empty parameters
		if filtered != "" {
			processed[key] = filtered
		}
	}

	return processed
}

// ExtractPrimaryEntity extracts the most likely entity name from the sentence
// Uses leaf nodes and entity heads from the dependency tree
func (ep *EntityPostProcessor) ExtractPrimaryEntity() string {
	// Strategy 1: Look for entities marked as PATIENT role
	if patients, ok := ep.srl.Arguments[Patient]; ok && len(patients) > 0 {
		return ep.tokens[patients[0]]
	}

	// Strategy 2: Extract leaf nodes (likely to be proper nouns/entity names)
	leaves := ep.tree.ExtractLeafNodes()
	for _, leaf := range leaves {
		lower := strings.ToLower(leaf)
		// Skip function words and keywords
		if !ep.stopWordFilter.IntentKeywords[lower] && !ep.stopWordFilter.ObjectKeywords[lower] {
			return leaf
		}
	}

	// Strategy 3: Find proper nouns (NNP)
	for i, pos := range ep.posTags {
		if pos == "NNP" {
			return ep.tokens[i]
		}
	}

	// Strategy 4: Find entity heads from dependency tree
	heads := ep.tree.FindEntityHeads()
	if len(heads) > 0 {
		return ep.tokens[heads[0]]
	}

	return ""
}

// ExtractEntitiesByRole extracts all entities for a given semantic role
func (ep *EntityPostProcessor) ExtractEntitiesByRole(role SemanticRole) []string {
	var entities []string

	if indices, ok := ep.srl.Arguments[role]; ok {
		for _, idx := range indices {
			entity := ep.tokens[idx]
			lower := strings.ToLower(entity)

			// Skip keywords and punctuation
			if !ep.stopWordFilter.IntentKeywords[lower] && !ep.stopWordFilter.ObjectKeywords[lower] {
				// Filter out punctuation
				entity = strings.TrimRight(entity, ".,!?;:")
				if entity != "" {
					entities = append(entities, entity)
				}
			}
		}
	}

	return entities
}

// ValidateAndCleanEntity validates an entity against stop-word filter
func (ep *EntityPostProcessor) ValidateAndCleanEntity(entity string) (string, bool) {
	cleaned := strings.TrimSpace(entity)
	cleaned = strings.TrimRight(cleaned, ".,!?;:")

	// Check if it contains only keywords
	tokens := strings.Fields(cleaned)
	hasNonKeyword := false

	for _, token := range tokens {
		cleanToken := strings.ToLower(strings.TrimRight(token, ".,!?;:"))
		if !ep.stopWordFilter.IntentKeywords[cleanToken] && !ep.stopWordFilter.ObjectKeywords[cleanToken] {
			hasNonKeyword = true
			break
		}
	}

	return cleaned, hasNonKeyword && cleaned != ""
}

// ========== INTEGRATED NER WITH SRL ==========

// NertaggerWithSRL performs Named Entity Recognition with Semantic Role Labeling
// This combines IOB tagging with semantic understanding for better accuracy
func NertaggerWithSRL(t tag.Tag) tag.Tag {
	// First, do IOB tagging
	t = NertaggerWithIOB(t)

	// Then, enhance with semantic role labeling
	postProcessor := NewEntityPostProcessor(t.Tokens, t.PosTag)

	// Enhance the NER tags with SRL information
	enhancedTags := make([]string, len(t.Tokens))
	copy(enhancedTags, t.NerTag)

	for role, indices := range postProcessor.srl.Arguments {
		for _, idx := range indices {
			if idx < len(enhancedTags) {
				// Map semantic roles to NER tags
				switch role {
				case Agent, Patient:
					if !strings.HasPrefix(enhancedTags[idx], "B-NAME") {
						enhancedTags[idx] = "B-NAME"
					}
				case Predicate:
					if !strings.HasPrefix(enhancedTags[idx], "B-ACTION") {
						enhancedTags[idx] = "B-ACTION"
					}
				case Location:
					if !strings.HasPrefix(enhancedTags[idx], "B-PATH") {
						enhancedTags[idx] = "B-PATH"
					}
				case Instrument, Attribute:
					if !strings.HasPrefix(enhancedTags[idx], "B-ATTRIBUTE") {
						enhancedTags[idx] = "B-ATTRIBUTE"
					}
				}
			}
		}
	}

	t.NerTag = enhancedTags
	return t
}

// ExtractCleanParameters extracts and cleans parameters using SRL
func ExtractCleanParameters(tokens []string, posTags []string, paramTriggers map[string]string) map[string]string {
	postProcessor := NewEntityPostProcessor(tokens, posTags)
	rawParams := make(map[string]string)

	// Extract raw parameters using trigger words
	for i, token := range tokens {
		lower := strings.ToLower(token)
		if paramKey, isParam := paramTriggers[lower]; isParam {
			// Look ahead for the parameter value
			if i+1 < len(tokens) {
				value := tokens[i+1]
				// Skip function words
				j := i + 1
				for j < len(tokens) {
					if tokens[j] == "the" || tokens[j] == "a" || tokens[j] == "an" {
						j++
						continue
					}
					value = tokens[j]
					break
				}
				rawParams[paramKey] = value
			}
		}
	}

	// Clean and validate parameters
	return postProcessor.ProcessParameters(rawParams)
}

// ========== DEBUG/ANALYSIS FUNCTIONS ==========

// AnalyzeSentence provides detailed analysis of a sentence for debugging
func AnalyzeSentence(tokens []string, posTags []string) map[string]any {
	parser := NewSemanticParser(tokens, posTags)
	tree, srl := parser.Parse()
	postProcessor := NewEntityPostProcessor(tokens, posTags)

	analysis := map[string]any{
		"tokens":        tokens,
		"pos_tags":      posTags,
		"predicate":     srl.Predicate,
		"predicate_idx": srl.PredIdx,
		"dependency_tree": map[string]any{
			"edges":        tree.Edges,
			"parents":      tree.Parent,
			"leaf_nodes":   tree.ExtractLeafNodes(),
			"entity_heads": tree.FindEntityHeads(),
		},
		"semantic_roles": map[string][]string{},
		"primary_entity": postProcessor.ExtractPrimaryEntity(),
	}

	// Convert semantic roles to strings for JSON marshaling
	for role, indices := range srl.Arguments {
		var roleTokens []string
		for _, idx := range indices {
			roleTokens = append(roleTokens, tokens[idx])
		}
		analysis["semantic_roles"].(map[string][]string)[string(role)] = roleTokens
	}

	return analysis
}
