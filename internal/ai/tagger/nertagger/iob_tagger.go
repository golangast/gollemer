package nertagger

import (
	"regexp"
	"strings"
	"sync"

	"github.com/golangast/gollemer/internal/ai/tagger/tag"
)

// IOBTagType represents the IOB tag classification
type IOBTagType string

const (
	// IOB tag prefixes
	B_PREFIX IOBTagType = "B-" // Beginning of entity
	I_PREFIX IOBTagType = "I-" // Inside entity
	O_PREFIX IOBTagType = "O"  // Outside entity

	// Entity types for IOB tagging
	B_ACTION    = "B-ACTION"
	I_ACTION    = "I-ACTION"
	B_TYPE      = "B-TYPE"
	I_TYPE      = "I-TYPE"
	B_NAME      = "B-NAME"
	I_NAME      = "I-NAME"
	B_ATTRIBUTE = "B-ATTRIBUTE"
	I_ATTRIBUTE = "I-ATTRIBUTE"
	B_TARGET    = "B-TARGET"
	I_TARGET    = "I-TARGET"
	B_SOURCE    = "B-SOURCE"
	I_SOURCE    = "I-SOURCE"
	B_URL       = "B-URL"
	I_URL       = "I-URL"
	B_PATH      = "B-PATH"
	I_PATH      = "I-PATH"
	O_TAG       = "O"
)

// iobRule defines patterns for IOB entity recognition
type iobRule struct {
	pattern      *regexp.Regexp
	entityType   string
	triggerWords map[string]bool // Words that can trigger continuation
}

var (
	iobRules     []iobRule
	iobRulesOnce sync.Once
)

// initIOBRules initializes the IOB pattern matching rules
func initIOBRules() {
	iobRules = []iobRule{
		// COMMAND/ACTION patterns (create, add, delete, etc.)
		{
			pattern:    regexp.MustCompile(`(?i)\b(create|add|make|generate|initialize|init|new|setup|start|put|copy)\b`),
			entityType: "ACTION",
			triggerWords: map[string]bool{
				"a": false, "an": false, "the": false, "with": false, "for": false,
			},
		},
		// COMMAND/ACTION patterns (delete, remove, move)
		{
			pattern:    regexp.MustCompile(`(?i)\b(delete|remove|mv|move)\b`),
			entityType: "ACTION",
			triggerWords: map[string]bool{
				"a": false, "an": false, "the": false,
			},
		},
		// OBJECT_TYPE patterns (webserver, database, handler, etc.)
		{
			pattern:    regexp.MustCompile(`(?i)\b(webserver|website|site|application|app)\b`),
			entityType: "TYPE",
			triggerWords: map[string]bool{
				"named": true, "called": true,
			},
		},
		{
			pattern:    regexp.MustCompile(`(?i)\b(database|db|table)\b`),
			entityType: "TYPE",
			triggerWords: map[string]bool{
				"named": true, "called": true, "with": false,
			},
		},
		{
			pattern:    regexp.MustCompile(`(?i)\b(handler|endpoint|route)\b`),
			entityType: "TYPE",
			triggerWords: map[string]bool{
				"named": true, "called": true, "at": true, "url": true,
			},
		},
		{
			pattern:    regexp.MustCompile(`(?i)\b(page|view|component)\b`),
			entityType: "TYPE",
			triggerWords: map[string]bool{
				"named": true, "called": true,
			},
		},
		{
			pattern:    regexp.MustCompile(`(?i)\b(file|folder|directory|structure|data\s+structure)\b`),
			entityType: "TYPE",
			triggerWords: map[string]bool{
				"named": true, "called": true, "in": true, "into": true,
			},
		},
		// NAME_PREFIX patterns (named, called)
		{
			pattern:      regexp.MustCompile(`(?i)\b(named|called)\b`),
			entityType:   "NAME",
			triggerWords: map[string]bool{}, // Trigger for next token
		},
		// URL patterns
		{
			pattern:    regexp.MustCompile(`^/[a-zA-Z0-9/_-]*`),
			entityType: "URL",
			triggerWords: map[string]bool{
				"at": false, "url": false,
			},
		},
		// PATH patterns
		{
			pattern:    regexp.MustCompile(`^[a-zA-Z0-9_./\\-]+$`),
			entityType: "PATH",
			triggerWords: map[string]bool{
				"in": true, "into": true, "to": true,
			},
		},
	}
}

// IOBTagger performs IOB (Inside-Outside-Beginning) tagging on tokens
// This prevents the "greedy slot" problem by explicitly marking entity boundaries
type IOBTagger struct {
	tokens       []string
	posTags      []string
	iobTags      []string
	entityBounds map[int]string // Maps token index to entity type
}

// NewIOBTagger creates a new IOB tagger
func NewIOBTagger(tokens []string, posTags []string) *IOBTagger {
	iobRulesOnce.Do(initIOBRules)

	tagger := &IOBTagger{
		tokens:       tokens,
		posTags:      posTags,
		iobTags:      make([]string, len(tokens)),
		entityBounds: make(map[int]string),
	}

	// Initialize all tags as "O" (Outside)
	for i := range tagger.iobTags {
		tagger.iobTags[i] = O_TAG
	}

	return tagger
}

// Tag performs the IOB tagging process
func (t *IOBTagger) Tag() []string {
	// Phase 1: Identify entity beginnings
	t.markEntityBeginnings()

	// Phase 2: Extend entities (mark continuations as "I-")
	t.extendEntities()

	// Phase 3: Handle special cases and conflicts
	t.resolveConflicts()

	// Phase 4: Apply POS tag fallback for untagged tokens
	t.applyPOSFallback()

	return t.iobTags
}

// markEntityBeginnings identifies the start of entities (B- tags)
func (t *IOBTagger) markEntityBeginnings() {
	for i, token := range t.tokens {
		if t.iobTags[i] != O_TAG {
			continue // Already tagged
		}

		// Check against IOB rules
		for _, rule := range iobRules {
			if rule.pattern.MatchString(token) {
				// Determine the B- tag
				var btag string
				switch rule.entityType {
				case "ACTION":
					btag = B_ACTION
				case "TYPE":
					btag = B_TYPE
				case "NAME":
					btag = B_NAME
				case "ATTRIBUTE":
					btag = B_ATTRIBUTE
				case "TARGET":
					btag = B_TARGET
				case "SOURCE":
					btag = B_SOURCE
				case "URL":
					btag = B_URL
				case "PATH":
					btag = B_PATH
				default:
					btag = "B-" + rule.entityType
				}

				t.iobTags[i] = btag
				t.entityBounds[i] = rule.entityType
				break
			}
		}

		// Special handling: proper nouns become B-NAME
		if t.iobTags[i] == O_TAG && (t.posTags[i] == "NNP" || t.posTags[i] == "NNPS") {
			t.iobTags[i] = B_NAME
			t.entityBounds[i] = "NAME"
		}
	}
}

// extendEntities marks continuation tokens as "I-" (Inside)
func (t *IOBTagger) extendEntities() {
	var lastEntityType string
	var lastWasEntity bool

	for i := 0; i < len(t.tokens); i++ {
		currentTag := t.iobTags[i]

		// If current token is B-*, prepare for continuation
		if after, ok := strings.CutPrefix(currentTag, "B-"); ok {
			lastEntityType = after
			lastWasEntity = true
			continue
		}

		if currentTag == O_TAG && lastWasEntity && lastEntityType != "" {
			// Check if this token should continue the entity
			lower := strings.ToLower(t.tokens[i])

			// Heuristic 1: Don't continue past separators
			separators := map[string]bool{
				"named": true, "called": true, "in": true, "into": true,
				"to": true, "for": true, "with": true, "from": true,
				"at": true, "url": true, "is": true, "and": true,
			}

			if separators[lower] || t.posTags[i] == "IN" || t.posTags[i] == "CC" {
				lastWasEntity = false
				continue
			}

			// Heuristic 2: Continue if it's a noun or number after TYPE or NAME
			if (lastEntityType == "TYPE" || lastEntityType == "NAME") &&
				(t.posTags[i] == "NN" || t.posTags[i] == "NNS" || t.posTags[i] == "NNP" || t.posTags[i] == "NNPS" || t.posTags[i] == "CD") {
				itag := "I-" + lastEntityType
				t.iobTags[i] = itag
				continue
			}

			// Heuristic 3: Continue TYPE if next token is an adjective or noun
			if lastEntityType == "TYPE" && (t.posTags[i] == "JJ" || t.posTags[i] == "NN" || t.posTags[i] == "NNS") {
				itag := "I-" + lastEntityType
				t.iobTags[i] = itag
				continue
			}

			lastWasEntity = false
		} else if currentTag == O_TAG {
			lastWasEntity = false
		}
	}
}

// resolveConflicts handles overlapping entity boundaries and special cases
func (t *IOBTagger) resolveConflicts() {
	// Handle "data structure" as a single TYPE entity
	for i := 0; i < len(t.tokens)-1; i++ {
		if strings.ToLower(t.tokens[i]) == "data" && strings.ToLower(t.tokens[i+1]) == "structure" {
			if t.iobTags[i] == O_TAG {
				t.iobTags[i] = B_TYPE
				t.iobTags[i+1] = I_TYPE
				t.entityBounds[i] = "TYPE"
			}
		}
	}

	// Handle multi-word URLs
	for i := 0; i < len(t.tokens); i++ {
		if strings.HasPrefix(t.iobTags[i], "B-URL") {
			// Continue URL if next token is part of path
			j := i + 1
			for j < len(t.tokens) && (strings.Contains(t.tokens[j], "/") || strings.Contains(t.tokens[j], "-") || t.posTags[j] == "NN") {
				t.iobTags[j] = "I-URL"
				j++
			}
		}
	}

	// Ensure NAME tags only follow NAME_PREFIX or immediately follow TYPE
	for i := 1; i < len(t.tokens); i++ {
		if strings.HasPrefix(t.iobTags[i], "B-NAME") {
			prev := strings.ToLower(t.tokens[i-1])
			prevTag := t.iobTags[i-1]

			// NAME is valid after "named" or "called"
			if prev == "named" || prev == "called" {
				continue
			}

			// NAME is valid immediately after TYPE
			if strings.HasPrefix(prevTag, "B-TYPE") || strings.HasPrefix(prevTag, "I-TYPE") {
				continue
			}

			// NAME is valid for proper nouns
			if t.posTags[i] == "NNP" || t.posTags[i] == "NNPS" {
				continue
			}

			// Otherwise, reset to O
			if prev != "named" && prev != "called" {
				t.iobTags[i] = O_TAG
			}
		}
	}
}

// applyPOSFallback assigns POS tags to untagged tokens
func (t *IOBTagger) applyPOSFallback() {
	for i := 0; i < len(t.iobTags); i++ {
		if t.iobTags[i] == O_TAG {
			// Use POS tag as fallback
			t.iobTags[i] = t.posTags[i]
		}
	}
}

// TagTokensWithIOB performs IOB tagging on tokens
func TagTokensWithIOB(tokens []string, posTags []string) []string {
	tagger := NewIOBTagger(tokens, posTags)
	return tagger.Tag()
}

// NertaggerWithIOB extends the Tag struct with IOB tags
func NertaggerWithIOB(t tag.Tag) tag.Tag {
	// Store original NER tags for reference
	t.NerTag = TagTokensWithIOB(t.Tokens, t.PosTag)
	return t
}

// EntitySpan represents a recognized entity in the text
type EntitySpan struct {
	StartIdx int
	EndIdx   int
	Text     string
}

// ParseIOBTags extracts entity spans from IOB-tagged tokens
// Returns a map of entity types to slices of entity spans
func ParseIOBTags(tokens []string, iobTags []string) map[string][]EntitySpan {
	entities := make(map[string][]EntitySpan)

	var currentEntity string
	var currentStart int

	for i := range iobTags {
		tag := iobTags[i]

		if strings.HasPrefix(tag, "B-") {
			// Save previous entity if any
			if currentEntity != "" {
				entityType := currentEntity
				text := strings.Join(tokens[currentStart:i], " ")
				entities[entityType] = append(entities[entityType], EntitySpan{
					StartIdx: currentStart,
					EndIdx:   i,
					Text:     text,
				})
			}

			// Start new entity
			currentEntity = strings.TrimPrefix(tag, "B-")
			currentStart = i
		} else if after, ok := strings.CutPrefix(tag, "I-"); ok {
			entityType := after
			// Verify continuity
			if currentEntity != entityType {
				// This shouldn't happen in well-formed IOB, but handle it
				if currentEntity != "" {
					text := strings.Join(tokens[currentStart:i], " ")
					entities[currentEntity] = append(entities[currentEntity], EntitySpan{
						StartIdx: currentStart,
						EndIdx:   i,
						Text:     text,
					})
				}
				currentEntity = entityType
				currentStart = i
			}
		} else {
			// Tag is "O" or a POS tag - end current entity
			if currentEntity != "" {
				text := strings.Join(tokens[currentStart:i], " ")
				entities[currentEntity] = append(entities[currentEntity], EntitySpan{
					StartIdx: currentStart,
					EndIdx:   i,
					Text:     text,
				})
				currentEntity = ""
			}
		}
	}

	// Don't forget the last entity
	if currentEntity != "" {
		text := strings.Join(tokens[currentStart:], " ")
		entities[currentEntity] = append(entities[currentEntity], EntitySpan{
			StartIdx: currentStart,
			EndIdx:   len(tokens),
			Text:     text,
		})
	}

	return entities
}

// ValidateIOBSequence checks if an IOB tag sequence is valid
// Returns true if the sequence follows IOB rules
func ValidateIOBSequence(tags []string) bool {
	if len(tags) == 0 {
		return true
	}

	var lastType string

	for i, tag := range tags {
		if tag == "O" {
			lastType = ""
			continue
		}

		if after, ok := strings.CutPrefix(tag, "B-"); ok {
			lastType = after
			continue
		}

		if after, ok := strings.CutPrefix(tag, "I-"); ok {
			currentType := after
			// I- must follow B- of the same type, or another I- of the same type
			if lastType != currentType {
				// Check previous tag
				if i == 0 {
					return false // I- cannot be first tag
				}
				prevTag := tags[i-1]
				if !strings.HasPrefix(prevTag, "B-"+currentType) && !strings.HasPrefix(prevTag, "I-"+currentType) {
					return false
				}
			}
			lastType = currentType
			continue
		}

		// Unknown format
		return false
	}

	return true
}
