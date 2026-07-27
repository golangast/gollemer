package planner

import (
	"strings"

	"github.com/golangast/gollemer/internal/ai/knowledge"
)

// ConceptMatcher scans user queries for concept terms and synonyms,
// retrieving the corresponding AST mutation rules from the knowledge registry.
type ConceptMatcher struct {
	registry *knowledge.Registry
}

// NewConceptMatcher creates a matcher backed by the default knowledge registry.
func NewConceptMatcher() *ConceptMatcher {
	return &ConceptMatcher{
		registry: knowledge.NewRegistry(),
	}
}

// NewConceptMatcherWithRegistry creates a matcher with a custom registry.
func NewConceptMatcherWithRegistry(reg *knowledge.Registry) *ConceptMatcher {
	return &ConceptMatcher{
		registry: reg,
	}
}

// ExtractConcepts scans the query for known concept terms and synonyms,
// returning all matching concept templates. Matching is case-insensitive
// and uses substring containment for fuzzy-like matching.
func (cm *ConceptMatcher) ExtractConcepts(query string) []knowledge.ConceptTemplate {
	queryLower := strings.ToLower(query)
	var matched []knowledge.ConceptTemplate

	for _, concept := range cm.registry.All() {
		// Check primary term name
		if strings.Contains(queryLower, strings.ToLower(concept.Term)) {
			matched = append(matched, concept)
			continue
		}
		// Check synonyms
		for _, syn := range concept.Synonyms {
			if strings.Contains(queryLower, strings.ToLower(syn)) {
				matched = append(matched, concept)
				break
			}
		}
	}
	return matched
}

// ExtractConceptsWithScores performs concept extraction with a simple
// relevance scoring mechanism. Each match is scored based on:
// - Exact term match: 3 points
// - Synonym match: 2 points
// - Partial word overlap: 1 point
// Returns concepts sorted by score descending.
func (cm *ConceptMatcher) ExtractConceptsWithScores(query string) []ScoredConcept {
	queryLower := strings.ToLower(query)
	queryWords := strings.FieldsFunc(queryLower, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})

	var scored []ScoredConcept

	for _, concept := range cm.registry.All() {
		score := 0
		termLower := strings.ToLower(concept.Term)

		// Check for exact term match
		if strings.Contains(queryLower, termLower) {
			score += 3
		} else {
			// Check partial word overlap with term
			termWords := strings.FieldsFunc(termLower, func(r rune) bool {
				return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
			})
			for _, tw := range termWords {
				for _, qw := range queryWords {
					if tw == qw {
						score++
						break
					}
				}
			}
		}

		// Check synonyms
		for _, syn := range concept.Synonyms {
			synLower := strings.ToLower(syn)
			if strings.Contains(queryLower, synLower) {
				score += 2
				break
			}
			// Check partial word overlap with synonym
			synWords := strings.FieldsFunc(synLower, func(r rune) bool {
				return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
			})
			for _, sw := range synWords {
				for _, qw := range queryWords {
					if sw == qw {
						score++
						break
					}
				}
			}
		}

		if score > 0 {
			scored = append(scored, ScoredConcept{
				Concept: concept,
				Score:   score,
			})
		}
	}

	// Sort by score descending
	for i := 0; i < len(scored); i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].Score > scored[i].Score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}

	return scored
}

// ScoredConcept pairs a concept template with its relevance score.
type ScoredConcept struct {
	Concept knowledge.ConceptTemplate
	Score   int
}

// Registry returns the underlying knowledge registry for inspection.
func (cm *ConceptMatcher) Registry() *knowledge.Registry {
	return cm.registry
}

// IngestBookFromFile reads a Go textbook file (Markdown, TXT, or PDF text dump),
// extracts concepts, idioms, and patterns, and registers them into the matcher's
// knowledge registry. Returns the ingestion result with all extracted concepts.
//
// Example usage:
//
//	result, err := matcher.IngestBookFromFile("the-go-programming-language.md")
//	fmt.Printf("Extracted %d concepts from %s\n", len(result.Concepts), result.BookTitle)
func (cm *ConceptMatcher) IngestBookFromFile(path string) (*knowledge.IngestionResult, error) {
	ingester := knowledge.NewBookIngester(cm.registry)
	return ingester.IngestFile(path)
}

// IngestBookFromText processes raw Go book content as a string, extracting
// concepts and registering them into the matcher's knowledge registry.
//
// Example usage:
//
//	content := `# Go Concurrency\n\n## Worker Pools\n\nA worker pool uses...`
//	result := matcher.IngestBookFromText(content, "Go Concurrency Guide")
func (cm *ConceptMatcher) IngestBookFromText(text string, bookTitle string) *knowledge.IngestionResult {
	ingester := knowledge.NewBookIngester(cm.registry)
	return ingester.IngestText(text, bookTitle)
}

// ExportConcepts serializes all registered concepts to a JSON file for reuse.
// This allows saving book-derived knowledge and reloading it later without
// re-parsing the source material.
func (cm *ConceptMatcher) ExportConcepts(outputPath string) error {
	ingester := knowledge.NewBookIngester(cm.registry)
	return ingester.ExportConcepts(outputPath)
}

// ImportConcepts loads previously exported concepts from a JSON file and
// merges them into the matcher's registry. Existing concepts are not overwritten.
func (cm *ConceptMatcher) ImportConcepts(path string) error {
	ingester := knowledge.NewBookIngester(cm.registry)
	return ingester.ImportConcepts(path)
}
