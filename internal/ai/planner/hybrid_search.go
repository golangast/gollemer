package planner

import (
	"math"
	"regexp"
	"sort"
	"strings"
	"sync"
)

type searchResult struct {
	candidate string
	bm25Score float64
	vecScore  float64
	combined  float64
}

// HybridSearchConfig holds parameters for the hybrid search.
type HybridSearchConfig struct {
	Alpha           float64
	VecDim          int
	TopK            int
	StopWords       map[string]bool
	IdentifierRegex *regexp.Regexp
}

// NewDefaultHybridSearchConfig creates a config with sensible defaults.
func NewDefaultHybridSearchConfig() *HybridSearchConfig {
	return &HybridSearchConfig{
		Alpha:     0.4,
		VecDim:    128,
		TopK:      20,
		StopWords: defaultStopWords(),
		IdentifierRegex: regexp.MustCompile(
			`[a-zA-Z_][a-zA-Z0-9_]*([A-Z][a-zA-Z0-9_]*)*`,
		),
	}
}

func defaultStopWords() map[string]bool {
	return map[string]bool{
		"add": true, "create": true, "remove": true, "delete": true, "update": true,
		"modify": true, "fix": true, "implement": true, "layer": true, "to": true,
		"in": true, "for": true, "the": true, "a": true, "an": true, "and": true,
		"or": true, "resolution": true, "caching": true, "cache": true, "struct": true,
		"method": true, "methods": true, "function": true, "file": true,
	}
}

// HybridIndex stores symbol embeddings and text for hybrid retrieval.
type HybridIndex struct {
	mu      sync.RWMutex
	symbols []hybridSymbol
	dim     int
	embedFn func(text string) []float64
}

type hybridSymbol struct {
	name       string
	signature  string
	docComment string
	kind       string
	file       string
	line       int
	vector     []float64
}

// NewHybridIndex creates a new hybrid search index.
func NewHybridIndex(dim int, embedFn func(text string) []float64) *HybridIndex {
	return &HybridIndex{
		symbols: make([]hybridSymbol, 0),
		dim:     dim,
		embedFn: embedFn,
	}
}

// IndexSymbol adds a symbol to the index.
func (hi *HybridIndex) IndexSymbol(name, signature, docComment, kind, file string, line int) {
	hi.mu.Lock()
	defer hi.mu.Unlock()

	text := name + " " + signature + " " + docComment
	vec := hi.embedFn(text)

	hi.symbols = append(hi.symbols, hybridSymbol{
		name:       name,
		signature:  signature,
		docComment: docComment,
		kind:       kind,
		file:       file,
		line:       line,
		vector:     vec,
	})
}

// IndexSymbols bulk-indexes symbols from a slice.
func (hi *HybridIndex) IndexSymbols(symbols []*SymbolEntry) {
	hi.mu.Lock()
	defer hi.mu.Unlock()

	for _, sym := range symbols {
		text := sym.Name + " " + sym.Signature + " " + sym.DocComment
		vec := hi.embedFn(text)

		hi.symbols = append(hi.symbols, hybridSymbol{
			name:       sym.Name,
			signature:  sym.Signature,
			docComment: sym.DocComment,
			kind:       sym.Kind,
			file:       sym.File,
			line:       sym.Line,
			vector:     vec,
		})
	}
}

// Search performs hybrid BM25 + vector similarity search.
func (hi *HybridIndex) Search(query string, cfg *HybridSearchConfig) []searchResult {
	hi.mu.RLock()
	defer hi.mu.RUnlock()

	if len(hi.symbols) == 0 {
		return nil
	}

	queryWords := extractQueryTokens(query, cfg)
	queryVec := hi.embedFn(query)

	bm25AvgDocLen := 0.0
	if len(hi.symbols) > 0 {
		for _, s := range hi.symbols {
			bm25AvgDocLen += float64(len(s.name) + len(s.signature))
		}
		bm25AvgDocLen /= float64(len(hi.symbols))
	}

	var results []searchResult
	for _, sym := range hi.symbols {
		bm25 := bm25Score(queryWords, sym, bm25AvgDocLen)
		vecSim := cosineSimilarity(queryVec, sym.vector)
		combined := cfg.Alpha*bm25 + (1.0-cfg.Alpha)*vecSim

		results = append(results, searchResult{
			candidate: sym.name,
			bm25Score: bm25,
			vecScore:  vecSim,
			combined:  combined,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].combined > results[j].combined
	})

	if len(results) > cfg.TopK {
		results = results[:cfg.TopK]
	}

	return results
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

type SymbolEntry struct {
	Name       string
	Signature  string
	DocComment string
	Kind       string
	File       string
	Line       int
}

func extractQueryTokens(query string, cfg *HybridSearchConfig) []string {
	words := strings.FieldsFunc(query, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_')
	})

	var tokens []string
	for _, w := range words {
		lower := strings.ToLower(w)
		if len(w) <= 2 || cfg.StopWords[lower] {
			continue
		}
		tokens = append(tokens, lower)
	}
	return tokens
}

func bm25Score(queryWords []string, sym hybridSymbol, avgDocLen float64) float64 {
	if len(queryWords) == 0 || avgDocLen <= 0 {
		return 0
	}

	const (
		k1 = 1.5
		b  = 0.75
	)

	docLen := float64(len(sym.name) + len(sym.signature))
	score := 0.0

	for _, qw := range queryWords {
		qwLower := strings.ToLower(qw)
		count := strings.Count(strings.ToLower(sym.name), qwLower)
		count += strings.Count(strings.ToLower(sym.signature), qwLower)
		count += strings.Count(strings.ToLower(sym.docComment), qwLower)

		if count == 0 {
			continue
		}

		idf := math.Log(1.0 + float64(len(sym.name)+1.0)/0.5)
		numerator := float64(count) * (k1 + 1.0)
		denominator := float64(count) + k1*(1.0-b+b*docLen/avgDocLen)
		score += idf * numerator / denominator
	}

	return score
}
