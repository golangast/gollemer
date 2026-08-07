package llm

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// IntentCorpusEntry represents a single intent category and its examples.
type IntentCorpusEntry struct {
	Intent   string   `json:"intent"`
	Examples []string `json:"examples"`
}

// CachedIntentEmbedding stores the precomputed embedding for a specific example.
type CachedIntentEmbedding struct {
	Intent    string
	Example   string
	Embedding []float64
}

// SemanticRouter handles dynamic loading and matching of trained intent embeddings.
type SemanticRouter struct {
	Embeddings []CachedIntentEmbedding
}

// LoadOrComputeIntentEmbeddings loads the cached embeddings from the gob file if they are up to date.
// Otherwise, it reads the intent corpus JSON, computes new embeddings, and caches them.
func LoadOrComputeIntentEmbeddings(projectRoot string, client *GollemerMoEClient) (*SemanticRouter, error) {
	jsonPath := filepath.Join(projectRoot, "data", "training", "intent_corpus.json")
	gobPath := filepath.Join(projectRoot, "data", "models", "gob_models", "intent_embeddings.gob")

	router := &SemanticRouter{}

	// Check if gob cache exists and is newer than json
	jsonStat, jsonErr := os.Stat(jsonPath)
	if jsonErr != nil {
		return nil, fmt.Errorf("could not stat corpus JSON: %w", jsonErr)
	}
	gobStat, gobErr := os.Stat(gobPath)

	useCache := false
	if gobErr == nil {
		if gobStat.ModTime().After(jsonStat.ModTime()) {
			useCache = true
		}
	}

	if useCache {
		f, err := os.Open(gobPath)
		if err == nil {
			defer f.Close()
			dec := gob.NewDecoder(f)
			if err := dec.Decode(&router.Embeddings); err == nil {
				return router, nil
			}
		}
	}

	// Compute from JSON
	jsonData, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read corpus JSON: %w", err)
	}

	var corpus []IntentCorpusEntry
	if err := json.Unmarshal(jsonData, &corpus); err != nil {
		return nil, fmt.Errorf("failed to parse corpus JSON: %w", err)
	}

	for _, entry := range corpus {
		for _, ex := range entry.Examples {
			emb := client.getSentenceEmbedding(ex)
			if emb != nil {
				router.Embeddings = append(router.Embeddings, CachedIntentEmbedding{
					Intent:    entry.Intent,
					Example:   ex,
					Embedding: emb,
				})
			}
		}
	}

	// Cache to gob
	os.MkdirAll(filepath.Dir(gobPath), 0755)
	f, err := os.Create(gobPath)
	if err == nil {
		defer f.Close()
		enc := gob.NewEncoder(f)
		enc.Encode(router.Embeddings)
	}

	return router, nil
}
