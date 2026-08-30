package memory

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// Record represents a single entry in the vector database.
type Record struct {
	ID        string    `json:"id"`
	Text      string    `json:"text"`
	Type      string    `json:"type,omitempty"`
	Category  string    `json:"category,omitempty"`
	Vector    []float64 `json:"vector"`
	Timestamp string    `json:"timestamp"`
}

// SearchResult holds a matched record and its similarity score.
type SearchResult struct {
	Record Record
	Score  float64
}

// VectorDB provides in-memory vector storage with JSON persistence.
type VectorDB struct {
	mu      sync.RWMutex
	records []Record
	dim     int
	dbPath  string
}

// NewVectorDB creates a new vector database and loads existing data.
func NewVectorDB(dim int, path string) *VectorDB {
	db := &VectorDB{
		records: make([]Record, 0),
		dim:     dim,
		dbPath:  path,
	}
	db.Load()
	return db
}

// Embed generates a fast character n-gram hash vector for the text.
func (db *VectorDB) Embed(text string) []float64 {
	vec := make([]float64, db.dim)
	text = strings.ToLower(strings.TrimSpace(text))

	runes := []rune(text)
	for i := 0; i < len(runes)-2; i++ {
		trigram := string(runes[i : i+3])
		hash := sha256Hash(trigram)
		for j := 0; j < len(hash)-1; j += 2 {
			idx := (int(hash[j])<<8 | int(hash[j+1])) % db.dim
			vec[idx] += 1.0
		}
	}

	words := strings.Fields(text)
	for _, w := range words {
		hash := sha256Hash(w)
		for j := 0; j < len(hash)-1; j += 2 {
			idx := (int(hash[j])<<8 | int(hash[j+1])) % db.dim
			vec[idx] += 1.5
		}
	}

	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range vec {
			vec[i] /= norm
		}
	}

	return vec
}

// Insert adds a new record to the database.
func (db *VectorDB) Insert(text, recordType, category string) {
	db.mu.Lock()
	defer db.mu.Unlock()

	vec := db.Embed(text)
	db.records = append(db.records, Record{
		ID:        fmt.Sprintf("%d", len(db.records)),
		Text:      text,
		Type:      recordType,
		Category:  category,
		Vector:    vec,
		Timestamp: "2026-06-14T01:34:19.648047652-05:00",
	})
	db.save()
}

// Search returns the top K most similar records.
func (db *VectorDB) Search(text string, topK int) []SearchResult {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if len(db.records) == 0 {
		return nil
	}

	queryVec := db.Embed(text)
	var results []SearchResult
	for _, rec := range db.records {
		score := cosineSimilarity(queryVec, rec.Vector)
		if score > 0.1 {
			results = append(results, SearchResult{
				Record: rec,
				Score:  score,
			})
		}
	}

	sortSearchResults(results)
	if len(results) > topK {
		results = results[:topK]
	}
	return results
}

// RetrieveContext searches the vector database for the top-k most relevant
// entries for a given query and returns a formatted context string suitable
// for prepending to a prompt buffer before generation.
func (db *VectorDB) RetrieveContext(query string, topK int) string {
	results := db.Search(query, topK)
	if len(results) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("[RETRIEVED_CONTEXT]\n")
	for i, r := range results {
		sb.WriteString(fmt.Sprintf("%d. %s (score: %.4f)\n", i+1, r.Record.Text, r.Score))
	}
	sb.WriteString("[/RETRIEVED_CONTEXT]\n")
	return sb.String()
}
func (db *VectorDB) ResolveCoreference(pronoun, context string) string {
	db.mu.RLock()
	defer db.mu.RUnlock()

	lowerPronoun := strings.ToLower(strings.TrimSpace(pronoun))
	lowerContext := strings.ToLower(strings.TrimSpace(context))

	results := db.Search(lowerContext, 5)
	if len(results) == 0 {
		return ""
	}

	for _, r := range results {
		if strings.Contains(lowerPronoun, r.Record.Text) ||
			strings.Contains(r.Record.Text, lowerPronoun) {
			return r.Record.Text
		}
	}

	return results[0].Record.Text
}

// StoreEntity stores a named entity for later coreference resolution.
func (db *VectorDB) StoreEntity(name, category string) {
	db.Insert(name, "entity", category)
}

// Load reads the database from disk.
func (db *VectorDB) Load() error {
	if db.dbPath == "" {
		return nil
	}
	data, err := os.ReadFile(db.dbPath)
	if err != nil {
		return err
	}
	db.mu.Lock()
	defer db.mu.Unlock()
	return json.Unmarshal(data, &db.records)
}

// save writes the database to disk.
func (db *VectorDB) save() error {
	if db.dbPath == "" {
		return nil
	}
	dir := filepath.Dir(db.dbPath)
	os.MkdirAll(dir, 0755)

	data, err := json.MarshalIndent(db.records, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(db.dbPath, data, 0644)
}

func sha256Hash(s string) [32]byte {
	return sha256.Sum256([]byte(s))
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func sortSearchResults(results []SearchResult) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
}
