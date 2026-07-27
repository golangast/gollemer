package memory

import (
	"crypto/sha256"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type Record struct {
	ID        string    `json:"id"`
	Text      string    `json:"text"`
	Vector    []float64 `json:"vector"`
	Timestamp time.Time `json:"timestamp"`
}

// VectorDB is a lightweight, pure Go vector database for local contextual memory.
type VectorDB struct {
	mu      sync.RWMutex
	records []Record
	dim     int
	dbPath  string
}

func NewVectorDB(dim int, path string) *VectorDB {
	db := &VectorDB{
		records: make([]Record, 0),
		dim:     dim,
		dbPath:  path,
	}
	db.Load()
	return db
}

// Embed generates a fast, pure Go vector representation of the text using a
// character n-gram hashing trick. This avoids heavy ML dependencies.
func (db *VectorDB) Embed(text string) []float64 {
	vec := make([]float64, db.dim)
	text = strings.ToLower(strings.TrimSpace(text))

	// Trigram hashing
	runes := []rune(text)
	for i := 0; i < len(runes)-2; i++ {
		trigram := string(runes[i : i+3])
		hash := sha256.Sum256([]byte(trigram))

		// Map hash bytes into the vector
		for j := 0; j < len(hash)-1; j += 2 {
			idx := (int(hash[j])<<8 | int(hash[j+1])) % db.dim
			vec[idx] += 1.0
		}
	}

	// Also hash full words
	words := strings.Fields(text)
	for _, w := range words {
		hash := sha256.Sum256([]byte(w))
		for j := 0; j < len(hash)-1; j += 2 {
			idx := (int(hash[j])<<8 | int(hash[j+1])) % db.dim
			vec[idx] += 1.5
		}
	}

	// L2 Normalize
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

func (db *VectorDB) Insert(text string) {
	db.mu.Lock()
	defer db.mu.Unlock()

	vec := db.Embed(text)
	db.records = append(db.records, Record{
		ID:        time.Now().Format(time.RFC3339Nano),
		Text:      text,
		Vector:    vec,
		Timestamp: time.Now(),
	})

	db.save()
}

type SearchResult struct {
	Record Record
	Score  float64
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
		// Only include somewhat relevant matches
		if score > 0.1 {
			results = append(results, SearchResult{
				Record: rec,
				Score:  score,
			})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > topK {
		results = results[:topK]
	}
	return results
}

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
