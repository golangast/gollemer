// Package tokenizer provides a Byte-Pair Encoding (BPE) tokenizer
// trained on Go source files and conversational datasets.
//
// BPE learns a ~16k subword vocabulary that covers both code tokens
// (identifiers, keywords, operators) and natural language tokens,
// enabling a single unified tokenizer for the coding+conversation agent.
package tokenizer

import (
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// BPEMerge records a single BPE merge operation: the two symbols being merged.
type BPEMerge struct {
	Left  string
	Right string
}

// BPETokenizer implements a Byte-Pair Encoding tokenizer with a target vocabulary size.
type BPETokenizer struct {
	// Vocabulary maps subword tokens to IDs (compatible with mainvocab.Vocabulary)
	Vocab *mainvocab.Vocabulary

	// Merges ordered list of learned merges (applied in order during encoding)
	Merges []BPEMerge

	// MaxVocabSize target vocabulary size (default ~16k)
	MaxVocabSize int

	// SpecialTokens tracks control tokens that should never be split
	SpecialTokens map[string]bool
}

// NewBPETokenizer creates a BPE tokenizer with the given target vocab size.
func NewBPETokenizer(vocabSize int) *BPETokenizer {
	return &BPETokenizer{
		Vocab:         mainvocab.NewVocabulary(),
		Merges:        make([]BPEMerge, 0),
		MaxVocabSize:  vocabSize,
		SpecialTokens: make(map[string]bool),
	}
}

// AddSpecialToken registers a token that should never be split by BPE.
func (b *BPETokenizer) AddSpecialToken(tok string) {
	b.SpecialTokens[tok] = true
	b.Vocab.AddToken(tok)
}

// collectCorpus scans Go source files and CSV text data to build a word frequency map.
func (b *BPETokenizer) collectCorpus(projectRoot string) (map[string]int, error) {
	freq := make(map[string]int)

	// 1. Collect Go source files
	err := filepath.Walk(filepath.Join(projectRoot, "internal"), func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		words := tokenizeSource(string(data))
		for _, w := range words {
			freq[w]++
		}
		return nil
	})
	if err != nil {
		log.Printf("⚠️  BPE corpus: error walking internal/ (Go files): %v", err)
	}

	// 2. Collect Go source files from cmd/ as well
	filepath.Walk(filepath.Join(projectRoot, "cmd"), func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		words := tokenizeSource(string(data))
		for _, w := range words {
			freq[w]++
		}
		return nil
	})

	// 3. Collect CSV training data
	csvFiles := []string{
		filepath.Join(projectRoot, "data/training/trainingdata/conversations.csv"),
		filepath.Join(projectRoot, "data/training/trainingdata/conversing.csv"),
		filepath.Join(projectRoot, "data/training/trainingdata/synthetic_pairs.csv"),
	}
	for _, csvPath := range csvFiles {
		f, err := os.Open(csvPath)
		if err != nil {
			log.Printf("⚠️  BPE corpus: cannot read %s: %v", csvPath, err)
			continue
		}
		reader := csv.NewReader(f)
		reader.FieldsPerRecord = -1
		reader.LazyQuotes = true
		records, err := reader.ReadAll()
		f.Close()
		if err != nil {
			log.Printf("⚠️  BPE corpus: error parsing %s: %v", csvPath, err)
			continue
		}

		for i, rec := range records {
			if i == 0 && len(rec) > 0 && (strings.EqualFold(rec[0], "query") || strings.EqualFold(rec[0], "role")) {
				continue // skip header row
			}
			// Ingest query (col 0) and answer (col 1) text ONLY
			for col := 0; col < len(rec) && col < 2; col++ {
				text := strings.TrimSpace(rec[col])
				if text == "" {
					continue
				}
				words := tokenizeText(text)
				for _, w := range words {
					freq[w]++
				}
			}
		}
	}

	return freq, nil
}

// tokenizeSource splits Go source code into tokens (preserving identifiers, keywords, operators).
func tokenizeSource(text string) []string {
	var tokens []string
	var current strings.Builder
	runes := []rune(text)

	flush := func() {
		if current.Len() > 0 {
			tokens = append(tokens, current.String())
			current.Reset()
		}
	}

	for _, r := range runes {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			current.WriteRune(r)
		} else {
			flush()
			if !unicode.IsSpace(r) {
				// Preserve single punctuation/operator characters as tokens
				tokens = append(tokens, string(r))
			}
		}
	}
	flush()
	return tokens
}

// tokenizeText splits natural language text into word tokens.
func tokenizeText(text string) []string {
	var tokens []string
	var current strings.Builder

	flush := func() {
		if current.Len() > 0 {
			word := strings.ToLower(current.String())
			if word != "" {
				tokens = append(tokens, word)
			}
			current.Reset()
		}
	}

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' || r == '_' {
			current.WriteRune(r)
		} else {
			flush()
			if !unicode.IsSpace(r) {
				tokens = append(tokens, string(r))
			}
		}
	}
	flush()
	return tokens
}

// Train runs the BPE training algorithm on the corpus and builds the vocabulary.
func (b *BPETokenizer) Train(projectRoot string) error {
	log.Printf("📚 BPE: Collecting corpus (Go source + CSV text data)...")
	freq, err := b.collectCorpus(projectRoot)
	if err != nil {
		return fmt.Errorf("BPE corpus collection failed: %w", err)
	}
	log.Printf("📚 BPE: Collected %d unique word types from corpus", len(freq))

	// Filter out special tokens from word frequencies
	for tok := range b.SpecialTokens {
		delete(freq, tok)
	}

	// Add special tokens first
	for tok := range b.SpecialTokens {
		b.Vocab.AddToken(tok)
	}

	// Convert words to character-level representations for BPE
	type wordEntry struct {
		text  string
		freq  int
		chars []string // current segmentation
	}
	entries := make([]*wordEntry, 0, len(freq))
	for word, count := range freq {
		if count < 1 || word == "" {
			continue
		}
		chars := make([]string, 0)
		for _, r := range word {
			chars = append(chars, string(r))
			// Add each character to vocab
			b.Vocab.AddToken(string(r))
		}
		entries = append(entries, &wordEntry{text: word, freq: count, chars: chars})
	}

	// Sort entries by frequency descending so common words drive merges
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].freq > entries[j].freq
	})

	log.Printf("📚 BPE: Starting merge training (target vocab size: %d)...", b.MaxVocabSize)

	// Track current unique subword set size
	uniqueSubwords := make(map[string]bool)
	for _, e := range entries {
		for _, c := range e.chars {
			uniqueSubwords[c] = true
		}
	}

	// Iteratively merge the most frequent pair
	for len(uniqueSubwords) < b.MaxVocabSize && len(entries) > 0 {
		// Count adjacent pair frequencies across all words
		pairFreq := make(map[[2]string]int)
		for _, e := range entries {
			for i := 0; i < len(e.chars)-1; i++ {
				key := [2]string{e.chars[i], e.chars[i+1]}
				pairFreq[key] += e.freq
			}
		}

		if len(pairFreq) == 0 {
			break
		}

		// Find the most frequent pair
		var bestPair [2]string
		bestCount := 0
		for pair, count := range pairFreq {
			if count > bestCount {
				bestCount = count
				bestPair = pair
			}
		}

		if bestCount < 2 {
			break // no meaningful merges left
		}

		// Record the merge
		merged := bestPair[0] + bestPair[1]
		b.Merges = append(b.Merges, BPEMerge{Left: bestPair[0], Right: bestPair[1]})
		b.Vocab.AddToken(merged)
		uniqueSubwords[merged] = true

		// Apply the merge to all entries
		for _, e := range entries {
			var newChars []string
			i := 0
			for i < len(e.chars) {
				if i+1 < len(e.chars) && e.chars[i] == bestPair[0] && e.chars[i+1] == bestPair[1] {
					newChars = append(newChars, merged)
					i += 2
				} else {
					newChars = append(newChars, e.chars[i])
					i++
				}
			}
			e.chars = newChars
		}
	}

	log.Printf("📚 BPE: Training complete. Vocab size: %d, Merges: %d", b.Vocab.Size(), len(b.Merges))
	return nil
}

// Encode converts a text string into a slice of token IDs using special token matching and BPE merges.
func (b *BPETokenizer) Encode(text string) []int {
	if len(text) == 0 {
		return nil
	}

	var ids []int
	remaining := text

	for len(remaining) > 0 {
		matchedSpecial := false

		// 1. Check if 'remaining' starts with any registered Special Token (longest match first)
		for specTok := range b.SpecialTokens {
			if strings.HasPrefix(remaining, specTok) {
				if id := b.Vocab.GetTokenID(specTok); id >= 0 {
					ids = append(ids, id)
					remaining = remaining[len(specTok):]
					matchedSpecial = true
					break
				}
			}
		}

		if matchedSpecial {
			continue
		}

		// 2. If no special token prefix matched, find the distance to the next potential special token match
		nextSpecialIdx := len(remaining)
		for specTok := range b.SpecialTokens {
			if idx := strings.Index(remaining, specTok); idx > 0 && idx < nextSpecialIdx {
				nextSpecialIdx = idx
			}
		}

		// 3. Extract standard text chunk up to the next special token
		chunk := remaining[:nextSpecialIdx]
		remaining = remaining[nextSpecialIdx:]

		// 4. Tokenize standard chunk through traditional word splitter + BPE merges
		words := tokenizeText(chunk)
		for _, word := range words {
			if word == "\n" {
				if id := b.Vocab.GetTokenID(word); id >= 0 {
					ids = append(ids, id)
				}
				continue
			}

			segments := b.segmentWord(word)
			for _, seg := range segments {
				id := b.Vocab.GetTokenID(seg)
				if id < 0 {
					id = b.Vocab.UnkID
				}
				ids = append(ids, id)
			}
		}
	}

	return ids
}

// segmentWord applies learned BPE merges to segment a single word into subword units.
func (b *BPETokenizer) segmentWord(word string) []string {
	if word == "" {
		return nil
	}

	// Start with character-level segmentation
	segments := make([]string, 0, len(word))
	for _, r := range word {
		segments = append(segments, string(r))
	}

	// Greedily apply merges in the order they were learned
	for _, merge := range b.Merges {
		var newSegments []string
		i := 0
		for i < len(segments) {
			if i+1 < len(segments) && segments[i] == merge.Left && segments[i+1] == merge.Right {
				newSegments = append(newSegments, merge.Left+merge.Right)
				i += 2
			} else {
				newSegments = append(newSegments, segments[i])
				i++
			}
		}
		segments = newSegments
	}

	return segments
}

// Decode converts token IDs back to text.
func (b *BPETokenizer) Decode(ids []int) string {
	var parts []string
	for _, id := range ids {
		word := b.Vocab.GetWord(id)
		if word == "" || word == "<pad>" || word == "UNK" {
			continue
		}
		if word == "</s>" || word == "<s>" {
			continue
		}
		// Skip BOS/EOS control tokens
		if id == b.Vocab.BosID || id == b.Vocab.EosID {
			continue
		}
		parts = append(parts, word)
	}
	return strings.Join(parts, " ")
}

// Save persists the BPE tokenizer (vocab + merges) to a GOB file.
func (b *BPETokenizer) Save(path string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(b)
}

// LoadBPETokenizer loads a BPE tokenizer from a GOB file.
func LoadBPETokenizer(path string) (*BPETokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()
	var b BPETokenizer
	if err := gob.NewDecoder(f).Decode(&b); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}
	return &b, nil
}

// TrainBPETokenizer is a convenience function that creates, trains, and saves
// a BPE tokenizer. Returns the path to the saved model.
func TrainBPETokenizer(projectRoot string, vocabSize int) (string, error) {
	bpe := NewBPETokenizer(vocabSize)

	// Register special tokens that should never be split
	bpe.AddSpecialToken("<|im_start|>")
	bpe.AddSpecialToken("<|im_end|>")
	bpe.AddSpecialToken("<pad>")
	bpe.AddSpecialToken("<s>")
	bpe.AddSpecialToken("</s>")
	bpe.AddSpecialToken("UNK")
	bpe.AddSpecialToken("\n")

	if err := bpe.Train(projectRoot); err != nil {
		return "", fmt.Errorf("BPE training failed: %w", err)
	}

	// Save the tokenizer
	path := filepath.Join(projectRoot, "data/models/gob_models/bpe_tokenizer.gob")
	if err := bpe.Save(path); err != nil {
		return "", fmt.Errorf("BPE save failed: %w", err)
	}

	// Also save vocabulary separately for backward compatibility
	vocabPath := filepath.Join(projectRoot, "data/models/gob_models/bpe_vocabulary.gob")
	if err := bpe.Vocab.Save(vocabPath); err != nil {
		log.Printf("⚠️  BPE vocab save warning: %v", err)
	}

	log.Printf("✅ BPE Tokenizer saved to %s (vocab=%d, merges=%d)", path, bpe.Vocab.Size(), len(bpe.Merges))
	return path, nil
}

// EstimatedVocabSize returns a heuristic vocabulary size based on the dataset.
// ~16k is a good balance for code + natural language.
const DefaultBPEVocabSize = 16384
