// Package vocab provides functions for creating and managing vocabularies
// for natural language processing tasks.

package vocab

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// Vocabulary represents a mapping from words to integer IDs.
type Vocabulary struct {
	WordToToken    map[string]int
	TokenToWord    []string
	PaddingTokenID int
	BosID          int
	EosID          int
	UnkID          int
}

// TrainingDataJSON is kept for backward compatibility with GOB-encoded models.
type TrainingDataJSON struct {
	Sentences []map[string]interface{} `json:"sentences"`
}

// NewVocabulary creates a new Vocabulary instance.
func NewVocabulary() *Vocabulary {
	v := &Vocabulary{
		WordToToken: make(map[string]int),
		TokenToWord: []string{},
	}
	// Initialize special tokens with fixed IDs
	v.AddToken("<pad>") // ID 0
	v.PaddingTokenID = 0
	v.AddToken("UNK") // ID 1
	v.UnkID = 1
	v.AddToken("<s>") // ID 2
	v.BosID = 2
	v.AddToken("</s>") // ID 3
	v.EosID = 3

	return v
}

func NewVocabularyFromMap(wordMap map[string]int) *Vocabulary {
	v := NewVocabulary()
	for word, id := range wordMap {
		v.WordToToken[word] = id
		if id >= len(v.TokenToWord) {
			v.TokenToWord = append(v.TokenToWord, make([]string, id-len(v.TokenToWord)+1)...)
		}
		v.TokenToWord[id] = word
	}
	return v
}

// AddToken adds a word to the vocabulary if it doesn't already exist.
func (v *Vocabulary) AddToken(word string) int {
	if id, ok := v.WordToToken[word]; ok {
		return id
	}
	id := len(v.TokenToWord)
	v.WordToToken[word] = id
	v.TokenToWord = append(v.TokenToWord, word)
	return id
}

// GetTokenID returns the token ID for a given word, or the UNK id if not found.
func (v *Vocabulary) GetTokenID(word string) int {
	if id, ok := v.WordToToken[word]; ok {
		return id
	}
	return v.UnkID
}

// GetWord returns the word for a given token ID, or an empty string if not found.
func (v *Vocabulary) GetWord(id int) string {
	if id >= 0 && id < len(v.TokenToWord) {
		return v.TokenToWord[id]
	}
	return ""
}

// Size returns the number of unique tokens in the vocabulary.
func (v *Vocabulary) Size() int {
	return len(v.TokenToWord)
}

// Decode converts a slice of token IDs back to a string.
func (v *Vocabulary) Decode(tokenIDs []int) string {
	words := make([]string, 0, len(tokenIDs))
	for _, id := range tokenIDs {
		if id == v.EosID {
			break
		}
		if id == v.PaddingTokenID {
			continue
		}
		words = append(words, v.GetWord(id))
	}
	return strings.Join(words, " ")
}

// CreateAndSaveVocab creates a vocabulary from a list of token slices and saves it as a GOB file.
func CreateAndSaveVocab(sentences [][]string, vocabPath string) (*Vocabulary, error) {
	tokenVocab := NewVocabulary()
	for _, tokens := range sentences {
		for _, token := range tokens {
			tokenVocab.AddToken(token)
		}
	}

	err := tokenVocab.Save(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to save vocabulary: %w", err)
	}

	return tokenVocab, nil
}

// Save saves a Vocabulary to a GOB file.
func (v *Vocabulary) Save(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(v)
}

// LoadVocabulary loads a Vocabulary from a GOB file.
func LoadVocabulary(filePath string) (*Vocabulary, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var v Vocabulary
	err = decoder.Decode(&v)
	if err != nil {
		return nil, err
	}

	// 🛡️ Robust ID restoration
	v.PaddingTokenID = -1
	if id, ok := v.WordToToken["<pad>"]; ok {
		v.PaddingTokenID = id
	} else {
		v.PaddingTokenID = v.AddToken("<pad>")
	}

	v.UnkID = -1
	if id, ok := v.WordToToken["UNK"]; ok {
		v.UnkID = id
	} else {
		v.UnkID = v.AddToken("UNK")
	}

	v.BosID = -1
	if id, ok := v.WordToToken["<s>"]; ok {
		v.BosID = id
	} else {
		v.BosID = v.AddToken("<s>")
	}

	v.EosID = -1
	if id, ok := v.WordToToken["</s>"]; ok {
		v.EosID = id
	} else {
		v.EosID = v.AddToken("</s>")
	}

	return &v, nil
}

// LoadTrainingDataJSON loads training data from a JSON file.
func LoadTrainingDataJSON(filePath string) (*TrainingDataJSON, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, err
	}

	var trainingData TrainingDataJSON
	err = json.Unmarshal(data, &trainingData)
	if err != nil {
		return nil, err
	}
	file.Close()

	return &trainingData, nil
}
