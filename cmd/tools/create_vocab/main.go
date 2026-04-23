package main

import (
	"bufio"
	"fmt"
	"log"
	"os"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

type IntentTrainingExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
	Description  string `json:"description"`
	Sentence     string `json:"sentence"`
}

func main() {
	// Define paths
	const corpusPath = "data/training/clean_corpus.txt"
	const vocabPath = "data/models/gob_models/query_vocabulary.gob"

	tokenVocab := vocab.NewVocabulary()

	// Process clean_corpus.txt
	file, err := os.Open(corpusPath)
	if err != nil {
		log.Fatalf("Failed to open corpus: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		tokens := tokenizer.Tokenize(line)
		for _, token := range tokens {
			tokenVocab.AddToken(token)
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading corpus: %v", err)
	}

	// Save the updated vocabulary
	err = tokenVocab.Save(vocabPath)
	if err != nil {
		log.Fatalf("Failed to save vocabulary: %v", err)
	}

	fmt.Printf("Vocabulary created from %s. Size: %d\n", corpusPath, tokenVocab.Size())
}
