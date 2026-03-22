package main

import (
	"bufio"
	"encoding/json"
	"fmt"

	"log"
	"os"
	"strings"

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
	const intentDataPath = "data/training/trainingdata/intent_data.json"
	const wikiQADataPath = "data/training/trainingdata/WikiQA-train.txt"
	const vocabPath = "data/models/gob_models/query_vocabulary.gob"

	tokenVocab := vocab.NewVocabulary()

	// Process intent_data.json
	intentFile, err := os.ReadFile(intentDataPath)
	if err != nil {
		log.Fatalf("Failed to read intent training data: %v", err)
	}

	var intents []IntentTrainingExample
	if err := json.Unmarshal(intentFile, &intents); err != nil {
		log.Fatalf("Failed to unmarshal intent training data: %v", err)
	}

	for _, intent := range intents {
		tokens := tokenizer.Tokenize(intent.Query)
		for _, token := range tokens {
			tokenVocab.AddToken(token)
		}
	}

	// Process WikiQA-train.txt
	wikiQAFile, err := os.Open(wikiQADataPath)
	if err != nil {
		log.Fatalf("Failed to open WikiQA training data: %v", err)
	}
	defer wikiQAFile.Close()

	scanner := bufio.NewScanner(wikiQAFile)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) >= 2 {
			// Tokenize question part
			questionTokens := tokenizer.Tokenize(parts[0])
			for _, token := range questionTokens {
				tokenVocab.AddToken(token)
			}
			// Tokenize answer part
			answerTokens := tokenizer.Tokenize(parts[1])
			for _, token := range answerTokens {
				tokenVocab.AddToken(token)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading WikiQA training data: %v", err)
	}

	// Process conversing.csv
	convFile, err := os.Open("data/training/trainingdata/conversing.csv")
	if err == nil {
		defer convFile.Close()
		scanner = bufio.NewScanner(convFile)
		for scanner.Scan() {
			line := scanner.Text()
			parts := strings.Split(line, ",")
			if len(parts) >= 2 {
				pattern := strings.Trim(parts[1], "\"")
				tokens := tokenizer.Tokenize(pattern)
				for _, token := range tokens {
					tokenVocab.AddToken(token)
				}
			}
		}
	}

	// Save the updated vocabulary
	err = tokenVocab.Save(vocabPath)
	if err != nil {
		log.Fatalf("Failed to save vocabulary: %v", err)
	}

	fmt.Println("vocabs", tokenVocab.TokenToWord)
	fmt.Println("vocab size", tokenVocab.Size())

	fmt.Println("Vocabulary created and saved successfully.")
}
