package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

type IntentTrainingExample struct {
	Query          string `json:"query"`
	SemanticOutput struct {
		Social struct {
			Intent    string `json:"intent"`
			SubIntent string `json:"sub_intent"`
		} `json:"social"`
	} `json:"semantic_output"`
}

// IntentTrainingData represents the structure of the old intent training data JSON.
type IntentTrainingData []IntentTrainingExample

// TaggedTrainingExample represents the new format for training the NER/tagging model.
type TaggedTrainingExample struct {
	Query  string   `json:"query"`
	Intent string   `json:"intent"`
	Tokens []string `json:"tokens"`
	Tags   []string `json:"tags"` // IOB format (Inside, Outside, Beginning)
}

// LoadIntentTrainingData loads the intent training data from a JSON file.
func LoadIntentTrainingData(filePath string) (*IntentTrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)
	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data IntentTrainingData
	err = json.Unmarshal(bytes, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal training data JSON from %s: %w", filePath, err)
	}

	return &data, nil
}

func main() {
	const intentDataPath = "data/training/tiny_chat.json"
	const taggedDataPath = "data/training/trainingdata/tagged_training_data.json"

	// Load the original training data
	trainingData, err := LoadIntentTrainingData(intentDataPath)
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}

	var taggedTrainingData []TaggedTrainingExample

	for _, example := range *trainingData {
		tokens := tokenizer.Tokenize(example.Query)
		tags := make([]string, len(tokens))
		for i := range tags {
			tags[i] = "O" // Default to Outside
		}

		// Extract intent
		intent := example.SemanticOutput.Social.Intent
		if intent == "" {
			intent = "social"
		}

		taggedExample := TaggedTrainingExample{
			Query:  example.Query,
			Intent: intent,
			Tokens: tokens,
			Tags:   tags,
		}
		taggedTrainingData = append(taggedTrainingData, taggedExample)
	}

	// Save the new tagged data
	file, err := os.Create(taggedDataPath)
	if err != nil {
		log.Fatalf("Failed to create tagged training data file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(taggedTrainingData); err != nil {
		log.Fatalf("Failed to encode tagged training data: %v", err)
	}

	log.Printf("Successfully converted %d examples to tagged format.", len(taggedTrainingData))
}
