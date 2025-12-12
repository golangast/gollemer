package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/zendrulat/nlptagger/neural/tokenizer"
)

type IntentTrainingExample struct {
	Query          string      `json:"query"`
	SemanticOutput interface{} `json:"semantic_output"`
}

type IntentTrainingData []IntentTrainingExample

func main() {
	// Load Q&A data
	data, err := LoadIntentTrainingData("../../trainingdata/qa_semantic_output.json")
	if err != nil {
		log.Fatal(err)
	}

	maxLen := 0
	sumLen := 0
	count := 0
	over32 := 0
	over64 := 0
	over128 := 0

	for _, example := range *data {
		semanticOutputJSON, _ := json.Marshal(example.SemanticOutput)
		trainingSemanticOutput := "<s> " + string(semanticOutputJSON) + " </s>"

		// Use the actual tokenizer logic
		tokens := tokenizer.Tokenize(trainingSemanticOutput)

		length := len(tokens)
		if length > maxLen {
			maxLen = length
		}
		sumLen += length
		count++
		if length > 32 {
			over32++
		}
		if length > 64 {
			over64++
		}
		if length > 128 {
			over128++
		}

		if count <= 5 {
			fmt.Printf("Example %d length: %d\n", count, length)
			// fmt.Printf("Tokens: %v\n", tokens)
		}
	}

	fmt.Printf("\nTotal examples: %d\n", count)
	fmt.Printf("Max length: %d\n", maxLen)
	fmt.Printf("Average length: %.2f\n", float64(sumLen)/float64(count))
	fmt.Printf("Examples > 32 tokens: %d (%.2f%%)\n", over32, float64(over32)/float64(count)*100)
	fmt.Printf("Examples > 64 tokens: %d (%.2f%%)\n", over64, float64(over64)/float64(count)*100)
	fmt.Printf("Examples > 128 tokens: %d (%.2f%%)\n", over128, float64(over128)/float64(count)*100)
}

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
