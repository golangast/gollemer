//go:build ignore

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Println("Usage: go run add_missing_tokens_to_vocab.go <vocab.json> <missing_tokens.txt>")
		os.Exit(1)
	}
	vocabPath := os.Args[1]
	missingTokensPath := os.Args[2]

	// Read and parse the vocabulary file
	vocabFile, err := ioutil.ReadFile(vocabPath)
	if err != nil {
		fmt.Printf("Error reading vocab file: %v\n", err)
		os.Exit(1)
	}

	var vocab map[string]int
	if err := json.Unmarshal(vocabFile, &vocab); err != nil {
		fmt.Printf("Error unmarshalling vocab json: %v\n", err)
		os.Exit(1)
	}

	// Read and parse the missing tokens file
	missingFile, err := os.Open(missingTokensPath)
	if err != nil {
		fmt.Printf("Error opening missing tokens file: %v\n", err)
		os.Exit(1)
	}
	defer missingFile.Close()

	var missingTokens []string
	scanner := bufio.NewScanner(missingFile)
	for scanner.Scan() {
		line := scanner.Text()
		if parts := strings.SplitN(line, ":", 2); len(parts) > 0 {
			token := strings.TrimSpace(parts[0])
			missingTokens = append(missingTokens, token)
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Printf("Error reading missing tokens file: %v\n", err)
		os.Exit(1)
	}

	// Find the max ID and add new tokens
	maxID := 0
	for _, id := range vocab {
		if id > maxID {
			maxID = id
		}
	}

	addedCount := 0
	for _, token := range missingTokens {
		if _, exists := vocab[token]; !exists {
			maxID++
			vocab[token] = maxID
			addedCount++
		}
	}

	// Write the updated vocabulary back to the file
	updatedVocab, err := json.MarshalIndent(vocab, "", "  ")
	if err != nil {
		fmt.Printf("Error marshalling updated vocab: %v\n", err)
		os.Exit(1)
	}

	if err := ioutil.WriteFile(vocabPath, updatedVocab, 0644); err != nil {
		fmt.Printf("Error writing updated vocab file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Added %d missing tokens to %s\n", addedCount, vocabPath)
}
