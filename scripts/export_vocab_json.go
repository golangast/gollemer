package main

import (
	"encoding/json"
	"fmt"
	"github.com/zendrulat/nlptagger/neural/nnu/vocab"
	"os"
)

func main() {
	vocabPath := "gob_models/sentence_vocabulary.gob"
	outputPath := "gob_models/sentence_vocab.json"
	v, err := vocab.LoadVocabulary(vocabPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load vocab: %v\n", err)
		os.Exit(1)
	}
	f, err := os.Create(outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create output file: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v.WordToToken); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to encode vocab: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Exported vocabulary to %s\n", outputPath)
}
