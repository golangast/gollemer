package main

import (
	"fmt"

	"github.com/golangast/gollemer/neural/nnu/vocab"
)

func main() {
	v, err := vocab.LoadVocabulary("gob_models/semantic_output_vocabulary.gob")
	if err == nil {
		fmt.Printf("Semantic Output Vocabulary size: %d\n", len(v.WordToToken))
	}

	v2, err := vocab.LoadVocabulary("gob_models/query_vocabulary.gob")
	if err == nil {
		fmt.Printf("Query Vocabulary size: %d\n", len(v2.WordToToken))
	}
}
