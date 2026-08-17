package main

import (
	"flag"
	"log"

	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

func main() {
	vocab := flag.Int("vocab", tokenizer.DefaultBPEVocabSize, "Target BPE vocab size")
	root := flag.String("root", ".", "Project root to scan and save tokenizer")
	flag.Parse()

	_, err := tokenizer.TrainBPETokenizer(*root, *vocab)
	if err != nil {
		log.Fatalf("BPE training failed: %v", err)
	}
}
