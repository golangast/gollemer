package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/semantic"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

type IntentTrainingExample struct {
	Query          string                  `json:"query"`
	SemanticOutput semantic.SemanticOutput `json:"semantic_output"`
	FlatOutput     string                  `json:"flat_output"`
}

type IntentTrainingData []IntentTrainingExample

func main() {
	const trainingDataPath = "data/training/trainingdata/semantic_output_data_flat.json"
	const vocabSavePath = "data/models/gob_models/semantic_output_vocabulary.gob"

	fmt.Println("🏗️ Initializing NLP Structure...")

	// 1. Generate Semantic Output Vocabulary
	fmt.Printf("Generating vocabulary from %s...\n", trainingDataPath)
	file, err := os.Open(trainingDataPath)
	if err != nil {
		log.Fatalf("Failed to open training data: %v", err)
	}
	defer file.Close()

	bytes, _ := io.ReadAll(file)
	var data IntentTrainingData
	if err := json.Unmarshal(bytes, &data); err != nil {
		log.Fatalf("Failed to unmarshal training data: %v", err)
	}

	semanticOutputVocabulary := vocab.NewVocabulary()
	
	// Add special tokens
	semanticOutputVocabulary.AddToken("<s>")
	semanticOutputVocabulary.AddToken("</s>")
	semanticOutputVocabulary.AddToken("<pad>")
	semanticOutputVocabulary.AddToken("UNK")
	
	for _, pair := range data {
		// Tokenize Flat Output
		trainingSemanticOutput := "<s> " + pair.FlatOutput + " </s>"
		tokens := tokenizer.Tokenize(trainingSemanticOutput)
		for _, token := range tokens {
			semanticOutputVocabulary.AddToken(token)
		}
	}

	semanticOutputVocabulary.BosID = semanticOutputVocabulary.GetTokenID("<s>")
	semanticOutputVocabulary.EosID = semanticOutputVocabulary.GetTokenID("</s>")
	semanticOutputVocabulary.PaddingTokenID = semanticOutputVocabulary.GetTokenID("<pad>")
	semanticOutputVocabulary.UnkID = semanticOutputVocabulary.GetTokenID("UNK")

	if err := semanticOutputVocabulary.Save(vocabSavePath); err != nil {
		log.Fatalf("Failed to save vocabulary: %v", err)
	}
	fmt.Printf("✅ Semantic Output Vocabulary saved to %s (%d tokens)\n", vocabSavePath, semanticOutputVocabulary.Size())

	// 2. Verify Internal structure
	if _, err := os.Stat("internal/ai/moe"); err == nil {
		fmt.Println("✅ Internal MoE structure detected.")
	} else {
		fmt.Println("⚠️  Internal MoE structure missing. Expected at internal/moe.")
	}

	fmt.Println("\n🚀 NLP Structure Initialization Complete!")
}
