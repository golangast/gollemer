package main

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	"github.com/golangast/gollemer/internal/ai/tagger/tag"
)

// LLMClient handles interaction with the MoE model
type LLMClient struct {
	model                    *moe.IntentMoE
	tokenizer                *tokenizer.Tokenizer
	semanticOutputTokenizer  *tokenizer.Tokenizer
	vocabulary               *mainvocab.Vocabulary
	semanticOutputVocabulary *mainvocab.Vocabulary
}

// NewLLMClient initializes the LLM client by loading the model and vocabularies
func NewLLMClient(vocabPath, outputVocabPath, modelPath string) (*LLMClient, error) {
	// Load vocabularies
	vocabulary, err := mainvocab.LoadVocabulary(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load input vocabulary: %w", err)
	}

	semanticOutputVocabulary, err := mainvocab.LoadVocabulary(outputVocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load semantic output vocabulary: %w", err)
	}

	// Create tokenizers
	tok, err := tokenizer.NewTokenizer(vocabulary)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	semanticOutputTokenizer, err := tokenizer.NewTokenizer(semanticOutputVocabulary)
	if err != nil {
		return nil, fmt.Errorf("failed to create semantic output tokenizer: %w", err)
	}

	// Load the trained MoE model
	model, err := moe.LoadIntentMoEModelFromGOB(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load MoE model: %w", err)
	}

	// Dynamic Vocab Synchronization: Resize model layers if they don't match the loaded vocab files
	if vocabulary.Size() > model.Embedding.VocabSize {
		model.ResizeEmbeddings(vocabulary.Size())
	}
	if semanticOutputVocabulary.Size() > model.SentenceVocabSize {
		model.Decoder.ResizeOutputLayer(semanticOutputVocabulary.Size())
		model.SentenceVocabSize = semanticOutputVocabulary.Size()
	}

	return &LLMClient{
		model:                    model,
		tokenizer:                tok,
		semanticOutputTokenizer:  semanticOutputTokenizer,
		vocabulary:               vocabulary,
		semanticOutputVocabulary: semanticOutputVocabulary,
	}, nil
}

// Query sends a text query to the model and returns the generated response
func (c *LLMClient) Query(input string) (string, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return "", fmt.Errorf("empty input")
	}

	// Encode the query
	tokenIDs, err := c.tokenizer.Encode(input)
	if err != nil {
		return "", fmt.Errorf("failed to encode query: %w", err)
	}

	// Pad or truncate the sequence to a fixed length
	maxSeqLength := 32 // Consistent with training/inference
	if len(tokenIDs) > maxSeqLength {
		tokenIDs = tokenIDs[:maxSeqLength]
	} else {
		for len(tokenIDs) < maxSeqLength {
			tokenIDs = append(tokenIDs, c.vocabulary.PaddingTokenID)
		}
	}

	// Create input tensor
	inputData := make([]float32, len(tokenIDs))
	for i, id := range tokenIDs {
		inputData[i] = float32(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputData)}, inputData, false)

	// Create a dummy target tensor for inference (required by Forward signature)
	dummyTargetTokenIDs := make([]float32, maxSeqLength)
	for i := range maxSeqLength {
		dummyTargetTokenIDs[i] = float32(c.semanticOutputVocabulary.PaddingTokenID) // Fixed: use correct vocab
	}
	dummyTargetTensor := tensor.NewTensor([]int{1, maxSeqLength}, dummyTargetTokenIDs, false)

	// Forward pass to get the context vector
	_, contextVector, err := c.model.Forward(0.0, inputTensor, dummyTargetTensor)
	if err != nil {
		return "", fmt.Errorf("model forward pass failed: %w", err)
	}

	// Greedy search decode
	sosID := c.semanticOutputVocabulary.GetTokenID("<s>")
	eosID := c.semanticOutputVocabulary.GetTokenID("</s>")
	predictedIDs, err := c.model.GreedySearchDecode(contextVector, maxSeqLength, sosID, eosID, 1.2, 0.0, 100, tag.Tag{})
	if err != nil {
		return "", fmt.Errorf("greedy search decode failed: %w", err)
	}

	// Decode the predicted IDs to a sentence
	predictedSentence, err := c.semanticOutputTokenizer.Decode(predictedIDs)
	if err != nil {
		return "", fmt.Errorf("failed to decode predicted IDs: %w", err)
	}

	return predictedSentence, nil
}
