package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/golangast/gollemer/internal/ai/moe"
	. "github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	tensor "github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// TaggedTrainingExample represents a single training example for the tagger model.
type TaggedTrainingExample struct {
	Query  string   `json:"query"`
	Intent string   `json:"intent"`
	Tokens []string `json:"tokens"`
	Tags   []string `json:"tags"`
}

// TaggedTrainingData represents the structure of the tagged training data JSON.
type TaggedTrainingData []TaggedTrainingExample

// LoadTaggedTrainingData loads the tagged training data from a JSON file.
func LoadTaggedTrainingData(filePath string) (*TaggedTrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)
	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data TaggedTrainingData
	err = json.Unmarshal(bytes, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal training data JSON from %s: %w", filePath, err)
	}

	return &data, nil
}

// BuildVocabularies builds vocabularies from the tagged training data.
func BuildVocabularies(taggedDataPath string) (*mainvocab.Vocabulary, *mainvocab.Vocabulary, *mainvocab.Vocabulary, error) {
	queryVocabulary := mainvocab.NewVocabulary()
	intentVocabulary := mainvocab.NewVocabulary()
	tagVocabulary := mainvocab.NewVocabulary()

	taggedData, err := LoadTaggedTrainingData(taggedDataPath)
	if err != nil {
		return nil, nil, nil, err
	}

	for _, example := range *taggedData {
		// Query vocabulary
		for _, word := range example.Tokens {
			queryVocabulary.AddToken(word)
		}

		// Intent vocabulary
		intentVocabulary.AddToken(example.Intent)

		// Tag vocabulary
		for _, tag := range example.Tags {
			tagVocabulary.AddToken(tag)
		}
	}

	return queryVocabulary, intentVocabulary, tagVocabulary, nil
}

// TrainTaggerModel trains the IntentTagger model.
func TrainTaggerModel(model *moe.IntentTagger, data *TaggedTrainingData, epochs int, learningRate float32, batchSize int, queryVocab, intentVocab, tagVocab *mainvocab.Vocabulary, queryTokenizer *tokenizer.Tokenizer, maxSequenceLength int) error {
	if model == nil {
		return errors.New("cannot train a nil model")
	}
	if data == nil || len(*data) == 0 {
		return errors.New("no training data provided")
	}

	optimizer := NewOptimizer(model.Parameters(), learningRate, 1.0) // Using a clip value of 1.0

	for epoch := range epochs {
		log.Printf("Epoch %d/%d", epoch+1, epochs)
		var totalLoss float32 = 0.0
		numBatches := 0
		for i := 0; i < len(*data); i += batchSize {
			end := min(i+batchSize, len(*data))
			batch := (*data)[i:end]

			loss, err := trainTaggerBatch(model, optimizer, batch, queryVocab, intentVocab, tagVocab, queryTokenizer, maxSequenceLength)
			if err != nil {
				log.Printf("Error training batch: %v", err)
				continue
			}
			totalLoss += loss
			numBatches++
		}
		if numBatches > 0 {
			log.Printf("Epoch %d, Average Loss: %f", epoch+1, totalLoss/float32(numBatches))
		}
	}

	return nil
}

func trainTaggerBatch(model *moe.IntentTagger, optimizer Optimizer, batch TaggedTrainingData, queryVocab, intentVocab, tagVocab *mainvocab.Vocabulary, queryTokenizer *tokenizer.Tokenizer, maxSequenceLength int) (float32, error) {
	optimizer.ZeroGrad()

	batchSize := len(batch)

	inputIDsBatch := make([]int, batchSize*maxSequenceLength)
	targetIntentIDs := make([]int, batchSize)
	targetTagIDsBatch := make([]int, batchSize*maxSequenceLength)

	for i, example := range batch {
		// Tokenize and pad query
		queryTokens, err := TokenizeAndConvertToIDs(example.Query, queryTokenizer, queryVocab, maxSequenceLength)
		if err != nil {
			return 0, fmt.Errorf("query tokenization failed for item %d: %w", i, err)
		}
		copy(inputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], queryTokens)

		// Get target intent ID
		targetIntentIDs[i] = intentVocab.GetTokenID(example.Intent)

		// Convert tags to IDs and pad
		tagIDs := make([]int, len(example.Tags))
		for j, tag := range example.Tags {
			tagIDs[j] = tagVocab.GetTokenID(tag)
		}
		if len(tagIDs) < maxSequenceLength {
			padding := make([]int, maxSequenceLength-len(tagIDs))
			for j := range padding {
				padding[j] = tagVocab.PaddingTokenID
			}
			tagIDs = append(tagIDs, padding...)
		}
		copy(targetTagIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], tagIDs)
	}

	inputTensor := tensor.NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat32s(inputIDsBatch), false)

	// Forward pass
	intentLogits, tagLogits, err := model.Forward(inputTensor)
	if err != nil {
		return 0, fmt.Errorf("model forward pass failed: %w", err)
	}

	// Calculate intent loss
	intentLoss, intentGrad := tensor.CrossEntropyLoss(intentLogits, targetIntentIDs, -1, 0.0) // No padding for intents

	// Calculate tag loss
	var tagLoss float32 = 0.0
	allTagGrads := make([]float32, batchSize*maxSequenceLength*tagVocab.Size())

	for t := range maxSequenceLength {
		targets := make([]int, batchSize)
		for i := range batchSize {
			targets[i] = targetTagIDsBatch[i*maxSequenceLength+t]
		}

		// Slice the 3D tensor to get [batchSize, tagVocabSize] for timestep t
		stepLogits, err := tagLogits.Slice(1, t, t+1)
		if err != nil {
			return 0, fmt.Errorf("failed to slice tag logits: %w", err)
		}
		// Reshape to 2D
		stepLogits.Reshape([]int{batchSize, tagVocab.Size()})

		loss, grad := tensor.CrossEntropyLoss(stepLogits, targets, tagVocab.PaddingTokenID, 0.0)
		tagLoss += loss

		// Copy gradients into the flat buffer for the 3D gradient tensor [batchSize, maxSequenceLength, tagVocabSize]
		vocabSize := tagVocab.Size()
		for b := 0; b < batchSize; b++ {
			for v := 0; v < vocabSize; v++ {
				destIdx := b*(maxSequenceLength*vocabSize) + t*vocabSize + v
				srcIdx := b*vocabSize + v
				allTagGrads[destIdx] = grad.Data[srcIdx]
			}
		}
	}

	totalLoss := intentLoss + tagLoss

	// Backward pass
	tagGradsTensor := tensor.NewTensor([]int{batchSize, maxSequenceLength, tagVocab.Size()}, allTagGrads, false)

	if err := model.Backward(intentGrad, tagGradsTensor); err != nil {
		return 0, fmt.Errorf("model backward pass failed: %w", err)
	}

	optimizer.Step()

	return totalLoss, nil
}

// TokenizeAndConvertToIDs tokenizes a text and converts tokens to their corresponding IDs, handling padding/truncation.
func TokenizeAndConvertToIDs(text string, tokenizer *tokenizer.Tokenizer, vocabulary *mainvocab.Vocabulary, maxLen int) ([]int, error) {
	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("failed to encode text: %w", err)
	}

	if len(tokenIDs) == 0 && maxLen > 0 {
		tokenIDs = make([]int, maxLen)
		for i := range tokenIDs {
			tokenIDs[i] = vocabulary.PaddingTokenID
		}
		return tokenIDs, nil
	} else if len(tokenIDs) == 0 {
		return []int{vocabulary.PaddingTokenID}, nil
	}

	if maxLen > 0 {
		if len(tokenIDs) > maxLen {
			tokenIDs = tokenIDs[:maxLen]
		} else if len(tokenIDs) < maxLen {
			paddingSize := maxLen - len(tokenIDs)
			padding := make([]int, paddingSize)
			for i := range padding {
				padding[i] = vocabulary.PaddingTokenID
			}
			tokenIDs = append(tokenIDs, padding...)
		}
	}
	return tokenIDs, nil
}

func convertIntsToFloat32s(input []int) []float32 {
	output := make([]float32, len(input))
	for i, v := range input {
		output[i] = float32(v)
	}
	return output
}

func main() {
	const taggedDataPath = "data/training/trainingdata/tagged_training_data.json"

	// Build vocabularies
	queryVocab, intentVocab, tagVocab, err := BuildVocabularies(taggedDataPath)
	if err != nil {
		log.Fatalf("Failed to build vocabularies: %v", err)
	}

	log.Printf("Query Vocabulary Size: %d", queryVocab.Size())
	log.Printf("Intent Vocabulary Size: %d", intentVocab.Size())
	log.Printf("Tag Vocabulary Size: %d", tagVocab.Size())

	// Load tagged training data
	trainingData, err := LoadTaggedTrainingData(taggedDataPath)
	if err != nil {
		log.Fatalf("Failed to load tagged training data: %v", err)
	}

	// Define training parameters
	epochs := 200
	var learningRate float32 = 0.001
	batchSize := 32

	// After vocabularies are fully populated, determine vocab sizes and create/load model
	inputVocabSize := queryVocab.Size()
	intentVocabSize := intentVocab.Size()
	tagVocabSize := tagVocab.Size()
	embeddingDim := 64
	numExperts := 4
	maxSequenceLength := 128

	log.Printf("Query Vocabulary Size: %d", inputVocabSize)
	log.Printf("Intent Vocabulary Size: %d", intentVocabSize)
	log.Printf("Tag Vocabulary Size: %d", tagVocabSize)
	log.Printf("Embedding Dimension: %d", embeddingDim)

	// Create a new Tagger model
	taggerModel, err := moe.NewIntentTagger(inputVocabSize, embeddingDim, numExperts, intentVocabSize, tagVocabSize)
	if err != nil {
		log.Fatalf("Failed to create new Tagger model: %v", err)
	}

	// Create tokenizers
	queryTokenizer, err := tokenizer.NewTokenizer(queryVocab)
	if err != nil {
		log.Fatalf("Failed to create query tokenizer: %v", err)
	}

	// Train the model
	err = TrainTaggerModel(taggerModel, trainingData, epochs, learningRate, batchSize, queryVocab, intentVocab, tagVocab, queryTokenizer, maxSequenceLength)
	if err != nil {
		log.Fatalf("Failed to train Tagger model: %v", err)
	}

	// Save the trained model and vocabularies
	log.Println("Training complete. Saving model and vocabularies...")
	// TODO: Implement saving for IntentTagger model
	// modelSavePath := "data/models/gob_models/tagger_model.gob"
	// modelFile, err := os.Create(modelSavePath)
	// if err != nil {
	// 	log.Fatalf("Failed to create model file: %v", err)
	// }
	// defer modelFile.Close()
	// err = moe.SaveIntentTaggerModel(taggerModel, modelFile) // To be implemented
	// if err != nil {
	// 	log.Fatalf("Failed to save Tagger model: %v", err)
	// }

	queryVocab.Save("data/models/gob_models/tagger_query_vocabulary.gob")
	intentVocab.Save("data/models/gob_models/tagger_intent_vocabulary.gob")
	tagVocab.Save("data/models/gob_models/tagger_tag_vocabulary.gob")

	log.Println("Done.")
}
