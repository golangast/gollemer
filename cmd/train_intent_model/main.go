package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/golangast/gollemer/neural/moe"
	"github.com/golangast/gollemer/neural/nn"
	mainvocab "github.com/golangast/gollemer/neural/nnu/vocab"
	"github.com/golangast/gollemer/neural/nnu/word2vec"
	"github.com/golangast/gollemer/neural/tensor"
	"github.com/golangast/gollemer/neural/tokenizer"
)

// IntentTrainingExample represents a single training example with a query and its intents.
type IntentTrainingExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
	Description  string `json:"description"`
	Sentence     string `json:"sentence"`
}

// IntentTrainingData represents the structure of the intent training data JSON.
type IntentTrainingData []IntentTrainingExample

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
	// Define paths
	const trainingDataPath = "trainingdata/intent_data.json"
	const queryVocabPath = "gob_models/query_vocabulary.gob"
	const parentIntentVocabPath = "gob_models/parent_intent_vocabulary.gob"
	const childIntentVocabPath = "gob_models/child_intent_vocabulary.gob"
	const sentenceVocabPath = "gob_models/sentence_vocabulary.gob"
	const modelSavePath = "gob_models/moe_classification_model.gob"

	// Load training data
	trainingData, err := LoadIntentTrainingData(trainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load intent training data: %v", err)
	}

	// Load or create vocabularies
	queryVocab, err := mainvocab.LoadVocabulary(queryVocabPath)
	if err != nil {
		log.Println("Failed to load query vocabulary, creating a new one.")
		queryVocab = mainvocab.NewVocabulary()
	}

	parentIntentVocab, err := mainvocab.LoadVocabulary(parentIntentVocabPath)
	if err != nil {
		log.Println("Failed to load parent intent vocabulary, creating a new one.")
		parentIntentVocab = mainvocab.NewVocabulary()
	}

	childIntentVocab, err := mainvocab.LoadVocabulary(childIntentVocabPath)
	if err != nil {
		log.Println("Failed to load child intent vocabulary, creating a new one.")
		childIntentVocab = mainvocab.NewVocabulary()
	}

	sentenceVocab, err := mainvocab.LoadVocabulary(sentenceVocabPath)
	if err != nil {
		log.Println("Failed to load sentence vocabulary, creating a new one.")
		sentenceVocab = mainvocab.NewVocabulary()
	}

	for _, example := range *trainingData {
		words := strings.Fields(strings.ToLower(example.Query))
		for _, word := range words {
			queryVocab.AddToken(word)
		}
		parentIntentVocab.AddToken(example.ParentIntent)
		childIntentVocab.AddToken(example.ChildIntent)
		sentenceWords := strings.Fields(strings.ToLower(example.Sentence))
		for _, word := range sentenceWords {
			sentenceVocab.AddToken(word)
		}
	}

	// Save vocabularies
	queryVocab.Save(queryVocabPath)
	parentIntentVocab.Save(parentIntentVocabPath)
	childIntentVocab.Save(childIntentVocabPath)
	sentenceVocab.Save(sentenceVocabPath)

	log.Printf("Query vocabulary size: %d", queryVocab.Size())
	log.Printf("Parent intent vocabulary size: %d", parentIntentVocab.Size())
	log.Printf("Child intent vocabulary size: %d", childIntentVocab.Size())
	log.Printf("Sentence vocabulary size: %d", sentenceVocab.Size())

	// Load Word2Vec model (assuming it's needed for embeddings)
	word2vecModel, err := word2vec.LoadModel("gob_models/word2vec_model.gob")
	if err != nil {
		log.Fatalf("Failed to load Word2Vec model: %v", err)
	}

	// Model hyperparameters
	embeddingDim := 128
	hiddenSize := 256
	maxAttentionHeads := 4
	numLayers := 2
	dropoutRate := 0.1

	// 1. Embedding
	embedding := nn.NewEmbedding(queryVocab.Size(), embeddingDim)
	embedding.LoadPretrainedWeights(word2vecModel.WordVectors)

	// 2. Simple RNN Encoder (replacing MoE in this example, adjust if actual MoE is desired)
	encoder, err := moe.NewSimpleRNNEncoder(embeddingDim, hiddenSize, numLayers)
	if err != nil {
		log.Fatalf("Failed to create SimpleRNNEncoder: %v", err)
	}

	// 3. RNN Decoder with increased capacity and dropout
	decoder, err := moe.NewRNNDecoder(embeddingDim, sentenceVocab.Size(), hiddenSize, maxAttentionHeads, numLayers, dropoutRate)
	if err != nil {
		log.Fatalf("Failed to create decoder: %v", err)
	}

	// 4. Create IntentMoE model
	model := &moe.IntentMoE{
		Embedding:         embedding,
		Encoder:           encoder,
		Decoder:           decoder,
		SentenceVocabSize: sentenceVocab.Size(),
	}
	if err != nil {
		log.Fatalf("Failed to create new MoE model: %v", err)
	}

	// Train the model
	TrainIntentModel(model, trainingData, queryVocab, parentIntentVocab, childIntentVocab, sentenceVocab, 100, 0.001, 32, 32)

	// Save the trained model
	log.Printf("Saving MoE Classification model to %s", modelSavePath)
	outputFile, err := os.Create(modelSavePath)
	if err != nil {
		log.Fatalf("Failed to create model file %s: %v", modelSavePath, err)
	}
	defer outputFile.Close()

	err = moe.SaveIntentMoEModelToGOB(model, outputFile)
	if err != nil {
		log.Fatalf("Failed to save MoE model: %v", err)
	}

	log.Println("Training complete.")
}

// TrainIntentModel trains the MoEClassificationModel for intent classification.
func TrainIntentModel(model *moe.IntentMoE, data *IntentTrainingData, queryVocab, parentIntentVocab, childIntentVocab, sentenceVocab *mainvocab.Vocabulary, epochs int, learningRate float64, batchSize int, maxSeqLength int) {
	optimizer := nn.NewOptimizer(model.Parameters(), learningRate, 5.0)

	for epoch := 0; epoch < epochs; epoch++ {
		log.Printf("Epoch %d/%d", epoch+1, epochs)
		totalLoss := 0.0
		numBatches := 0

		for i := 0; i < len(*data); i += batchSize {
			end := i + batchSize
			if end > len(*data) {
				end = len(*data)
			}
			batch := (*data)[i:end]

			loss, err := trainIntentModelBatch(model, optimizer, batch, queryVocab, parentIntentVocab, childIntentVocab, sentenceVocab, maxSeqLength)
			if err != nil {
				log.Printf("Error training batch: %v", err)
				continue
			}
			totalLoss += loss
			numBatches++
		}
		if numBatches > 0 {
			log.Printf("Epoch %d, Average Loss: %f", epoch+1, totalLoss/float64(numBatches))
		}
	}
}

// trainIntentModelBatch performs a single training step on a batch of intent data.
func trainIntentModelBatch(model *moe.IntentMoE, optimizer nn.Optimizer, batch IntentTrainingData, queryVocab, parentIntentVocab, childIntentVocab, sentenceVocab *mainvocab.Vocabulary, maxSeqLength int) (float64, error) {
	optimizer.ZeroGrad()

	batchSize := len(batch)

	inputIDsBatch := make([]int, batchSize*maxSeqLength)
	parentIntentIDs := make([]int, batchSize)
	childIntentIDs := make([]int, batchSize)
	targetSentenceIDsBatch := make([]int, batchSize*maxSeqLength)

	tok, err := tokenizer.NewTokenizer(queryVocab)
	if err != nil {
		return 0, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	sentenceTokenizer, err := tokenizer.NewTokenizer(sentenceVocab)
	if err != nil {
		return 0, fmt.Errorf("failed to create sentence tokenizer: %w", err)
	}

	for i, example := range batch {
		tokenIDs, err := tok.Encode(example.Query)
		if err != nil {
			return 0, fmt.Errorf("query tokenization failed for item %d: %w", i, err)
		}

		if len(tokenIDs) > maxSeqLength {
			tokenIDs = tokenIDs[:maxSeqLength]
		} else {
			padding := make([]int, maxSeqLength-len(tokenIDs))
			for j := range padding {
				padding[j] = queryVocab.PaddingTokenID
			}
			tokenIDs = append(tokenIDs, padding...)
		}
		copy(inputIDsBatch[i*maxSeqLength:(i+1)*maxSeqLength], tokenIDs)

		sentenceTokenIDs, err := sentenceTokenizer.Encode(example.Sentence)
		if err != nil {
			return 0, fmt.Errorf("sentence tokenization failed for item %d: %w", i, err)
		}

		if len(sentenceTokenIDs) > maxSeqLength {
			sentenceTokenIDs = sentenceTokenIDs[:maxSeqLength]
		} else {
			padding := make([]int, maxSeqLength-len(sentenceTokenIDs))
			for j := range padding {
				padding[j] = sentenceVocab.PaddingTokenID
			}
			sentenceTokenIDs = append(sentenceTokenIDs, padding...)
		}
		copy(targetSentenceIDsBatch[i*maxSeqLength:(i+1)*maxSeqLength], sentenceTokenIDs)

		parentIntentIDs[i] = parentIntentVocab.GetTokenID(example.ParentIntent)
		childIntentIDs[i] = childIntentVocab.GetTokenID(example.ChildIntent)
	}

	inputTensor := tensor.NewTensor([]int{batchSize, maxSeqLength}, convertIntsToFloat64s(inputIDsBatch), false)
	targetSentenceTensor := tensor.NewTensor([]int{batchSize, maxSeqLength}, convertIntsToFloat64s(targetSentenceIDsBatch), false)

	sentenceLogits, _, err := model.Forward(0.0, inputTensor, targetSentenceTensor)
	if err != nil {
		return 0, fmt.Errorf("model forward pass failed: %w", err)
	}

	sentenceLoss := 0.0
	sentenceGrads := make([]*tensor.Tensor, maxSeqLength-1)

	for t := 0; t < maxSeqLength-1; t++ {
		targets := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			targets[i] = int(targetSentenceIDsBatch[i*maxSeqLength+t+1])
		}
		loss, grad := tensor.CrossEntropyLoss(sentenceLogits[t], targets, sentenceVocab.PaddingTokenID, 0.1)
		sentenceLoss += loss
		sentenceGrads[t] = grad
	}

	totalLoss := sentenceLoss

	err = model.Backward(sentenceGrads...)
	if err != nil {
		return 0, fmt.Errorf("model backward pass failed: %w", err)
	}

	optimizer.Step()

	return totalLoss, nil
}

func convertIntsToFloat64s(input []int) []float64 {
	output := make([]float64, len(input))
	for i, v := range input {
		output[i] = float64(v)
	}
	return output
}

func SequenceCrossEntropyLoss(predictions *tensor.Tensor, targets []int, paddingID int) (float64, *tensor.Tensor) {
	batchSize := predictions.Shape[0]
	seqLen := predictions.Shape[1]
	vocabSize := predictions.Shape[2]

	predictionsFlat, _ := predictions.Reshape([]int{batchSize * seqLen, vocabSize})
	softmax, _ := predictionsFlat.Softmax(1)
	logSoftmax, _ := softmax.Log()

	targetsFlat := targets

	totalLoss := 0.0
	numTokens := 0

	grad := make([]float64, len(logSoftmax.Data))

	for i := 0; i < batchSize*seqLen; i++ {
		targetID := targetsFlat[i]

		if targetID == paddingID {
			continue
		}

		numTokens++
		predictedLogProb := logSoftmax.Data[i*vocabSize+targetID]
		totalLoss -= predictedLogProb

		for j := 0; j < vocabSize; j++ {
			prob := logSoftmax.Data[i*vocabSize+j]
			if j == targetID {
				grad[i*vocabSize+j] = prob - 1
			} else {
				grad[i*vocabSize+j] = prob
			}
		}
	}

	if numTokens == 0 {
		return 0.0, nil
	}

	avgLoss := totalLoss / float64(numTokens)
	gradTensor := tensor.NewTensor(logSoftmax.Shape, grad, true)

	return avgLoss, gradTensor
}
