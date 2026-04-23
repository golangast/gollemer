package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/word2vec"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
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
	FlatOutput string `json:"flat_output"`
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
	const trainingDataPath = "data/training/tiny_chat.json"
	const queryVocabPath = "data/models/gob_models/query_vocabulary.gob"
	const parentIntentVocabPath = "data/models/gob_models/parent_intent_vocabulary.gob"
	const childIntentVocabPath = "data/models/gob_models/child_intent_vocabulary.gob"
	const sentenceVocabPath = "data/models/gob_models/sentence_vocabulary.gob"
	const modelSavePath = "data/models/gob_models/moe_classification_model.gob"

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
		words := tokenizer.Tokenize(strings.ToLower(example.Query))
		for _, word := range words {
			queryVocab.AddToken(word)
		}
		parentIntentVocab.AddToken(example.SemanticOutput.Social.Intent)
		childIntentVocab.AddToken(example.SemanticOutput.Social.SubIntent)
		sentenceWords := tokenizer.Tokenize(strings.ToLower(example.FlatOutput))
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
	word2vecModel, err := word2vec.LoadModel("data/models/gob_models/word2vec_model.gob")
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
	// Convert Word2Vec vectors to float32
	f32Vectors := make(map[int][]float32)
	for id, vec := range word2vecModel.WordVectors {
		f32Vec := make([]float32, len(vec))
		for i, v := range vec {
			f32Vec[i] = float32(v)
		}
		f32Vectors[id] = f32Vec
	}
	embedding.LoadPretrainedWeights(f32Vectors)

	// 2. Simple RNN Encoder (replacing MoE in this example, adjust if actual MoE is desired)
	encoder, err := moe.NewSimpleRNNEncoder(embeddingDim, hiddenSize, numLayers)
	if err != nil {
		log.Fatalf("Failed to create SimpleRNNEncoder: %v", err)
	}

	// 3. RNN Decoder with increased capacity and dropout (numExperts=1 for legacy decoder)
	decoder, err := moe.NewRNNDecoder(hiddenSize, sentenceVocab.Size(), hiddenSize, maxAttentionHeads, numLayers, float32(dropoutRate), 1)
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

	err = moe.SaveIntentMoEModelToGOB(model, modelSavePath)
	if err != nil {
		log.Fatalf("Failed to save MoE model: %v", err)
	}

	log.Println("Training complete.")
}

// TrainIntentModel trains the MoEClassificationModel for intent classification.
func TrainIntentModel(model *moe.IntentMoE, data *IntentTrainingData, queryVocab, parentIntentVocab, childIntentVocab, sentenceVocab *mainvocab.Vocabulary, epochs int, learningRate float32, batchSize int, maxSeqLength int) {
	optimizer := nn.NewOptimizer(model.Parameters(), learningRate, 5.0)

	for epoch := range epochs {
		log.Printf("Epoch %d/%d", epoch+1, epochs)
		var totalLoss float32 = 0.0
		numBatches := 0

		for i := 0; i < len(*data); i += batchSize {
			end := min(i+batchSize, len(*data))
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
			log.Printf("Epoch %d, Average Loss: %f", epoch+1, totalLoss/float32(numBatches))
		}
	}
}

// trainIntentModelBatch performs a single training step on a batch of intent data.
func trainIntentModelBatch(model *moe.IntentMoE, optimizer nn.Optimizer, batch IntentTrainingData, queryVocab, parentIntentVocab, childIntentVocab, sentenceVocab *mainvocab.Vocabulary, maxSeqLength int) (float32, error) {
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

		parentIntentIDs[i] = parentIntentVocab.GetTokenID(example.SemanticOutput.Social.Intent)
		childIntentIDs[i] = childIntentVocab.GetTokenID(example.SemanticOutput.Social.SubIntent)
		
		sentenceTokenIDs, err := sentenceTokenizer.Encode(example.FlatOutput)
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
	}

	inputTensor := tensor.NewTensor([]int{batchSize, maxSeqLength}, convertIntsToFloat32s(inputIDsBatch), false)
	targetSentenceTensor := tensor.NewTensor([]int{batchSize, maxSeqLength}, convertIntsToFloat32s(targetSentenceIDsBatch), false)

	sentenceLogits, _, err := model.Forward(0.0, inputTensor, targetSentenceTensor)
	if err != nil {
		return 0, fmt.Errorf("model forward pass failed: %w", err)
	}

	var sentenceLoss float32 = 0.0
	var sentenceGrads []*tensor.Tensor

	if len(sentenceLogits) == 1 && len(sentenceLogits[0].Shape) == 3 {
		// Vectorized 3D loss
		logits := sentenceLogits[0]
		// targets for the whole sequence (shifted by 1)
		fullTargets := make([]int, batchSize*(maxSeqLength-1))
		for i := 0; i < batchSize; i++ {
			for t := 0; t < maxSeqLength-1; t++ {
				fullTargets[i*(maxSeqLength-1)+t] = targetSentenceIDsBatch[i*maxSeqLength+t+1]
			}
		}
		
		loss, grad := tensor.CrossEntropyLoss(logits, fullTargets, sentenceVocab.PaddingTokenID, 0.1)
		sentenceLoss = loss
		sentenceGrads = []*tensor.Tensor{grad}
	} else {
		sentenceGrads = make([]*tensor.Tensor, maxSeqLength-1)
		for t := 0; t < maxSeqLength-1; t++ {
			targets := make([]int, batchSize)
			for i := 0; i < batchSize; i++ {
				targets[i] = int(targetSentenceIDsBatch[i*maxSeqLength+t+1])
			}
			loss, grad := tensor.CrossEntropyLoss(sentenceLogits[t], targets, sentenceVocab.PaddingTokenID, 0.1)
			sentenceLoss += loss
			sentenceGrads[t] = grad
		}
	}

	totalLoss := sentenceLoss

	err = model.Backward(sentenceGrads...)
	if err != nil {
		return 0, fmt.Errorf("model backward pass failed: %w", err)
	}

	optimizer.Step()

	return totalLoss, nil
}

func convertIntsToFloat32s(input []int) []float32 {
	output := make([]float32, len(input))
	for i, v := range input {
		output[i] = float32(v)
	}
	return output
}

func SequenceCrossEntropyLoss(predictions *tensor.Tensor, targets []int, paddingID int) (float32, *tensor.Tensor) {
	batchSize := predictions.Shape[0]
	seqLen := predictions.Shape[1]
	vocabSize := predictions.Shape[2]

	predictionsFlat, _ := predictions.Reshape([]int{batchSize * seqLen, vocabSize})
	softmax, _ := predictionsFlat.Softmax(1)
	logSoftmax, _ := softmax.Log()

	targetsFlat := targets

	var totalLoss float32 = 0.0
	numTokens := 0

	grad := make([]float32, len(logSoftmax.Data))

	for i := 0; i < batchSize*seqLen; i++ {
		targetID := targetsFlat[i]

		if targetID == paddingID {
			continue
		}

		numTokens++
		predictedLogProb := logSoftmax.Data[i*vocabSize+targetID]
		totalLoss -= predictedLogProb

		for j := range vocabSize {
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

	avgLoss := totalLoss / float32(numTokens)
	gradTensor := tensor.NewTensor(logSoftmax.Shape, grad, true)

	return avgLoss, gradTensor
}
