package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/signal"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"syscall"

	"github.com/golangast/gollemer/neural/moe"
	"github.com/golangast/gollemer/neural/nn"
	mainvocab "github.com/golangast/gollemer/neural/nnu/vocab"
	"github.com/golangast/gollemer/neural/nnu/word2vec"
	"github.com/golangast/gollemer/neural/semantic"
	. "github.com/golangast/gollemer/neural/tensor"
	"github.com/golangast/gollemer/neural/tokenizer"
	"github.com/golangast/gollemer/tagger/tag"
)

// IntentTrainingExample represents a single training example with a query and its intents.
type IntentTrainingExample struct {
	Query          string                  `json:"query"`
	SemanticOutput semantic.SemanticOutput `json:"semantic_output"`
	FlatOutput     string                  `json:"flat_output"`
}

// IntentTrainingData represents the structure of the intent training data JSON.
type IntentTrainingData []IntentTrainingExample

// TokenizedTrainingExample represents a pre-tokenized training example.
type TokenizedTrainingExample struct {
	QueryIDs          []float64
	SemanticOutputIDs []float64
}

// EnhancedTrainingExample includes SRL and ASG annotations
type EnhancedTrainingExample struct {
	Query         string                          `json:"query"`
	FlatOutput    string                          `json:"flat_output"`
	SemanticRoles map[string]interface{}          `json:"semantic_roles"`
	ASG           *semantic.AbstractSemanticGraph `json:"abstract_semantic_graph"`
	ExecutionPlan map[string]interface{}          `json:"execution_plan"`
}

// TokenizeTrainingData pre-tokenizes the training data in parallel.
func TokenizeTrainingData(data *IntentTrainingData, queryTokenizer, semanticOutputTokenizer *tokenizer.Tokenizer, queryVocab, semanticOutputVocab *mainvocab.Vocabulary, maxLen int) ([]TokenizedTrainingExample, error) {
	tokenizedData := make([]TokenizedTrainingExample, len(*data))
	var wg sync.WaitGroup
	var errMutex sync.Mutex
	var firstErr error

	numWorkers := runtime.NumCPU()
	batchSize := (len(*data) + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * batchSize
		end := start + batchSize
		if end > len(*data) {
			end = len(*data)
		}
		if start >= end {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				example := (*data)[i]

				// Tokenize Query
				qIDs, err := TokenizeAndConvertToIDs(example.Query, queryTokenizer, queryVocab, maxLen)
				if err != nil {
					errMutex.Lock()
					if firstErr == nil {
						firstErr = err
					}
					errMutex.Unlock()
					return
				}

				// Tokenize Flat Output (simplified format)
				trainingSemanticOutput := "<s> " + example.FlatOutput + " </s>"
				sIDs, err := TokenizeAndConvertToIDs(trainingSemanticOutput, semanticOutputTokenizer, semanticOutputVocab, maxLen)
				if err != nil {
					errMutex.Lock()
					if firstErr == nil {
						firstErr = err
					}
					errMutex.Unlock()
					return
				}

				tokenizedData[i] = TokenizedTrainingExample{
					QueryIDs:          convertIntsToFloat64s(qIDs),
					SemanticOutputIDs: convertIntsToFloat64s(sIDs),
				}
			}
		}(start, end)
	}
	wg.Wait()
	return tokenizedData, firstErr
}

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

// EnhanceTrainingDataWithSRLAndASG adds semantic role labeling and abstract semantic graphs to training data
func EnhanceTrainingDataWithSRLAndASG(data *IntentTrainingData) ([]EnhancedTrainingExample, error) {
	enhanced := make([]EnhancedTrainingExample, len(*data))
	srl := semantic.NewSemanticRoleLabeler()
	asgGen := semantic.NewASGGenerator()

	for i, example := range *data {
		// Extract semantic roles from query
		roles, err := srl.ExtractRoles(example.Query)
		if err != nil {
			log.Printf("Warning: failed to extract roles for query '%s': %v", example.Query, err)
			roles = make(map[string]interface{})
		}

		// Generate ASG from extracted roles
		operation := ""
		if op, exists := roles["operation"]; exists {
			operation = op.(string)
		}

		resources := make([]map[string]string, 0)
		if res, exists := roles["resources"]; exists {
			if resSlice, ok := res.([]map[string]string); ok {
				resources = resSlice
			}
		}

		arguments := make([]map[string]string, 0)
		if args, exists := roles["arguments"]; exists {
			if argsSlice, ok := args.([]map[string]string); ok {
				arguments = argsSlice
			}
		}

		modifiers := make([]map[string]string, 0)
		if mods, exists := roles["modifiers"]; exists {
			if modsSlice, ok := mods.([]string); ok {
				for _, mod := range modsSlice {
					modifiers = append(modifiers, map[string]string{"value": mod})
				}
			}
		}

		// Build ASG
		asg := asgGen.GenerateFromSemanticRoles(operation, resources, arguments, modifiers)

		// Validate ASG
		if err := asgGen.ValidateASG(asg); err != nil {
			log.Printf("Warning: ASG validation failed for query '%s': %v", example.Query, err)
		}

		// Generate execution plan from ASG
		executionPlan := asgGen.GenerateExecutionPlan(asg)

		enhanced[i] = EnhancedTrainingExample{
			Query:         example.Query,
			FlatOutput:    example.FlatOutput,
			SemanticRoles: roles,
			ASG:           asg,
			ExecutionPlan: executionPlan,
		}
	}

	return enhanced, nil
}

// TokenizeAndConvertToIDs tokenizes a text and converts tokens to their corresponding IDs, handling padding/truncation.
func TokenizeAndConvertToIDs(text string, tokenizer *tokenizer.Tokenizer, vocabulary *mainvocab.Vocabulary, maxLen int) ([]int, error) {

	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("failed to encode text: %w", err)
	}

	if maxLen <= 0 {
		if len(tokenIDs) == 0 {
			return []int{vocabulary.PaddingTokenID}, nil
		}
		return tokenIDs, nil
	}

	// Pre-allocate a slice of maxLen.
	finalTokenIDs := make([]int, maxLen)

	// Fill with padding token initially
	for i := range finalTokenIDs {
		finalTokenIDs[i] = vocabulary.PaddingTokenID
	}

	// Copy the actual token IDs, truncating if they are longer than maxLen.
	copy(finalTokenIDs, tokenIDs)

	return finalTokenIDs, nil
}

// TrainIntentMoEModel trains the MoEClassificationModel.
func TrainIntentMoEModel(model *moe.IntentMoE, data []TokenizedTrainingExample, epochs int, learningRate float64, batchSize int, maxSequenceLength int, semanticOutputVocab *mainvocab.Vocabulary) error {
	cpuProfileFile, err := os.Create("cpu.prof")
	if err != nil {
		log.Fatal("could not create CPU profile: ", err)
	}
	if err := pprof.StartCPUProfile(cpuProfileFile); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}

	if model == nil {
		return errors.New("cannot train a nil model")
	}
	if len(data) == 0 {
		return errors.New("no training data provided")
	}

	optimizer := nn.NewOptimizer(model.Parameters(), learningRate, 1.0) // Using a clip value of 1.0

	// Learning rate scheduling parameters
	baseLR := learningRate
	minLR := learningRate / 10.0 // 0.00001
	totalBatches := (len(data) + batchSize - 1) / batchSize
	totalSteps := epochs * totalBatches
	warmupSteps := totalBatches * 2 // Warmup for first 2 epochs
	currentStep := 0

	for epoch := 0; epoch < epochs; epoch++ {
		// Calculate scheduled sampling probability for logging
		scheduledSamplingProb := math.Min(0.5, float64(epoch)/float64(epochs*2))
		log.Printf("Epoch %d/%d (Scheduled Sampling: %.1f%%)", epoch+1, epochs, scheduledSamplingProb*100)
		totalLoss := 0.0
		numBatches := 0
		// Create batches for training
		for i := 0; i < len(data); i += batchSize {
			end := i + batchSize
			if end > len(data) {
				end = len(data)
			}
			batch := data[i:end]

			// Update learning rate with scheduling
			currentLR := calculateLearningRate(currentStep, totalSteps, warmupSteps, baseLR, minLR)
			if adamOpt, ok := optimizer.(*nn.Adam); ok {
				adamOpt.SetLearningRate(currentLR)
			}
			currentStep++

			loss, err := trainIntentMoEBatch(model, optimizer, batch, maxSequenceLength, epoch, epochs, semanticOutputVocab)
			if err != nil {
				log.Printf("Error training batch: %v", err)
				continue // Or handle error more strictly
			}
			totalLoss += loss
			numBatches++

			// Log gradient norms every batch for debugging
			if numBatches%5 == 0 {
				gradNorm := computeGradientNorm(model.Parameters())
				log.Printf("Batch %d: Loss=%.2f, GradNorm=%.4f, LR=%.6f", numBatches, loss, gradNorm, currentLR)
			}

			// if numBatches == 1 && memProfileFile != nil {
			// 	if err := pprof.WriteHeapProfile(memProfileFile); err != nil {
			// 		log.Printf("could not write memory profile for batch 1: %v", err)
			// 	}
			// }
		}
		if numBatches > 0 {
			log.Printf("Epoch %d, Average Loss: %f", epoch+1, totalLoss/float64(numBatches))
		}
		if epoch == 0 {
			pprof.StopCPUProfile()
			cpuProfileFile.Close()
			log.Println("CPU profile saved to cpu.prof")

			memProfileFile, err := os.Create("mem.prof")
			if err != nil {
				log.Fatal("could not create memory profile: ", err)
			}
			runtime.GC() // get up-to-date statistics
			if err := pprof.WriteHeapProfile(memProfileFile); err != nil {
				log.Fatal("could not write memory profile: ", err)
			}
			memProfileFile.Close()
			log.Println("Memory profile saved to mem.prof")
		}
	}

	return nil
}

// EnhancedTokenizedExample holds tokenized inputs plus original enhanced metadata
type EnhancedTokenizedExample struct {
	QueryIDs          []float64
	SemanticOutputIDs []float64
	Enhanced          *EnhancedTrainingExample
}

// TokenizeEnhancedTrainingData tokenizes enhanced examples and preserves ASG/roles
func TokenizeEnhancedTrainingData(data []EnhancedTrainingExample, semanticOutputTokenizer *tokenizer.Tokenizer, queryVocab, semanticOutputVocab *mainvocab.Vocabulary, maxLen int) ([]EnhancedTokenizedExample, error) {
	tokenized := make([]EnhancedTokenizedExample, len(data))
	for i, ex := range data {
		qIDs, err := TokenizeAndConvertToIDs(ex.Query, semanticOutputTokenizer, queryVocab, maxLen)
		if err != nil {
			return nil, fmt.Errorf("failed to tokenize query: %w", err)
		}

		// For flat output (legacy) we still use the FlatOutput field
		trainingSemanticOutput := "<s> " + ex.FlatOutput + " </s>"
		sIDs, err := TokenizeAndConvertToIDs(trainingSemanticOutput, semanticOutputTokenizer, semanticOutputVocab, maxLen)
		if err != nil {
			return nil, fmt.Errorf("failed to tokenize semantic output: %w", err)
		}

		tokenized[i] = EnhancedTokenizedExample{
			QueryIDs:          convertIntsToFloat64s(qIDs),
			SemanticOutputIDs: convertIntsToFloat64s(sIDs),
			Enhanced:          &data[i],
		}
	}
	return tokenized, nil
}

// TrainIntentMoEModelWithEnhancedData trains the model using enhanced examples (with SRL/ASG annotations).
// This integrates StructuredSemanticLoss as an auxiliary structural penalty (non-differentiable at present).
func TrainIntentMoEModelWithEnhancedData(model *moe.IntentMoE, enhancedData []EnhancedTrainingExample, epochs int, learningRate float64, batchSize int, maxSequenceLength int, semanticOutputVocab *mainvocab.Vocabulary) error {
	if model == nil {
		return errors.New("cannot train a nil model")
	}
	if len(enhancedData) == 0 {
		return errors.New("no enhanced training data provided")
	}

	// Build a query vocabulary from enhanced data and create tokenizers
	queryVocab := mainvocab.NewVocabulary()
	// Add tokens from queries and flat outputs to respective vocabs
	for _, ex := range enhancedData {
		for _, tok := range tokenizer.Tokenize(strings.ToLower(ex.Query)) {
			queryVocab.AddToken(tok)
		}
		for _, tok := range tokenizer.Tokenize(strings.ToLower(ex.FlatOutput)) {
			semanticOutputVocab.AddToken(tok)
		}
	}

	// Ensure BOS/EOS exist
	semanticOutputVocab.AddToken("<s>")
	semanticOutputVocab.AddToken("</s>")

	queryTok, err := tokenizer.NewTokenizer(queryVocab)
	if err != nil {
		return fmt.Errorf("failed to create query tokenizer: %w", err)
	}
	semTok, err := tokenizer.NewTokenizer(semanticOutputVocab)
	if err != nil {
		return fmt.Errorf("failed to create semantic tokenizer: %w", err)
	}

	// Tokenize enhanced data
	tokenized, err := TokenizeEnhancedTrainingData(enhancedData, queryTok, queryVocab, semanticOutputVocab, maxSequenceLength)
	if err != nil {
		return fmt.Errorf("failed to tokenize enhanced training data: %w", err)
	}

	// Convert to simple tokenized examples for re-use of optimizer creation
	simple := make([]TokenizedTrainingExample, len(tokenized))
	for i, ex := range tokenized {
		simple[i] = TokenizedTrainingExample{QueryIDs: ex.QueryIDs, SemanticOutputIDs: ex.SemanticOutputIDs}
	}

	// Reuse the standard training loop but call enhanced batch trainer
	optimizer := nn.NewOptimizer(model.Parameters(), learningRate, 1.0)
	baseLR := learningRate
	minLR := learningRate / 10.0
	totalBatches := (len(simple) + batchSize - 1) / batchSize
	totalSteps := epochs * totalBatches
	warmupSteps := totalBatches * 2
	currentStep := 0

	ssl := semantic.NewStructuredSemanticLoss()
	asgGen := semantic.NewASGGenerator()
	srl := semantic.NewSemanticRoleLabeler()

	for epoch := 0; epoch < epochs; epoch++ {
		scheduledSamplingProb := math.Min(0.5, float64(epoch)/float64(epochs*2))
		log.Printf("Enhanced Epoch %d/%d (Scheduled Sampling: %.1f%%)", epoch+1, epochs, scheduledSamplingProb*100)
		totalLoss := 0.0
		numBatches := 0

		for i := 0; i < len(simple); i += batchSize {
			end := i + batchSize
			if end > len(simple) {
				end = len(simple)
			}
			batch := simple[i:end]
			enhancedBatch := tokenized[i:end]

			currentLR := calculateLearningRate(currentStep, totalSteps, warmupSteps, baseLR, minLR)
			if adamOpt, ok := optimizer.(*nn.Adam); ok {
				adamOpt.SetLearningRate(currentLR)
			}
			currentStep++

			loss, err := trainIntentMoEBatchEnhanced(model, optimizer, batch, enhancedBatch, maxSequenceLength, epoch, epochs, semanticOutputVocab, semTok, ssl, srl, asgGen)
			if err != nil {
				log.Printf("Error training enhanced batch: %v", err)
				continue
			}
			totalLoss += loss
			numBatches++
			if numBatches%5 == 0 {
				gradNorm := computeGradientNorm(model.Parameters())
				log.Printf("Enhanced Batch %d: Loss=%.2f, GradNorm=%.4f, LR=%.6f", numBatches, loss, gradNorm, currentLR)
			}
		}
		if numBatches > 0 {
			log.Printf("Enhanced Epoch %d, Average Loss: %f", epoch+1, totalLoss/float64(numBatches))
		}
	}

	return nil
}

// trainIntentMoEBatchEnhanced trains on a batch of enhanced tokenized examples and
// applies an auxiliary structure penalty from StructuredSemanticLoss by decoding predictions
// and comparing ASG structure validity. Note: the structure penalty is non-differentiable
// because it is computed after greedy decoding; it serves as a training signal but will not
// produce gradients through the model predictions.
func trainIntentMoEBatchEnhanced(intentMoEModel *moe.IntentMoE, optimizer nn.Optimizer, batch []TokenizedTrainingExample, enhancedBatch []EnhancedTokenizedExample, maxSequenceLength int, epoch, totalEpochs int, semanticOutputVocab *mainvocab.Vocabulary, semTok *tokenizer.Tokenizer, ssl *semantic.StructuredSemanticLoss, srl *semantic.SemanticRoleLabeler, asgGen *semantic.ASGGenerator) (float64, error) {
	optimizer.ZeroGrad()

	batchSize := len(batch)

	inputIDsBatch := make([]float64, batchSize*maxSequenceLength)
	semanticOutputIDsBatch := make([]float64, batchSize*maxSequenceLength)

	for i, example := range batch {
		copy(inputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], example.QueryIDs)
		copy(semanticOutputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], example.SemanticOutputIDs)
	}

	inputTensor := NewTensor([]int{batchSize, maxSequenceLength}, inputIDsBatch, false)
	semanticOutputTensor := NewTensor([]int{batchSize, maxSequenceLength}, semanticOutputIDsBatch, false)

	scheduledSamplingProb := math.Min(0.5, float64(epoch)/float64(totalEpochs*2))

	semanticOutputLogits, contextVector, err := intentMoEModel.Forward(scheduledSamplingProb, inputTensor, semanticOutputTensor)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model forward pass failed: %w", err)
	}

	semanticOutputLoss := 0.0
	semanticOutputGrads := make([]*Tensor, maxSequenceLength-1)

	for t := 0; t < maxSequenceLength-1; t++ {
		targets := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			targets[i] = int(semanticOutputIDsBatch[i*maxSequenceLength+t+1])
		}
		loss, grad := CrossEntropyLoss(semanticOutputLogits[t], targets, semanticOutputVocab.PaddingTokenID, 0.1)
		semanticOutputLoss += loss
		semanticOutputGrads[t] = grad
	}

	// Auxiliary: decode each example greedily, build ASG and compute structure validity
	structurePenalty := 0.0
	for i := 0; i < batchSize; i++ {
		// Slice context vector for the single example
		ctxSlice, err := contextVector.Slice(0, i, i+1)
		if err != nil {
			continue
		}

		predIDs, err := intentMoEModel.GreedySearchDecode(ctxSlice, maxSequenceLength, semanticOutputVocab.BosID, semanticOutputVocab.EosID, 1.0, 100, tag.Tag{}) // topK=100
		if err != nil {
			continue
		}

		// Decode predicted token IDs to string
		predStr, err := semTok.Decode(predIDs)
		if err != nil {
			continue
		}

		// Extract semantic roles and ASG from predicted string
		predRoles, _ := srl.ExtractRoles(predStr)
		predASG := asgGen.GenerateFromSemanticRoles("", nil, nil, nil)
		if predRoles != nil {
			// Attempt to map roles into the generator inputs conservatively
			op := ""
			if o, ok := predRoles["operation"].(string); ok {
				op = o
			}
			var resources []map[string]string
			if r, ok := predRoles["resources"].([]map[string]string); ok {
				resources = r
			}
			var arguments []map[string]string
			if a, ok := predRoles["arguments"].([]map[string]string); ok {
				arguments = a
			}
			var modifiers []map[string]string
			if m, ok := predRoles["modifiers"].([]string); ok {
				for _, mm := range m {
					modifiers = append(modifiers, map[string]string{"value": mm})
				}
			}
			predASG = asgGen.GenerateFromSemanticRoles(op, resources, arguments, modifiers)
		}

		// Compare predicted ASG to ground truth
		gtASG := enhancedBatch[i].Enhanced.ASG
		metrics := ssl.ComputeMetrics(predASG, gtASG)
		// Penalize invalid structure (1.0 means valid -> penalty 0)
		structVal := metrics["structure_validity"]
		structurePenalty += (1.0 - structVal) * 0.5 // weight 0.5 for structure penalty per-example
	}

	entropyWeight := 0.01
	totalLoss := semanticOutputLoss + entropyWeight*0.0 + structurePenalty

	// Backward using token-level gradients (structurePenalty is non-differentiable)
	err = intentMoEModel.Backward(semanticOutputGrads...)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model backward pass failed: %w", err)
	}

	optimizer.Step()

	return totalLoss, nil
}

// trainIntentMoEBatch performs a single training step on a batch of data.
func trainIntentMoEBatch(intentMoEModel *moe.IntentMoE, optimizer nn.Optimizer, batch []TokenizedTrainingExample, maxSequenceLength int, epoch, totalEpochs int, semanticOutputVocab *mainvocab.Vocabulary) (float64, error) {
	optimizer.ZeroGrad()

	batchSize := len(batch)

	inputIDsBatch := make([]float64, batchSize*maxSequenceLength)
	semanticOutputIDsBatch := make([]float64, batchSize*maxSequenceLength)

	for i, example := range batch {
		copy(inputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], example.QueryIDs)
		copy(semanticOutputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], example.SemanticOutputIDs)
	}

	// Convert input IDs to a Tensor (embeddings will be handled by the model)
	inputTensor := NewTensor([]int{batchSize, maxSequenceLength}, inputIDsBatch, false)
	semanticOutputTensor := NewTensor([]int{batchSize, maxSequenceLength}, semanticOutputIDsBatch, false)

	// Calculate scheduled sampling probability: gradually increase from 0% to 50%
	// Formula: min(0.5, epoch / (totalEpochs * 2))
	scheduledSamplingProb := math.Min(0.5, float64(epoch)/float64(totalEpochs*2))

	// Forward pass through the IntentMoE model with scheduled sampling
	semanticOutputLogits, _, err := intentMoEModel.Forward(scheduledSamplingProb, inputTensor, semanticOutputTensor)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model forward pass failed: %w", err)
	}

	// Calculate loss for the semantic output
	semanticOutputLoss := 0.0
	// The decoder now produces maxSequenceLength-1 outputs
	semanticOutputGrads := make([]*Tensor, maxSequenceLength-1)
	entropyLoss := 0.0 // Entropy regularization term

	for t := 0; t < maxSequenceLength-1; t++ {
		targets := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			// Target for step t (input t) is token at t+1
			targets[i] = int(semanticOutputIDsBatch[i*maxSequenceLength+t+1])
		}
		loss, grad := CrossEntropyLoss(semanticOutputLogits[t], targets, semanticOutputVocab.PaddingTokenID, 0.1)
		semanticOutputLoss += loss
		semanticOutputGrads[t] = grad
	}

	// Combine losses with entropy regularization weight
	entropyWeight := 0.01 // Small weight to not dominate main loss
	totalLoss := semanticOutputLoss + entropyWeight*entropyLoss

	// Backward pass
	err = intentMoEModel.Backward(semanticOutputGrads...)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model backward pass failed: %w", err)
	}

	optimizer.Step()

	// Per-batch example logging commented out for speed
	// Only log loss, not decoded examples
	// predictedIDs, err := intentMoEModel.GreedySearchDecode(contextVector, 20, semanticOutputVocab.GetTokenID("<s>"), semanticOutputVocab.GetTokenID("</s>"), 1.0, 100) // topK=100
	// if err != nil {
	// 	log.Printf("Error decoding guessed sentence: %v", err)
	// } else {
	// 	guessedSentence, err := semanticOutputTokenizer.Decode(predictedIDs)
	// 	if err != nil {
	// 		log.Printf("Error decoding guessed sentence: %v", err)
	// 	} else {
	// 		log.Printf("Guessed semantic output: %s", guessedSentence)
	// 	}
	// 	targetJSON, _ := json.Marshal(batch[0].SemanticOutput)
	// 	log.Printf("Target semantic output: %s", string(targetJSON))
	// }

	return totalLoss, nil
}

// computeGradientNorm calculates the L2 norm of all parameter gradients
func computeGradientNorm(params []*Tensor) float64 {
	totalNorm := 0.0
	for _, param := range params {
		if param.Grad != nil {
			for _, g := range param.Grad.Data {
				totalNorm += g * g
			}
		}
	}
	return math.Sqrt(totalNorm)
}

// calculateLearningRate computes the learning rate with warmup and cosine decay
func calculateLearningRate(step, totalSteps, warmupSteps int, baseLR, minLR float64) float64 {
	if step < warmupSteps {
		// Linear warmup
		return baseLR * float64(step) / float64(warmupSteps)
	}
	// Cosine decay after warmup
	progress := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
	return minLR + (baseLR-minLR)*0.5*(1+math.Cos(math.Pi*progress))
}

func convertIntsToFloat64s(input []int) []float64 {
	output := make([]float64, len(input))
	for i, v := range input {
		output[i] = float64(v)
	}
	return output
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func convertW2VVocab(w2vVocab map[string]int) *mainvocab.Vocabulary {
	vocab := mainvocab.NewVocabulary()
	vocab.WordToToken = w2vVocab
	maxID := 0
	for _, id := range w2vVocab {
		if id > maxID {
			maxID = id
		}
	}
	vocab.TokenToWord = make([]string, maxID+1)
	for token, id := range w2vVocab {
		vocab.TokenToWord[id] = token
	}
	return vocab
}

func BuildVocabularies(dataPath string) (*mainvocab.Vocabulary, *mainvocab.Vocabulary, error) {
	queryVocabulary := mainvocab.NewVocabulary()
	semanticOutputVocabulary := mainvocab.NewVocabulary()

	semanticTrainingData, err := LoadIntentTrainingData(dataPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load semantic training data from %s: %w", dataPath, err)
	}

	for _, pair := range *semanticTrainingData {
		// Use the same tokenizer logic as during inference to build the vocabulary
		tokenizedQuery := tokenizer.Tokenize(strings.ToLower(pair.Query))
		for _, word := range tokenizedQuery {
			queryVocabulary.AddToken(word)
		}

		semanticOutputJSON, err := json.Marshal(pair.SemanticOutput)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal semantic output: %w", err)
		}

		// Add BOS and EOS tokens to the sentence when building the vocabulary
		trainingSemanticOutput := "<s> " + string(semanticOutputJSON) + " </s>"
		tokenizedSemanticOutput := tokenizer.Tokenize(trainingSemanticOutput)
		for _, word := range tokenizedSemanticOutput {
			semanticOutputVocabulary.AddToken(word)
		}
	}

	// Explicitly add BOS and EOS tokens to the sentence vocabulary
	semanticOutputVocabulary.BosID = semanticOutputVocabulary.GetTokenID("<s>")
	semanticOutputVocabulary.EosID = semanticOutputVocabulary.GetTokenID("</s>")

	return queryVocabulary, semanticOutputVocabulary, nil
}

func main() {
	const trainingDataPath = "./trainingdata/intent_data.json"
	const semanticTrainingDataPath = "./trainingdata/semantic_output_data_flat.json"
	const word2vecModelPath = "gob_models/word2vec_model.gob"

	// Set up a channel to listen for interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Goroutine to handle graceful shutdown on signal
	go func() {
		<-sigChan // Block until a signal is received
		log.Println("Received interrupt signal. Exiting.")
		os.Exit(0) // Exit gracefully
	}()

	// Define training parameters
	epochs := 5           // Reasonable for full dataset
	learningRate := 0.001 // Standard learning rate
	batchSize := 16       // Reduced from 32 to avoid dimension errors with large dataset
	semanticOutputVocabularySavePath := "gob_models/semantic_output_vocabulary.gob"

	// Load Word2Vec model
	word2vecModel, err := word2vec.LoadModel(word2vecModelPath)
	if err != nil {
		log.Fatalf("Failed to load Word2Vec model: %v", err)
	}

	// Create query vocabulary from word2vec model
	queryVocabulary := convertW2VVocab(word2vecModel.Vocabulary)

	// Add missing tokens from our specific domain
	extraTokens := []string{"jill", "webserver", "jack", "go", "8080", "create", "named", "jim", "test", "data", "handler"}
	for _, token := range extraTokens {
		if _, exists := queryVocabulary.WordToToken[token]; !exists {
			queryVocabulary.AddToken(token)
			log.Printf("Added extra token to query vocab: %s", token)
		}
	}

	// Try to load other vocabularies first
	semanticOutputVocabulary, err := mainvocab.LoadVocabulary(semanticOutputVocabularySavePath)
	if err != nil {
		log.Println("Failed to load semantic output vocabulary, creating a new one.")
	}

	if semanticOutputVocabulary == nil {
		log.Println("Building vocabularies from scratch...")
		_, semanticOutputVocabulary, err = BuildVocabularies(semanticTrainingDataPath)
		if err != nil {
			log.Fatalf("Failed to build vocabularies: %v", err)
		}
	}

	log.Printf("Query Vocabulary (after load/create): Size=%d", len(queryVocabulary.WordToToken))
	log.Printf("Semantic Output Vocabulary (after load/create): Size=%d", len(semanticOutputVocabulary.WordToToken))

	// Load Intent training data
	semanticTrainingData, err := LoadIntentTrainingData(semanticTrainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load semantic training data from %s: %v", semanticTrainingDataPath, err)
	}
	log.Printf("Loaded %d training examples from %s.", len(*semanticTrainingData), semanticTrainingDataPath)

	// TEMPORARILY DISABLED: Load only small dataset for debugging
	/*
		// Load WikiQA training data
		const wikiQATrainingDataPath = "./trainingdata/generated_wikiqa_intents.json"
		wikiQATrainingData, err := LoadIntentTrainingData(wikiQATrainingDataPath)
		if err != nil {
			log.Printf("Warning: Failed to load WikiQA training data from %s: %v. Proceeding without it.", wikiQATrainingDataPath, err)
		} else {
			log.Printf("Loaded %d training examples from %s.", len(*wikiQATrainingData), wikiQATrainingDataPath)
			// Use full WikiQA dataset for better generalization
			// Merge datasets
			*semanticTrainingData = append(*semanticTrainingData, *wikiQATrainingData...)
			log.Printf("Total training examples after merging WikiQA: %d", len(*semanticTrainingData))
		}

		// Load Q&A training data from converted TSV
		const qaTrainingDataPath = "./trainingdata/qa_semantic_output.json"
		qaTrainingData, err := LoadIntentTrainingData(qaTrainingDataPath)
		if err != nil {
			log.Printf("Warning: Failed to load Q&A training data from %s: %v. Proceeding without it.", qaTrainingDataPath, err)
		} else {
			log.Printf("Loaded %d Q&A training examples from %s.", len(*qaTrainingData), qaTrainingDataPath)
			// Use full Q&A dataset for better generalization
			// Merge Q&A dataset
			*semanticTrainingData = append(*semanticTrainingData, *qaTrainingData...)
			log.Printf("Total training examples after merging Q&A: %d", len(*semanticTrainingData))
		}
	*/

	log.Printf("Training on small dataset only: %d examples", len(*semanticTrainingData))

	// After vocabularies are fully populated, determine vocab sizes and create/load model
	inputVocabSize := len(queryVocabulary.WordToToken)
	semanticOutputVocabSize := len(semanticOutputVocabulary.WordToToken)
	embeddingDim := 128      // Increased back to 128
	numExperts := 4          // Increased back to 4
	maxSequenceLength := 120 // Increased to 120
	maxAttentionHeads := 4   // Increased back to 4

	log.Printf("Query Vocabulary Size: %d", inputVocabSize)
	log.Printf("Semantic Output Vocabulary Size: %d", semanticOutputVocabSize)
	log.Printf("Embedding Dimension: %d", embeddingDim)
	log.Printf("Word2Vec Model Vocab Size: %d", word2vecModel.VocabSize)
	log.Printf("Word2Vec Model Vector Size: %d", word2vecModel.VectorSize)
	log.Printf("Number of Experts: %d", numExperts)

	var intentMoEModel *moe.IntentMoE // Declare intentMoEModel here

	modelSavePath := "gob_models/moe_classification_model.gob"

	// Always create a new IntentMoE model for now to debug gob loading
	log.Printf("Creating a new IntentMoE model.")
	// Model hyperparameters - ORIGINAL SIZE (stable with flat format)
	embeddingDim = 128    // Original size
	hiddenSize := 256     // Original size
	maxAttentionHeads = 4 // Keep at 4
	numLayers := 2        // Original size
	dropoutRate := 0.1    // Keep at 0.1

	// 1. Embedding
	embedding := nn.NewEmbedding(inputVocabSize, embeddingDim)
	if word2vecModel != nil {
		embedding.LoadPretrainedWeights(word2vecModel.WordVectors)
	}

	// 2. Simple RNN Encoder (replacing MoE)
	log.Println("Using SimpleRNNEncoder instead of MoE")
	encoder, err := moe.NewSimpleRNNEncoder(embeddingDim, hiddenSize, numLayers)
	if err != nil {
		log.Fatalf("Failed to create SimpleRNNEncoder: %v", err)
	}

	// 3. RNN Decoder with increased capacity and dropout
	decoder, err := moe.NewRNNDecoder(embeddingDim, semanticOutputVocabSize, hiddenSize, maxAttentionHeads, numLayers, dropoutRate)
	if err != nil {
		log.Fatalf("Failed to create decoder: %v", err)
	}

	// 4. Create IntentMoE model
	intentMoEModel = &moe.IntentMoE{
		Embedding:         embedding,
		Encoder:           encoder,
		Decoder:           decoder,
		SentenceVocabSize: semanticOutputVocabSize,
	}

	// Training Loop
	// epochs = 5 // Removed redundant assignment

	// Create tokenizers once after vocabularies are loaded/created
	queryTokenizer, err := tokenizer.NewTokenizer(queryVocabulary)
	if err != nil {
		log.Fatalf("Failed to create query tokenizer: %v", err)
	}
	semanticOutputTokenizer, err := tokenizer.NewTokenizer(semanticOutputVocabulary)
	if err != nil {
		log.Fatalf("Failed to create semantic output tokenizer: %v", err)
	}

	log.Println("Pre-tokenizing training data...")
	tokenizedData, err := TokenizeTrainingData(semanticTrainingData, queryTokenizer, semanticOutputTokenizer, queryVocabulary, semanticOutputVocabulary, maxSequenceLength)
	if err != nil {
		log.Fatalf("Failed to tokenize training data: %v", err)
	}
	log.Println("Pre-tokenization complete.")

	// Enhance training data with semantic role labeling and ASG
	log.Println("Enhancing training data with semantic role labeling and abstract semantic graphs...")
	enhancedData, err := EnhanceTrainingDataWithSRLAndASG(semanticTrainingData)
	if err != nil {
		log.Fatalf("Failed to enhance training data: %v", err)
	}
	log.Printf("Enhanced %d training examples with SRL and ASG annotations.\n", len(enhancedData))

	// Save enhanced training data for analysis
	enhancedDataFile, err := os.Create("trainingdata/enhanced_training_data.json")
	if err != nil {
		log.Printf("Warning: Could not save enhanced training data: %v\n", err)
	} else {
		defer enhancedDataFile.Close()
		encoder := json.NewEncoder(enhancedDataFile)
		if err := encoder.Encode(enhancedData[:min(len(enhancedData), 10)]); err != nil {
			log.Printf("Warning: Could not write enhanced training data: %v\n", err)
		}
	}

	// Train the model
	err = TrainIntentMoEModel(intentMoEModel, tokenizedData, epochs, learningRate, batchSize, maxSequenceLength, semanticOutputVocabulary)
	if err != nil {
		log.Fatalf("Failed to train IntentMoE model: %v", err)
	}

	// Detach the model from the computation graph to allow for clean serialization
	log.Println("Detaching model from computation graph...")
	DetachModel(intentMoEModel)

	// Save the trained model
	fmt.Printf("Saving IntentMoE model to %s\n", modelSavePath)
	modelFile, err := os.Create(modelSavePath)
	if err != nil {
		log.Fatalf("Failed to create model file: %v", err)
	}
	defer modelFile.Close()
	err = moe.SaveIntentMoEModelToGOB(intentMoEModel, modelFile)
	if err != nil {
		log.Fatalf("Failed to save IntentMoE model: %v", err)
	}

	// Save the vocabularies
	queryVocabularySavePath := "gob_models/query_vocabulary.gob"
	err = queryVocabulary.Save(queryVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save query vocabulary: %v", err)
	}
	err = semanticOutputVocabulary.Save(semanticOutputVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save semantic output vocabulary: %v", err)
	}
}

// DetachModel removes the computation graph (gradients and creators) from the model parameters
// to ensure that only the weights are saved. This prevents serialization issues and reduces file size.
func DetachModel(model *moe.IntentMoE) {
	params := model.Parameters()
	for _, param := range params {
		param.Grad = nil
		param.Creator = nil
		param.Mask = nil
		param.Operation = nil
		// We keep RequiresGrad as is, or set it to false if we want to freeze the model.
		// For saving, it doesn't strictly matter for gob if we don't save Creator,
		// but setting Creator to nil is the key.
	}

	// Clear decoder state which might hold references to the computation graph
	if model.Decoder != nil {
		model.Decoder.InitialHiddenState = nil
		model.Decoder.InitialCellState = nil

		// Clear LSTM cells state
		if model.Decoder.LSTM != nil {
			for _, layer := range model.Decoder.LSTM.Cells {
				for _, cell := range layer {
					cell.InputTensor = nil
					cell.PrevHidden = nil
					cell.PrevCell = nil
				}
			}
		}
	}

	log.Println("Model detached from computation graph.")
	runtime.GC() // Force garbage collection to free up memory before saving
}
