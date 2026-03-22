package main

import (
	"encoding/gob"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/golangast/gollemer/internal/moe"
	"github.com/golangast/gollemer/neural/nn"
	mainvocab "github.com/golangast/gollemer/neural/nnu/vocab"
	"github.com/golangast/gollemer/neural/nnu/word2vec"
	"github.com/golangast/gollemer/neural/semantic"
	. "github.com/golangast/gollemer/neural/tensor"
	"github.com/golangast/gollemer/neural/tokenizer"
	"github.com/golangast/gollemer/tagger/tag"
)

func init() {
	gob.Register(&moe.MoEEncoder{})
}

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
	SemanticRoles map[string]any                  `json:"semantic_roles"`
	ASG           *semantic.AbstractSemanticGraph `json:"abstract_semantic_graph"`
	ExecutionPlan map[string]any                  `json:"execution_plan"`
}

// TokenizeTrainingData pre-tokenizes the training data in parallel.
func TokenizeTrainingData(data *IntentTrainingData, queryTokenizer, semanticOutputTokenizer *tokenizer.Tokenizer, queryVocab, semanticOutputVocab *mainvocab.Vocabulary, maxLen int) ([]TokenizedTrainingExample, error) {
	tokenizedData := make([]TokenizedTrainingExample, len(*data))
	var wg sync.WaitGroup
	var errMutex sync.Mutex
	var firstErr error

	numWorkers := runtime.NumCPU()
	batchSize := (len(*data) + numWorkers - 1) / numWorkers

	for w := range numWorkers {
		start := w * batchSize
		end := min(start+batchSize, len(*data))
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
			roles = make(map[string]any)
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
func TrainIntentMoEModel(model *moe.IntentMoE, data []TokenizedTrainingExample, epochs int, learningRate float64, batchSize int, maxSequenceLength int, semanticOutputVocab *mainvocab.Vocabulary, checkpointPath string, checkpointInterval int) error {
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

	for epoch := range epochs {
		// Calculate scheduled sampling probability for logging
		scheduledSamplingProb := math.Min(0.5, float64(epoch+1)/float64(epochs*8))
		log.Printf("Epoch %d/%d (Scheduled Sampling: %.1f%%)", epoch+1, epochs, scheduledSamplingProb*100)
		totalLoss := 0.0
		numBatches := 0
		// Create batches for training
		for i := 0; i < len(data); i += batchSize {
			batchStartTime := time.Now()
			end := min(i+batchSize, len(data))
			batch := data[i:end]

			// Update learning rate with scheduling
			currentLR := calculateLearningRate(currentStep, totalSteps, warmupSteps, baseLR, minLR)
			if adamOpt, ok := optimizer.(*nn.Adam); ok {
				adamOpt.SetLearningRate(currentLR)
			}
			currentStep++

			loss, err := trainIntentMoEBatch(model, optimizer, batch, maxSequenceLength, epoch, epochs, semanticOutputVocab, numBatches)
			if err != nil {
				log.Printf("Error training batch: %v", err)
				continue // Or handle error more strictly
			}
			totalLoss += loss
			
			numBatches++

			// Checkpointing
			if checkpointInterval > 0 && numBatches%checkpointInterval == 0 {
				if err := saveCheckpoint(model, checkpointPath, epoch, numBatches); err != nil {
					log.Printf("Failed to save checkpoint: %v", err)
				}
			}

			// Log gradient norms every batch for debugging
			if numBatches%5 == 0 {
				gradNorm := computeGradientNorm(model.Parameters())
				percent := float64(numBatches) / float64(totalBatches) * 100
				log.Printf("Batch %d/%d (%.1f%%): Loss=%.2f, GradNorm=%.4f, LR=%.6f, Time=%v", numBatches, totalBatches, percent, loss, gradNorm, currentLR, time.Since(batchStartTime))

				for i, layer := range moe.ActiveLayers {
					fmt.Printf("Layer %d ", i)
					layer.VisualizeUtilization()
				}
				// runtime.GC() // Removed explicit GC for performance
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

	for epoch := range epochs {
		scheduledSamplingProb := math.Min(0.5, float64(epoch+1)/float64(epochs*4))
		log.Printf("Enhanced Epoch %d/%d (Scheduled Sampling: %.1f%%)", epoch+1, epochs, scheduledSamplingProb*100)
		totalLoss := 0.0
		numBatches := 0

		for i := 0; i < len(simple); i += batchSize {
			end := min(i+batchSize, len(simple))
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
			
			// Aggressively clear computation graph after each batch
			DetachModel(model)
			numBatches++
			if numBatches%5 == 0 {
				gradNorm := computeGradientNorm(model.Parameters())
				log.Printf("Enhanced Batch %d: Loss=%.2f, GradNorm=%.4f, LR=%.6f", numBatches, loss, gradNorm, currentLR)
				// runtime.GC() // Removed explicit GC for performance
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

	scheduledSamplingProb := math.Min(0.5, float64(epoch+1)/float64(totalEpochs*2))

	semanticOutputLogits, contextVector, err := intentMoEModel.Forward(scheduledSamplingProb, inputTensor, semanticOutputTensor)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model forward pass failed: %w", err)
	}

	semanticOutputLoss := 0.0
	semanticOutputGrads := make([]*Tensor, maxSequenceLength-1)

	targets := make([]int, batchSize)
	for t := 0; t < maxSequenceLength-1; t++ {
		for i := range batchSize {
			targets[i] = int(semanticOutputIDsBatch[i*maxSequenceLength+t+1])
		}
		loss, grad := CrossEntropyLoss(semanticOutputLogits[t], targets, semanticOutputVocab.PaddingTokenID, 0.1)
		semanticOutputLoss += loss
		semanticOutputGrads[t] = grad
	}

	// Auxiliary: decode each example greedily, build ASG and compute structure validity
	structurePenalty := 0.0
	for i := range batchSize {
		// Slice context vector for the single example
		ctxSlice, err := contextVector.Slice(0, i, i+1)
		if err != nil {
			continue
		}

		predIDs, err := intentMoEModel.GreedySearchDecode(ctxSlice, maxSequenceLength, semanticOutputVocab.BosID, semanticOutputVocab.EosID, 1.0, 0.0, 100, tag.Tag{}) // topK=100
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

	// Clip gradients to prevent exploding gradients
	clipGradients(intentMoEModel.Parameters(), 10.0)

	optimizer.Step()

	return totalLoss, nil
}

// trainIntentMoEBatch performs a single training step on a batch of data.
func trainIntentMoEBatch(intentMoEModel *moe.IntentMoE, optimizer nn.Optimizer, batch []TokenizedTrainingExample, maxSequenceLength int, epoch, totalEpochs int, semanticOutputVocab *mainvocab.Vocabulary, batchIndex int) (float64, error) {
	start := time.Now()
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

	prepTime := time.Since(start)
	tForward := time.Now()

	// Calculate scheduled sampling probability: gradually increase from 0% to 50%
	// Formula: min(0.5, (epoch + 1) / (totalEpochs * 2)) - Faster increase to force model recovery
	scheduledSamplingProb := math.Min(0.5, float64(epoch+1)/float64(totalEpochs*2))

	// Forward pass through the IntentMoE model with scheduled sampling
	semanticOutputLogits, _, err := intentMoEModel.Forward(scheduledSamplingProb, inputTensor, semanticOutputTensor)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model forward pass failed: %w", err)
	}
	forwardTime := time.Since(tForward)
	tLoss := time.Now()

	// Calculate loss for the semantic output
	semanticOutputLoss := 0.0
	// The decoder now produces maxSequenceLength-1 outputs in non-vectorized mode,
	// or a single 3D tensor in vectorized mode.
	var semanticOutputGrads []*Tensor
	entropyLoss := 0.0
	
	if len(semanticOutputLogits) == 1 && len(semanticOutputLogits[0].Shape) == 3 {
		// Vectorized 3D loss path
		logits3D := semanticOutputLogits[0]
		numSteps := maxSequenceLength - 1
		allTargets := make([]int, batchSize*numSteps)
		for b := 0; b < batchSize; b++ {
			for t := 0; t < numSteps; t++ {
				allTargets[b*numSteps+t] = int(semanticOutputIDsBatch[b*maxSequenceLength+t+1])
			}
		}
		
		loss, grad3D := CrossEntropyLoss(logits3D, allTargets, semanticOutputVocab.PaddingTokenID, 0.1)
		semanticOutputLoss = loss
		
		// For backward pass, we might need a slice of grads if the model expects it,
		// but since we are in vectorized mode, we should update IntentMoE.Backward to handle 3D.
		// For now, let's keep it as a single-element slice for the model's Backward to handle.
		semanticOutputGrads = []*Tensor{grad3D}
	} else {
		// Traditional step-by-step loss path
		semanticOutputGrads = make([]*Tensor, len(semanticOutputLogits))
		var wg sync.WaitGroup
		var lossMutex sync.Mutex

		for t := range semanticOutputLogits {
			wg.Add(1)
			go func(t int) {
				defer wg.Done()
				targets := make([]int, batchSize)
				for i := 0; i < batchSize; i++ {
					targets[i] = int(semanticOutputIDsBatch[i*maxSequenceLength+t+1])
				}
				loss, grad := CrossEntropyLoss(semanticOutputLogits[t], targets, semanticOutputVocab.PaddingTokenID, 0.1)
				if grad == nil {
					// Handle case where loss function returns nil gradient (e.g. all padding)
					grad = NewTensor(semanticOutputLogits[t].Shape, make([]float64, len(semanticOutputLogits[t].Data)), false)
				}
				lossMutex.Lock()
				semanticOutputLoss += loss
				semanticOutputGrads[t] = grad
				lossMutex.Unlock()
			}(t)
		}
		wg.Wait()
		
		// Normalize by number of timesteps to be consistent with vectorized path
		if len(semanticOutputLogits) > 0 {
			div := float64(len(semanticOutputLogits))
			semanticOutputLoss /= div
			for t := range semanticOutputGrads {
				for i := range semanticOutputGrads[t].Data {
					semanticOutputGrads[t].Data[i] /= div
				}
			}
		}
	}

	// Combine losses with entropy regularization weight
	entropyWeight := 0.01 // Small weight to not dominate main loss

	lbLoss := 0.0
	if moeEnc, ok := intentMoEModel.Encoder.(*moe.MoEEncoder); ok {
		lbLoss = moeEnc.Layer.LoadBalancingLoss * moeEnc.Layer.LoadBalancingWeight
	}

	totalLoss := semanticOutputLoss + entropyWeight*entropyLoss + lbLoss

	lossTime := time.Since(tLoss)
	tBackward := time.Now()

	// Backward pass
	err = intentMoEModel.Backward(semanticOutputGrads...)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model backward pass failed: %w", err)
	}

	backwardTime := time.Since(tBackward)
	tOptim := time.Now()

	// Clip gradients to prevent exploding gradients
	clipGradients(intentMoEModel.Parameters(), 1.0)

	optimizer.Step()

	optimTime := time.Since(tOptim)

	if batchIndex%10 == 0 {
		log.Printf("Batch %d Profile: Prep=%v, Fwd=%v, Loss=%v, Bwd=%v, Opt=%v", batchIndex, prepTime, forwardTime, lossTime, backwardTime, optimTime)
	}

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

// clipGradients scales the gradients of the model parameters if the norm exceeds maxNorm.
func clipGradients(params []*Tensor, maxNorm float64) {
	totalNorm := computeGradientNorm(params)
	if totalNorm > maxNorm {
		scale := maxNorm / totalNorm
		for _, param := range params {
			if param.Grad != nil {
				for i := range param.Grad.Data {
					param.Grad.Data[i] *= scale
				}
			}
		}
	}
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
		return minLR + (baseLR-minLR)*float64(step)/float64(warmupSteps)
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

func BuildVocabularies(semanticTrainingData *IntentTrainingData) (*mainvocab.Vocabulary, *mainvocab.Vocabulary, error) {
	queryVocabulary := mainvocab.NewVocabulary()
	semanticOutputVocabulary := mainvocab.NewVocabulary()

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
	const semanticTrainingDataPath = "./trainingdata/semantic_output_data_flat.json"
	const word2vecModelPath = "gob_models/word2vec_model.gob"

	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

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
	dryRun := flag.Bool("dry-run", false, "Run a quick test with 100 examples for 5 epochs")
	flagLR := flag.Float64("lr", 0.00001, "Learning rate for training (default 1e-5)")
	flagEpochs := flag.Int("epochs", 50, "Number of epochs to train (default 50)")

	flag.Parse()

	epochs := *flagEpochs
	learningRate := *flagLR
	batchSize := 8        // Reduced to avoid OOM
	semanticOutputVocabularySavePath := "gob_models/semantic_output_vocabulary.gob"

	// Load Word2Vec model
	word2vecModel, err := word2vec.LoadModel(word2vecModelPath)
	if err != nil {
		log.Fatalf("Failed to load Word2Vec model: %v", err)
	}

	// Load Intent training data
	semanticTrainingData, err := LoadIntentTrainingData(semanticTrainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load semantic training data from %s: %v", semanticTrainingDataPath, err)
	}
	log.Printf("Loaded %d training examples from %s.", len(*semanticTrainingData), semanticTrainingDataPath)

	if *dryRun {
		log.Println("🏃 DRY RUN ENABLED: Using only 100 examples and 5 epochs.")
		epochs = 5
		if len(*semanticTrainingData) > 100 {
			subset := (*semanticTrainingData)[:100]
			*semanticTrainingData = subset
		}
	}

	// Create query vocabulary from word2vec model
	queryVocabulary := convertW2VVocab(word2vecModel.Vocabulary)

	// Frequency-based Vocabulary Pruning for domain tokens
	log.Println("Pruning noisy domain tokens...")
	tokenCounts := make(map[string]int)
	for _, example := range *semanticTrainingData {
		for _, tok := range tokenizer.Tokenize(strings.ToLower(example.Query)) {
			tokenCounts[tok]++
		}
	}

	// Add missing tokens from our specific domain IF they appear enough or are known special tokens
	extraTokens := []string{"jill", "webserver", "jack", "go", "8080", "create", "named", "jim", "test", "data", "handler"}
	for _, token := range extraTokens {
		if _, exists := queryVocabulary.WordToToken[token]; !exists {
			queryVocabulary.AddToken(token)
		}
	}

	// Prune tokens that are NOT in Word2Vec and have very low frequency
	missCount := 0
	for tok, count := range tokenCounts {
		if _, exists := word2vecModel.Vocabulary[tok]; !exists {
			if count < 2 { // Prune singleton tokens that are also Word2Vec misses
				missCount++
				continue
			}
			queryVocabulary.AddToken(tok)
		}
	}
	log.Printf("Pruned %d low-frequency 'garbage' tokens missing from Word2Vec.", missCount)

	// Balance data: Oversample semantic data to match WikiQA scale
	log.Println("Balancing dataset (Oversampling semantic data)...")
	originalSemantic := *semanticTrainingData
	// Reduced oversampling from 6x to 3x to mitigate overfitting
	for i := 0; i < 1; i++ { // 2x total
		*semanticTrainingData = append(*semanticTrainingData, originalSemantic...)
	}
	log.Printf("Semantic training data size after balancing: %d", len(*semanticTrainingData))

	// Load WikiQA training data if available
	const wikiQATrainingDataPath = "./trainingdata/generated_wikiqa_intents.json"
	if _, err := os.Stat(wikiQATrainingDataPath); err == nil {
		wikiQATrainingData, err := LoadIntentTrainingData(wikiQATrainingDataPath)
		if err == nil {
			*semanticTrainingData = append(*semanticTrainingData, *wikiQATrainingData...)
			log.Printf("Merged %d WikiQA examples. Total: %d", len(*wikiQATrainingData), len(*semanticTrainingData))
		}
	}

	// Load Q&A training data if available
	const qaTrainingDataPath = "./trainingdata/qa_semantic_output.json"
	if _, err := os.Stat(qaTrainingDataPath); err == nil {
		qaTrainingData, err := LoadIntentTrainingData(qaTrainingDataPath)
		if err == nil {
			*semanticTrainingData = append(*semanticTrainingData, *qaTrainingData...)
			log.Printf("Merged %d Q&A examples. Total: %d", len(*qaTrainingData), len(*semanticTrainingData))
		}
	}

	// Load Conversational training data
	const conversationalDataPath = "./trainingdata/conversational_intents.json"
	if _, err := os.Stat(conversationalDataPath); err == nil {
		convData, err := LoadIntentTrainingData(conversationalDataPath)
		if err == nil {
			*semanticTrainingData = append(*semanticTrainingData, *convData...)
			log.Printf("Merged %d Conversational examples. Total: %d", len(*convData), len(*semanticTrainingData))
		}
	}

	// Load Help training data
	const helpDataPath = "./trainingdata/help_intents.json"
	if _, err := os.Stat(helpDataPath); err == nil {
		helpData, err := LoadIntentTrainingData(helpDataPath)
		if err == nil {
			*semanticTrainingData = append(*semanticTrainingData, *helpData...)
			log.Printf("Merged %d Help examples. Total: %d", len(*helpData), len(*semanticTrainingData))
		}
	}

	// Try to load other vocabularies first
	semanticOutputVocabulary, err := mainvocab.LoadVocabulary(semanticOutputVocabularySavePath)
	if err != nil {
		log.Println("Failed to load semantic output vocabulary, creating a new one.")
	}

	if semanticOutputVocabulary == nil {
		log.Println("Building vocabularies from scratch...")
		_, semanticOutputVocabulary, err = BuildVocabularies(semanticTrainingData)
		if err != nil {
			log.Fatalf("Failed to build vocabularies: %v", err)
		}
	}

	log.Printf("Query Vocabulary (after load/create): Size=%d", len(queryVocabulary.WordToToken))
	log.Printf("Semantic Output Vocabulary (after load/create): Size=%d", len(semanticOutputVocabulary.WordToToken))

	// After vocabularies are fully populated, determine vocab sizes and create/load model
	inputVocabSize := len(queryVocabulary.WordToToken)
	semanticOutputVocabSize := len(semanticOutputVocabulary.WordToToken)
	embeddingDim := word2vecModel.VectorSize // Match Word2Vec dimension
	numExperts := 4          // Increased back to 4
	maxSequenceLength := 50 // Reduced to 50

	log.Printf("Query Vocabulary Size: %d", inputVocabSize)
	log.Printf("Semantic Output Vocabulary Size: %d", semanticOutputVocabSize)
	log.Printf("Embedding Dimension: %d", embeddingDim)
	log.Printf("Word2Vec Model Vocab Size: %d", word2vecModel.VocabSize)
	log.Printf("Word2Vec Model Vector Size: %d", word2vecModel.VectorSize)
	log.Printf("Number of Experts: %d", numExperts)

	var intentMoEModel *moe.IntentMoE // Declare intentMoEModel here

	modelSavePath := "gob_models/moe_classification_model.gob"

	// Try to load existing model first
	if _, err := os.Stat(modelSavePath); err == nil {
		log.Printf("Loading existing IntentMoE model from %s...", modelSavePath)
		intentMoEModel, err = moe.LoadIntentMoEModelFromGOB(modelSavePath)
		if err != nil {
			log.Printf("Failed to load existing model: %v. Creating new one.", err)
			intentMoEModel = nil
		} else {
			log.Println("Successfully loaded existing model.")
			
			// Dynamic Resizing for Vocab Growth
			if inputVocabSize > intentMoEModel.Embedding.VocabSize {
				log.Printf("Resizing model embeddings: %d -> %d", intentMoEModel.Embedding.VocabSize, inputVocabSize)
				intentMoEModel.ResizeEmbeddings(inputVocabSize)
			}
			if semanticOutputVocabSize > intentMoEModel.SentenceVocabSize {
				log.Printf("Resizing model output layer: %d -> %d", intentMoEModel.SentenceVocabSize, semanticOutputVocabSize)
				intentMoEModel.Decoder.ResizeOutputLayer(semanticOutputVocabSize)
				intentMoEModel.SentenceVocabSize = semanticOutputVocabSize
			}
		}
	}

	if intentMoEModel == nil {
		// Always create a new IntentMoE model for now to debug gob loading
		log.Printf("Creating a new IntentMoE model. (SIMD Enabled: %v)", IsSIMDEnabled())
		// Model hyperparameters - INCREASED CAPACITY
		// embeddingDim is set above based on W2V model
		hiddenSize := 512              // Increased from 256 for more capacity
		maxAttentionHeads := 4         // Keep at 4
		numLayers := 2                 // Original size
		dropoutRate := 0.1             // Keep at 0.1

		// 1. Embedding
		embedding := nn.NewEmbedding(inputVocabSize, embeddingDim)
		
		// Always Initialize with Xavier/Glorot first to ensure unknown tokens are not zero
		log.Println("Initializing embeddings with Xavier/Glorot...")
		fanIn := inputVocabSize
		fanOut := embeddingDim
		limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
		
		// Create initial weights map
		initialWeights := make(map[int][]float64)
		for i := 0; i < inputVocabSize; i++ {
			initialWeights[i] = make([]float64, embeddingDim)
			for j := 0; j < embeddingDim; j++ {
				initialWeights[i][j] = (rand.Float64() * 2 * limit) - limit
			}
		}
		embedding.LoadPretrainedWeights(initialWeights)

		if word2vecModel != nil && word2vecModel.VectorSize == embeddingDim {
			log.Printf("Loading pretrained Word2Vec weights (dim=%d)...", embeddingDim)
			embedding.LoadPretrainedWeights(word2vecModel.WordVectors)

			// Fix Word2Vec Misses: Re-initialize UNK embedding with noise
			if unkID, ok := queryVocabulary.WordToToken["UNK"]; ok {
				log.Printf("Re-initializing UNK token (ID %d) with random noise...", unkID)
				limit := math.Sqrt(6.0 / float64(inputVocabSize+embeddingDim))
				start := unkID * embeddingDim
				end := start + embeddingDim
				for j := start; j < end; j++ {
					embedding.Weight.Data[j] = (rand.Float64() * 2 * limit) - limit
				}
			}
		} else if word2vecModel != nil {
			log.Printf("Word2Vec model vector size %d does not match embedding dim %d. Skipping loading pretrained weights.", word2vecModel.VectorSize, embeddingDim)
		}

		// 2. MoE Encoder
		log.Println("Using MoE Encoder")
		encoder, err := moe.NewMoEEncoder(embeddingDim, hiddenSize, numLayers, numExperts)
		if err != nil {
			log.Fatalf("Failed to create MoEEncoder: %v", err)
		}

		// Adjust MoE settings for training
		encoder.Layer.LoadBalancingWeight = 0.5
		encoder.Layer.CapacityFactor = 1.5
		encoder.Layer.RouterTemperature = 1.0

		// 3. RNN Decoder with increased capacity and dropout
		decoder, err := moe.NewRNNDecoder(hiddenSize, semanticOutputVocabSize, hiddenSize, maxAttentionHeads, numLayers, dropoutRate, numExperts)
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
	}

	// Adjust MoE settings for training to prevent token dropping and encourage diversity
	for _, layer := range moe.ActiveLayers {
		layer.CapacityFactor = 1.2       // Increased to prevents tokens being dropped (Suggested 1.0+)
		layer.LoadBalancingWeight = 2.5  // Increased to force router to distribute tokens more evenly (Suggested 2.0-2.5)
		layer.RouterTemperature = 1.2    // Higher temp for softer routing, helping underutilized experts (Suggested 1.0-1.2)
		layer.ExpertDropoutRate = 0.1    // Keep noise enabled
		layer.SetMode(true)              // Enable training mode (noise)
	}
	log.Println("🔧 Adjusted MoE: Capacity=1.2, LB=2.5, Temp=1.2, Dropout=0.1")

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
	// log.Println("Enhancing training data with semantic role labeling and abstract semantic graphs...")
	// enhancedData, err := EnhanceTrainingDataWithSRLAndASG(semanticTrainingData)
	// if err != nil {
	// 	log.Fatalf("Failed to enhance training data: %v", err)
	// }
	// log.Printf("Enhanced %d training examples with SRL and ASG annotations.\n", len(enhancedData))

	// Save enhanced training data for analysis
	// enhancedDataFile, err := os.Create("trainingdata/enhanced_training_data.json")
	// if err != nil {
	// 	log.Printf("Warning: Could not save enhanced training data: %v\n", err)
	// } else {
	// 	defer enhancedDataFile.Close()
	// 	encoder := json.NewEncoder(enhancedDataFile)
	// 	if err := encoder.Encode(enhancedData[:min(len(enhancedData), 10)]); err != nil {
	// 		log.Printf("Warning: Could not write enhanced training data: %v\n", err)
	// 	}
	// }

	// Force GC to free up memory from loading large JSON files before training
	runtime.GC()

	// Train the model
	checkpointInterval := 500 // Save every 500 batches
	err = TrainIntentMoEModel(intentMoEModel, tokenizedData, epochs, learningRate, batchSize, maxSequenceLength, semanticOutputVocabulary, modelSavePath, checkpointInterval)
	if err != nil {
		log.Fatalf("Failed to train IntentMoE model: %v", err)
	}

	// Detach the model from the computation graph to allow for clean serialization
	log.Println("Detaching model from computation graph...")
	DetachModel(intentMoEModel)

	// Save the trained model
	fmt.Printf("Saving IntentMoE model to %s\n", modelSavePath)
	err = moe.SaveIntentMoEModelToGOB(intentMoEModel, modelSavePath)
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

// saveCheckpoint saves the model state to a file.
func saveCheckpoint(model *moe.IntentMoE, basePath string, epoch, batch int) error {
	filename := fmt.Sprintf("%s.epoch%d.batch%d", basePath, epoch+1, batch)
	log.Printf("Saving checkpoint to %s...", filename)
	return moe.SaveIntentMoEModelToGOB(model, filename)
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

	// Clear encoder state
	if enc, ok := model.Encoder.(*moe.MoEEncoder); ok {
		enc.ClearState()
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
