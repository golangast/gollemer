// Package training implements Fill-In-The-Middle (FIM) training for Gollemer's
// MoE model. FIM training teaches the model to insert code at specific points
// given surrounding context, which is critical for surgical code editing.
//
// The training format is:
//
//	Input:  <PRE>code_before<SUF>code_after<MID>
//	Output: code_to_insert
//
// This enables the model to generate localized patches that don't break
// surrounding functions or types.
package training

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"
)

// FIMConfig holds hyperparameters for FIM training.
type FIMConfig struct {
	LearningRate  float64 `json:"learning_rate"`
	BatchSize     int     `json:"batch_size"`
	Epochs        int     `json:"epochs"`
	MaxSeqLen     int     `json:"max_seq_len"`
	WarmupSteps   int     `json:"warmup_steps"`
	ValInterval   int     `json:"val_interval"`  // Validate every N steps
	SaveInterval  int     `json:"save_interval"` // Save checkpoint every N steps
	OutputDir     string  `json:"output_dir"`
	FIMPentalty   float64 `json:"fim_penalty"`    // Penalty for incorrect FIM predictions
	ConceptWeight float64 `json:"concept_weight"` // Weight for concept-guided examples
}

// DefaultFIMConfig returns sensible defaults for FIM training.
func DefaultFIMConfig() FIMConfig {
	return FIMConfig{
		LearningRate:  1e-4,
		BatchSize:     8,
		Epochs:        10,
		MaxSeqLen:     1024,
		WarmupSteps:   100,
		ValInterval:   50,
		SaveInterval:  200,
		OutputDir:     "models/fim_checkpoints",
		FIMPentalty:   0.1,
		ConceptWeight: 0.3,
	}
}

// FIMExample is a single FIM training example.
type FIMExample struct {
	Prefix      string   `json:"prefix"`
	Suffix      string   `json:"suffix"`
	Middle      string   `json:"middle"`
	Instruction string   `json:"instruction,omitempty"`
	Concepts    []string `json:"concepts,omitempty"`
}

// FIMDataset holds all FIM training data.
type FIMDataset struct {
	Examples []FIMExample    `json:"examples"`
	Vocab    map[string]int  `json:"vocab"`
	Stats    FIMDatasetStats `json:"stats"`
}

// FIMDatasetStats tracks dataset statistics.
type FIMDatasetStats struct {
	TotalExamples  int     `json:"total_examples"`
	AvgPrefixLen   float64 `json:"avg_prefix_len"`
	AvgSuffixLen   float64 `json:"avg_suffix_len"`
	AvgMiddleLen   float64 `json:"avg_middle_len"`
	VocabSize      int     `json:"vocab_size"`
	ConceptCovered int     `json:"concept_covered"`
}

// FIMTrainer orchestrates FIM training for the MoE model.
type FIMTrainer struct {
	config FIMConfig
	model  FIMTrainableModel
	vocab  map[string]int
	mu     sync.Mutex
}

// FIMTrainableModel is the interface the MoE model must implement for FIM training.
type FIMTrainableModel interface {
	// ForwardFIM runs a forward pass with FIM-formatted input.
	// Returns token logits and loss.
	ForwardFIM(prefix, suffix, middle []int) (loss float64, logits [][]float64, err error)

	// Backward updates model weights from the computed loss.
	Backward(loss float64) error

	// SaveCheckpoint saves the current model state.
	SaveCheckpoint(path string) error

	// LoadCheckpoint loads a saved model state.
	LoadCheckpoint(path string) error

	// GetLearningRate returns the current learning rate.
	GetLearningRate() float64

	// SetLearningRate updates the learning rate.
	SetLearningRate(lr float64)
}

// NewFIMTrainer creates a new FIM trainer.
func NewFIMTrainer(config FIMConfig, model FIMTrainableModel) *FIMTrainer {
	return &FIMTrainer{
		config: config,
		model:  model,
		vocab:  make(map[string]int),
	}
}

// TrainFromFile loads a dataset and runs FIM training.
func (ft *FIMTrainer) TrainFromFile(datasetPath string) error {
	log.Printf("Loading FIM dataset from %s...", datasetPath)

	data, err := os.ReadFile(datasetPath)
	if err != nil {
		return fmt.Errorf("read dataset: %w", err)
	}

	var dataset struct {
		Train []interface{} `json:"train"`
		Val   []interface{} `json:"val"`
		Meta  struct {
			FIMExamples int `json:"fim_examples"`
		} `json:"meta"`
	}
	if err := json.Unmarshal(data, &dataset); err != nil {
		return fmt.Errorf("unmarshal dataset: %w", err)
	}

	// Extract FIM examples from the training data
	var fimExamples []FIMExample
	for _, item := range dataset.Train {
		if ex := extractFIMExample(item); ex != nil {
			fimExamples = append(fimExamples, *ex)
		}
	}

	log.Printf("Extracted %d FIM examples from training data", len(fimExamples))

	// Build vocabulary
	ft.buildVocab(fimExamples)

	// Run training
	return ft.train(fimExamples)
}

// TrainFromExamples runs FIM training directly from provided examples.
func (ft *FIMTrainer) TrainFromExamples(examples []FIMExample) error {
	ft.buildVocab(examples)
	return ft.train(examples)
}

// train runs the main FIM training loop.
func (ft *FIMTrainer) train(examples []FIMExample) error {
	if len(examples) == 0 {
		return fmt.Errorf("no training examples provided")
	}

	// Create output directory
	if err := os.MkdirAll(ft.config.OutputDir, 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Split into train/val
	rand.Shuffle(len(examples), func(i, j int) {
		examples[i], examples[j] = examples[j], examples[i]
	})
	nVal := len(examples) / 10
	if nVal < 1 {
		nVal = 1
	}
	trainSet := examples[nVal:]
	valSet := examples[:nVal]

	log.Printf("Training: %d examples, Validation: %d examples", len(trainSet), len(valSet))
	log.Printf("Vocab size: %d", len(ft.vocab))
	log.Printf("Config: LR=%.6f, Batch=%d, Epochs=%d, MaxSeqLen=%d",
		ft.config.LearningRate, ft.config.BatchSize, ft.config.Epochs, ft.config.MaxSeqLen)

	// Training loop
	step := 0
	bestValLoss := math.MaxFloat64

	for epoch := 1; epoch <= ft.config.Epochs; epoch++ {
		rand.Shuffle(len(trainSet), func(i, j int) {
			trainSet[i], trainSet[j] = trainSet[j], trainSet[i]
		})

		epochLoss := 0.0
		batchCount := 0

		for i := 0; i < len(trainSet); i += ft.config.BatchSize {
			end := i + ft.config.BatchSize
			if end > len(trainSet) {
				end = len(trainSet)
			}
			batch := trainSet[i:end]

			// Forward pass for each example in batch
			batchLoss := 0.0
			for _, ex := range batch {
				loss, err := ft.forwardExample(ex)
				if err != nil {
					log.Printf("Warning: forward pass failed: %v", err)
					continue
				}
				batchLoss += loss
			}

			avgBatchLoss := batchLoss / float64(len(batch))

			// Apply learning rate schedule (cosine decay with warmup)
			lr := ft.lrSchedule(step)
			ft.model.SetLearningRate(lr)

			// Backward pass
			if err := ft.model.Backward(avgBatchLoss); err != nil {
				log.Printf("Warning: backward pass failed: %v", err)
				continue
			}

			epochLoss += avgBatchLoss
			batchCount++
			step++

			// Explicit memory cleanup every 50 steps to prevent OOM
			if step%50 == 0 {
				runtime.GC()
			}

			// Validation
			if step%ft.config.ValInterval == 0 {
				valLoss := ft.evaluate(valSet)
				log.Printf("Step %d: train_loss=%.6f val_loss=%.6f lr=%.8f",
					step, avgBatchLoss, valLoss, lr)

				if valLoss < bestValLoss {
					bestValLoss = valLoss
					ft.saveCheckpoint("best")
				}
			}

			// Save checkpoint
			if step%ft.config.SaveInterval == 0 {
				ft.saveCheckpoint(fmt.Sprintf("step_%d", step))
			}
		}

		avgEpochLoss := epochLoss / float64(batchCount)
		log.Printf("Epoch %d/%d complete: avg_loss=%.6f", epoch, ft.config.Epochs, avgEpochLoss)
	}

	log.Printf("Training complete! Best validation loss: %.6f", bestValLoss)
	return nil
}

// forwardExample runs a single FIM example through the model.
func (ft *FIMTrainer) forwardExample(ex FIMExample) (float64, error) {
	// Tokenize prefix, suffix, middle
	prefixTokens := ft.tokenize(ex.Prefix)
	suffixTokens := ft.tokenize(ex.Suffix)
	middleTokens := ft.tokenize(ex.Middle)

	// Truncate to max sequence length
	maxContext := ft.config.MaxSeqLen / 2
	if len(prefixTokens) > maxContext {
		prefixTokens = prefixTokens[len(prefixTokens)-maxContext:]
	}
	if len(suffixTokens) > maxContext {
		suffixTokens = suffixTokens[:maxContext]
	}
	if len(middleTokens) > ft.config.MaxSeqLen/4 {
		middleTokens = middleTokens[:ft.config.MaxSeqLen/4]
	}

	// Apply concept weighting if concepts are present
	loss, _, err := ft.model.ForwardFIM(prefixTokens, suffixTokens, middleTokens)
	if err != nil {
		return 0, fmt.Errorf("forward FIM: %w", err)
	}

	// Apply FIM penalty for longer middle sections (harder to predict)
	if len(middleTokens) > 10 {
		penalty := ft.config.FIMPentalty * float64(len(middleTokens)) / float64(ft.config.MaxSeqLen)
		loss += penalty
	}

	// Apply concept weight bonus if example has concepts
	if len(ex.Concepts) > 0 {
		loss *= (1.0 - ft.config.ConceptWeight)
	}

	return loss, nil
}

// evaluate runs validation on a set of examples.
func (ft *FIMTrainer) evaluate(examples []FIMExample) float64 {
	if len(examples) == 0 {
		return 0
	}

	totalLoss := 0.0
	for _, ex := range examples {
		loss, err := ft.forwardExample(ex)
		if err != nil {
			continue
		}
		totalLoss += loss
	}
	return totalLoss / float64(len(examples))
}

// lrSchedule computes the learning rate with cosine decay and warmup.
// Uses a more conservative decay that never drops below 5% of the initial LR,
// and extends totalSteps to prevent premature decay to zero.
func (ft *FIMTrainer) lrSchedule(step int) float64 {
	// Use a generous estimate: ~100 batches per epoch on average, times epochs.
	// This prevents LR from decaying to zero before training completes.
	estBatchesPerEpoch := 100
	totalSteps := ft.config.Epochs * estBatchesPerEpoch
	// Ensure a minimum positive total steps
	if totalSteps < ft.config.WarmupSteps+1 {
		totalSteps = ft.config.WarmupSteps + 100
	}

	// Minimum LR: never drop below 5% of initial learning rate
	minLR := ft.config.LearningRate * 0.05

	if step < ft.config.WarmupSteps {
		// Linear warmup
		return ft.config.LearningRate * float64(step) / float64(ft.config.WarmupSteps)
	}

	// Cosine decay, but floor at minLR so it never hits zero
	progress := float64(step-ft.config.WarmupSteps) / float64(totalSteps-ft.config.WarmupSteps)
	if progress > 1.0 {
		progress = 1.0
	}
	cosine := ft.config.LearningRate * 0.5 * (1.0 + math.Cos(math.Pi*progress))
	if cosine < minLR {
		return minLR
	}
	return cosine
}

// buildVocab constructs a token vocabulary from the training examples.
func (ft *FIMTrainer) buildVocab(examples []FIMExample) {
	ft.mu.Lock()
	defer ft.mu.Unlock()

	ft.vocab = make(map[string]int)
	ft.vocab["<PAD>"] = 0
	ft.vocab["<UNK>"] = 1
	ft.vocab["<PRE>"] = 2
	ft.vocab["<SUF>"] = 3
	ft.vocab["<MID>"] = 4
	ft.vocab["<EOS>"] = 5
	nextID := 6

	// Count word frequencies
	freq := make(map[string]int)
	for _, ex := range examples {
		for _, word := range tokenizeText(ex.Prefix) {
			freq[word]++
		}
		for _, word := range tokenizeText(ex.Suffix) {
			freq[word]++
		}
		for _, word := range tokenizeText(ex.Middle) {
			freq[word]++
		}
	}

	// Add words that appear more than once
	for word, count := range freq {
		if count >= 2 {
			ft.vocab[word] = nextID
			nextID++
		}
	}

	log.Printf("Built vocabulary: %d tokens", len(ft.vocab))
}

// tokenize converts text to token IDs.
func (ft *FIMTrainer) tokenize(text string) []int {
	ft.mu.Lock()
	defer ft.mu.Unlock()

	words := tokenizeText(text)
	tokens := make([]int, 0, len(words))
	for _, w := range words {
		if id, ok := ft.vocab[w]; ok {
			tokens = append(tokens, id)
		} else {
			tokens = append(tokens, ft.vocab["<UNK>"])
		}
	}
	return tokens
}

// tokenizeText splits text into tokens (words and punctuation).
func tokenizeText(text string) []string {
	// Simple whitespace + punctuation tokenization
	var tokens []string
	current := strings.Builder{}

	for _, r := range text {
		if r == ' ' || r == '\n' || r == '\t' {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			if r == '\n' {
				tokens = append(tokens, "<NEWLINE>")
			}
		} else if isPunctuation(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
		} else {
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
}

// isPunctuation checks if a rune is code-relevant punctuation.
func isPunctuation(r rune) bool {
	return strings.ContainsRune("(){}[];,.:=+-*/<>!&|^~", r)
}

// saveCheckpoint saves the current model state.
func (ft *FIMTrainer) saveCheckpoint(name string) {
	path := fmt.Sprintf("%s/%s_fim.bin", ft.config.OutputDir, name)
	if err := ft.model.SaveCheckpoint(path); err != nil {
		log.Printf("Warning: failed to save checkpoint %s: %v", name, err)
		return
	}
	log.Printf("Checkpoint saved: %s", path)
}

// extractFIMExample attempts to extract a FIM example from a generic interface{}.
func extractFIMExample(item interface{}) *FIMExample {
	switch v := item.(type) {
	case map[string]interface{}:
		// Check if it has FIM fields
		if prompt, ok := v["prompt"].(string); ok {
			if completion, ok := v["completion"].(string); ok {
				// Parse <PRE>prefix<SUF>suffix<MID> format
				prefix, suffix, middle := parseFIMPrompt(prompt, completion)
				if middle != "" {
					ex := &FIMExample{
						Prefix: prefix,
						Suffix: suffix,
						Middle: middle,
					}
					if inst, ok := v["instruction"].(string); ok {
						ex.Instruction = inst
					}
					return ex
				}
			}
		}
		// Check if it has before_code/target_patch (SEARCH/REPLACE format)
		if before, ok := v["before_code"].(string); ok {
			if patch, ok := v["target_patch"].(string); ok {
				after := extractAfterFromPatch(patch)
				if after != "" {
					prefix, middle, suffix := splitIntoFIM(before, after)
					if middle != "" {
						ex := &FIMExample{
							Prefix: prefix,
							Suffix: suffix,
							Middle: middle,
						}
						if inst, ok := v["instruction"].(string); ok {
							ex.Instruction = inst
						}
						if concepts, ok := v["concepts"].([]interface{}); ok {
							for _, c := range concepts {
								if cs, ok := c.(string); ok {
									ex.Concepts = append(ex.Concepts, cs)
								}
							}
						}
						return ex
					}
				}
			}
		}
	}
	return nil
}

// parseFIMPrompt extracts prefix, suffix, middle from FIM-formatted prompt.
func parseFIMPrompt(prompt, completion string) (string, string, string) {
	// Format: <PRE>prefix<SUF>suffix<MID>
	preIdx := strings.Index(prompt, "<PRE>")
	sufIdx := strings.Index(prompt, "<SUF>")
	midIdx := strings.Index(prompt, "<MID>")

	if preIdx == -1 || sufIdx == -1 || midIdx == -1 {
		return "", "", ""
	}

	prefix := prompt[preIdx+5 : sufIdx]
	suffix := prompt[sufIdx+5 : midIdx]
	middle := completion

	return prefix, suffix, middle
}

// splitIntoFIM splits before/after code into prefix/middle/suffix.
func splitIntoFIM(before, after string) (string, string, string) {
	beforeLines := strings.Split(before, "\n")
	afterLines := strings.Split(after, "\n")

	// Find first diff
	firstDiff := -1
	minLen := len(beforeLines)
	if len(afterLines) < minLen {
		minLen = len(afterLines)
	}
	for i := 0; i < minLen; i++ {
		if beforeLines[i] != afterLines[i] {
			firstDiff = i
			break
		}
	}
	if firstDiff == -1 {
		if len(beforeLines) != len(afterLines) {
			firstDiff = minLen
		} else {
			return "", "", ""
		}
	}

	// Find last diff
	lastDiff := len(afterLines)
	for i := 0; i < minLen; i++ {
		bi := len(beforeLines) - 1 - i
		ai := len(afterLines) - 1 - i
		if bi < 0 || ai < 0 || beforeLines[bi] != afterLines[ai] {
			lastDiff = len(afterLines) - i
			break
		}
	}

	prefix := strings.Join(afterLines[:firstDiff], "\n")
	middle := strings.Join(afterLines[firstDiff:lastDiff], "\n")
	suffix := strings.Join(afterLines[lastDiff:], "\n")

	return prefix, middle, suffix
}

// GenerateFIMInference creates a FIM prompt for inference-time code insertion.
func GenerateFIMInference(prefix, suffix string) string {
	return fmt.Sprintf("<PRE>%s<SUF>%s<MID>", prefix, suffix)
}

// ParseFIMOutput extracts the generated middle code from model output.
func ParseFIMOutput(output string) string {
	// The model generates the middle code directly
	return strings.TrimSpace(output)
}
