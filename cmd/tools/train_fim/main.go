// TrainFIM trains Gollemer's MoE model on mined Go patch datasets using
// Fill-In-The-Middle (FIM) format. This teaches the model to insert code
// at specific points given surrounding context.
//
// Usage:
//
//	go run ./cmd/tools/train_fim/main.go \
//	  -data="data/training/fim_dataset.json" \
//	  -epochs=10 \
//	  -batch=8
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/debug"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/training"
)

func main() {
	datasetPath := flag.String("data", "data/training/fim_dataset.json", "Path to FIM dataset JSON")
	epochs := flag.Int("epochs", 10, "Number of training epochs")
	batchSize := flag.Int("batch", 8, "Batch size")
	lr := flag.Float64("lr", 1e-4, "Learning rate")
	outputDir := flag.String("output", "models/fim_checkpoints", "Output directory for checkpoints")
	embedDim := flag.Int("embed-dim", 64, "Embedding dimension (smaller = less memory)")
	vocabSize := flag.Int("vocab-size", 1000, "Vocabulary size")
	flag.Parse()

	if err := run(*datasetPath, *epochs, *batchSize, *lr, *outputDir, *embedDim, *vocabSize); err != nil {
		log.Fatalf("Fatal: %v", err)
	}
}

func run(datasetPath string, epochs, batchSize int, lr float64, outputDir string, embedDim, vocabSize int) error {
	log.Printf("Loading FIM dataset from %s...", datasetPath)

	// Load the dataset
	data, err := os.ReadFile(datasetPath)
	if err != nil {
		return fmt.Errorf("read dataset: %w", err)
	}

	var dataset struct {
		Train []interface{} `json:"train"`
		Val   []interface{} `json:"val"`
		Meta  struct {
			FIMExamples   int `json:"fim_examples"`
			SearchReplace int `json:"search_replace"`
		} `json:"meta"`
	}
	if err := json.Unmarshal(data, &dataset); err != nil {
		return fmt.Errorf("unmarshal dataset: %w", err)
	}

	log.Printf("Dataset loaded: %d train examples, %d val examples", len(dataset.Train), len(dataset.Val))
	log.Printf("  FIM examples: %d, SEARCH/REPLACE: %d", dataset.Meta.FIMExamples, dataset.Meta.SearchReplace)

	// Extract FIM examples from the training data
	var fimExamples []training.FIMExample
	for _, item := range dataset.Train {
		if ex := extractFIMExample(item); ex != nil {
			fimExamples = append(fimExamples, *ex)
		}
	}
	log.Printf("Extracted %d FIM examples from training data", len(fimExamples))

	if len(fimExamples) == 0 {
		return fmt.Errorf("no FIM examples found in dataset")
	}

	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Create a minimal MoE model for FIM training
	log.Printf("Creating MoE model (embed_dim=%d, vocab_size=%d)...", embedDim, vocabSize)
	model := createMOEModel(embedDim, vocabSize)

	// Create FIM adapter
	adapter := training.NewFIMAdapter(model, embedDim, vocabSize)
	adapter.SetLearningRate(lr)

	// Reduce memory before training starts
	runtime.GC()
	debug.SetGCPercent(50) // More aggressive GC to prevent OOM

	// Configure FIM training with memory-safe settings
	config := training.DefaultFIMConfig()
	config.Epochs = epochs
	config.BatchSize = batchSize
	config.LearningRate = lr
	config.OutputDir = outputDir
	config.MaxSeqLen = 512    // Reduced from 1024 to prevent OOM
	config.ValInterval = 100  // Validate less frequently
	config.SaveInterval = 500 // Save less frequently

	log.Printf("Starting FIM training:")
	log.Printf("  Epochs: %d", config.Epochs)
	log.Printf("  Batch size: %d", config.BatchSize)
	log.Printf("  Learning rate: %.6f", config.LearningRate)
	log.Printf("  Output dir: %s", config.OutputDir)

	// Create trainer and run
	trainer := training.NewFIMTrainer(config, adapter)
	if err := trainer.TrainFromExamples(fimExamples); err != nil {
		return fmt.Errorf("FIM training failed: %w", err)
	}

	log.Printf("FIM training complete! Checkpoints saved to %s", outputDir)
	return nil
}

// createMOEModel creates a minimal MoE model for FIM training.
// Uses small dimensions to fit within available memory.
func createMOEModel(embedDim, vocabSize int) *moe.MoEStack {
	// Create a single MoE layer with minimal experts
	numExperts := 2 // Reduced from 4 to save memory
	k := 1          // Top-1 routing

	// Use the expert builder pattern that NewMoELayer expects
	// NewFeedForwardExpert(inputDim, hiddenDim, outputDim)
	// Use smaller hidden dim to reduce memory
	hiddenDim := embedDim // No expansion to save memory
	expertBuilder := func(idx int) (moe.Expert, error) {
		return moe.NewFeedForwardExpert(embedDim, hiddenDim, embedDim)
	}

	layer, err := moe.NewMoELayer(embedDim, embedDim, numExperts, k, expertBuilder)
	if err != nil {
		log.Fatalf("Failed to create MoE layer: %v", err)
	}
	layer.Training = true

	return moe.NewMoEStack(layer)
}

// extractFIMExample attempts to extract a FIM example from a generic interface{}.
func extractFIMExample(item interface{}) *training.FIMExample {
	switch v := item.(type) {
	case map[string]interface{}:
		// Check if it has FIM fields (prompt/completion format)
		if prompt, ok := v["prompt"].(string); ok {
			if completion, ok := v["completion"].(string); ok {
				prefix, suffix, middle := parseFIMPrompt(prompt, completion)
				if middle != "" {
					ex := &training.FIMExample{
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
						ex := &training.FIMExample{
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
		}
	}
	return nil
}

// parseFIMPrompt extracts prefix, suffix, middle from FIM-formatted prompt.
func parseFIMPrompt(prompt, completion string) (string, string, string) {
	preIdx := indexOf(prompt, "<PRE>")
	sufIdx := indexOf(prompt, "<SUF>")
	midIdx := indexOf(prompt, "<MID>")

	if preIdx == -1 || sufIdx == -1 || midIdx == -1 {
		return "", "", ""
	}

	prefix := prompt[preIdx+5 : sufIdx]
	suffix := prompt[sufIdx+5 : midIdx]
	middle := completion

	return prefix, suffix, middle
}

// extractAfterFromPatch extracts the REPLACE section from a SEARCH/REPLACE patch.
func extractAfterFromPatch(patch string) string {
	parts := splitOn(patch, "=======\n")
	if len(parts) != 2 {
		return ""
	}
	after := trimSuffix(parts[1], "\n>>>>>>> REPLACE")
	after = trimSuffix(after, ">>>>>>> REPLACE")
	return after
}

// splitIntoFIM splits before/after code into prefix/middle/suffix.
// Finds the changed region by comparing before and after line-by-line.
// Returns empty strings if no diff found.
func splitIntoFIM(before, after string) (string, string, string) {
	beforeLines := splitLines(before)
	afterLines := splitLines(after)

	if len(beforeLines) == 0 && len(afterLines) == 0 {
		return "", "", ""
	}

	// Find the first differing line from the start
	firstDiff := -1
	maxCommon := len(beforeLines)
	if len(afterLines) < maxCommon {
		maxCommon = len(afterLines)
	}
	for i := 0; i < maxCommon; i++ {
		if beforeLines[i] != afterLines[i] {
			firstDiff = i
			break
		}
	}
	if firstDiff == -1 {
		// All common lines match - diff is purely in extra lines
		if len(beforeLines) < len(afterLines) {
			// Lines were added at position maxCommon
			prefix := joinLines(afterLines[:maxCommon])
			middle := joinLines(afterLines[maxCommon:])
			return prefix, middle, ""
		} else if len(beforeLines) > len(afterLines) {
			// Lines were removed, the middle is empty (entire change is a deletion)
			prefix := joinLines(afterLines)
			return prefix, "", ""
		}
		return "", "", "" // Identical
	}

	// Find the last differing line from the end
	lastDiff := len(afterLines)
	maxCommonEnd := len(beforeLines)
	if len(afterLines) < maxCommonEnd {
		maxCommonEnd = len(afterLines)
	}
	for i := 1; i <= maxCommonEnd; i++ {
		bi := len(beforeLines) - i
		ai := len(afterLines) - i
		if bi < firstDiff || ai < firstDiff || beforeLines[bi] != afterLines[ai] {
			lastDiff = len(afterLines) - i + 1
			break
		}
	}
	// Ensure lastDiff is at least firstDiff
	if lastDiff < firstDiff {
		lastDiff = firstDiff
	}
	// Ensure lastDiff doesn't exceed afterLines length
	if lastDiff > len(afterLines) {
		lastDiff = len(afterLines)
	}

	prefix := joinLines(afterLines[:firstDiff])
	middle := joinLines(afterLines[firstDiff:lastDiff])
	suffix := joinLines(afterLines[lastDiff:])

	return prefix, middle, suffix
}

// Helper functions to avoid importing strings in this simple CLI
func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func splitOn(s, sep string) []string {
	idx := indexOf(s, sep)
	if idx == -1 {
		return []string{s}
	}
	return []string{s[:idx], s[idx+len(sep):]}
}

func trimSuffix(s, suffix string) string {
	if len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix {
		return s[:len(s)-len(suffix)]
	}
	return s
}

func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	lines = append(lines, s[start:])
	return lines
}

func joinLines(lines []string) string {
	if len(lines) == 0 {
		return ""
	}
	result := lines[0]
	for _, l := range lines[1:] {
		result += "\n" + l
	}
	return result
}
