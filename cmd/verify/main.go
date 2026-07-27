package main

import (
	"log"
	"path/filepath"
	"sort"

	"github.com/golangast/gollemer/internal/ai/moe"
)

func main() {
	// 1. Find the latest checkpoint
	checkpoints, _ := filepath.Glob("data/models/checkpoints/*.bin")
	if len(checkpoints) == 0 {
		log.Fatalf("No binary checkpoints found in data/models/checkpoints/")
	}
	sort.Strings(checkpoints)
	latest := checkpoints[len(checkpoints)-1]
	log.Printf("Verifying latest model: %s", latest)

	// 2. Load the model
	inputDim := 128
	model, err := moe.LoadSparseModel(latest, inputDim, 8, 2)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// 3. Simple Test Set
	log.Println("--- 🩺 Running Diagnostic Suite ---")

	testInput := make([]float32, inputDim)
	for i := range testInput {
		testInput[i] = 1.0 // Unit signal
	}

	prediction, indices := model.Predict(testInput)
	log.Printf("Prediction Mean: %.4f | Experts: %v", sliceMean(prediction), indices)

	// 4. Trace Output (Simulated Tokenizer)
	dummyTokenizer := &DummyTokenizer{dim: inputDim}
	model.Trace("The quick brown fox", dummyTokenizer)

	log.Println("✅ Verification Complete.")
}

type DummyTokenizer struct {
	dim int
}

func (t *DummyTokenizer) Tokenize(s string) []string {
	return []string{"The", "quick", "brown", "fox"}
}

func (t *DummyTokenizer) Embed(s string) []float32 {
	v := make([]float32, t.dim)
	for i := range v {
		v[i] = float32(len(s)) / 10.0 // Deterministic signal for trace
	}
	return v
}

func sliceMean(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x
	}
	return sum / float32(len(v))
}
