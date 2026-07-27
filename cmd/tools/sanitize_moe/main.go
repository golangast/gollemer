package main

import (
	"fmt"
	"log"
	"math"
	"path/filepath"

	"github.com/golangast/gollemer/internal/ai/llm"
	"github.com/golangast/gollemer/internal/ai/moe"
)

func main() {
	projectRoot, err := llm.FindProjectRoot()
	if err != nil {
		log.Fatalf("Failed to find project root: %v", err)
	}

	modelPath := filepath.Join(projectRoot, "data/models/gob_models/moe_classification_model.gob")
	model, err := moe.LoadIntentMoEModelWithFallback(modelPath)
	if err != nil {
		log.Fatalf("Failed to load MoE model: %v", err)
	}

	fmt.Printf("📊 Analyzing MoE Model: %s\n", modelPath)
	fmt.Println("-------------------------------------------")

	// Find MoE layers
	var layers []*moe.MoELayer
	if ml, ok := model.Encoder.(*moe.MoELayer); ok {
		layers = append(layers, ml)
	} else if hybrid, ok := model.Encoder.(*moe.HybridLLMGNNEncoder); ok {
		if ml, ok := hybrid.LLMEncoder.(*moe.MoELayer); ok {
			layers = append(layers, ml)
		}
	}
	if model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	for i, layer := range layers {
		fmt.Printf("Layer %d:\n", i)
		analyzeLayer(layer)
	}
}

func analyzeLayer(m *moe.MoELayer) {
	for i, expert := range m.Experts {
		params := expert.Parameters()
		var totalWeight float32
		var count int
		hasNaN := false

		for _, p := range params {
			for _, v := range p.Data {
				if math.IsNaN(float64(v)) {
					hasNaN = true
				}
				totalWeight += v * v
				count++
			}
		}

		norm := math.Sqrt(float64(totalWeight))
		status := "✅ OK"
		if hasNaN {
			status = "❌ NaN DETECTED"
		}

		fmt.Printf("  Expert %d: Norm=%.4f, Params=%d [%s]\n", i, norm, count, status)
	}

	// Analyze Router
	rParams := m.GatingNetwork.Linear.Parameters()
	var rWeight float32
	for _, p := range rParams {
		for _, v := range p.Data {
			rWeight += v * v
		}
	}
	fmt.Printf("  Router Norm: %.4f\n", math.Sqrt(float64(rWeight)))
	fmt.Println()
}
