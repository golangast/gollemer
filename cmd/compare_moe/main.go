package main

import (
	"flag"
	"fmt"
	"log"
	"math"

	"github.com/golangast/gollemer/neural/moe"
)

func main() {
	pathA := flag.String("a", "", "Path to first model")
	pathB := flag.String("b", "", "Path to second model")
	flag.Parse()

	if *pathA == "" || *pathB == "" {
		fmt.Println("Usage: go run cmd/compare_moe/main.go -a model_old.gob -b model_new.gob")
		return
	}

	modelA, err := moe.LoadIntentMoEModelFromGOB(*pathA)
	if err != nil {
		log.Fatalf("Failed to load model A: %v", err)
	}
	modelB, err := moe.LoadIntentMoEModelFromGOB(*pathB)
	if err != nil {
		log.Fatalf("Failed to load model B: %v", err)
	}

	fmt.Printf("📊 Comparing: %s vs %s\n", *pathA, *pathB)
	fmt.Println("---------------------------------------------------------")

	// Helper to find MoE layers
	findMoE := func(m *moe.IntentMoE) []*moe.MoELayer {
		var layers []*moe.MoELayer
		if ml, ok := m.Encoder.(*moe.MoELayer); ok {
			layers = append(layers, ml)
		} else if hybrid, ok := m.Encoder.(*moe.HybridLLMGNNEncoder); ok {
			if ml, ok := hybrid.LLMEncoder.(*moe.MoELayer); ok {
				layers = append(layers, ml)
			}
		}
		if m.Decoder.OutputMoE != nil {
			layers = append(layers, m.Decoder.OutputMoE)
		}
		return layers
	}

	layersA := findMoE(modelA)
	layersB := findMoE(modelB)

	for i := range layersA {
		if i >= len(layersB) {
			break
		}
		fmt.Printf("Layer %d:\n", i)
		compareLayers(layersA[i], layersB[i])
	}
}

func compareLayers(mA, mB *moe.MoELayer) {
	for i := 0; i < len(mA.Experts); i++ {
		normA := calculateNorm(mA.Experts[i])
		normB := calculateNorm(mB.Experts[i])
		diff := normB - normA
		
		status := "✅ Stable"
		if math.Abs(diff) > 0.5 {
			status = "⚠️  DRIFTING"
		}
		if math.IsNaN(diff) {
			status = "❌ CRITICAL (NaN)"
		}

		fmt.Printf("  Expert %d: NormA: %.4f | NormB: %.4f | Change: %+.4f [%s]\n", 
			i, normA, normB, diff, status)
	}

	// Router
	rNormA := calculateRouterNorm(mA)
	rNormB := calculateRouterNorm(mB)
	fmt.Printf("  Router  : NormA: %.4f | NormB: %.4f | Change: %+.4f\n\n", 
		rNormA, rNormB, rNormB-rNormA)
}

func calculateNorm(expert moe.Expert) float64 {
	var sum float64
	for _, p := range expert.Parameters() {
		for _, v := range p.Data {
			sum += v * v
		}
	}
	return math.Sqrt(sum)
}

func calculateRouterNorm(m *moe.MoELayer) float64 {
	var sum float64
	for _, p := range m.GatingNetwork.Linear.Parameters() {
		for _, v := range p.Data {
			sum += v * v
		}
	}
	return math.Sqrt(sum)
}
