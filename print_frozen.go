package main

import (
	"fmt"
	"path/filepath"

	"github.com/golangast/gollemer/internal/ai/moe"
)

func Printfrozen() {
	intentModel, _ := moe.LoadIntentMoEModelWithFallback(filepath.Join("data", "models", "gob_models", "moe_social_model.gob"))
	if intentModel == nil {
		fmt.Println("failed to load model")
		return
	}
	layers := intentModel.Encoder.GetMoELayers()
	for i, l := range layers {
		fmt.Printf("Layer %d experts: %d\n", i, len(l.Experts))
		fmt.Printf("Layer %d frozen len: %d\n", i, len(l.ExpertFrozen))
	}
	fmt.Printf("Decoder experts: %d\n", len(intentModel.Decoder.OutputMoE.Experts))
	fmt.Printf("Decoder output frozen len: %d\n", len(intentModel.Decoder.OutputMoE.ExpertFrozen))
}
