package main

import (
	"fmt"
	"log"

	"github.com/golangast/gollemer/internal/ai/moe"
)

func main() {
	modelPath := "data/models/gob_models/moe_social_model.gob"
	ckpt, err := moe.LoadIntentMoECheckpoint(modelPath)
	if err != nil {
		log.Fatalf("Failed to load: %v", err)
	}

	for i, l := range ckpt.Model.Encoder.GetMoELayers() {
		fmt.Printf("Encoder Layer %d has %d experts\n", i, len(l.Experts))
	}
	if ckpt.Model.Decoder != nil && ckpt.Model.Decoder.OutputMoE != nil {
		fmt.Printf("Decoder OutputMoE has %d experts\n", len(ckpt.Model.Decoder.OutputMoE.Experts))
	}
}
