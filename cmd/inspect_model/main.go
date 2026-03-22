package main

import (
	"fmt"
	"log"
	"math"

	"github.com/golangast/gollemer/internal/moe"
	"github.com/golangast/gollemer/neural/tensor"
)

func main() {
	modelPath := "gob_models/moe_classification_model.gob"
	model, err := moe.LoadIntentMoEModelFromGOB(modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	fmt.Printf("🏗️ Inspecting Model Environment: %s\n", modelPath)

	// Check Embedding Layer
	emb := model.Embedding
	if emb == nil {
		fmt.Println("❌ Embedding Layer: Missing")
	} else {
		norm := math.Sqrt(tensor.DotProduct(emb.Weight.Data, emb.Weight.Data))
		fmt.Printf("✅ Embedding Layer: VocabSize=%d, Dim=%d, Full L2 Norm=%.6f\n", emb.VocabSize, emb.DimModel, norm)
		
		// Check first few token embeddings
		for i := 0; i < 5; i++ {
			row := emb.Weight.Data[i*emb.DimModel : (i+1)*emb.DimModel]
			rowNorm := math.Sqrt(tensor.DotProduct(row, row))
			fmt.Printf("   Token %d Norm: %.6f\n", i, rowNorm)
		}
	}

	// Check Encoder
	fmt.Printf("✅ Encoder structure: %T\n", model.Encoder)
	if moeEnc, ok := model.Encoder.(*moe.MoEEncoder); ok {
		params := moeEnc.Layer.Parameters()
		paramNorm := 0.0
		for _, p := range params {
			paramNorm += tensor.DotProduct(p.Data, p.Data)
		}
		fmt.Printf("✅ MoE Layer: NumExperts=%d, Parameter Norm=%.6f\n", len(moeEnc.Layer.Experts), math.Sqrt(paramNorm))
	}

	fmt.Println("\n🚀 Model Inspection Complete!")
}
