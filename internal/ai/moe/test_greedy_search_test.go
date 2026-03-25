package moe

import (
	"log"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/tagger/tag"
)

func TestGreedySearchDecode(t *testing.T) {
	// Load the trained MoEClassificationModel model
	model, err := LoadIntentMoEModelFromGOB("../../../data/models/gob_models/moe_classification_model.gob")
	if err != nil {
		t.Skipf("Skipping: Model load failed (%v). Train the model first to match the new MoE structure.", err)
	}

	// Create a dummy context vector
	contextVector := tensor.NewTensor([]int{1, 32, 128}, make([]float64, 32*128), false)

	// Call GreedySearchDecode
	predictedIDs, err := model.GreedySearchDecode(contextVector, 32, 0, 1, 1.0, 0.0, 100, tag.Tag{})
	if err != nil {
		t.Fatalf("Greedy search decode failed: %v", err)
	}

	log.Printf("Predicted token IDs: %v", predictedIDs)
}
