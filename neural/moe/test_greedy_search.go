package moe

import (
	"github.com/zendrulat/nlptagger/neural/tensor"
	"github.com/zendrulat/nlptagger/tagger/tag"
	"log"
	"testing"
)

func TestGreedySearchDecode(t *testing.T) {
	// Load the trained MoEClassificationModel model
	model, err := LoadIntentMoEModelFromGOB("../../gob_models/moe_classification_model.gob")
	if err != nil {
		t.Fatalf("Failed to load MoE model: %v", err)
	}

	// Create a dummy context vector
	contextVector := tensor.NewTensor([]int{1, 32, 128}, make([]float64, 32*128), false)

	// Call GreedySearchDecode
	predictedIDs, err := model.GreedySearchDecode(contextVector, 32, 0, 1, 1.0, 100, tag.Tag{})
	if err != nil {
		t.Fatalf("Greedy search decode failed: %v", err)
	}

	log.Printf("Predicted token IDs: %v", predictedIDs)
}
