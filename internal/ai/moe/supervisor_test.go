package moe

import (
	"testing"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func TestSupervisor_Interventions(t *testing.T) {
	// Initialize a small vocabulary
	vocab := mainvocab.NewVocabulary()
	vocab.AddToken("what")
	vocab.AddToken("who")
	vocab.AddToken("you")

	// Initialize model
	vocabSize := vocab.Size()
	embeddingDim := 64
	numExperts := 8
	model, err := NewHybridIntentMoE(vocabSize, embeddingDim, numExperts, 5, 5, vocabSize, 2, nil)
	if err != nil {
		t.Fatalf("Failed to create HybridIntentMoE: %v", err)
	}
	model.SentenceVocab = vocab

	// Initialize supervisor
	supervisor := NewSupervisor()

	// 1. Test SpawnSpecializedExpert
	// We want to spawn expert ID 25 on Layer 0.
	supervisor.SpawnSpecializedExpert(model, 0, "IDENTITY", 25)

	layer0 := model.Encoder.GetMoELayers()[0]
	if len(layer0.Experts) < 26 {
		t.Errorf("Expected at least 26 experts in layer 0 after spawning specialized expert, got %d", len(layer0.Experts))
	}

	roleName := layer0.ExpertRole[25]
	if roleName != "IDENTITY" {
		t.Errorf("Expected expert 25 role to be 'IDENTITY', got '%s'", roleName)
	}

	// 2. Test AdjustRoutingAffinity
	// Before adjusting, check weight alignment/value
	weightsBefore := layer0.GatingNetwork.Linear.Weights.Data[0*layer0.GatingNetwork.Linear.Weights.Shape[1]+25]

	supervisor.AdjustRoutingAffinity(model, "what", 25, 2.5)

	weightsAfter := layer0.GatingNetwork.Linear.Weights.Data[0*layer0.GatingNetwork.Linear.Weights.Shape[1]+25]
	if weightsBefore == weightsAfter {
		t.Errorf("Expected routing weights to change after AdjustRoutingAffinity, but they remained equal (value = %f)", weightsAfter)
	}

	// 3. Test ClearFailureLogs
	supervisor.FailureLogs["E16"] = 10
	supervisor.ClearFailureLogs(model)

	if supervisor.FailureLogs["E16"] != 0 {
		t.Errorf("Expected FailureLogs to be cleared, but E16 still has count %d", supervisor.FailureLogs["E16"])
	}
}
