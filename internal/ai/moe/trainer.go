package moe

import (
	"log"
	"time"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

type TrainingStats struct {
	Epoch          int
	CurrentLoss    float32
	Perplexity     float32
	BestPerplexity float32
	Layer0Counts   []int   // SIMD-accumulated expert hits for L0
	MaxDominance   float32 // The utilization % of the most used expert
	StepConfidence float32 // Average probability of the Top-1 token
}

type Trainer struct {
	BestModelPath        string
	CollapseCount        int
	LastSafeLR           float32
	ObservabilityEnabled bool
}

func (t *Trainer) AutoHeal(m *IntentMoE, opt *nn.Adam, stats TrainingStats) {
	usage := stats.MaxDominance

	// Level 1: Early Warning - Apply Router Reset
	if usage > 0.85 {
		log.Printf("💊 Level 1 Heal: Expert Dominance at %.2f%%. Resetting Routers...\n", usage*100)
		for _, layer := range ActiveLayers {
			layer.ResetRouterWeights()
		}
		t.CollapseCount++
	} else {
		if t.CollapseCount > 0 {
			t.CollapseCount--
		}
	}

	// Level 2: Poisoned Gradients - Revert to Golden Checkpoint
	if t.CollapseCount >= 3 || (stats.BestPerplexity > 0 && stats.Perplexity > stats.BestPerplexity*2.0) {
		log.Println("🚨 Level 2 Heal: Perplexity Spike / Persistent Collapse. Rolling back...")
		if t.BestModelPath != "" {
			ckpt, err := LoadIntentMoECheckpoint(t.BestModelPath)
			if err == nil {
				*m = *ckpt.Model
				log.Printf("✨ Successfully rolled back to %s", t.BestModelPath)
			} else {
				log.Printf("❌ Failed to roll back: %v", err)
			}
		}

		// Level 3: Learning Rate Surgery
		opt.SetLearningRate(opt.GetLearningRate() * 0.7)
		opt.SetRouterLR(opt.GetRouterLR() * 0.5)
		log.Printf("📉 New LR: %e | RouterLR: %e", opt.GetLearningRate(), opt.GetRouterLR())
		t.CollapseCount = 0
	}
}

func (t *Trainer) SaveGoldenCheckpoint(m *IntentMoE, stats TrainingStats, currentStep int, profile nn.TrainingProfile, tokens int64, duration time.Duration) {
	if stats.MaxDominance > 0.75 {
		log.Printf("🚫 Save Aborted: Expert Dominance too high (%.2f%%)\n", stats.MaxDominance*100)
		return
	}
	if stats.StepConfidence < 0.15 && stats.Epoch > 5 {
		log.Printf("🚫 Save Aborted: Step Confidence too low (%.2f%%)\n", stats.StepConfidence*100)
		return
	}
	if stats.Perplexity < stats.BestPerplexity || stats.BestPerplexity == 0 {
		log.Println("💾 Saving New Golden Checkpoint: Perplexity Improved!")
		checkpointPath := "data/models/gob_models/golden_checkpoint.gob"

		ckpt := &Checkpoint{
			Model:           m,
			StepCount:       currentStep,
			LastProfile:     profile,
			Commitment:      m.CalculateCommitment(),
			TokensProcessed: tokens,
			TotalDuration:   duration,
			Version:         "gollemer-v1.2-simd",
		}

		err := SaveIntentMoECheckpoint(ckpt, checkpointPath)
		if err != nil {
			log.Printf("Error saving checkpoint: %v\n", err)
			return
		}
		t.BestModelPath = checkpointPath
	}
}

// InitializeObservability is a no-op stub (observability removed).
func (t *Trainer) InitializeObservability(numExperts int, windowSize int, vocab *vocab.Vocabulary) {
	_ = numExperts
	_ = windowSize
	_ = vocab
	log.Println("✅ MoE Trainer initialized")
}

// RecordTrainingStep is a no-op stub.
func (t *Trainer) RecordTrainingStep(expertIDs, tokenIDs []int, loss float32) {}

// UpdateWeightVelocity is a no-op stub.
func (t *Trainer) UpdateWeightVelocity(layerName string, currentWeights []float32, embeddingTensor *tensor.Tensor) {
}

// RecordWeightSnapshot is a no-op stub.
func (t *Trainer) RecordWeightSnapshot(layerName string, weights []float32) {}

// FinishEpoch is a no-op stub.
func (t *Trainer) FinishEpoch(vocab *vocab.Vocabulary) {}

// ExportObservabilityMetrics is a no-op stub.
func (t *Trainer) ExportObservabilityMetrics(vocab *vocab.Vocabulary) (string, error) {
	return "{}", nil
}

// RecordTokenTrajectory is a no-op stub.
func (t *Trainer) RecordTokenTrajectory(tokenID int, tokenStr string, expertPath []int, confidences []float32) {
}

// UpdateEmbeddingGalaxy is a no-op stub.
func (t *Trainer) UpdateEmbeddingGalaxy(vocab *vocab.Vocabulary, embeddingTensor *tensor.Tensor, topN int) {
}

// RecordExpertForSimilarity is a no-op stub.
func (t *Trainer) RecordExpertForSimilarity(expertID int, weights []float32) {}

// ComputeExpertRedundancy is a no-op stub.
func (t *Trainer) ComputeExpertRedundancy(numExperts int) {}
