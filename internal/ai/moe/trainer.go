package moe

import (
	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/word2vec"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"log"
	"time"
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
	Observability        *MoEObservability
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
		// Recovery: If usage is healthy, slowly bleed off the collapse counter
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
	// 1. Check for Expert Dominance (The "Dictator" Check)
	if stats.MaxDominance > 0.75 {
		log.Printf("🚫 Save Aborted: Expert Dominance too high (%.2f%%)\n", stats.MaxDominance*100)
		return
	}

	// 2. Check for Confidence (The "Word Salad" Check)
	if stats.StepConfidence < 0.15 && stats.Epoch > 5 {
		log.Printf("🚫 Save Aborted: Step Confidence too low (%.2f%%)\n", stats.StepConfidence*100)
		return
	}

	// 3. Check for Improvement (The "Progress" Check)
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

		// Update the trainer's benchmark
		t.BestModelPath = checkpointPath
	}
}

// InitializeObservability sets up advanced observability tracking.
func (t *Trainer) InitializeObservability(numExperts int, windowSize int, vocab *vocab.Vocabulary, w2vModel *word2vec.SimpleWord2Vec) {
	t.Observability = NewMoEObservability(numExperts, windowSize, vocab, w2vModel)
	t.ObservabilityEnabled = true
	log.Println("✅ MoE Observability initialized with 4 advanced metrics")
}

// RecordTrainingStep records a single training step for observability tracking.
func (t *Trainer) RecordTrainingStep(expertIDs, tokenIDs []int, loss float32) {
	if !t.ObservabilityEnabled || t.Observability == nil {
		return
	}
	t.Observability.RecordStep(expertIDs, tokenIDs, loss)
}

// UpdateWeightVelocity updates weight velocity tracking for a layer.
func (t *Trainer) UpdateWeightVelocity(layerName string, currentWeights []float32, embeddingTensor *tensor.Tensor) {
	if !t.ObservabilityEnabled || t.Observability == nil {
		return
	}
	t.Observability.FinishStep(layerName, currentWeights, embeddingTensor)
}

// RecordWeightSnapshot captures baseline weights for next delta calculation.
func (t *Trainer) RecordWeightSnapshot(layerName string, weights []float32) {
	if !t.ObservabilityEnabled || t.Observability == nil {
		return
	}
	t.Observability.RecordWeights(layerName, weights)
}

// FinishEpoch resets windowed metrics and logs observability report.
func (t *Trainer) FinishEpoch(vocab *vocab.Vocabulary) {
	if !t.ObservabilityEnabled || t.Observability == nil {
		return
	}
	t.Observability.ResetForEpoch()
	t.Observability.Log(vocab)
}

// ExportObservabilityMetrics exports all observability metrics as JSON.
func (t *Trainer) ExportObservabilityMetrics(vocab *vocab.Vocabulary) (string, error) {
	if !t.ObservabilityEnabled || t.Observability == nil {
		return "{}", nil
	}
	return t.Observability.ExportMetricsJSON(vocab)
}

// RecordTokenTrajectory records the full routing path of a token through expert layers.
func (t *Trainer) RecordTokenTrajectory(tokenID int, tokenStr string, expertPath []int, confidences []float32) {
	if !t.ObservabilityEnabled || t.Observability == nil {
		return
	}
	t.Observability.RecordTokenTrajectory(tokenID, tokenStr, expertPath, confidences)
}

// UpdateEmbeddingGalaxy updates the 2D PCA projection of the embedding space.
func (t *Trainer) UpdateEmbeddingGalaxy(vocab *vocab.Vocabulary, embeddingTensor *tensor.Tensor, topN int) {
	if !t.ObservabilityEnabled || t.Observability == nil {
		return
	}
	t.Observability.UpdateEmbeddingProjection(vocab, embeddingTensor, topN)
}

// RecordExpertForSimilarity records expert weights for redundancy detection.
func (t *Trainer) RecordExpertForSimilarity(expertID int, weights []float32) {
	if !t.ObservabilityEnabled || t.Observability == nil {
		return
	}
	t.Observability.RecordExpertWeights(expertID, weights)
}

// ComputeExpertRedundancy computes the expert similarity matrix and detects redundancy.
func (t *Trainer) ComputeExpertRedundancy(numExperts int) {
	if !t.ObservabilityEnabled || t.Observability == nil {
		return
	}
	t.Observability.ComputeExpertSimilarity()
}
