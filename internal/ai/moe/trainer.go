package moe

import (
	"log"
	"time"
	"github.com/golangast/gollemer/internal/ai/neural/nn"
)

type TrainingStats struct {
	Epoch           int
	CurrentLoss     float32
	Perplexity      float64
	BestPerplexity  float64
	Layer0Counts    []int   // SIMD-accumulated expert hits for L0
	MaxDominance    float32 // The utilization % of the most used expert
	StepConfidence  float32 // Average probability of the Top-1 token
}

type Trainer struct {
	BestModelPath string
	CollapseCount int
	LastSafeLR    float64
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
		if t.CollapseCount > 0 { t.CollapseCount-- }
	}

	// Level 2: Poisoned Gradients - Revert to Golden Checkpoint
	if t.CollapseCount >= 3 || (stats.BestPerplexity > 0 && stats.Perplexity > stats.BestPerplexity * 2.0) {
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
