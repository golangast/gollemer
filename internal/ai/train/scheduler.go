package train

import (
	"math"
)

// GetLR calculates the current learning rate based on a Cosine Decay with Warmup.
// This prevents the model from diverging early (warmup) and helps fine-tune weights
// toward the end of training (cosine decay), preventing the model from getting 
// stuck in local minima too early as often happens with step decay.
func GetLR(currentStep, totalSteps int, baseLR float32) float32 {
	// 2% of total steps for linear warmup (faster takeoff for smaller datasets)
	warmupSteps := totalSteps / 50 
	if warmupSteps == 0 { warmupSteps = 1 } // Safety floor
	
	if currentStep < warmupSteps {
		// Linear Warmup: Scale LR from 0 to baseLR
		return baseLR * float32(currentStep) / float32(warmupSteps)
	}
	
	// Cosine Decay: Scale LR from baseLR to 0 over the remaining steps
	if currentStep >= totalSteps {
		return 0.0 // Training finished
	}
	
	// Progress range: [0, 1] for the decay portion
	progress := float32(currentStep-warmupSteps) / float32(totalSteps-warmupSteps)
	
	// Cosine formula: 0.5 * baseLR * (1 + cos(pi * progress))
	return float32(0.5 * float64(baseLR) * (1.0 + math.Cos(math.Pi*float64(progress))))
}
