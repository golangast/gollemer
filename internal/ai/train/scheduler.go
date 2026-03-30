package train

import (
	"math"
)

// GetLR calculates the current learning rate based on a Cosine Decay with Warmup.
// This prevents the model from diverging early (warmup) and helps fine-tune weights
// toward the end of training (cosine decay), preventing the model from getting 
// stuck in local minima too early as often happens with step decay.
func GetLR(currentStep, totalSteps int, baseLR float64) float64 {
	// 10% of total steps for linear warmup
	warmupSteps := totalSteps / 10 
	if warmupSteps == 0 { warmupSteps = 1 } // Safety floor
	
	if currentStep < warmupSteps {
		// Linear Warmup: Scale LR from 0 to baseLR
		return baseLR * float64(currentStep) / float64(warmupSteps)
	}
	
	// Cosine Decay: Scale LR from baseLR to 0 over the remaining steps
	if currentStep >= totalSteps {
		return 0.0 // Training finished
	}
	
	// Progress range: [0, 1] for the decay portion
	progress := float64(currentStep-warmupSteps) / float64(totalSteps-warmupSteps)
	
	// Cosine formula: 0.5 * baseLR * (1 + cos(pi * progress))
	return 0.5 * baseLR * (1 + math.Cos(math.Pi*progress))
}
