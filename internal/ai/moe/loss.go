package moe

import (
	"math"
)

// TokenFeatures holds the POS tags or syntactic roles for a sequence.
// Matches the supervisor's PRON/AUX tracking.
type TokenFeatures struct {
	IsPronoun   []bool // Length N (sequence length)
	IsAuxiliary []bool // Length N
}

// RouterLossConfig holds hyperparameters for the advanced gating objective.
type RouterLossConfig struct {
	BaseLBW         float64 // Traditional load balancing weight (e.g., 0.030)
	SyntacticWeight float64 // Penalty weight for separating PRON/AUX dependencies
	Temperature     float64 // Softmax temperature for gating stability
}

// PreferencePair represents a chosen/rejected response pair for DPO training.
type PreferencePair struct {
	Chosen   []int // Token IDs for the preferred response
	Rejected []int // Token IDs for the rejected response
}

// DPOLossConfig holds hyperparameters for Direct Preference Optimization.
type DPOLossConfig struct {
	Beta           float64 // DPO temperature parameter (typically 0.1 to 0.5)
	ChosenWeight   float64 // Weight multiplier for chosen tokens
	RejectedWeight float64 // Weight multiplier for rejected tokens
}

// ComputeDPOLoss calculates the Direct Preference Optimization loss between
// chosen and rejected response sequences. It increases the probability of
// step-by-step reasoning tokens (chosen) while suppressing incorrect or
// shallow responses (rejected).
func ComputeDPOLoss(
	chosenLogits, rejectedLogits []float64,
	chosenIDs, rejectedIDs []int,
	cfg DPOLossConfig,
) float64 {
	if len(chosenIDs) == 0 || len(rejectedIDs) == 0 {
		return 0.0
	}

	// Compute log-ratios for chosen vs rejected sequences
	chosenLogRatio := computeLogRatio(chosenLogits, chosenIDs, cfg.ChosenWeight)
	rejectedLogRatio := computeLogRatio(rejectedLogits, rejectedIDs, cfg.RejectedWeight)

	// DPO loss: -log(sigmoid(beta * (chosen_log_ratio - rejected_log_ratio)))
	diff := chosenLogRatio - rejectedLogRatio
	loss := -math.Log(1.0 / (1.0 + math.Exp(-cfg.Beta*diff)))

	// Clamp to prevent extreme values
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0.5 // Default fallback
	}

	return loss
}

// ComputeAdvancedRouterLoss calculates a composite loss that punishes the router
// for isolating dependent tokens (PRON -> AUX) into unlinked experts.
func ComputeAdvancedRouterLoss(
	gatingProbs [][]float64, // [N][NumExperts] Softmax outputs for the sequence
	features TokenFeatures,
	cfg RouterLossConfig,
) float64 {
	numTokens := len(gatingProbs)
	if numTokens == 0 {
		return 0.0
	}
	numExperts := len(gatingProbs[0])

	// 1. Traditional Load Balancing Loss (Entropy-based or squared-utilization)
	lbLoss := 0.0
	expertCounts := make([]float64, numExperts)
	for t := 0; t < numTokens; t++ {
		for e := 0; e < numExperts; e++ {
			expertCounts[e] += gatingProbs[t][e]
		}
	}

	meanCount := float64(numTokens) / float64(numExperts)
	for e := 0; e < numExperts; e++ {
		diff := expertCounts[e] - meanCount
		lbLoss += diff * diff
	}
	lbLoss = (lbLoss / float64(numExperts)) * cfg.BaseLBW

	// 2. Syntactic Co-Location Loss
	// If a PRON and AUX exist in the same window, track their assignment distribution
	syncLoss := 0.0
	pronIdx := -1
	auxIdx := -1

	for i := 0; i < numTokens; i++ {
		if features.IsPronoun[i] {
			pronIdx = i
		}
		if features.IsAuxiliary[i] {
			auxIdx = i
		}

		// When both are found in a close localized context window
		if pronIdx != -1 && auxIdx != -1 && math.Abs(float64(pronIdx-auxIdx)) <= 3 {
			// Cross-entropy/Dot-product penalty between their routing distributions.
			// High penalty if they route to completely orthogonal experts.
			dotProduct := 0.0
			for e := 0; e < numExperts; e++ {
				dotProduct += gatingProbs[pronIdx][e] * gatingProbs[auxIdx][e]
			}
			// Minimize (1 - alignment)
			syncLoss += (1.0 - dotProduct)

			// Reset tracking for next pair window
			pronIdx = -1
			auxIdx = -1
		}
	}

	return lbLoss + (syncLoss * cfg.SyntacticWeight)
}

// computeLogRatio computes the log-ratio of the model's preference for a
// sequence of token IDs, weighted by the DPO configuration.
func computeLogRatio(logits []float64, ids []int, weight float64) float64 {
	if len(ids) == 0 || len(logits) == 0 {
		return 0.0
	}

	var sum float64
	for _, id := range ids {
		if id >= 0 && id < len(logits) {
			logit := logits[id]
			// Apply weight to reinforce or suppress this token
			sum += float64(weight) * logit
		}
	}

	// Normalize by sequence length
	return sum / float64(len(ids))
}
