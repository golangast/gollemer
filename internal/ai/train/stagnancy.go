package train

import (
	"math"
	"math/rand"
	"time"
)

// StagnancyTracker keeps a moving average of weight gradients to identify "Timid" units.
type StagnancyTracker struct {
	Epsilon float32 // Threshold below which a weight is considered stagnant
}

// CalculateTimidMask identifies weights where the gradient velocity has dropped below Epsilon.
// Uses a simple moving average to avoid noise-driven triggers.
func (t *StagnancyTracker) CalculateTimidMask(accGrads []float32, currentGrads []float32) []bool {
	mask := make([]bool, len(currentGrads))

	// Help compiler with Bounds Check Elimination
	if len(accGrads) != len(currentGrads) {
		return mask
	}

	const alpha float32 = 0.01
	const invAlpha float32 = 1.0 - alpha

	for i := range currentGrads {
		// Update Moving Average of absolute gradients: acc = (acc * 0.99) + (|grad| * 0.01)
		g := float32(math.Abs(float64(currentGrads[i])))
		accGrads[i] = (accGrads[i] * invAlpha) + (g * alpha)

		// Set mask if the accumulated movement is below the threshold
		mask[i] = accGrads[i] < t.Epsilon
	}
	return mask
}

// PerturbStagnantWeights applies small Gaussian noise to weights identified as stagnant.
// This "shakes" the weights out of local minima where they might be stuck during a plateau.
func PerturbStagnantWeights(weights []float32, timidMask []bool, intensity float32) {
	if len(weights) != len(timidMask) {
		return
	}

	seed := time.Now().UnixNano()
	r := rand.New(rand.NewSource(seed))

	for i, isTimid := range timidMask {
		if isTimid {
			// Apply a small Gaussian nudge
			nudge := float32(r.NormFloat64()) * intensity
			weights[i] += nudge
		}
	}
}
